"""Parse collection HTML and classify resource candidates and pagination links."""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import (
    unquote,
    urljoin,
    urlparse,
)

from ...models import (
    CollectionDiscoveryConfidence,
    CollectionDiscoveryContext,
    CollectionMembershipDecision,
    CollectionResourceKind,
    CollectionScopePolicy,
)
from ..acquisition import normalize_doi
from .urls import (
    CollectionError,
    canonical_origin,
    canonicalize_url,
)

DOI_DISCOVERY_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)


ARTICLE_PATH_MARKERS = ("/article/", "/doi/", "/paper/", "/publication/", "/content/")


NEXT_LABELS = {"next", "next page", "older", "more", "下一页", "下页"}


POSITIVE_MEMBERSHIP_CONTEXTS = frozenset(
    {
        "publication",
        "publications",
        "paper",
        "papers",
        "article",
        "articles",
        "result",
        "results",
        "research-output",
        "publication-item",
    }
)


REFERENCE_CONTEXTS = frozenset(
    {
        "references",
        "reference",
        "bibliography",
        "citations",
        "cited-by",
        "references-list",
    }
)


NAVIGATION_CONTEXTS = frozenset(
    {"footer", "navigation", "nav", "related-articles", "recommended"}
)


VOID_HTML_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)


CONTEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class _Link:
    href: str
    text: str
    rel: tuple[str, ...]
    media_type: str | None
    css_class: tuple[str, ...]
    context_tokens: frozenset[str]


@dataclass(frozen=True)
class _TextFragment:
    text: str
    context_tokens: frozenset[str]


class _CollectionHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[_Link] = []
        self.metadata: list[tuple[str, str]] = []
        self._active_link: dict[str, object] | None = None
        self._link_text: list[str] = []
        self.script_count = 0
        self.text_fragments: list[_TextFragment] = []
        self._hidden_depth = 0
        self._context_stack: list[tuple[str, frozenset[str]]] = []

    @staticmethod
    def _element_context(tag: str, attrs: dict[str, str]) -> frozenset[str]:
        values = [tag]
        values.extend(
            attrs.get(key, "")
            for key in ("id", "class", "role", "aria-label", "name")
        )
        tokens: set[str] = set()
        for value in values:
            normalized = value.casefold().replace("_", "-")
            for match in CONTEXT_TOKEN_PATTERN.findall(normalized):
                tokens.add(match)
                tokens.update(part for part in match.split("-") if part)
        return frozenset(tokens)

    def _current_context(self) -> frozenset[str]:
        return frozenset(
            token for _tag, context in self._context_stack for token in context
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value or "" for key, value in attrs}
        normalized_tag = tag.casefold()
        element_context = self._element_context(normalized_tag, values)
        full_context = self._current_context() | element_context
        if normalized_tag == "a" and values.get("href"):
            self._active_link = {**values, "context_tokens": full_context}
            self._link_text = []
        elif normalized_tag == "link" and values.get("href"):
            self._append_link({**values, "context_tokens": full_context}, "")
        elif normalized_tag == "meta":
            key = (values.get("name") or values.get("property") or "").casefold()
            value = values.get("content", "").strip()
            if key and value:
                self.metadata.append((key, value))
        if normalized_tag in {"script", "style"}:
            self._hidden_depth += 1
            if normalized_tag == "script":
                self.script_count += 1
        if normalized_tag not in VOID_HTML_TAGS:
            self._context_stack.append((normalized_tag, element_context))

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() not in VOID_HTML_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag == "a" and self._active_link is not None:
            self._append_link(self._active_link, " ".join(self._link_text))
            self._active_link = None
            self._link_text = []
        if normalized_tag in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1
        for index in range(len(self._context_stack) - 1, -1, -1):
            if self._context_stack[index][0] == normalized_tag:
                del self._context_stack[index:]
                break

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._active_link is not None:
            self._link_text.append(text)
        if not self._hidden_depth:
            self.text_fragments.append(_TextFragment(text, self._current_context()))

    def _append_link(self, attrs: dict[str, object], text: str) -> None:
        rel = tuple(sorted(set(str(attrs.get("rel", "")).casefold().split())))
        css_class = tuple(sorted(set(str(attrs.get("class", "")).casefold().split())))
        self.links.append(
            _Link(
                href=str(attrs["href"]).strip(),
                text=" ".join(text.casefold().split()),
                rel=rel,
                media_type=str(attrs.get("type", "")).casefold() or None,
                css_class=css_class,
                context_tokens=frozenset(attrs.get("context_tokens", ())),
            )
        )


@dataclass(frozen=True)
class _Candidate:
    kind: CollectionResourceKind
    original_identifier: str
    normalized_identifier: str
    discovery_method: str
    discovery_confidence: CollectionDiscoveryConfidence
    discovery_context: CollectionDiscoveryContext
    membership_eligible: bool
    membership_decision: CollectionMembershipDecision
    membership_reason: str
    doi: str | None = None
    url: str | None = None
    media_type: str | None = None

    @property
    def key(self) -> tuple[int, str]:
        priority = {
            CollectionResourceKind.DOI: 0,
            CollectionResourceKind.ARTICLE_URL: 1,
            CollectionResourceKind.PDF_URL: 2,
            CollectionResourceKind.METADATA_RECORD: 3,
        }[self.kind]
        return priority, self.normalized_identifier


def _clean_doi_candidate(value: str) -> str | None:
    candidate = unquote(value).rstrip(".,;:)]}")
    try:
        return normalize_doi(candidate)
    except ValueError:
        return None


def _doi_inside(value: str) -> str | None:
    match = DOI_DISCOVERY_PATTERN.search(unquote(value))
    return _clean_doi_candidate(match.group(0)) if match else None


def _resource_candidate(
    *,
    kind: CollectionResourceKind,
    original: str,
    method: str,
    base_url: str,
    confidence: CollectionDiscoveryConfidence,
    context: CollectionDiscoveryContext,
    membership_eligible: bool,
    membership_decision: CollectionMembershipDecision,
    membership_reason: str,
    media_type: str | None = None,
) -> _Candidate | None:
    if kind == CollectionResourceKind.DOI:
        doi = _clean_doi_candidate(original)
        if doi is None:
            return None
        return _Candidate(
            kind,
            original,
            doi,
            method,
            confidence,
            context,
            membership_eligible,
            membership_decision,
            membership_reason,
            doi=doi,
        )
    absolute = canonicalize_url(urljoin(base_url, original))
    embedded_doi = _doi_inside(absolute)
    if embedded_doi is not None:
        return _Candidate(
            CollectionResourceKind.DOI,
            original,
            embedded_doi,
            f"{method}:embedded_doi",
            confidence,
            context,
            membership_eligible,
            membership_decision,
            membership_reason,
            doi=embedded_doi,
        )
    return _Candidate(
        kind,
        original,
        absolute,
        method,
        confidence,
        context,
        membership_eligible,
        membership_decision,
        membership_reason,
        url=absolute,
        media_type=media_type,
    )


def _membership_context(
    context_tokens: frozenset[str],
) -> tuple[
    CollectionMembershipDecision,
    str,
    CollectionDiscoveryContext,
    bool,
]:
    if context_tokens & REFERENCE_CONTEXTS:
        return (
            CollectionMembershipDecision.EXCLUDED_REFERENCE_CONTEXT,
            "negative reference or bibliography context",
            CollectionDiscoveryContext.REFERENCE_CONTEXT,
            False,
        )
    if context_tokens & NAVIGATION_CONTEXTS:
        return (
            CollectionMembershipDecision.EXCLUDED_NAVIGATION_CONTEXT,
            "negative navigation, related-content, or footer context",
            CollectionDiscoveryContext.NAVIGATION_CONTEXT,
            False,
        )
    if context_tokens & POSITIVE_MEMBERSHIP_CONTEXTS:
        return (
            CollectionMembershipDecision.ELIGIBLE,
            "local publication-member context",
            CollectionDiscoveryContext.PUBLICATION_MEMBER_CONTEXT,
            True,
        )
    return (
        CollectionMembershipDecision.UNVERIFIED,
        "no local collection-membership context",
        CollectionDiscoveryContext.UNVERIFIED_CONTEXT,
        False,
    )


def _extract_page(
    body: bytes,
    page_url: str,
    policy: CollectionScopePolicy,
) -> tuple[tuple[_Candidate, ...], tuple[str, ...], bool, bool]:
    try:
        html = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CollectionError("collection page is not valid UTF-8") from error
    parser = _CollectionHTMLParser()
    parser.feed(html)
    candidates: list[_Candidate] = []
    for fragment in parser.text_fragments:
        decision, reason, context, membership_eligible = _membership_context(
            fragment.context_tokens
        )
        for match in DOI_DISCOVERY_PATTERN.finditer(fragment.text):
            candidate = _resource_candidate(
                kind=CollectionResourceKind.DOI,
                original=match.group(0),
                method="doi_text",
                base_url=page_url,
                confidence=(
                    CollectionDiscoveryConfidence.MEDIUM
                    if membership_eligible
                    else CollectionDiscoveryConfidence.LOW
                ),
                context=context,
                membership_eligible=membership_eligible,
                membership_decision=decision,
                membership_reason=reason,
            )
            if candidate is not None:
                candidates.append(candidate)
    metadata_article_keys = {
        "citation_fulltext_html_url",
        "citation_abstract_html_url",
    }
    metadata_dois = tuple(
        sorted(
            {
                doi
                for key, value in parser.metadata
                if key in {"citation_doi", "dc.identifier", "prism.doi"}
                for doi in (_clean_doi_candidate(value),)
                if doi is not None
            }
        )
    )
    for key, value in parser.metadata:
        kind: CollectionResourceKind | None = None
        media_type: str | None = None
        if key in {"citation_doi", "dc.identifier", "prism.doi"}:
            kind = CollectionResourceKind.DOI
        elif key == "citation_pdf_url":
            kind = CollectionResourceKind.PDF_URL
            media_type = "application/pdf"
        elif key in metadata_article_keys:
            kind = CollectionResourceKind.ARTICLE_URL
            media_type = "text/html"
        if kind is not None:
            if kind != CollectionResourceKind.DOI and len(metadata_dois) == 1:
                candidate = _Candidate(
                    CollectionResourceKind.DOI,
                    value,
                    metadata_dois[0],
                    f"metadata:{key}:associated_doi",
                    CollectionDiscoveryConfidence.HIGH,
                    CollectionDiscoveryContext.CITATION_METADATA,
                    True,
                    CollectionMembershipDecision.ELIGIBLE,
                    "citation metadata identifies the page's primary work",
                    doi=metadata_dois[0],
                )
            else:
                candidate = _resource_candidate(
                    kind=kind,
                    original=value,
                    method=f"metadata:{key}",
                    base_url=page_url,
                    confidence=CollectionDiscoveryConfidence.HIGH,
                    context=CollectionDiscoveryContext.CITATION_METADATA,
                    membership_eligible=True,
                    membership_decision=CollectionMembershipDecision.ELIGIBLE,
                    membership_reason="citation metadata identifies the page's primary work",
                    media_type=media_type,
                )
            if candidate is not None:
                candidates.append(candidate)
    pagination: set[str] = set()
    unresolved_pagination = False
    for link in parser.links:
        is_next = (
            "next" in link.rel
            or link.text in NEXT_LABELS
            or "next" in link.css_class
        )
        try:
            absolute = canonicalize_url(urljoin(page_url, link.href))
        except CollectionError:
            if is_next:
                unresolved_pagination = True
            continue
        if is_next:
            pagination.add(absolute)
            continue
        parsed = urlparse(absolute)
        path = parsed.path.casefold()
        kind: CollectionResourceKind | None = None
        media_type: str | None = None
        method = "link"
        if _doi_inside(link.href):
            kind = CollectionResourceKind.DOI
            base_confidence = CollectionDiscoveryConfidence.HIGH
            method = "explicit_doi_link"
        elif link.media_type == "application/pdf" or path.endswith(".pdf"):
            kind = CollectionResourceKind.PDF_URL
            media_type = "application/pdf"
            method = "direct_pdf_link"
            base_confidence = CollectionDiscoveryConfidence.HIGH
        elif "article" in link.rel or any(marker in path for marker in ARTICLE_PATH_MARKERS):
            kind = CollectionResourceKind.ARTICLE_URL
            media_type = "text/html"
            method = "article_link"
            base_confidence = CollectionDiscoveryConfidence.MEDIUM
        if kind is None:
            continue
        if (
            kind != CollectionResourceKind.DOI
            and canonical_origin(absolute) != canonical_origin(page_url)
            and not policy.allow_external_literature_links
        ):
            continue
        decision, reason, context, membership_eligible = _membership_context(
            link.context_tokens
        )
        candidate = _resource_candidate(
            kind=kind,
            original=link.href,
            method=method,
            base_url=page_url,
            confidence=(
                base_confidence
                if membership_eligible
                else CollectionDiscoveryConfidence.LOW
            ),
            context=context,
            membership_eligible=membership_eligible,
            membership_decision=decision,
            membership_reason=reason,
            media_type=media_type,
        )
        if candidate is not None:
            candidates.append(candidate)
    visible = " ".join(fragment.text for fragment in parser.text_fragments).casefold()
    dynamic_only = (
        parser.script_count > 0
        and not candidates
        and not pagination
        and any(marker in visible for marker in ("enable javascript", "requires javascript", "javascript required"))
    )
    return (
        tuple(candidates),
        tuple(sorted(pagination)),
        dynamic_only,
        unresolved_pagination,
    )
