from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import posixpath
import re
from typing import Callable, Protocol
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlparse, urlunparse

from pydantic import ValidationError

from ..immutable import FrozenDict
from ..models import (
    AcquisitionStatus,
    CollectionAcquisitionLink,
    CollectionCompletenessStatus,
    CollectionDefinition,
    CollectionDiff,
    CollectionDiscoveryConfidence,
    CollectionDiscoveryContext,
    CollectionDiscoveryResult,
    CollectionImportOutcome,
    CollectionImportRecord,
    CollectionImportStatus,
    CollectionLiteratureMembership,
    CollectionMembershipDecision,
    CollectionPageRecord,
    CollectionResourceKind,
    CollectionResourceOccurrence,
    CollectionResourceImportResult,
    CollectionScopePolicy,
    CollectionSnapshot,
    CollectionSourceKind,
    CollectionTraversalStatus,
    DiscoveredCollectionResource,
)
from ..repositories import EvidenceStore, KnowledgeRepositories
from ..serialization import content_hash
from .acquisition import (
    SUPPORTED_HTML_TYPES,
    AcquisitionError,
    LiteratureAcquisitionService,
    SafeHTTPError,
    SafeHTTPFetcher,
    normalize_doi,
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


class CollectionError(ValueError):
    pass


def canonicalize_url(value: str) -> str:
    parsed = urlparse(value.strip())
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise CollectionError("collection URLs must use HTTP/HTTPS and include a host")
    host = parsed.hostname.rstrip(".").casefold()
    try:
        port = parsed.port
    except ValueError as error:
        raise CollectionError("collection URL port is invalid") from error
    default_port = 443 if scheme == "https" else 80
    netloc = host if port in {None, default_port} else f"{host}:{port}"
    decoded_path = unquote(parsed.path or "/")
    normalized_path = posixpath.normpath(decoded_path)
    if decoded_path.endswith("/") and normalized_path != "/":
        normalized_path += "/"
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    path = quote(normalized_path, safe="/%:@+~!$&'()*,-.;=")
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)), doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def canonical_origin(value: str) -> str:
    parsed = urlparse(canonicalize_url(value))
    return f"{parsed.scheme}://{parsed.netloc}"


def _canonical_path_prefix(value: str) -> str:
    path = unquote(value.strip())
    if not path.startswith("/"):
        path = f"/{path}"
    normalized = posixpath.normpath(path)
    if path.endswith("/") and normalized != "/":
        normalized += "/"
    return normalized


def _default_path_prefix(entry_path: str) -> str:
    canonical = _canonical_path_prefix(entry_path or "/")
    if canonical == "/" or canonical.endswith("/"):
        return canonical
    leaf = posixpath.basename(canonical)
    if "." in leaf:
        parent = posixpath.dirname(canonical) or "/"
        return parent if parent == "/" else f"{parent}/"
    return canonical


def path_matches_prefix(path: str, prefix: str) -> bool:
    candidate = _canonical_path_prefix(path)
    boundary = _canonical_path_prefix(prefix)
    if boundary == "/":
        return True
    base = boundary.rstrip("/")
    return candidate == base or candidate.startswith(f"{base}/")


def make_collection_scope_policy(
    entry_url: str,
    *,
    max_pages: int = 100,
    max_depth: int = 10,
    max_resources: int = 1000,
    allow_external_literature_links: bool = False,
    allowed_origins: tuple[str, ...] | None = None,
    allowed_path_prefixes: tuple[str, ...] | None = None,
) -> CollectionScopePolicy:
    canonical_entry = canonicalize_url(entry_url)
    parsed = urlparse(canonical_entry)
    origin_values = tuple(
        sorted({canonical_origin(item) for item in (allowed_origins or (canonical_entry,))})
    )
    prefix_values = allowed_path_prefixes or (_default_path_prefix(parsed.path),)
    prefixes = tuple(sorted({_canonical_path_prefix(item) for item in prefix_values}))
    identity = {
        "allowed_origins": origin_values,
        "allowed_path_prefixes": prefixes,
        "allow_external_literature_links": allow_external_literature_links,
        "max_pages": max_pages,
        "max_depth": max_depth,
        "max_resources": max_resources,
        "pagination_policy": "explicit_next_only",
        "content_types": tuple(sorted(SUPPORTED_HTML_TYPES)),
    }
    return CollectionScopePolicy(**identity, content_hash=content_hash(identity))


def make_collection_definition(
    entry_url: str,
    domain: str,
    *,
    name: str | None = None,
    source_kind: CollectionSourceKind = CollectionSourceKind.COLLECTION_URL,
    connector_id: str = "generic-html-collection",
    connector_version: str = "1.1.0",
    connector_completeness_contract: str = "explicit-pagination-terminal-v1",
    scope_policy: CollectionScopePolicy | None = None,
) -> CollectionDefinition:
    canonical_entry = canonicalize_url(entry_url)
    policy = scope_policy or make_collection_scope_policy(canonical_entry)
    identity = {
        "source_kind": source_kind,
        "entry_url": canonical_entry,
        "name": name or canonical_entry,
        "domain": domain,
        "connector_id": connector_id,
        "connector_version": connector_version,
        "connector_completeness_contract": connector_completeness_contract,
        "scope_policy": policy,
    }
    collection_id = f"literature-collection-{content_hash(identity)[:24]}"
    payload = {"collection_id": collection_id, **identity}
    return CollectionDefinition(**payload, content_hash=content_hash(payload))


class CollectionConnector(Protocol):
    connector_id: str
    connector_version: str

    def discover(
        self,
        definition: CollectionDefinition,
        repositories: KnowledgeRepositories,
        fetcher: SafeHTTPFetcher,
    ) -> CollectionDiscoveryResult:
        ...


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


def _make_bound(model_type, prefix: str, identity: dict):
    identity = {key: value for key, value in identity.items() if value is not None}
    record_id = f"{prefix}-{content_hash(identity)[:24]}"
    id_field = {
        "collection-page": "page_id",
        "collection-resource": "discovered_resource_id",
        "collection-occurrence": "occurrence_id",
        "collection-snapshot": "snapshot_id",
        "collection-acquisition-link": "link_id",
        "collection-resource-import": "result_id",
        "collection-membership": "membership_id",
        "collection-import": "import_id",
        "collection-diff": "diff_id",
    }[prefix]
    payload = {id_field: record_id, **identity}
    return model_type(**payload, content_hash=content_hash(payload))


class GenericHTMLCollectionConnector:
    connector_id = "generic-html-collection"
    connector_version = "1.1.0"
    completeness_contract = "explicit-pagination-terminal-v1"

    def discover(
        self,
        definition: CollectionDefinition,
        repositories: KnowledgeRepositories,
        fetcher: SafeHTTPFetcher,
    ) -> CollectionDiscoveryResult:
        if (
            definition.connector_id != self.connector_id
            or definition.connector_version != self.connector_version
            or definition.connector_completeness_contract
            != self.completeness_contract
        ):
            raise CollectionError("collection definition does not bind this connector")
        if definition.source_kind == CollectionSourceKind.LOCAL_MANIFEST:
            raise CollectionError("GenericHTMLCollectionConnector does not read local manifests")
        repositories.collection_definitions.put(definition.collection_id, definition)
        policy = definition.scope_policy
        queue: list[tuple[str, int]] = [(canonicalize_url(definition.entry_url), 0)]
        queued = {queue[0][0]}
        visited: set[str] = set()
        pages: list[CollectionPageRecord] = []
        resources: dict[tuple[int, str], DiscoveredCollectionResource] = {}
        occurrences: list[CollectionResourceOccurrence] = []
        reasons: list[str] = []
        authentication_blocked = False
        resource_limit_reached = False
        saw_explicit_pagination = False
        while queue and len(pages) < policy.max_pages:
            requested_url, depth = queue.pop(0)
            queued.discard(requested_url)
            if requested_url in visited:
                if "pagination_cycle_detected" not in reasons:
                    reasons.append("pagination_cycle_detected")
                continue
            visited.add(requested_url)
            if not self._page_in_scope(requested_url, policy):
                reasons.append("unresolved_pagination_outside_scope")
                continue
            try:
                response = fetcher.fetch(
                    requested_url,
                    allowed_content_types=frozenset(policy.content_types),
                )
            except SafeHTTPError as error:
                authentication_blocked = error.code == "REQUIRES_AUTHENTICATION"
                reason = (
                    "collection_requires_authentication"
                    if authentication_blocked
                    else f"page_retrieval_failed:{error.code}"
                )
                reasons.append(reason)
                page = self._page_record(
                    definition,
                    requested_url=requested_url,
                    resolved_url=requested_url,
                    http_status=error.status or 0,
                    media_type="unavailable",
                    response_sha256=hashlib.sha256(b"").hexdigest(),
                    resource_ids=(),
                    pagination=(),
                )
                repositories.collection_pages.put(page.page_id, page)
                pages.append(page)
                continue
            try:
                candidates, pagination, dynamic_only, unresolved_pagination = _extract_page(
                    response.body, response.url, policy
                )
            except CollectionError:
                candidates, pagination, dynamic_only, unresolved_pagination = (), (), False, False
                reasons.append("unsupported_page_encoding")
            if pagination:
                saw_explicit_pagination = True
            page_resource_ids: list[str] = []
            accepted_candidates: list[_Candidate] = []
            for candidate in candidates:
                if candidate.key not in resources and len(resources) >= policy.max_resources:
                    resource_limit_reached = True
                    continue
                resource = resources.get(candidate.key)
                if resource is None:
                    stable_identity = {
                        "collection_id": definition.collection_id,
                        "resource_kind": candidate.kind,
                        "normalized_identifier": candidate.normalized_identifier,
                    }
                    resource_id = f"collection-resource-{content_hash(stable_identity)[:24]}"
                    payload = {
                        "discovered_resource_id": resource_id,
                        "collection_id": definition.collection_id,
                        "resource_kind": candidate.kind,
                        "normalized_identifier": candidate.normalized_identifier,
                        "doi": candidate.doi,
                        "url": candidate.url,
                        "media_type": candidate.media_type,
                        "highest_discovery_confidence": candidate.discovery_confidence,
                        "membership_eligible": candidate.membership_eligible,
                        "membership_decision": candidate.membership_decision,
                        "membership_reason": candidate.membership_reason,
                    }
                    payload = {key: value for key, value in payload.items() if value is not None}
                    resource = DiscoveredCollectionResource(
                        **payload, content_hash=content_hash(payload)
                    )
                    resources[candidate.key] = resource
                else:
                    confidence_rank = {
                        CollectionDiscoveryConfidence.LOW: 0,
                        CollectionDiscoveryConfidence.MEDIUM: 1,
                        CollectionDiscoveryConfidence.HIGH: 2,
                    }
                    highest = (
                        candidate.discovery_confidence
                        if confidence_rank[candidate.discovery_confidence]
                        > confidence_rank[resource.highest_discovery_confidence]
                        else resource.highest_discovery_confidence
                    )
                    membership_eligible = (
                        resource.membership_eligible or candidate.membership_eligible
                    )
                    decision_rank = {
                        CollectionMembershipDecision.EXCLUDED_NAVIGATION_CONTEXT: 0,
                        CollectionMembershipDecision.EXCLUDED_REFERENCE_CONTEXT: 0,
                        CollectionMembershipDecision.UNVERIFIED: 1,
                        CollectionMembershipDecision.ELIGIBLE: 2,
                    }
                    current_decision = resource.membership_decision
                    candidate_wins = decision_rank[candidate.membership_decision] > (
                        decision_rank[current_decision]
                    )
                    membership_decision = (
                        candidate.membership_decision
                        if candidate_wins
                        else current_decision
                    )
                    membership_reason = (
                        candidate.membership_reason
                        if candidate_wins
                        else resource.membership_reason
                    )
                    if (
                        highest != resource.highest_discovery_confidence
                        or membership_eligible != resource.membership_eligible
                        or membership_decision != resource.membership_decision
                        or membership_reason != resource.membership_reason
                    ):
                        payload = resource.model_dump(
                            mode="json", exclude={"content_hash"}, exclude_none=True
                        )
                        payload.update(
                            highest_discovery_confidence=highest,
                            membership_eligible=membership_eligible,
                            membership_decision=membership_decision,
                            membership_reason=membership_reason,
                        )
                        resource = DiscoveredCollectionResource(
                            **payload, content_hash=content_hash(payload)
                        )
                        resources[candidate.key] = resource
                page_resource_ids.append(resource.discovered_resource_id)
                accepted_candidates.append(candidate)
            if dynamic_only:
                reasons.append("unsupported_dynamic_collection")
            if unresolved_pagination:
                reasons.append("unresolved_pagination")
            valid_pagination: list[str] = []
            for next_url in pagination:
                if not self._page_in_scope(next_url, policy):
                    reasons.append("unresolved_pagination_outside_scope")
                    continue
                if next_url in visited or next_url in queued:
                    reasons.append("pagination_cycle_detected")
                    continue
                if depth >= policy.max_depth:
                    reasons.append("max_depth_reached")
                    continue
                valid_pagination.append(next_url)
                queue.append((next_url, depth + 1))
                queued.add(next_url)
            queue.sort(key=lambda item: (item[1], item[0]))
            media_type = (
                response.headers.get("content-type", "").split(";", 1)[0]
                or "unknown"
            )
            page_artifact = repositories.collection_page_artifacts.put(
                response.body,
                collection_id=definition.collection_id,
                requested_url=canonicalize_url(requested_url),
                resolved_url=canonicalize_url(response.url),
                media_type=media_type,
            )
            page = self._page_record(
                definition,
                requested_url=requested_url,
                resolved_url=response.url,
                http_status=response.status,
                media_type=media_type,
                response_sha256=hashlib.sha256(response.body).hexdigest(),
                page_artifact_id=page_artifact.artifact_id,
                page_artifact_hash=page_artifact.content_hash,
                resource_ids=tuple(sorted(set(page_resource_ids))),
                pagination=tuple(sorted(set(pagination))),
            )
            repositories.collection_pages.put(page.page_id, page)
            pages.append(page)
            for index, (candidate, resource_id) in enumerate(
                zip(accepted_candidates, page_resource_ids)
            ):
                occurrence = _make_bound(
                    CollectionResourceOccurrence,
                    "collection-occurrence",
                    {
                        "collection_id": definition.collection_id,
                        "discovered_resource_id": resource_id,
                        "discovery_page_id": page.page_id,
                        "discovery_index": index,
                        "original_identifier": candidate.original_identifier,
                        "discovery_method": candidate.discovery_method,
                        "discovery_confidence": candidate.discovery_confidence,
                        "discovery_context": candidate.discovery_context,
                        "membership_eligible": candidate.membership_eligible,
                        "membership_decision": candidate.membership_decision,
                        "membership_reason": candidate.membership_reason,
                    },
                )
                repositories.collection_occurrences.put(occurrence.occurrence_id, occurrence)
                occurrences.append(occurrence)
        if queue:
            reasons.append("max_pages_reached")
        if resource_limit_reached:
            reasons.append("max_resources_reached")
        unique_reasons = tuple(dict.fromkeys(reasons))
        successful_pages = sum(200 <= page.http_status < 300 for page in pages)
        if authentication_blocked and successful_pages == 0:
            traversal_status = CollectionTraversalStatus.REQUIRES_AUTHENTICATION
            completeness_status = CollectionCompletenessStatus.REQUIRES_AUTHENTICATION
            completeness_basis = ("authentication_prevented_scope_exhaustion",)
        elif successful_pages == 0 and unique_reasons:
            traversal_status = CollectionTraversalStatus.FAILED
            completeness_status = CollectionCompletenessStatus.FAILED
            completeness_basis = ("no_collection_page_was_retrieved",)
        elif "unsupported_dynamic_collection" in unique_reasons:
            traversal_status = CollectionTraversalStatus.UNSUPPORTED_DYNAMIC
            completeness_status = CollectionCompletenessStatus.UNSUPPORTED_DYNAMIC
            completeness_basis = ("generic_connector_cannot_execute_dynamic_continuation",)
        elif unique_reasons:
            traversal_status = CollectionTraversalStatus.LIMITED
            completeness_status = CollectionCompletenessStatus.PARTIAL
            completeness_basis = tuple(f"scope_not_exhausted:{reason}" for reason in unique_reasons)
        elif saw_explicit_pagination:
            traversal_status = CollectionTraversalStatus.EXHAUSTED
            completeness_status = CollectionCompletenessStatus.PROVEN_COMPLETE
            completeness_basis = ("explicit_pagination_chain_reached_terminal_page",)
        else:
            traversal_status = CollectionTraversalStatus.EXHAUSTED
            completeness_status = CollectionCompletenessStatus.POLICY_EXHAUSTED_UNVERIFIED
            completeness_basis = ("no_explicit_completeness_evidence",)
            unique_reasons = ("collection_completeness_unverified",)
        completeness = {
            CollectionCompletenessStatus.PROVEN_COMPLETE: CollectionImportStatus.COMPLETE,
            CollectionCompletenessStatus.REQUIRES_AUTHENTICATION: CollectionImportStatus.REQUIRES_AUTHENTICATION,
            CollectionCompletenessStatus.FAILED: CollectionImportStatus.FAILED,
        }.get(completeness_status, CollectionImportStatus.PARTIAL)
        ordered_pages = tuple(sorted(pages, key=lambda item: item.page_id))
        ordered_resources = tuple(
            sorted(resources.values(), key=lambda item: item.discovered_resource_id)
        )
        for resource in ordered_resources:
            repositories.collection_resources.put(
                resource.discovered_resource_id, resource
            )
        ordered_occurrences = tuple(
            sorted(occurrences, key=lambda item: item.occurrence_id)
        )
        snapshot = _make_bound(
            CollectionSnapshot,
            "collection-snapshot",
            {
                "collection_id": definition.collection_id,
                "connector_id": self.connector_id,
                "connector_version": self.connector_version,
                "connector_completeness_contract": self.completeness_contract,
                "scope_policy_hash": policy.content_hash,
                "page_record_hashes": FrozenDict(
                    {item.page_id: item.content_hash for item in ordered_pages}
                ),
                "discovered_resource_hashes": FrozenDict(
                    {
                        item.discovered_resource_id: item.content_hash
                        for item in ordered_resources
                    }
                ),
                "resource_provenance_hashes": FrozenDict(
                    {
                        resource.discovered_resource_id: content_hash(
                            {
                                "resource_hash": resource.content_hash,
                                "occurrence_hashes": tuple(
                                    sorted(
                                        occurrence.content_hash
                                        for occurrence in ordered_occurrences
                                        if occurrence.discovered_resource_id
                                        == resource.discovered_resource_id
                                    )
                                ),
                            }
                        )
                        for resource in ordered_resources
                    }
                ),
                "occurrence_hashes": FrozenDict(
                    {item.occurrence_id: item.content_hash for item in ordered_occurrences}
                ),
                "visited_page_count": len(ordered_pages),
                "unique_resource_count": len(ordered_resources),
                "traversal_status": traversal_status,
                "completeness_status": completeness_status,
                "completeness_basis": completeness_basis,
                "completeness": completeness,
                "completeness_reasons": unique_reasons,
            },
        )
        repositories.collection_snapshots.put(snapshot.snapshot_id, snapshot)
        return CollectionDiscoveryResult(
            definition=definition,
            pages=ordered_pages,
            resources=ordered_resources,
            occurrences=ordered_occurrences,
            snapshot=snapshot,
        )

    @staticmethod
    def _page_in_scope(url: str, policy: CollectionScopePolicy) -> bool:
        parsed = urlparse(canonicalize_url(url))
        return (
            canonical_origin(url) in set(policy.allowed_origins)
            and any(
                path_matches_prefix(parsed.path, prefix)
                for prefix in policy.allowed_path_prefixes
            )
        )

    @staticmethod
    def _page_record(
        definition: CollectionDefinition,
        *,
        requested_url: str,
        resolved_url: str,
        http_status: int,
        media_type: str,
        response_sha256: str,
        page_artifact_id: str | None = None,
        page_artifact_hash: str | None = None,
        resource_ids: tuple[str, ...],
        pagination: tuple[str, ...],
    ) -> CollectionPageRecord:
        return _make_bound(
            CollectionPageRecord,
            "collection-page",
            {
                "collection_id": definition.collection_id,
                "requested_url": canonicalize_url(requested_url),
                "resolved_url": canonicalize_url(resolved_url),
                "http_status": http_status,
                "media_type": media_type,
                "response_sha256": response_sha256,
                "page_artifact_id": page_artifact_id,
                "page_artifact_hash": page_artifact_hash,
                "connector_id": definition.connector_id,
                "connector_version": definition.connector_version,
                "discovered_resource_refs": tuple(sorted(set(resource_ids))),
                "pagination_refs": tuple(sorted(set(pagination))),
            },
        )


def diff_collection_snapshots(
    old: CollectionSnapshot,
    new: CollectionSnapshot,
) -> CollectionDiff:
    if old.collection_id != new.collection_id:
        raise CollectionError("collection snapshots belong to different collections")
    old_ids = set(old.discovered_resource_hashes)
    new_ids = set(new.discovered_resource_hashes)
    shared_ids = old_ids & new_ids
    changed_ids = {
        resource_id
        for resource_id in shared_ids
        if (
            old.discovered_resource_hashes[resource_id]
            != new.discovered_resource_hashes[resource_id]
            or old.resource_provenance_hashes.get(
                resource_id, old.discovered_resource_hashes[resource_id]
            )
            != new.resource_provenance_hashes.get(
                resource_id, new.discovered_resource_hashes[resource_id]
            )
        )
    }
    return _make_bound(
        CollectionDiff,
        "collection-diff",
        {
            "old_snapshot_id": old.snapshot_id,
            "old_snapshot_hash": old.content_hash,
            "new_snapshot_id": new.snapshot_id,
            "new_snapshot_hash": new.content_hash,
            "added_resource_ids": tuple(sorted(new_ids - old_ids)),
            "removed_resource_ids": tuple(sorted(old_ids - new_ids)),
            "changed_resource_ids": tuple(sorted(changed_ids)),
            "unchanged_resource_ids": tuple(sorted(shared_ids - changed_ids)),
        },
    )


EXPECTED_COLLECTION_ACQUISITION_EXCEPTIONS = (
    AcquisitionError,
    FileExistsError,
    OSError,
    ValidationError,
)


class CollectionImportService:
    def __init__(
        self,
        *,
        fetcher: SafeHTTPFetcher | None = None,
        connector: CollectionConnector | None = None,
        acquisition_service: LiteratureAcquisitionService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.fetcher = fetcher or SafeHTTPFetcher()
        self.connector = connector or GenericHTMLCollectionConnector()
        self.acquisition_service = acquisition_service or LiteratureAcquisitionService(
            self.fetcher
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def run(
        self,
        definition: CollectionDefinition,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        *,
        previous_snapshot: CollectionSnapshot | None = None,
    ) -> CollectionImportOutcome:
        discovery = self.connector.discover(definition, repositories, self.fetcher)
        links: list[CollectionAcquisitionLink] = []
        results: list[CollectionResourceImportResult] = []
        warnings = list(discovery.snapshot.completeness_reasons)
        counts = {
            "ingested": 0,
            "metadata_only": 0,
            "auth": 0,
            "unavailable": 0,
            "failed": 0,
            "unverified": 0,
        }
        for resource in discovery.resources:
            if not resource.membership_eligible:
                result = self._failed_resource_result(
                    definition,
                    discovery.snapshot,
                    resource,
                    status=AcquisitionStatus.DISCOVERED,
                    failure_code="UNVERIFIED_COLLECTION_MEMBERSHIP",
                )
                repositories.collection_resource_import_results.put(
                    result.result_id, result
                )
                results.append(result)
                counts["unverified"] += 1
                continue
            source = resource.doi or resource.url
            if source is None:
                result = self._failed_resource_result(
                    definition,
                    discovery.snapshot,
                    resource,
                    status=AcquisitionStatus.FAILED,
                    failure_code="MISSING_ACQUISITION_INPUT",
                )
                repositories.collection_resource_import_results.put(
                    result.result_id, result
                )
                results.append(result)
                counts["failed"] += 1
                continue
            try:
                outcome = self.acquisition_service.add(
                    source,
                    definition.domain,
                    repositories,
                    evidence_store,
                )
                acquisition = repositories.literature_acquisitions.get(
                    outcome.acquisition_id
                )
            except EXPECTED_COLLECTION_ACQUISITION_EXCEPTIONS as error:
                failure_code = f"EXPECTED_ACQUISITION_FAILURE:{type(error).__name__}"
                result = self._failed_resource_result(
                    definition,
                    discovery.snapshot,
                    resource,
                    status=AcquisitionStatus.FAILED,
                    failure_code=failure_code,
                )
                repositories.collection_resource_import_results.put(
                    result.result_id, result
                )
                results.append(result)
                counts["failed"] += 1
                warnings.append(f"{resource.discovered_resource_id}:{failure_code}")
                continue
            link = _make_bound(
                CollectionAcquisitionLink,
                "collection-acquisition-link",
                {
                    "collection_id": definition.collection_id,
                    "snapshot_id": discovery.snapshot.snapshot_id,
                    "discovered_resource_id": resource.discovered_resource_id,
                    "discovered_resource_hash": resource.content_hash,
                    "acquisition_id": acquisition.acquisition_id,
                    "acquisition_hash": acquisition.content_hash,
                    "resulting_literature_id": outcome.literature_id,
                    "acquisition_status": outcome.acquisition_status,
                },
            )
            repositories.collection_acquisition_links.put(link.link_id, link)
            links.append(link)
            result = _make_bound(
                CollectionResourceImportResult,
                "collection-resource-import",
                {
                    "collection_id": definition.collection_id,
                    "snapshot_id": discovery.snapshot.snapshot_id,
                    "discovered_resource_id": resource.discovered_resource_id,
                    "discovered_resource_hash": resource.content_hash,
                    "acquisition_link_id": link.link_id,
                    "acquisition_link_hash": link.content_hash,
                    "acquisition_status": outcome.acquisition_status,
                },
            )
            repositories.collection_resource_import_results.put(result.result_id, result)
            results.append(result)
            self._increment_count(counts, outcome.acquisition_status)
            warnings.extend(
                f"{resource.discovered_resource_id}:{warning}"
                for warning in outcome.warnings
            )
        memberships = self._build_memberships(discovery, links, repositories)
        total = len(discovery.resources)
        if (
            discovery.snapshot.completeness_status
            == CollectionCompletenessStatus.REQUIRES_AUTHENTICATION
            and total == 0
        ):
            status = CollectionImportStatus.REQUIRES_AUTHENTICATION
        elif (
            discovery.snapshot.completeness_status
            == CollectionCompletenessStatus.PROVEN_COMPLETE
            and counts["auth"]
            == counts["unavailable"]
            == counts["failed"]
            == counts["unverified"]
            == 0
        ):
            status = CollectionImportStatus.COMPLETE
        elif (
            discovery.snapshot.completeness_status
            == CollectionCompletenessStatus.FAILED
            and total == 0
        ):
            status = CollectionImportStatus.FAILED
        else:
            status = CollectionImportStatus.PARTIAL
        execution_time = self.clock().astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        record = _make_bound(
            CollectionImportRecord,
            "collection-import",
            {
                "executed_at": execution_time,
                "collection_id": definition.collection_id,
                "snapshot_id": discovery.snapshot.snapshot_id,
                "snapshot_hash": discovery.snapshot.content_hash,
                "acquisition_link_hashes": FrozenDict(
                    {item.link_id: item.content_hash for item in links}
                ),
                "resource_result_hashes": FrozenDict(
                    {item.result_id: item.content_hash for item in results}
                ),
                "membership_hashes": FrozenDict(
                    {item.membership_id: item.content_hash for item in memberships}
                ),
                "total_resources": total,
                "discovery_occurrence_count": len(discovery.occurrences),
                "discovered_resource_count": total,
                "logical_literature_count": len(memberships),
                "ingested_literature_count": len(memberships),
                "ingested_count": counts["ingested"],
                "metadata_only_count": counts["metadata_only"],
                "auth_required_count": counts["auth"],
                "unavailable_count": counts["unavailable"],
                "failed_count": counts["failed"],
                "unverified_membership_count": counts["unverified"],
                "status": status,
                "warnings": tuple(dict.fromkeys(warnings)),
            },
        )
        repositories.collection_imports.put(record.import_id, record)
        if previous_snapshot is not None:
            diff = diff_collection_snapshots(previous_snapshot, discovery.snapshot)
            repositories.collection_diffs.put(diff.diff_id, diff)
        return CollectionImportOutcome(
            collection_id=definition.collection_id,
            snapshot_id=discovery.snapshot.snapshot_id,
            completeness=discovery.snapshot.completeness,
            traversal_status=discovery.snapshot.traversal_status,
            completeness_status=discovery.snapshot.completeness_status,
            completeness_basis=discovery.snapshot.completeness_basis,
            visited_pages=discovery.snapshot.visited_page_count,
            discovered_resources=len(discovery.occurrences),
            unique_resources=total,
            discovery_occurrence_count=len(discovery.occurrences),
            discovered_resource_count=total,
            logical_literature_count=len(memberships),
            ingested_literature_count=len(memberships),
            ingested=counts["ingested"],
            metadata_only=counts["metadata_only"],
            requires_authentication=counts["auth"],
            unavailable=counts["unavailable"],
            failed=counts["failed"],
            unverified_membership=counts["unverified"],
            import_id=record.import_id,
            warnings=record.warnings,
        )

    @staticmethod
    def _failed_resource_result(
        definition: CollectionDefinition,
        snapshot: CollectionSnapshot,
        resource: DiscoveredCollectionResource,
        *,
        status: AcquisitionStatus,
        failure_code: str,
    ) -> CollectionResourceImportResult:
        return _make_bound(
            CollectionResourceImportResult,
            "collection-resource-import",
            {
                "collection_id": definition.collection_id,
                "snapshot_id": snapshot.snapshot_id,
                "discovered_resource_id": resource.discovered_resource_id,
                "discovered_resource_hash": resource.content_hash,
                "acquisition_status": status,
                "failure_code": failure_code,
            },
        )

    @staticmethod
    def _increment_count(counts: dict[str, int], status: AcquisitionStatus) -> None:
        if status == AcquisitionStatus.INGESTED:
            counts["ingested"] += 1
        elif status == AcquisitionStatus.METADATA_ONLY:
            counts["metadata_only"] += 1
        elif status == AcquisitionStatus.REQUIRES_AUTHENTICATION:
            counts["auth"] += 1
        elif status in {
            AcquisitionStatus.FULLTEXT_UNAVAILABLE,
            AcquisitionStatus.UNSUPPORTED_MEDIA,
        }:
            counts["unavailable"] += 1
        else:
            counts["failed"] += 1

    @staticmethod
    def _build_memberships(
        discovery: CollectionDiscoveryResult,
        links: list[CollectionAcquisitionLink],
        repositories: KnowledgeRepositories,
    ) -> tuple[CollectionLiteratureMembership, ...]:
        grouped: dict[str, list[CollectionAcquisitionLink]] = {}
        for link in links:
            if link.resulting_literature_id is not None:
                grouped.setdefault(link.resulting_literature_id, []).append(link)
        memberships: list[CollectionLiteratureMembership] = []
        for literature_id, literature_links in sorted(grouped.items()):
            resource_ids = {
                link.discovered_resource_id for link in literature_links
            }
            occurrence_ids = {
                occurrence.occurrence_id
                for occurrence in discovery.occurrences
                if occurrence.discovered_resource_id in resource_ids
            }
            membership = _make_bound(
                CollectionLiteratureMembership,
                "collection-membership",
                {
                    "collection_id": discovery.definition.collection_id,
                    "snapshot_id": discovery.snapshot.snapshot_id,
                    "literature_id": literature_id,
                    "discovered_resource_refs": tuple(sorted(resource_ids)),
                    "acquisition_refs": tuple(
                        sorted({link.acquisition_id for link in literature_links})
                    ),
                    "occurrence_refs": tuple(sorted(occurrence_ids)),
                },
            )
            repositories.collection_literature_memberships.put(
                membership.membership_id, membership
            )
            memberships.append(membership)
        return tuple(memberships)
