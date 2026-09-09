from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import posixpath
import re
from typing import Protocol
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlparse, urlunparse

from ..immutable import FrozenDict
from ..models import (
    AcquisitionStatus,
    CollectionAcquisitionLink,
    CollectionDefinition,
    CollectionDiff,
    CollectionDiscoveryResult,
    CollectionImportOutcome,
    CollectionImportRecord,
    CollectionImportStatus,
    CollectionPageRecord,
    CollectionResourceKind,
    CollectionResourceOccurrence,
    CollectionScopePolicy,
    CollectionSnapshot,
    CollectionSourceKind,
    DiscoveredCollectionResource,
)
from ..repositories import KnowledgeRepositories, SourceEvidenceStore
from ..serialization import content_hash
from .acquisition import (
    SUPPORTED_HTML_TYPES,
    LiteratureAcquisitionService,
    SafeHTTPError,
    SafeHTTPFetcher,
    normalize_doi,
)


DOI_DISCOVERY_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)
ARTICLE_PATH_MARKERS = ("/article/", "/doi/", "/paper/", "/publication/", "/content/")
NEXT_LABELS = {"next", "next page", "older", "more", "下一页", "下页"}


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
    prefix_values = allowed_path_prefixes or (parsed.path or "/",)
    prefixes = tuple(sorted({item if item.startswith("/") else f"/{item}" for item in prefix_values}))
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
    connector_version: str = "1.0.0",
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


class _CollectionHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[_Link] = []
        self.metadata: list[tuple[str, str]] = []
        self._active_link: dict[str, object] | None = None
        self._link_text: list[str] = []
        self.script_count = 0
        self.visible_text: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value or "" for key, value in attrs}
        normalized_tag = tag.casefold()
        if normalized_tag == "a" and values.get("href"):
            self._active_link = values
            self._link_text = []
        elif normalized_tag == "link" and values.get("href"):
            self._append_link(values, "")
        elif normalized_tag == "meta":
            key = (values.get("name") or values.get("property") or "").casefold()
            value = values.get("content", "").strip()
            if key and value:
                self.metadata.append((key, value))
        if normalized_tag in {"script", "style"}:
            self._hidden_depth += 1
            if normalized_tag == "script":
                self.script_count += 1

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag == "a" and self._active_link is not None:
            self._append_link(self._active_link, " ".join(self._link_text))
            self._active_link = None
            self._link_text = []
        if normalized_tag in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._active_link is not None:
            self._link_text.append(text)
        if not self._hidden_depth:
            self.visible_text.append(text)

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
            )
        )


@dataclass(frozen=True)
class _Candidate:
    kind: CollectionResourceKind
    original_identifier: str
    normalized_identifier: str
    discovery_method: str
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
    media_type: str | None = None,
) -> _Candidate | None:
    if kind == CollectionResourceKind.DOI:
        doi = _clean_doi_candidate(original)
        if doi is None:
            return None
        return _Candidate(kind, original, doi, method, doi=doi)
    absolute = canonicalize_url(urljoin(base_url, original))
    embedded_doi = _doi_inside(absolute)
    if embedded_doi is not None:
        return _Candidate(
            CollectionResourceKind.DOI,
            original,
            embedded_doi,
            f"{method}:embedded_doi",
            doi=embedded_doi,
        )
    return _Candidate(
        kind,
        original,
        absolute,
        method,
        url=absolute,
        media_type=media_type,
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
    for match in DOI_DISCOVERY_PATTERN.finditer(html):
        candidate = _resource_candidate(
            kind=CollectionResourceKind.DOI,
            original=match.group(0),
            method="doi_text",
            base_url=page_url,
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
                    doi=metadata_dois[0],
                )
            else:
                candidate = _resource_candidate(
                    kind=kind,
                    original=value,
                    method=f"metadata:{key}",
                    base_url=page_url,
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
        elif link.media_type == "application/pdf" or path.endswith(".pdf"):
            kind = CollectionResourceKind.PDF_URL
            media_type = "application/pdf"
            method = "direct_pdf_link"
        elif "article" in link.rel or any(marker in path for marker in ARTICLE_PATH_MARKERS):
            kind = CollectionResourceKind.ARTICLE_URL
            media_type = "text/html"
            method = "article_link"
        if kind is None:
            continue
        if canonical_origin(absolute) != canonical_origin(page_url) and not policy.allow_external_literature_links:
            continue
        candidate = _resource_candidate(
            kind=kind,
            original=link.href,
            method=method,
            base_url=page_url,
            media_type=media_type,
        )
        if candidate is not None:
            candidates.append(candidate)
    visible = " ".join(parser.visible_text).casefold()
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
        "collection-import": "import_id",
        "collection-diff": "diff_id",
    }[prefix]
    payload = {id_field: record_id, **identity}
    return model_type(**payload, content_hash=content_hash(payload))


class GenericHTMLCollectionConnector:
    connector_id = "generic-html-collection"
    connector_version = "1.0.0"

    def discover(
        self,
        definition: CollectionDefinition,
        repositories: KnowledgeRepositories,
        fetcher: SafeHTTPFetcher,
    ) -> CollectionDiscoveryResult:
        if definition.connector_id != self.connector_id or definition.connector_version != self.connector_version:
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
                    }
                    payload = {key: value for key, value in payload.items() if value is not None}
                    resource = DiscoveredCollectionResource(
                        **payload, content_hash=content_hash(payload)
                    )
                    resources[candidate.key] = resource
                    repositories.collection_resources.put(resource.discovered_resource_id, resource)
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
            page = self._page_record(
                definition,
                requested_url=requested_url,
                resolved_url=response.url,
                http_status=response.status,
                media_type=response.headers.get("content-type", "").split(";", 1)[0] or "unknown",
                response_sha256=hashlib.sha256(response.body).hexdigest(),
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
            completeness = CollectionImportStatus.REQUIRES_AUTHENTICATION
        elif successful_pages == 0 and unique_reasons:
            completeness = CollectionImportStatus.FAILED
        elif unique_reasons:
            completeness = CollectionImportStatus.PARTIAL
        else:
            completeness = CollectionImportStatus.COMPLETE
        ordered_pages = tuple(sorted(pages, key=lambda item: item.page_id))
        ordered_resources = tuple(
            sorted(resources.values(), key=lambda item: item.discovered_resource_id)
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
                "page_record_hashes": FrozenDict(
                    {item.page_id: item.content_hash for item in ordered_pages}
                ),
                "discovered_resource_hashes": FrozenDict(
                    {
                        item.discovered_resource_id: item.content_hash
                        for item in ordered_resources
                    }
                ),
                "occurrence_hashes": FrozenDict(
                    {item.occurrence_id: item.content_hash for item in ordered_occurrences}
                ),
                "visited_page_count": len(ordered_pages),
                "unique_resource_count": len(ordered_resources),
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
            and any(parsed.path.startswith(prefix) for prefix in policy.allowed_path_prefixes)
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
            "unchanged_resource_ids": tuple(sorted(old_ids & new_ids)),
        },
    )


class CollectionImportService:
    def __init__(
        self,
        *,
        fetcher: SafeHTTPFetcher | None = None,
        connector: CollectionConnector | None = None,
        acquisition_service: LiteratureAcquisitionService | None = None,
    ) -> None:
        self.fetcher = fetcher or SafeHTTPFetcher()
        self.connector = connector or GenericHTMLCollectionConnector()
        self.acquisition_service = acquisition_service or LiteratureAcquisitionService(
            self.fetcher
        )

    def run(
        self,
        definition: CollectionDefinition,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore,
        *,
        previous_snapshot: CollectionSnapshot | None = None,
    ) -> CollectionImportOutcome:
        discovery = self.connector.discover(definition, repositories, self.fetcher)
        links: list[CollectionAcquisitionLink] = []
        warnings = list(discovery.snapshot.completeness_reasons)
        counts = {
            "ingested": 0,
            "metadata_only": 0,
            "auth": 0,
            "unavailable": 0,
            "failed": 0,
        }
        for resource in discovery.resources:
            source = resource.doi or resource.url
            if source is None:
                warnings.append(f"{resource.discovered_resource_id}:missing_acquisition_input")
                continue
            outcome = self.acquisition_service.add(
                source,
                definition.domain,
                repositories,
                evidence_store,
            )
            acquisition = repositories.literature_acquisitions.get(outcome.acquisition_id)
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
            if outcome.acquisition_status == AcquisitionStatus.INGESTED:
                counts["ingested"] += 1
            elif outcome.acquisition_status == AcquisitionStatus.METADATA_ONLY:
                counts["metadata_only"] += 1
            elif outcome.acquisition_status == AcquisitionStatus.REQUIRES_AUTHENTICATION:
                counts["auth"] += 1
            elif outcome.acquisition_status in {
                AcquisitionStatus.FULLTEXT_UNAVAILABLE,
                AcquisitionStatus.UNSUPPORTED_MEDIA,
            }:
                counts["unavailable"] += 1
            else:
                counts["failed"] += 1
            warnings.extend(
                f"{resource.discovered_resource_id}:{warning}"
                for warning in outcome.warnings
            )
        total = len(discovery.resources)
        missing_links = total - len(links)
        counts["failed"] += missing_links
        if (
            discovery.snapshot.completeness == CollectionImportStatus.REQUIRES_AUTHENTICATION
            and total == 0
        ):
            status = CollectionImportStatus.REQUIRES_AUTHENTICATION
        elif (
            discovery.snapshot.completeness == CollectionImportStatus.COMPLETE
            and counts["auth"] == counts["unavailable"] == counts["failed"] == 0
        ):
            status = CollectionImportStatus.COMPLETE
        elif discovery.snapshot.completeness == CollectionImportStatus.FAILED and total == 0:
            status = CollectionImportStatus.FAILED
        else:
            status = CollectionImportStatus.PARTIAL
        link_hashes = FrozenDict({item.link_id: item.content_hash for item in links})
        record = _make_bound(
            CollectionImportRecord,
            "collection-import",
            {
                "collection_id": definition.collection_id,
                "snapshot_id": discovery.snapshot.snapshot_id,
                "snapshot_hash": discovery.snapshot.content_hash,
                "acquisition_link_hashes": link_hashes,
                "total_resources": total,
                "ingested_count": counts["ingested"],
                "metadata_only_count": counts["metadata_only"],
                "auth_required_count": counts["auth"],
                "unavailable_count": counts["unavailable"],
                "failed_count": counts["failed"],
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
            visited_pages=discovery.snapshot.visited_page_count,
            discovered_resources=len(discovery.occurrences),
            unique_resources=total,
            ingested=counts["ingested"],
            metadata_only=counts["metadata_only"],
            requires_authentication=counts["auth"],
            unavailable=counts["unavailable"],
            failed=counts["failed"],
            import_id=record.import_id,
            warnings=record.warnings,
        )
