from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import http.client
from io import BytesIO
import ipaddress
import json
from pathlib import Path
import re
import shutil
import socket
import ssl
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping, Protocol
from urllib.parse import quote, urljoin, urlparse

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..models import (
    AcquisitionAttemptRecord,
    AcquisitionInputKind,
    AcquisitionStatus,
    CanonicalHTMLTextArtifact,
    CanonicalTextBlock,
    FullTextAccessStatus,
    FullTextCandidate,
    FullTextSourceKind,
    HTMLLiteratureIngestionRecord,
    LiteratureAcquisitionOutcome,
    LiteratureAcquisitionRecord,
    LiteratureAcquisitionRequest,
    LiteratureDocument,
    LiteratureIngestionStatus,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    MetadataAlternative,
    MetadataFieldDecision,
    MetadataMergeManifest,
    MetadataRetrievalRecord,
    MetadataValueOrigin,
    RawHTMLLiteratureArtifact,
    ResolvedLiteratureResource,
    SourceDocument,
    literature_identity_id,
)
from ..repositories import KnowledgeRepositories, SourceEvidenceStore
from ..serialization import content_hash
from .ingestion import (
    LiteratureIngestionService,
    LiteratureMetadata,
    validate_literature_ingestion_chain,
)


DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
SUPPORTED_PDF_TYPES = frozenset({"application/pdf"})
SUPPORTED_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})


class AcquisitionError(ValueError):
    pass


class SafeHTTPError(AcquisitionError):
    def __init__(self, code: str, message: str, *, status: int | None = None) -> None:
        self.code = code
        self.status = status
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class HTTPResponse:
    url: str
    status: int
    headers: Mapping[str, str]
    body: bytes
    connected_ip: str


class HTTPTransport(Protocol):
    def request(
        self,
        url: str,
        *,
        validated_ips: tuple[str, ...],
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse:
        """Connect only to one of validated_ips; never resolve or follow redirects."""
        ...


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, hostname: str, port: int, pinned_ip: str, timeout: float) -> None:
        super().__init__(hostname, port, timeout=timeout)
        self.pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self.pinned_ip, self.port), self.timeout, self.source_address
        )


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    def __init__(
        self,
        hostname: str,
        port: int,
        pinned_ip: str,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(hostname, port, pinned_ip, timeout)
        self.context = context

    def connect(self) -> None:
        super().connect()
        if self.sock is None:
            raise OSError("pinned HTTPS socket was not created")
        self.sock = self.context.wrap_socket(self.sock, server_hostname=self.host)


class PinnedHTTPTransport:
    """HTTP transport that connects to a prevalidated IP and preserves Host/TLS SNI."""

    def __init__(self, *, ssl_context: ssl.SSLContext | None = None) -> None:
        self.ssl_context = ssl_context or ssl.create_default_context()

    def request(
        self,
        url: str,
        *,
        validated_ips: tuple[str, ...],
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse:
        if not validated_ips:
            raise SafeHTTPError("DNS_RESOLUTION_FAILED", "no validated IP was supplied")
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        try:
            port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
        except ValueError as error:
            raise SafeHTTPError("INVALID_URL", "URL port is invalid") from error
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        host_header = f"[{hostname}]" if ":" in hostname else hostname
        default_port = 443 if parsed.scheme.casefold() == "https" else 80
        if port != default_port:
            host_header = f"{hostname}:{port}"
        last_error: OSError | None = None
        for pinned_ip in validated_ips:
            connection: http.client.HTTPConnection
            if parsed.scheme.casefold() == "https":
                connection = _PinnedHTTPSConnection(
                    hostname, port, pinned_ip, timeout, self.ssl_context
                )
            else:
                connection = _PinnedHTTPConnection(
                    hostname, port, pinned_ip, timeout
                )
            try:
                connection.request(
                    "GET",
                    target,
                    headers={
                        "Host": host_header,
                        "User-Agent": "SPC-Literature-Acquisition/1.1",
                        "Accept": "*/*",
                        "Connection": "close",
                    },
                )
                response = connection.getresponse()
                body = response.read(max_bytes + 1)
                headers = MappingProxyType(
                    {
                        key.casefold(): value.strip()
                        for key, value in response.getheaders()
                    }
                )
                return HTTPResponse(
                    url=url,
                    status=response.status,
                    headers=headers,
                    body=body,
                    connected_ip=pinned_ip,
                )
            except OSError as error:
                last_error = error
            finally:
                connection.close()
        raise SafeHTTPError(
            "NETWORK_ERROR", "remote resource could not be fetched"
        ) from last_error


class SafeHTTPFetcher:
    def __init__(
        self,
        transport: HTTPTransport | None = None,
        *,
        timeout: float = 15.0,
        max_bytes: int = 20 * 1024 * 1024,
        max_redirects: int = 5,
        dns_resolver: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        if timeout <= 0 or max_bytes <= 0 or max_redirects < 0:
            raise ValueError("safe HTTP limits must be positive")
        self.transport = transport or PinnedHTTPTransport()
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.dns_resolver = dns_resolver or self._resolve_host

    def fetch(
        self, url: str, *, allowed_content_types: frozenset[str]
    ) -> HTTPResponse:
        current = url
        for redirect_count in range(self.max_redirects + 1):
            validated_ips = self._validated_ips(current)
            response = self.transport.request(
                current,
                validated_ips=validated_ips,
                timeout=self.timeout,
                max_bytes=self.max_bytes,
            )
            if response.connected_ip not in validated_ips:
                raise SafeHTTPError(
                    "CONNECTION_IP_MISMATCH",
                    "transport connected outside the validated IP set",
                )
            if response.url != current:
                raise SafeHTTPError(
                    "UNVALIDATED_REDIRECT",
                    "HTTP transport followed a redirect outside the safe fetcher",
                )
            headers = {
                str(key).casefold(): str(value).strip()
                for key, value in response.headers.items()
            }
            if response.status in {301, 302, 303, 307, 308}:
                location = headers.get("location")
                if not location:
                    raise SafeHTTPError("INVALID_REDIRECT", "redirect has no location")
                if redirect_count == self.max_redirects:
                    raise SafeHTTPError("TOO_MANY_REDIRECTS", "redirect limit exceeded")
                current = urljoin(current, location)
                continue
            if response.status in {401, 403}:
                raise SafeHTTPError(
                    "REQUIRES_AUTHENTICATION",
                    "remote resource requires authentication",
                    status=response.status,
                )
            if response.status < 200 or response.status >= 300:
                raise SafeHTTPError(
                    "HTTP_STATUS_ERROR",
                    f"remote resource returned HTTP {response.status}",
                    status=response.status,
                )
            declared_length = headers.get("content-length")
            if declared_length is not None:
                try:
                    if int(declared_length) > self.max_bytes:
                        raise SafeHTTPError(
                            "RESOURCE_TOO_LARGE", "declared resource size exceeds limit"
                        )
                except ValueError as error:
                    raise SafeHTTPError(
                        "INVALID_CONTENT_LENGTH", "invalid Content-Length header"
                    ) from error
            if len(response.body) > self.max_bytes:
                raise SafeHTTPError(
                    "RESOURCE_TOO_LARGE", "downloaded resource exceeds limit"
                )
            media_type = headers.get("content-type", "").split(";", 1)[0]
            media_type = media_type.strip().casefold()
            if media_type not in allowed_content_types:
                raise SafeHTTPError(
                    "UNSUPPORTED_CONTENT_TYPE",
                    f"content type is not allowed: {media_type or 'missing'}",
                )
            return HTTPResponse(
                url=current,
                status=response.status,
                headers=MappingProxyType(headers),
                body=response.body,
                connected_ip=response.connected_ip,
            )
        raise SafeHTTPError("TOO_MANY_REDIRECTS", "redirect limit exceeded")

    def _validated_ips(self, url: str) -> tuple[str, ...]:
        parsed = urlparse(url)
        if parsed.scheme.casefold() not in {"http", "https"}:
            raise SafeHTTPError("UNSAFE_URL_SCHEME", "only HTTP/HTTPS are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise SafeHTTPError("URL_CREDENTIALS_REJECTED", "URL credentials are forbidden")
        hostname = parsed.hostname
        if not hostname:
            raise SafeHTTPError("INVALID_URL", "URL has no hostname")
        normalized_host = hostname.rstrip(".").casefold()
        if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
            raise SafeHTTPError("LOCALHOST_REJECTED", "localhost is forbidden")
        try:
            addresses = (str(ipaddress.ip_address(normalized_host)),)
        except ValueError:
            try:
                addresses = self.dns_resolver(normalized_host)
            except OSError as error:
                raise SafeHTTPError("DNS_RESOLUTION_FAILED", "hostname could not resolve") from error
        if not addresses:
            raise SafeHTTPError("DNS_RESOLUTION_FAILED", "hostname resolved to no addresses")
        validated: list[str] = []
        for address in addresses:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as error:
                raise SafeHTTPError("INVALID_DNS_ADDRESS", "DNS returned an invalid address") from error
            if not parsed_address.is_global:
                raise SafeHTTPError(
                    "PRIVATE_ADDRESS_REJECTED",
                    "loopback, private, link-local, or non-global address is forbidden",
                )
            validated.append(str(parsed_address))
        return tuple(sorted(set(validated)))

    @staticmethod
    def _resolve_host(hostname: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item[4][0]
                    for item in socket.getaddrinfo(
                        hostname, None, type=socket.SOCK_STREAM
                    )
                }
            )
        )


def normalize_doi(value: str) -> str:
    normalized = value.strip()
    lowered = normalized.casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "https://dx.doi.org/",
        "doi:",
    ):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    normalized = normalized.casefold()
    if not DOI_PATTERN.fullmatch(normalized) or any(char.isspace() for char in normalized):
        raise AcquisitionError("invalid DOI")
    return normalized


def detect_acquisition_input(value: str) -> AcquisitionInputKind:
    candidate_path = Path(value)
    if candidate_path.is_file():
        return AcquisitionInputKind.LOCAL_FILE
    try:
        normalize_doi(value)
    except AcquisitionError:
        pass
    else:
        return AcquisitionInputKind.DOI
    parsed = urlparse(value)
    if parsed.scheme.casefold() in {"http", "https"}:
        return AcquisitionInputKind.URL
    raise AcquisitionError("input is not a DOI, HTTP(S) URL, or existing local file")


def _make_candidate(**identity) -> FullTextCandidate:
    candidate_id = f"fulltext-candidate-{content_hash(identity)[:24]}"
    payload = {"candidate_id": candidate_id, **identity}
    return FullTextCandidate(**payload, content_hash=content_hash(payload))


def _make_resource(**identity) -> ResolvedLiteratureResource:
    candidates = tuple(
        sorted(
            identity.get("fulltext_candidates", ()),
            key=lambda item: (item.priority, item.candidate_id),
        )
    )
    identity = {
        **identity,
        "metadata_retrieval_refs": tuple(
            identity.get("metadata_retrieval_refs", ())
        ),
        "metadata_retrieval_hashes": tuple(
            identity.get("metadata_retrieval_hashes", ())
        ),
        "fulltext_candidates": candidates,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    resource_id = f"resolved-literature-{content_hash(identity)[:24]}"
    payload = {"resource_id": resource_id, **identity}
    return ResolvedLiteratureResource(**payload, content_hash=content_hash(payload))


def _make_metadata_retrieval(
    response: HTTPResponse,
    *,
    resolver_id: str,
    resolver_version: str,
) -> MetadataRetrievalRecord:
    media_type = response.headers.get("content-type", "").split(";", 1)[0]
    response_payload_utf8: str | None = None
    if media_type in {"application/json", *SUPPORTED_HTML_TYPES}:
        try:
            response_payload_utf8 = response.body.decode("utf-8")
        except UnicodeDecodeError:
            response_payload_utf8 = None
    identity = {
        "source_url": response.url,
        "response_sha256": hashlib.sha256(response.body).hexdigest(),
        "response_byte_size": len(response.body),
        "media_type": media_type,
        "resolver_id": resolver_id,
        "resolver_version": resolver_version,
        "response_payload_utf8": response_payload_utf8,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    retrieval_id = f"metadata-retrieval-{content_hash(identity)[:24]}"
    payload = {"retrieval_id": retrieval_id, **identity}
    return MetadataRetrievalRecord(
        **payload, content_hash=content_hash(payload)
    )


def _make_request(
    original_input: str,
    input_kind: AcquisitionInputKind,
    domain: str,
) -> LiteratureAcquisitionRequest:
    identity = {
        "original_input": original_input.strip(),
        "input_kind": input_kind,
        "requested_domain": domain,
    }
    request_id = f"literature-acquisition-request-{content_hash(identity)[:24]}"
    payload = {"request_id": request_id, **identity}
    return LiteratureAcquisitionRequest(**payload, content_hash=content_hash(payload))


def _reprioritize_candidates(
    candidates: tuple[FullTextCandidate, ...],
) -> tuple[FullTextCandidate, ...]:
    records: list[FullTextCandidate] = []
    seen: set[tuple[str | None, str | None, str]] = set()
    for candidate in candidates:
        key = (candidate.url, candidate.local_path_ref, candidate.media_type)
        if key in seen:
            continue
        seen.add(key)
        identity = candidate.model_dump(
            mode="python",
            exclude={"candidate_id", "content_hash", "priority"},
            exclude_none=True,
        )
        records.append(_make_candidate(**identity, priority=len(records)))
    return tuple(records)


def _pdf_metadata(content: bytes) -> dict[str, object]:
    try:
        reader = PdfReader(BytesIO(content), strict=True)
    except PdfReadError as error:
        raise AcquisitionError("PDF metadata could not be read") from error
    metadata = reader.metadata or {}
    title = str(metadata.get("/Title") or "").strip() or None
    author = str(metadata.get("/Author") or "").strip()
    authors = (author,) if author else ()
    creation_date = str(metadata.get("/CreationDate") or "")
    year_match = re.search(r"D:(\d{4})", creation_date)
    year = int(year_match.group(1)) if year_match else None
    return {"title": title, "authors": authors, "year": year}


class ResourceResolver(Protocol):
    resolver_id: str
    resolver_version: str

    def resolve(self, request: LiteratureAcquisitionRequest) -> ResolvedLiteratureResource:
        ...


class LocalPDFResolver:
    resolver_id = "local-pdf-resolver"
    resolver_version = "1.0.0"

    def resolve(self, request: LiteratureAcquisitionRequest) -> ResolvedLiteratureResource:
        if request.input_kind != AcquisitionInputKind.LOCAL_FILE:
            raise AcquisitionError("LocalPDFResolver requires local_file input")
        path = Path(request.original_input).resolve()
        if path.is_symlink() or not path.is_file():
            raise AcquisitionError("local literature input must be a regular file")
        content = path.read_bytes()
        if not content.startswith(b"%PDF-"):
            return _make_resource(
                canonical_identifier=f"local:{hashlib.sha256(content).hexdigest()}",
                title=path.stem,
                authors=(),
                metadata_source="local-file-inspection",
                fulltext_candidates=(),
                resolution_status=AcquisitionStatus.UNSUPPORTED_MEDIA,
            )
        metadata = _pdf_metadata(content)
        candidate = _make_candidate(
            local_path_ref=str(path),
            media_type="application/pdf",
            access_status=FullTextAccessStatus.ACCESSIBLE,
            source_kind=FullTextSourceKind.LOCAL_FILE,
            discovered_by=self.resolver_id,
            priority=0,
            content_sha256=hashlib.sha256(content).hexdigest(),
        )
        return _make_resource(
            canonical_identifier=f"sha256:{hashlib.sha256(content).hexdigest()}",
            **metadata,
            metadata_source="local-pdf-metadata",
            fulltext_candidates=(candidate,),
            resolution_status=AcquisitionStatus.FULLTEXT_FOUND,
        )


class DOIResolver:
    resolver_id = "crossref-doi-resolver"
    resolver_version = "1.0.0"

    def __init__(self, fetcher: SafeHTTPFetcher) -> None:
        self.fetcher = fetcher
        self.metadata_retrievals: tuple[MetadataRetrievalRecord, ...] = ()

    def resolve(self, request: LiteratureAcquisitionRequest) -> ResolvedLiteratureResource:
        doi = normalize_doi(request.original_input)
        response = self.fetcher.fetch(
            f"https://api.crossref.org/works/{quote(doi, safe='')}",
            allowed_content_types=frozenset({"application/json"}),
        )
        crossref_retrieval = _make_metadata_retrieval(
            response,
            resolver_id=self.resolver_id,
            resolver_version=self.resolver_version,
        )
        self.metadata_retrievals = (crossref_retrieval,)
        try:
            payload = json.loads(response.body.decode("utf-8"))
            message = payload["message"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise AcquisitionError("DOI metadata response is invalid") from error
        title_values = message.get("title") or []
        title = str(title_values[0]).strip() if title_values else None
        authors = tuple(
            name
            for item in message.get("author") or []
            if (
                name := " ".join(
                    part
                    for part in (str(item.get("given", "")).strip(), str(item.get("family", "")).strip())
                    if part
                )
            )
        )
        year = self._year(message)
        journal_values = message.get("container-title") or []
        journal = str(journal_values[0]).strip() if journal_values else None
        landing_url = str(message.get("URL") or f"https://doi.org/{doi}")
        candidates: list[FullTextCandidate] = []
        for priority, link in enumerate(message.get("link") or []):
            media_type = str(link.get("content-type") or "").split(";", 1)[0].casefold()
            url = str(link.get("URL") or "").strip()
            if not url or media_type not in SUPPORTED_PDF_TYPES | SUPPORTED_HTML_TYPES:
                continue
            candidates.append(
                _make_candidate(
                    url=url,
                    media_type=media_type,
                    access_status=FullTextAccessStatus.DISCOVERED,
                    source_kind=FullTextSourceKind.METADATA_LINK,
                    discovered_by=self.resolver_id,
                    discovery_source_url=response.url,
                    priority=priority,
                )
            )
        fallback: ResolvedLiteratureResource | None = None
        if not candidates and landing_url:
            fallback_resolver = ArticleURLResolver(self.fetcher)
            fallback_request = _make_request(
                landing_url, AcquisitionInputKind.URL, request.requested_domain
            )
            try:
                fallback = fallback_resolver.resolve(fallback_request)
            except AcquisitionError:
                fallback = None
            self.metadata_retrievals = (
                crossref_retrieval,
                *fallback_resolver.metadata_retrievals,
            )
        merged_candidates = _reprioritize_candidates(
            (
                *tuple(candidates),
                *(fallback.fulltext_candidates if fallback is not None else ()),
            )
        )
        return _make_resource(
            canonical_identifier=doi,
            doi=doi,
            title=title or (fallback.title if fallback is not None else None),
            authors=authors or (fallback.authors if fallback is not None else ()),
            year=year or (fallback.year if fallback is not None else None),
            journal=journal or (fallback.journal if fallback is not None else None),
            landing_url=landing_url,
            metadata_source=(
                "crossref+article-url" if fallback is not None else "crossref"
            ),
            metadata_retrieval_refs=tuple(
                item.retrieval_id for item in self.metadata_retrievals
            ),
            metadata_retrieval_hashes=tuple(
                item.content_hash for item in self.metadata_retrievals
            ),
            fulltext_candidates=merged_candidates,
            resolution_status=(
                AcquisitionStatus.FULLTEXT_FOUND
                if merged_candidates
                else AcquisitionStatus.FULLTEXT_UNAVAILABLE
            ),
        )

    @staticmethod
    def _year(message: Mapping[str, object]) -> int | None:
        for key in ("published-print", "published-online", "issued"):
            value = message.get(key)
            if not isinstance(value, Mapping):
                continue
            parts = value.get("date-parts")
            if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                try:
                    return int(parts[0][0])
                except (TypeError, ValueError):
                    continue
        return None


class _MetadataHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.links: list[dict[str, str]] = []
        self.article_depth = 0
        self.has_article_text = False

    def handle_starttag(self, tag: str, attrs) -> None:
        values = {str(key).casefold(): str(value) for key, value in attrs if value is not None}
        if tag.casefold() == "meta":
            name = (values.get("name") or values.get("property") or "").casefold()
            content = values.get("content", "").strip()
            if name and content:
                self.meta.setdefault(name, []).append(content)
        elif tag.casefold() == "link":
            self.links.append(values)
        elif tag.casefold() == "article":
            self.article_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "article" and self.article_depth:
            self.article_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.article_depth and data.strip():
            self.has_article_text = True


class ArticleURLResolver:
    resolver_id = "article-url-resolver"
    resolver_version = "1.0.0"

    def __init__(self, fetcher: SafeHTTPFetcher) -> None:
        self.fetcher = fetcher
        self.metadata_retrievals: tuple[MetadataRetrievalRecord, ...] = ()

    def resolve(self, request: LiteratureAcquisitionRequest) -> ResolvedLiteratureResource:
        response = self.fetcher.fetch(
            request.original_input,
            allowed_content_types=SUPPORTED_PDF_TYPES | SUPPORTED_HTML_TYPES,
        )
        retrieval = _make_metadata_retrieval(
            response,
            resolver_id=self.resolver_id,
            resolver_version=self.resolver_version,
        )
        self.metadata_retrievals = (retrieval,)
        media_type = response.headers.get("content-type", "").split(";", 1)[0].casefold()
        if media_type == "application/pdf":
            metadata = _pdf_metadata(response.body)
            candidate = _make_candidate(
                url=response.url,
                media_type=media_type,
                access_status=FullTextAccessStatus.ACCESSIBLE,
                source_kind=FullTextSourceKind.DIRECT_PDF,
                discovered_by=self.resolver_id,
                discovery_source_url=response.url,
                priority=0,
                content_sha256=hashlib.sha256(response.body).hexdigest(),
            )
            return _make_resource(
                canonical_identifier=response.url,
                **metadata,
                landing_url=response.url,
                metadata_source="direct-pdf-metadata",
                metadata_retrieval_refs=(retrieval.retrieval_id,),
                metadata_retrieval_hashes=(retrieval.content_hash,),
                fulltext_candidates=(candidate,),
                resolution_status=AcquisitionStatus.FULLTEXT_FOUND,
            )
        try:
            html_text = response.body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AcquisitionError("HTML resource is not UTF-8") from error
        parser = _MetadataHTMLParser()
        parser.feed(html_text)
        doi_values = parser.meta.get("citation_doi", ())
        doi = normalize_doi(doi_values[0]) if doi_values else None
        title_values = parser.meta.get("citation_title") or parser.meta.get("og:title") or ()
        title = title_values[0].strip() if title_values else None
        authors = tuple(parser.meta.get("citation_author", ()))
        date_values = parser.meta.get("citation_publication_date", ())
        year_match = re.search(r"(\d{4})", date_values[0]) if date_values else None
        journal_values = parser.meta.get("citation_journal_title", ())
        candidates: list[FullTextCandidate] = []
        pdf_urls = list(parser.meta.get("citation_pdf_url", ()))
        pdf_urls.extend(
            item.get("href", "")
            for item in parser.links
            if "pdf" in item.get("type", "").casefold()
        )
        for priority, pdf_url in enumerate(dict.fromkeys(pdf_urls)):
            if pdf_url:
                candidates.append(
                    _make_candidate(
                        url=urljoin(response.url, pdf_url),
                        media_type="application/pdf",
                        access_status=FullTextAccessStatus.DISCOVERED,
                        source_kind=FullTextSourceKind.LANDING_PAGE_PDF,
                        discovered_by=self.resolver_id,
                        discovery_source_url=response.url,
                        priority=priority,
                    )
                )
        if parser.has_article_text:
            candidates.append(
                _make_candidate(
                    url=response.url,
                    media_type="text/html",
                    access_status=FullTextAccessStatus.ACCESSIBLE,
                    source_kind=FullTextSourceKind.HTML_ARTICLE,
                    discovered_by=self.resolver_id,
                    discovery_source_url=response.url,
                    priority=len(candidates),
                    content_sha256=hashlib.sha256(response.body).hexdigest(),
                )
            )
        return _make_resource(
            canonical_identifier=doi or response.url,
            doi=doi,
            title=title,
            authors=authors,
            year=int(year_match.group(1)) if year_match else None,
            journal=journal_values[0].strip() if journal_values else None,
            landing_url=response.url,
            metadata_source="article-html-metadata",
            metadata_retrieval_refs=(retrieval.retrieval_id,),
            metadata_retrieval_hashes=(retrieval.content_hash,),
            fulltext_candidates=tuple(candidates),
            resolution_status=(
                AcquisitionStatus.FULLTEXT_FOUND
                if candidates
                else AcquisitionStatus.FULLTEXT_UNAVAILABLE
            ),
        )


@dataclass(frozen=True)
class HTMLTextExtraction:
    source_url: str
    artifact_sha256: str
    canonical_text: str
    text_sha256: str
    blocks: tuple[CanonicalTextBlock, ...]


class _CanonicalHTMLParser(HTMLParser):
    BLOCK_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.active: list[tuple[str, list[str]]] = []
        self.blocks: list[tuple[str, str]] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and normalized in self.BLOCK_TAGS:
            self.active.append((normalized, []))

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
            return
        if self.ignored_depth or normalized not in self.BLOCK_TAGS:
            return
        for index in range(len(self.active) - 1, -1, -1):
            active_tag, content = self.active[index]
            if active_tag == normalized:
                text = " ".join("".join(content).split())
                if text:
                    self.blocks.append((normalized, text))
                del self.active[index]
                break

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth and self.active:
            self.active[-1][1].append(data)


class HTMLLiteratureTextExtractor:
    extractor_id = "html-canonical-text"
    extractor_version = "1.0.0"
    extractor_config_hash = content_hash(
        {
            "block_tags": tuple(sorted(_CanonicalHTMLParser.BLOCK_TAGS)),
            "ignored_tags": ("noscript", "script", "style"),
            "encoding": "utf-8",
            "block_separator": "two_lf",
        }
    )

    def extract(self, source_url: str, html_bytes: bytes) -> HTMLTextExtraction:
        try:
            html_text = html_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AcquisitionError("HTML resource is not UTF-8") from error
        parser = _CanonicalHTMLParser()
        parser.feed(html_text)
        if not parser.blocks:
            raise AcquisitionError("HTML article contains no canonical text blocks")
        texts = [text for _, text in parser.blocks]
        canonical_text = "\n\n".join(texts)
        blocks: list[CanonicalTextBlock] = []
        cursor = 0
        for block_type, block_text in parser.blocks:
            end = cursor + len(block_text)
            identity = {
                "page_number": 1,
                "block_type": f"html_{block_type}",
                "section_path": (),
                "start_offset": cursor,
                "end_offset": end,
                "text_hash": hashlib.sha256(block_text.encode("utf-8")).hexdigest(),
            }
            blocks.append(
                CanonicalTextBlock(
                    block_id=f"canonical-block-{content_hash(identity)[:24]}",
                    **identity,
                )
            )
            cursor = end + 2
        return HTMLTextExtraction(
            source_url=source_url,
            artifact_sha256=hashlib.sha256(html_bytes).hexdigest(),
            canonical_text=canonical_text,
            text_sha256=hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
            blocks=tuple(blocks),
        )


METADATA_FIELDS = (
    "authors",
    "citation_refs",
    "doi",
    "journal",
    "keywords",
    "title",
    "topics",
    "url",
    "year",
)


def _normalized_metadata_value(field_name: str, value: object) -> object | None:
    if value is None:
        return None
    if field_name in {"authors", "citation_refs", "keywords", "topics"}:
        if isinstance(value, str):
            values = (value.strip(),) if value.strip() else ()
        elif isinstance(value, (list, tuple)):
            values = tuple(str(item).strip() for item in value if str(item).strip())
        else:
            raise AcquisitionError(f"metadata field {field_name} must be a list")
        return values or None
    if field_name == "year":
        try:
            year = int(value)
        except (TypeError, ValueError) as error:
            raise AcquisitionError("metadata year must be an integer") from error
        if year < 1000 or year > 9999:
            raise AcquisitionError("metadata year is out of range")
        return year
    text = str(value).strip()
    return text or None


def _metadata_from_resource(resource: ResolvedLiteratureResource) -> dict[str, object]:
    return {
        "title": resource.title,
        "authors": resource.authors,
        "year": resource.year,
        "journal": resource.journal,
        "doi": resource.doi,
        "url": resource.landing_url,
    }


def _merge_metadata(
    explicit: Mapping[str, object] | None,
    resolved: Mapping[str, object],
    embedded: Mapping[str, object],
    *,
    domain: str,
) -> tuple[LiteratureMetadata | None, MetadataMergeManifest]:
    explicit_values = dict(explicit or {})
    unknown = set(explicit_values) - set(METADATA_FIELDS)
    if unknown:
        raise AcquisitionError(
            f"unsupported explicit metadata fields: {', '.join(sorted(unknown))}"
        )
    sources = (
        (MetadataValueOrigin.EXPLICIT, explicit_values),
        (MetadataValueOrigin.RESOLVED, resolved),
        (MetadataValueOrigin.EMBEDDED, embedded),
    )
    selected: dict[str, object] = {}
    decisions: list[MetadataFieldDecision] = []
    for field_name in METADATA_FIELDS:
        values: list[tuple[MetadataValueOrigin, object]] = []
        for origin, source in sources:
            value = _normalized_metadata_value(field_name, source.get(field_name))
            if value is not None:
                values.append((origin, value))
        if not values:
            decisions.append(
                MetadataFieldDecision(
                    field_name=field_name,
                    selected_origin=MetadataValueOrigin.UNRESOLVED,
                )
            )
            continue
        selected_origin, selected_value = values[0]
        selected[field_name] = selected_value
        alternatives = tuple(
            MetadataAlternative(origin=origin, value=value)
            for origin, value in values[1:]
            if value != selected_value
        )
        decisions.append(
            MetadataFieldDecision(
                field_name=field_name,
                selected_origin=selected_origin,
                selected_value=selected_value,
                alternatives=alternatives,
            )
        )
    identity = {"decisions": tuple(decisions)}
    manifest_id = f"metadata-merge-{content_hash(identity)[:24]}"
    payload = {"manifest_id": manifest_id, **identity}
    manifest = MetadataMergeManifest(
        **payload, content_hash=content_hash(payload)
    )
    if not all(selected.get(field) for field in ("title", "authors", "year")):
        return None, manifest
    return (
        LiteratureMetadata.model_validate({**selected, "domain": domain}),
        manifest,
    )


@dataclass(frozen=True)
class _CandidateResult:
    status: AcquisitionStatus
    failure_code: str | None
    http_status: int | None
    manifest: MetadataMergeManifest
    warnings: tuple[str, ...] = ()
    literature_id: str | None = None
    raw_artifact_id: str | None = None
    raw_artifact_hash: str | None = None
    canonical_text_id: str | None = None
    canonical_text_hash: str | None = None
    ingestion_id: str | None = None
    ingestion_hash: str | None = None
    representation: LiteratureRepresentationReference | None = None


def validate_literature_representation(
    representation: LiteratureRepresentationReference,
    repositories: KnowledgeRepositories,
    evidence_store: SourceEvidenceStore,
) -> LiteratureRepresentationReference:
    stored = repositories.literature_representation_refs.get(
        representation.representation_id
    )
    if stored != representation:
        raise ValueError("literature representation differs from repository record")
    if representation.representation_kind == LiteratureRepresentationKind.PDF:
        ingestion = repositories.literature_ingestions.get(
            representation.ingestion_id
        )
        chain = validate_literature_ingestion_chain(
            ingestion, repositories, evidence_store
        )
        if (
            representation.ingestion_hash != ingestion.content_hash
            or representation.raw_artifact_id != chain.raw_artifact.artifact_id
            or representation.raw_artifact_hash != chain.raw_artifact.content_hash
            or representation.canonical_text_id
            != chain.canonical_text.canonical_text_id
            or representation.canonical_text_hash
            != chain.canonical_text.content_hash
            or representation.source_id != chain.source.source_id
            or representation.source_version != chain.source.version
        ):
            raise ValueError("PDF representation binding is invalid")
        return representation
    ingestion = repositories.html_literature_ingestions.get(
        representation.ingestion_id
    )
    raw = repositories.raw_html_literature_artifacts.get(
        representation.raw_artifact_id
    )
    canonical = repositories.canonical_html_text_artifacts.get(
        representation.canonical_text_id or ""
    )
    source = evidence_store.source_records.get(
        f"{representation.source_id}--{representation.source_version}"
    )
    evidence_store.verify_source_integrity(source)
    if (
        representation.ingestion_hash != ingestion.content_hash
        or representation.literature_id != ingestion.literature_id
        or representation.raw_artifact_hash != raw.content_hash
        or representation.canonical_text_hash != canonical.content_hash
        or ingestion.raw_artifact_id != raw.artifact_id
        or ingestion.raw_artifact_hash != raw.content_hash
        or ingestion.canonical_text_id != canonical.canonical_text_id
        or ingestion.canonical_text_hash != canonical.content_hash
        or canonical.raw_artifact_id != raw.artifact_id
        or canonical.raw_artifact_hash != raw.content_hash
        or ingestion.source_id != source.source_id
        or ingestion.source_version != source.version
        or source.content_sha256 != canonical.text_sha256
    ):
        raise ValueError("HTML representation binding is invalid")
    return representation


class LiteratureAcquisitionService:
    def __init__(
        self,
        fetcher: SafeHTTPFetcher | None = None,
        *,
        html_extractor: HTMLLiteratureTextExtractor | None = None,
    ) -> None:
        self.fetcher = fetcher or SafeHTTPFetcher()
        self.html_extractor = html_extractor or HTMLLiteratureTextExtractor()

    def add(
        self,
        original_input: str,
        domain: str,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore,
        explicit_metadata: Mapping[str, object] | None = None,
    ) -> LiteratureAcquisitionOutcome:
        input_kind = detect_acquisition_input(original_input)
        request = _make_request(original_input, input_kind, domain)
        repositories.acquisition_requests.put(request.request_id, request)
        resolver: ResourceResolver
        if input_kind == AcquisitionInputKind.LOCAL_FILE:
            resolver = LocalPDFResolver()
        elif input_kind == AcquisitionInputKind.DOI:
            resolver = DOIResolver(self.fetcher)
        else:
            resolver = ArticleURLResolver(self.fetcher)
        try:
            resource = resolver.resolve(request)
        except (AcquisitionError, SafeHTTPError) as error:
            status = AcquisitionStatus.FAILED
            if isinstance(error, SafeHTTPError):
                if error.code == "REQUIRES_AUTHENTICATION":
                    status = AcquisitionStatus.REQUIRES_AUTHENTICATION
                elif error.code == "UNSUPPORTED_CONTENT_TYPE":
                    status = AcquisitionStatus.UNSUPPORTED_MEDIA
            failure_retrievals = tuple(
                getattr(resolver, "metadata_retrievals", ())
            )
            for retrieval in failure_retrievals:
                repositories.metadata_retrievals.put(
                    retrieval.retrieval_id, retrieval
                )
            resource = _make_resource(
                canonical_identifier=request.original_input,
                authors=(),
                metadata_source=f"{resolver.resolver_id}:failed",
                metadata_retrieval_refs=tuple(
                    item.retrieval_id for item in failure_retrievals
                ),
                metadata_retrieval_hashes=tuple(
                    item.content_hash for item in failure_retrievals
                ),
                fulltext_candidates=(),
                resolution_status=status,
            )
            repositories.resolved_literature_resources.put(
                resource.resource_id, resource
            )
            return self._finish(
                request,
                resolver,
                resource,
                repositories,
                status=status,
                warnings=(str(error),),
            )
        retrievals = tuple(getattr(resolver, "metadata_retrievals", ()))
        expected_retrieval_ids = tuple(item.retrieval_id for item in retrievals)
        expected_retrieval_hashes = tuple(item.content_hash for item in retrievals)
        if (
            resource.metadata_retrieval_refs != expected_retrieval_ids
            or resource.metadata_retrieval_hashes != expected_retrieval_hashes
        ):
            raise AcquisitionError("resolved metadata provenance binding is incomplete")
        for retrieval in retrievals:
            repositories.metadata_retrievals.put(
                retrieval.retrieval_id, retrieval
            )
        repositories.resolved_literature_resources.put(resource.resource_id, resource)
        candidates = tuple(
            item
            for item in resource.fulltext_candidates
            if item.access_status
            in {
                FullTextAccessStatus.DISCOVERED,
                FullTextAccessStatus.ACCESSIBLE,
            }
        )
        if not candidates:
            status = (
                AcquisitionStatus.FULLTEXT_UNAVAILABLE
                if resource.resolution_status
                in {
                    AcquisitionStatus.FULLTEXT_UNAVAILABLE,
                    AcquisitionStatus.METADATA_RESOLVED,
                }
                else resource.resolution_status
            )
            return self._finish(
                request, resolver, resource, repositories, status=status
            )
        attempts: list[AcquisitionAttemptRecord] = []
        results: list[_CandidateResult] = []
        for attempt_index, candidate in enumerate(candidates):
            result = self._attempt_candidate(
                candidate,
                resource,
                request,
                repositories,
                evidence_store,
                explicit_metadata,
            )
            repositories.metadata_merge_manifests.put(
                result.manifest.manifest_id, result.manifest
            )
            attempt = self._make_attempt(
                request, candidate, attempt_index, result
            )
            repositories.acquisition_attempts.put(attempt.attempt_id, attempt)
            attempts.append(attempt)
            results.append(result)
            if result.status == AcquisitionStatus.INGESTED:
                return self._finish(
                    request,
                    resolver,
                    resource,
                    repositories,
                    status=AcquisitionStatus.INGESTED,
                    candidate=candidate,
                    manifest=result.manifest,
                    attempts=tuple(attempts),
                    literature_id=result.literature_id,
                    ingestion_id=result.ingestion_id,
                    canonical_text_id=result.canonical_text_id,
                    representation=result.representation,
                    warnings=result.warnings,
                )
        final_status = (
            AcquisitionStatus.METADATA_ONLY
            if any(item.status == AcquisitionStatus.METADATA_ONLY for item in results)
            else AcquisitionStatus.FULLTEXT_UNAVAILABLE
        )
        warnings = tuple(
            dict.fromkeys(
                warning
                for result in results
                for warning in (
                    *result.warnings,
                    *((result.failure_code,) if result.failure_code else ()),
                )
            )
        )
        return self._finish(
            request,
            resolver,
            resource,
            repositories,
            status=final_status,
            attempts=tuple(attempts),
            warnings=warnings,
        )

    def _attempt_candidate(
        self,
        candidate: FullTextCandidate,
        resource: ResolvedLiteratureResource,
        request: LiteratureAcquisitionRequest,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore,
        explicit_metadata: Mapping[str, object] | None,
    ) -> _CandidateResult:
        if candidate.media_type == "application/pdf":
            return self._attempt_pdf(
                candidate,
                resource,
                request,
                repositories,
                evidence_store,
                explicit_metadata,
            )
        if candidate.media_type in SUPPORTED_HTML_TYPES:
            return self._attempt_html(
                candidate,
                resource,
                request,
                repositories,
                evidence_store,
                explicit_metadata,
            )
        _, manifest = _merge_metadata(
            explicit_metadata,
            _metadata_from_resource(resource),
            {},
            domain=request.requested_domain,
        )
        return _CandidateResult(
            status=AcquisitionStatus.UNSUPPORTED_MEDIA,
            failure_code="UNSUPPORTED_MEDIA",
            http_status=None,
            manifest=manifest,
            warnings=("Selected full text uses unsupported media.",),
        )

    def _attempt_pdf(
        self,
        candidate: FullTextCandidate,
        resource: ResolvedLiteratureResource,
        request: LiteratureAcquisitionRequest,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore,
        explicit_metadata: Mapping[str, object] | None,
    ) -> _CandidateResult:
        http_status: int | None = None
        if candidate.local_path_ref is not None:
            pdf_path = Path(candidate.local_path_ref)
            if pdf_path.is_symlink() or not pdf_path.is_file():
                _, manifest = _merge_metadata(
                    explicit_metadata,
                    _metadata_from_resource(resource),
                    {},
                    domain=request.requested_domain,
                )
                return _CandidateResult(
                    status=AcquisitionStatus.FULLTEXT_UNAVAILABLE,
                    failure_code="LOCAL_FILE_UNAVAILABLE",
                    http_status=None,
                    manifest=manifest,
                )
            pdf_bytes = pdf_path.read_bytes()
        else:
            try:
                response = self.fetcher.fetch(
                    candidate.url or "", allowed_content_types=SUPPORTED_PDF_TYPES
                )
                pdf_bytes = response.body
                http_status = response.status
            except SafeHTTPError as error:
                _, manifest = _merge_metadata(
                    explicit_metadata,
                    _metadata_from_resource(resource),
                    {},
                    domain=request.requested_domain,
                )
                return _CandidateResult(
                    status=AcquisitionStatus.FULLTEXT_UNAVAILABLE,
                    failure_code=error.code,
                    http_status=error.status,
                    manifest=manifest,
                    warnings=(str(error),),
                )
        if not pdf_bytes.startswith(b"%PDF-"):
            _, manifest = _merge_metadata(
                explicit_metadata,
                _metadata_from_resource(resource),
                {},
                domain=request.requested_domain,
            )
            return _CandidateResult(
                status=AcquisitionStatus.UNSUPPORTED_MEDIA,
                failure_code="INVALID_PDF_SIGNATURE",
                http_status=http_status,
                manifest=manifest,
            )
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        if candidate.content_sha256 is not None and candidate.content_sha256 != digest:
            _, manifest = _merge_metadata(
                explicit_metadata,
                _metadata_from_resource(resource),
                {},
                domain=request.requested_domain,
            )
            return _CandidateResult(
                status=AcquisitionStatus.FAILED,
                failure_code="CONTENT_HASH_CHANGED",
                http_status=http_status,
                manifest=manifest,
            )
        try:
            embedded = _pdf_metadata(pdf_bytes)
        except AcquisitionError:
            embedded = {}
        metadata, manifest = _merge_metadata(
            explicit_metadata,
            _metadata_from_resource(resource),
            embedded,
            domain=request.requested_domain,
        )
        if metadata is None:
            return _CandidateResult(
                status=AcquisitionStatus.METADATA_ONLY,
                failure_code="INCOMPLETE_METADATA",
                http_status=http_status,
                manifest=manifest,
            )
        if candidate.local_path_ref is not None:
            return self._run_k1b(
                pdf_path,
                pdf_bytes,
                metadata,
                manifest,
                repositories,
                evidence_store,
                http_status,
            )
        staging_root = repositories.root / ".acquisition_staging"
        if staging_root.exists() and staging_root.is_symlink():
            raise AcquisitionError("acquisition staging root cannot be a symlink")
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="download-", dir=staging_root))
        try:
            pdf_path = staging / "resource.pdf"
            pdf_path.write_bytes(pdf_bytes)
            if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != digest:
                raise AcquisitionError("staged PDF integrity check failed")
            return self._run_k1b(
                pdf_path,
                pdf_bytes,
                metadata,
                manifest,
                repositories,
                evidence_store,
                http_status,
            )
        finally:
            if staging.exists():
                if not staging.resolve().is_relative_to(staging_root.resolve()):
                    raise AcquisitionError("acquisition staging path escaped its root")
                shutil.rmtree(staging)

    def _run_k1b(
        self,
        pdf_path: Path,
        pdf_bytes: bytes,
        metadata: LiteratureMetadata,
        manifest: MetadataMergeManifest,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore,
        http_status: int | None,
    ) -> _CandidateResult:
        if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != hashlib.sha256(
            pdf_bytes
        ).hexdigest():
            raise AcquisitionError("staged PDF integrity check failed")
        outcome = LiteratureIngestionService().ingest(
            pdf_path, metadata, repositories, evidence_store
        )
        ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
        raw = repositories.raw_literature_artifacts.get(outcome.artifact_id)
        if outcome.canonical_text_id is None:
            return _CandidateResult(
                status=AcquisitionStatus.FAILED,
                failure_code=(
                    "REQUIRES_OCR" if outcome.requires_ocr else "PDF_INGESTION_FAILED"
                ),
                http_status=http_status,
                manifest=manifest,
                warnings=outcome.warnings,
                literature_id=outcome.literature_id,
                raw_artifact_id=raw.artifact_id,
                raw_artifact_hash=raw.content_hash,
                ingestion_id=ingestion.ingestion_id,
                ingestion_hash=ingestion.content_hash,
            )
        canonical = repositories.canonical_text_artifacts.get(
            outcome.canonical_text_id
        )
        representation = self._representation(
            kind=LiteratureRepresentationKind.PDF,
            literature_id=outcome.literature_id,
            raw_artifact_id=raw.artifact_id,
            raw_artifact_hash=raw.content_hash,
            canonical_text_id=canonical.canonical_text_id,
            canonical_text_hash=canonical.content_hash,
            ingestion_id=ingestion.ingestion_id,
            ingestion_hash=ingestion.content_hash,
            source_id=outcome.source_id,
            source_version=outcome.source_version,
        )
        repositories.literature_representation_refs.put(
            representation.representation_id, representation
        )
        validate_literature_representation(
            representation, repositories, evidence_store
        )
        return _CandidateResult(
            status=AcquisitionStatus.INGESTED,
            failure_code=None,
            http_status=http_status,
            manifest=manifest,
            warnings=outcome.warnings,
            literature_id=outcome.literature_id,
            raw_artifact_id=raw.artifact_id,
            raw_artifact_hash=raw.content_hash,
            canonical_text_id=canonical.canonical_text_id,
            canonical_text_hash=canonical.content_hash,
            ingestion_id=ingestion.ingestion_id,
            ingestion_hash=ingestion.content_hash,
            representation=representation,
        )

    def _attempt_html(
        self,
        candidate: FullTextCandidate,
        resource: ResolvedLiteratureResource,
        request: LiteratureAcquisitionRequest,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore,
        explicit_metadata: Mapping[str, object] | None,
    ) -> _CandidateResult:
        _, base_manifest = _merge_metadata(
            explicit_metadata,
            _metadata_from_resource(resource),
            {},
            domain=request.requested_domain,
        )
        try:
            response = self.fetcher.fetch(
                candidate.url or "", allowed_content_types=SUPPORTED_HTML_TYPES
            )
            extraction = self.html_extractor.extract(response.url, response.body)
        except (AcquisitionError, SafeHTTPError) as error:
            return _CandidateResult(
                status=AcquisitionStatus.FULLTEXT_UNAVAILABLE,
                failure_code=(error.code if isinstance(error, SafeHTTPError) else "HTML_EXTRACTION_FAILED"),
                http_status=(error.status if isinstance(error, SafeHTTPError) else None),
                manifest=base_manifest,
                warnings=(str(error),),
            )
        if (
            candidate.content_sha256 is not None
            and candidate.content_sha256 != extraction.artifact_sha256
        ):
            return _CandidateResult(
                status=AcquisitionStatus.FAILED,
                failure_code="CONTENT_HASH_CHANGED",
                http_status=response.status,
                manifest=base_manifest,
            )
        metadata, manifest = _merge_metadata(
            explicit_metadata,
            _metadata_from_resource(resource),
            {},
            domain=request.requested_domain,
        )
        if metadata is None:
            return _CandidateResult(
                status=AcquisitionStatus.METADATA_ONLY,
                failure_code="INCOMPLETE_METADATA",
                http_status=response.status,
                manifest=manifest,
            )
        literature_id = literature_identity_id(
            title=metadata.title,
            authors=metadata.authors,
            year=metadata.year,
            doi=metadata.doi,
        )
        media_type = response.headers.get("content-type", "").split(";", 1)[0]
        raw = repositories.raw_html_literature_artifacts.put(
            response.body,
            source_url=response.url,
            literature_id=literature_id,
            media_type=media_type,
        )
        canonical = repositories.canonical_html_text_artifacts.put(
            extraction.canonical_text,
            raw_artifact=raw,
            extractor_id=self.html_extractor.extractor_id,
            extractor_version=self.html_extractor.extractor_version,
            extractor_config_hash=self.html_extractor.extractor_config_hash,
            blocks=extraction.blocks,
        )
        canonical_path = repositories.root.joinpath(
            *Path(canonical.stored_path).parts
        )
        source = evidence_store.ingest(
            canonical_path,
            f"source-{literature_id}",
            f"html-{canonical.canonical_text_id.removeprefix('canonical-html-text-')}",
            metadata.title,
            source_role="literature_author",
            source_type="literature_article",
        )
        ingestion = self._html_ingestion(raw, canonical, source)
        repositories.html_literature_ingestions.put(
            ingestion.ingestion_id, ingestion
        )
        self._finalize_html_document(
            metadata, raw, canonical, source, repositories
        )
        representation = self._representation(
            kind=LiteratureRepresentationKind.HTML,
            literature_id=literature_id,
            raw_artifact_id=raw.artifact_id,
            raw_artifact_hash=raw.content_hash,
            canonical_text_id=canonical.canonical_text_id,
            canonical_text_hash=canonical.content_hash,
            ingestion_id=ingestion.ingestion_id,
            ingestion_hash=ingestion.content_hash,
            source_id=source.source_id,
            source_version=source.version,
        )
        repositories.literature_representation_refs.put(
            representation.representation_id, representation
        )
        validate_literature_representation(
            representation, repositories, evidence_store
        )
        return _CandidateResult(
            status=AcquisitionStatus.INGESTED,
            failure_code=None,
            http_status=response.status,
            manifest=manifest,
            literature_id=literature_id,
            raw_artifact_id=raw.artifact_id,
            raw_artifact_hash=raw.content_hash,
            canonical_text_id=canonical.canonical_text_id,
            canonical_text_hash=canonical.content_hash,
            ingestion_id=ingestion.ingestion_id,
            ingestion_hash=ingestion.content_hash,
            representation=representation,
        )

    def _html_ingestion(
        self,
        raw: RawHTMLLiteratureArtifact,
        canonical: CanonicalHTMLTextArtifact,
        source: SourceDocument,
    ) -> HTMLLiteratureIngestionRecord:
        identity = {
            "literature_id": raw.literature_id,
            "raw_artifact_id": raw.artifact_id,
            "raw_artifact_hash": raw.content_hash,
            "canonical_text_id": canonical.canonical_text_id,
            "canonical_text_hash": canonical.content_hash,
            "source_id": source.source_id,
            "source_version": source.version,
            "extractor_id": canonical.extractor_id,
            "extractor_version": canonical.extractor_version,
            "extractor_config_hash": canonical.extractor_config_hash,
            "ingestion_status": LiteratureIngestionStatus.ACCEPTED,
        }
        ingestion_id = f"html-literature-ingestion-{content_hash(identity)[:24]}"
        payload = {"ingestion_id": ingestion_id, **identity}
        return HTMLLiteratureIngestionRecord(
            **payload, content_hash=content_hash(payload)
        )

    @staticmethod
    def _finalize_html_document(
        metadata: LiteratureMetadata,
        raw: RawHTMLLiteratureArtifact,
        canonical: CanonicalHTMLTextArtifact,
        source: SourceDocument,
        repositories: KnowledgeRepositories,
    ) -> LiteratureDocument:
        try:
            return repositories.literature_documents.get(raw.literature_id)
        except FileNotFoundError:
            pass
        identity = {
            **metadata.model_dump(mode="json", exclude_none=True),
            "literature_id": raw.literature_id,
            "raw_artifact_ref": raw.stored_path,
            "canonical_text_ref": canonical.stored_path,
            "source_id": source.source_id,
            "source_version": source.version,
        }
        document = LiteratureDocument(
            **identity, content_hash=content_hash(identity)
        )
        repositories.literature_documents.put(document.literature_id, document)
        return document

    @staticmethod
    def _representation(
        *,
        kind: LiteratureRepresentationKind,
        literature_id: str,
        raw_artifact_id: str,
        raw_artifact_hash: str,
        canonical_text_id: str | None,
        canonical_text_hash: str | None,
        ingestion_id: str,
        ingestion_hash: str,
        source_id: str | None,
        source_version: str | None,
    ) -> LiteratureRepresentationReference:
        identity = {
            "representation_kind": kind,
            "literature_id": literature_id,
            "raw_artifact_id": raw_artifact_id,
            "raw_artifact_hash": raw_artifact_hash,
            "canonical_text_id": canonical_text_id,
            "canonical_text_hash": canonical_text_hash,
            "ingestion_id": ingestion_id,
            "ingestion_hash": ingestion_hash,
            "source_id": source_id,
            "source_version": source_version,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        representation_id = f"literature-representation-{content_hash(identity)[:24]}"
        payload = {"representation_id": representation_id, **identity}
        return LiteratureRepresentationReference(
            **payload, content_hash=content_hash(payload)
        )

    @staticmethod
    def _make_attempt(
        request: LiteratureAcquisitionRequest,
        candidate: FullTextCandidate,
        attempt_index: int,
        result: _CandidateResult,
    ) -> AcquisitionAttemptRecord:
        identity = {
            "request_id": request.request_id,
            "request_hash": request.content_hash,
            "candidate_id": candidate.candidate_id,
            "candidate_hash": candidate.content_hash,
            "attempt_index": attempt_index,
            "status": result.status,
            "http_status": result.http_status,
            "media_type": candidate.media_type,
            "failure_code": result.failure_code,
            "raw_artifact_id": result.raw_artifact_id,
            "raw_artifact_hash": result.raw_artifact_hash,
            "canonical_text_id": result.canonical_text_id,
            "canonical_text_hash": result.canonical_text_hash,
            "ingestion_id": result.ingestion_id,
            "ingestion_hash": result.ingestion_hash,
            "representation_id": (
                result.representation.representation_id
                if result.representation is not None
                else None
            ),
            "representation_hash": (
                result.representation.content_hash
                if result.representation is not None
                else None
            ),
            "metadata_merge_manifest_id": result.manifest.manifest_id,
            "metadata_merge_manifest_hash": result.manifest.content_hash,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        attempt_id = f"acquisition-attempt-{content_hash(identity)[:24]}"
        payload = {"attempt_id": attempt_id, **identity}
        return AcquisitionAttemptRecord(
            **payload, content_hash=content_hash(payload)
        )

    @staticmethod
    def _request(
        original_input: str,
        input_kind: AcquisitionInputKind,
        domain: str,
    ) -> LiteratureAcquisitionRequest:
        return _make_request(original_input, input_kind, domain)

    @staticmethod
    def _finish(
        request: LiteratureAcquisitionRequest,
        resolver: ResourceResolver,
        resource: ResolvedLiteratureResource,
        repositories: KnowledgeRepositories,
        *,
        status: AcquisitionStatus,
        candidate: FullTextCandidate | None = None,
        manifest: MetadataMergeManifest | None = None,
        attempts: tuple[AcquisitionAttemptRecord, ...] = (),
        literature_id: str | None = None,
        ingestion_id: str | None = None,
        canonical_text_id: str | None = None,
        representation: LiteratureRepresentationReference | None = None,
        warnings: tuple[str, ...] = (),
    ) -> LiteratureAcquisitionOutcome:
        identity = {
            "request_id": request.request_id,
            "request_hash": request.content_hash,
            "resolver_id": resolver.resolver_id,
            "resolver_version": resolver.resolver_version,
            "resolved_resource_id": resource.resource_id,
            "resolved_resource_hash": resource.content_hash,
            "selected_candidate_id": candidate.candidate_id if candidate else None,
            "selected_candidate_hash": candidate.content_hash if candidate else None,
            "metadata_merge_manifest_id": manifest.manifest_id if manifest else None,
            "metadata_merge_manifest_hash": manifest.content_hash if manifest else None,
            "attempt_refs": tuple(item.attempt_id for item in attempts),
            "attempt_hashes": tuple(item.content_hash for item in attempts),
            "resulting_literature_id": literature_id,
            "resulting_ingestion_id": ingestion_id,
            "resulting_canonical_text_id": canonical_text_id,
            "resulting_representation_id": (
                representation.representation_id if representation else None
            ),
            "resulting_representation_hash": (
                representation.content_hash if representation else None
            ),
            "status": status,
            "warnings": warnings,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        acquisition_id = f"literature-acquisition-{content_hash(identity)[:24]}"
        payload = {"acquisition_id": acquisition_id, **identity}
        record = LiteratureAcquisitionRecord(
            **payload, content_hash=content_hash(payload)
        )
        repositories.literature_acquisitions.put(record.acquisition_id, record)
        return LiteratureAcquisitionOutcome(
            input_kind=request.input_kind,
            canonical_identifier=resource.canonical_identifier,
            doi=resource.doi,
            title=resource.title,
            acquisition_status=status,
            fulltext_status=(
                AcquisitionStatus.FULLTEXT_FOUND
                if status == AcquisitionStatus.INGESTED
                else resource.resolution_status
                if status == AcquisitionStatus.METADATA_ONLY
                else status
            ),
            literature_id=literature_id,
            ingestion_id=ingestion_id,
            canonical_text_id=canonical_text_id,
            representation_id=(
                representation.representation_id if representation else None
            ),
            acquisition_id=record.acquisition_id,
            attempt_ids=tuple(item.attempt_id for item in attempts),
            warnings=warnings,
        )
