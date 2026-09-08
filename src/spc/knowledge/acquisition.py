from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
from io import BytesIO
import ipaddress
import json
from pathlib import Path
import re
import shutil
import socket
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..models import (
    AcquisitionInputKind,
    AcquisitionStatus,
    CanonicalTextBlock,
    FullTextAccessStatus,
    FullTextCandidate,
    FullTextSourceKind,
    LiteratureAcquisitionOutcome,
    LiteratureAcquisitionRecord,
    LiteratureAcquisitionRequest,
    ResolvedLiteratureResource,
)
from ..repositories import KnowledgeRepositories, SourceEvidenceStore
from ..serialization import content_hash
from .ingestion import LiteratureIngestionService, LiteratureMetadata


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


class HTTPTransport(Protocol):
    def request(self, url: str, *, timeout: float, max_bytes: int) -> HTTPResponse:
        ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UrllibHTTPTransport:
    def request(self, url: str, *, timeout: float, max_bytes: int) -> HTTPResponse:
        request = Request(url, headers={"User-Agent": "SPC-Literature-Acquisition/1.0"})
        opener = build_opener(_NoRedirect)
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        except OSError as error:
            raise SafeHTTPError(
                "NETWORK_ERROR", "remote resource could not be fetched"
            ) from error
        with response:
            body = response.read(max_bytes + 1)
            headers = {
                key.casefold(): value.strip() for key, value in response.headers.items()
            }
            return HTTPResponse(
                url=response.geturl(),
                status=response.status,
                headers=MappingProxyType(headers),
                body=body,
            )


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
        self.transport = transport or UrllibHTTPTransport()
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.dns_resolver = dns_resolver or self._resolve_host

    def fetch(
        self, url: str, *, allowed_content_types: frozenset[str]
    ) -> HTTPResponse:
        current = url
        for redirect_count in range(self.max_redirects + 1):
            self._validate_url(current)
            response = self.transport.request(
                current, timeout=self.timeout, max_bytes=self.max_bytes
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
            )
        raise SafeHTTPError("TOO_MANY_REDIRECTS", "redirect limit exceeded")

    def _validate_url(self, url: str) -> None:
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
    identity = {**identity, "fulltext_candidates": candidates}
    identity = {key: value for key, value in identity.items() if value is not None}
    resource_id = f"resolved-literature-{content_hash(identity)[:24]}"
    payload = {"resource_id": resource_id, **identity}
    return ResolvedLiteratureResource(**payload, content_hash=content_hash(payload))


def _pdf_metadata(content: bytes, *, filename: str) -> dict[str, object]:
    try:
        reader = PdfReader(BytesIO(content), strict=True)
    except PdfReadError as error:
        raise AcquisitionError("PDF metadata could not be read") from error
    metadata = reader.metadata or {}
    title = str(metadata.get("/Title") or "").strip() or Path(filename).stem
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
        metadata = _pdf_metadata(content, filename=path.name)
        candidate = _make_candidate(
            local_path_ref=str(path),
            media_type="application/pdf",
            access_status=FullTextAccessStatus.ACCESSIBLE,
            source_kind=FullTextSourceKind.LOCAL_FILE,
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

    def resolve(self, request: LiteratureAcquisitionRequest) -> ResolvedLiteratureResource:
        doi = normalize_doi(request.original_input)
        response = self.fetcher.fetch(
            f"https://api.crossref.org/works/{quote(doi, safe='')}",
            allowed_content_types=frozenset({"application/json"}),
        )
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
                    priority=priority,
                )
            )
        return _make_resource(
            canonical_identifier=doi,
            doi=doi,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            landing_url=landing_url,
            metadata_source="crossref",
            fulltext_candidates=tuple(candidates),
            resolution_status=(
                AcquisitionStatus.FULLTEXT_FOUND
                if candidates
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

    def resolve(self, request: LiteratureAcquisitionRequest) -> ResolvedLiteratureResource:
        response = self.fetcher.fetch(
            request.original_input,
            allowed_content_types=SUPPORTED_PDF_TYPES | SUPPORTED_HTML_TYPES,
        )
        media_type = response.headers.get("content-type", "").split(";", 1)[0].casefold()
        if media_type == "application/pdf":
            metadata = _pdf_metadata(response.body, filename=Path(urlparse(response.url).path).name)
            candidate = _make_candidate(
                url=response.url,
                media_type=media_type,
                access_status=FullTextAccessStatus.ACCESSIBLE,
                source_kind=FullTextSourceKind.DIRECT_PDF,
                priority=0,
                content_sha256=hashlib.sha256(response.body).hexdigest(),
            )
            return _make_resource(
                canonical_identifier=response.url,
                **metadata,
                landing_url=response.url,
                metadata_source="direct-pdf-metadata",
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
    ) -> LiteratureAcquisitionOutcome:
        input_kind = detect_acquisition_input(original_input)
        request = self._request(original_input, input_kind, domain)
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
            resource = _make_resource(
                canonical_identifier=request.original_input,
                authors=(),
                metadata_source=f"{resolver.resolver_id}:failed",
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
        candidate = candidates[0]
        if candidate.media_type == "application/pdf":
            return self._ingest_pdf_candidate(
                request,
                resolver,
                resource,
                candidate,
                repositories,
                evidence_store,
            )
        if candidate.media_type in SUPPORTED_HTML_TYPES:
            try:
                response = self.fetcher.fetch(
                    candidate.url or "", allowed_content_types=SUPPORTED_HTML_TYPES
                )
                extraction = self.html_extractor.extract(response.url, response.body)
                if (
                    candidate.content_sha256 is not None
                    and candidate.content_sha256 != extraction.artifact_sha256
                ):
                    raise AcquisitionError("HTML artifact changed after resolution")
            except (AcquisitionError, SafeHTTPError) as error:
                return self._finish(
                    request,
                    resolver,
                    resource,
                    repositories,
                    status=AcquisitionStatus.FAILED,
                    candidate=candidate,
                    warnings=(str(error),),
                )
            return self._finish(
                request,
                resolver,
                resource,
                repositories,
                status=AcquisitionStatus.FULLTEXT_FOUND,
                candidate=candidate,
                warnings=(
                    "HTML full text canonicalized without K1B PDF ingestion; "
                    f"artifact_sha256={extraction.artifact_sha256}; "
                    f"text_sha256={extraction.text_sha256}; blocks={len(extraction.blocks)}",
                ),
            )
        return self._finish(
            request,
            resolver,
            resource,
            repositories,
            status=AcquisitionStatus.UNSUPPORTED_MEDIA,
            candidate=candidate,
            warnings=("Selected full text uses unsupported media.",),
        )

    def _ingest_pdf_candidate(
        self,
        request: LiteratureAcquisitionRequest,
        resolver: ResourceResolver,
        resource: ResolvedLiteratureResource,
        candidate: FullTextCandidate,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore,
    ) -> LiteratureAcquisitionOutcome:
        missing = tuple(
            name
            for name, value in (
                ("title", resource.title),
                ("authors", resource.authors),
                ("year", resource.year),
            )
            if not value
        )
        if missing:
            return self._finish(
                request,
                resolver,
                resource,
                repositories,
                status=AcquisitionStatus.METADATA_ONLY,
                candidate=candidate,
                warnings=(f"K1B ingestion requires metadata fields: {', '.join(missing)}",),
            )
        if candidate.local_path_ref is not None:
            pdf_path = Path(candidate.local_path_ref)
            return self._run_k1b(
                pdf_path,
                request,
                resolver,
                resource,
                candidate,
                repositories,
                evidence_store,
            )
        try:
            response = self.fetcher.fetch(
                candidate.url or "", allowed_content_types=SUPPORTED_PDF_TYPES
            )
        except SafeHTTPError as error:
            status = (
                AcquisitionStatus.REQUIRES_AUTHENTICATION
                if error.code == "REQUIRES_AUTHENTICATION"
                else AcquisitionStatus.FULLTEXT_UNAVAILABLE
            )
            return self._finish(
                request,
                resolver,
                resource,
                repositories,
                status=status,
                candidate=candidate,
                warnings=(str(error),),
            )
        if not response.body.startswith(b"%PDF-"):
            return self._finish(
                request,
                resolver,
                resource,
                repositories,
                status=AcquisitionStatus.UNSUPPORTED_MEDIA,
                candidate=candidate,
                warnings=("Remote resource declared PDF but has no PDF signature.",),
            )
        digest = hashlib.sha256(response.body).hexdigest()
        if candidate.content_sha256 is not None and candidate.content_sha256 != digest:
            return self._finish(
                request,
                resolver,
                resource,
                repositories,
                status=AcquisitionStatus.FAILED,
                candidate=candidate,
                warnings=("Remote PDF changed after resolution.",),
            )
        staging_root = repositories.root / ".acquisition_staging"
        if staging_root.exists() and staging_root.is_symlink():
            raise AcquisitionError("acquisition staging root cannot be a symlink")
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="download-", dir=staging_root))
        try:
            pdf_path = staging / "resource.pdf"
            pdf_path.write_bytes(response.body)
            if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != digest:
                raise AcquisitionError("staged PDF integrity check failed")
            return self._run_k1b(
                pdf_path,
                request,
                resolver,
                resource,
                candidate,
                repositories,
                evidence_store,
            )
        finally:
            if staging.exists():
                if not staging.resolve().is_relative_to(staging_root.resolve()):
                    raise AcquisitionError("acquisition staging path escaped its root")
                shutil.rmtree(staging)

    def _run_k1b(
        self,
        pdf_path: Path,
        request: LiteratureAcquisitionRequest,
        resolver: ResourceResolver,
        resource: ResolvedLiteratureResource,
        candidate: FullTextCandidate,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore,
    ) -> LiteratureAcquisitionOutcome:
        metadata = LiteratureMetadata(
            title=resource.title or "",
            authors=resource.authors,
            year=resource.year or 0,
            journal=resource.journal,
            doi=resource.doi,
            url=resource.landing_url,
            domain=request.requested_domain,
        )
        outcome = LiteratureIngestionService().ingest(
            pdf_path, metadata, repositories, evidence_store
        )
        return self._finish(
            request,
            resolver,
            resource,
            repositories,
            status=AcquisitionStatus.INGESTED,
            candidate=candidate,
            literature_id=outcome.literature_id,
            ingestion_id=outcome.ingestion_id,
            canonical_text_id=outcome.canonical_text_id,
            warnings=outcome.warnings,
        )

    @staticmethod
    def _request(
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
        return LiteratureAcquisitionRequest(
            **payload, content_hash=content_hash(payload)
        )

    @staticmethod
    def _finish(
        request: LiteratureAcquisitionRequest,
        resolver: ResourceResolver,
        resource: ResolvedLiteratureResource,
        repositories: KnowledgeRepositories,
        *,
        status: AcquisitionStatus,
        candidate: FullTextCandidate | None = None,
        literature_id: str | None = None,
        ingestion_id: str | None = None,
        canonical_text_id: str | None = None,
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
            "resulting_literature_id": literature_id,
            "resulting_ingestion_id": ingestion_id,
            "resulting_canonical_text_id": canonical_text_id,
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
            acquisition_id=record.acquisition_id,
            warnings=warnings,
        )
