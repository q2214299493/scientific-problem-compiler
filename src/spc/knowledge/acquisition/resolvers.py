"""Resolve DOI, article URL and local PDF inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol
from urllib.parse import quote, urljoin
import hashlib
import json
import re

from ...models import (
    AcquisitionInputKind,
    AcquisitionStatus,
    FullTextAccessStatus,
    FullTextCandidate,
    FullTextSourceKind,
    LiteratureAcquisitionRequest,
    MetadataRetrievalRecord,
    ResolvedLiteratureResource,
)
from ..literature_text import AcquisitionError
from .metadata import (
    SUPPORTED_HTML_TYPES,
    SUPPORTED_PDF_TYPES,
    _MetadataHTMLParser,
    _make_candidate,
    _make_metadata_retrieval,
    _make_resource,
    _pdf_metadata,
    _reprioritize_candidates,
    normalize_doi,
)
from .transport import SafeHTTPFetcher


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
    resolver_version = "1.1.0"

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
        merged_candidates = _reprioritize_candidates(tuple(candidates))
        return _make_resource(
            canonical_identifier=doi,
            doi=doi,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            landing_url=landing_url,
            metadata_source="crossref",
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
