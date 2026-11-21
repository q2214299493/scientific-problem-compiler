"""Acquisition input identity, candidate ordering and metadata provenance."""

from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse
import hashlib
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ...models import (
    AcquisitionInputKind,
    FullTextCandidate,
    LiteratureAcquisitionRequest,
    MetadataAlternative,
    MetadataFieldDecision,
    MetadataMergeManifest,
    MetadataRetrievalRecord,
    MetadataValueOrigin,
    ResolvedLiteratureResource,
)
from ...serialization import content_hash
from ..ingestion import LiteratureMetadata
from ..literature_text import AcquisitionError
from .transport import HTTPResponse


SUPPORTED_PDF_TYPES = frozenset({"application/pdf"})


SUPPORTED_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})


DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


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
