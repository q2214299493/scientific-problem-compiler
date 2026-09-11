from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from urllib.parse import urlparse

from ...models import (
    EvidenceSpan,
    ExpertAttributionRecord,
    ExpertProfile,
    RawLiteratureArtifact,
    SourceRole,
    SourceType,
)
from ...repositories import KnowledgeEvidenceStore, KnowledgeRepositories
from ...serialization import content_hash, require_safe_path_component
from ..acquisition import HTMLLiteratureTextExtractor, SafeHTTPFetcher
from ..ingestion import PAGE_SEPARATOR, PypdfLiteratureTextExtractor
from .contracts import ExpertSourcePage, ExpertSourceRecord


def _put_reuse(repository, key: str, record) -> None:
    try:
        existing = repository.get(key)
    except FileNotFoundError:
        repository.put(key, record)
        return
    if existing != record:
        raise FileExistsError(f"conflicting immutable expert record: {key}")


def make_expert_profile(
    display_name: str,
    *,
    organization: str | None,
    role: str | None,
    profile_source_refs: tuple[str, ...] = ("explicit-user-declaration",),
) -> ExpertProfile:
    identity = {
        "display_name": display_name,
        "organization": organization,
        "role": role,
    }
    expert_id = f"expert-{content_hash(identity)[:24]}"
    payload = {
        "expert_id": expert_id,
        "display_name": display_name,
        "affiliations": (organization,) if organization else (),
        "expertise_domains": (),
        "expertise_topics": (),
        "profile_source_refs": profile_source_refs,
        "organization": organization,
        "role": role,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    return ExpertProfile(**payload, content_hash=content_hash(payload))


def _source_semantics(source_type: str, *, is_url: bool) -> tuple[SourceRole, SourceType]:
    normalized = source_type.strip().casefold().replace("-", "_")
    if normalized in {"reviewer_comment", "reviewer_comments"}:
        return SourceRole.REVIEWER, SourceType.REVIEWER_COMMENT
    if is_url or normalized in {"webpage", "expert_webpage"}:
        return SourceRole.INTERNAL_RESEARCHER, SourceType.EXPERT_WEBPAGE
    if normalized in {
        "meeting_note",
        "meeting_notes",
        "internal_note",
        "local_text",
        "markdown",
        "pdf",
    }:
        return SourceRole.INTERNAL_RESEARCHER, SourceType.INTERNAL_NOTE
    raise ValueError(f"unsupported expert source type: {source_type}")


def _canonicalize_text(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("expert text source must be UTF-8") from error
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("expert source contains no usable text")
    return text


def _pdf_text(data: bytes) -> tuple[str, tuple[ExpertSourcePage, ...]]:
    digest = hashlib.sha256(data).hexdigest()
    identity = {"literature_id": "expert-source-staging", "sha256": digest}
    artifact_id = f"literature-artifact-{content_hash(identity)[:24]}"
    payload = {
        "artifact_id": artifact_id,
        "literature_id": "expert-source-staging",
        "original_filename": "expert-source.pdf",
        "media_type": "application/pdf",
        "byte_size": len(data),
        "sha256": digest,
        "stored_path": f"literature_artifacts/{artifact_id}/artifact.pdf",
    }
    artifact = RawLiteratureArtifact(**payload, content_hash=content_hash(payload))
    with tempfile.TemporaryDirectory(prefix="spc-expert-pdf-") as temporary:
        path = Path(temporary) / "source.pdf"
        path.write_bytes(data)
        extraction = PypdfLiteratureTextExtractor().extract(artifact, path)
    if extraction.requires_ocr or extraction.text is None:
        raise ValueError("expert PDF has no usable born-digital text; OCR is required")
    pages: list[ExpertSourcePage] = []
    cursor = 0
    for page_number, page_text in enumerate(extraction.page_texts, start=1):
        if page_text:
            pages.append(
                ExpertSourcePage(
                    page_number=page_number,
                    start_offset=cursor,
                    end_offset=cursor + len(page_text),
                )
            )
        cursor += len(page_text)
        if page_number < len(extraction.page_texts):
            cursor += len(PAGE_SEPARATOR)
    return extraction.text, tuple(pages)


def _immutable_copy(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("expert source artifact cannot be a symlink")
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"conflicting expert source artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"expert source staging path already exists: {temporary}")
    temporary.write_bytes(data)
    temporary.replace(path)


class ExpertSourceIngestionService:
    def __init__(self, *, fetcher: SafeHTTPFetcher | None = None) -> None:
        self.fetcher = fetcher or SafeHTTPFetcher()

    def ingest(
        self,
        expert_id: str,
        source: str | Path,
        source_type: str,
        repositories: KnowledgeRepositories,
        *,
        domain: str,
        source_relationship: str,
        attribution_basis: str,
        attribution_status: str = "confirmed",
    ) -> ExpertSourceRecord:
        require_safe_path_component(expert_id, field="expert_id")
        repositories.expert_profiles.get(expert_id)
        source_text = str(source)
        parsed = urlparse(source_text)
        is_url = parsed.scheme.casefold() in {"http", "https"}
        if parsed.scheme and not is_url and not Path(source_text).exists():
            raise ValueError("expert source must be a local file or HTTP/HTTPS URL")
        if is_url:
            response = self.fetcher.fetch(
                source_text,
                allowed_content_types=frozenset(
                    {"application/pdf", "text/html", "text/plain", "text/markdown"}
                ),
            )
            raw = response.body
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
            if media_type == "application/pdf":
                canonical_text, pages = _pdf_text(raw)
                input_kind = "remote_pdf"
            elif media_type == "text/html":
                extraction = HTMLLiteratureTextExtractor().extract(source_text, raw)
                canonical_text = extraction.canonical_text
                pages = (
                    ExpertSourcePage(
                        page_number=1,
                        start_offset=0,
                        end_offset=len(canonical_text),
                    ),
                )
                input_kind = "public_webpage"
            else:
                canonical_text = _canonicalize_text(raw)
                pages = ()
                input_kind = "remote_text"
        else:
            path = Path(source).resolve()
            if path.is_symlink() or not path.is_file():
                raise ValueError("expert source must be a regular non-symlink file")
            raw = path.read_bytes()
            if path.suffix.casefold() == ".pdf":
                canonical_text, pages = _pdf_text(raw)
                input_kind = "local_pdf"
            elif path.suffix.casefold() in {".txt", ".md", ".markdown"}:
                canonical_text = _canonicalize_text(raw)
                pages = ()
                input_kind = "local_text"
            else:
                raise ValueError("expert local source must be UTF-8 text/markdown or PDF")
        source_role, source_document_type = _source_semantics(source_type, is_url=is_url)
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        canonical_bytes = canonical_text.encode("utf-8")
        canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        artifact_key = content_hash(
            {
                "expert_id": expert_id,
                "raw_sha256": raw_sha256,
                "canonical_sha256": canonical_sha256,
                "source_type": source_type,
            }
        )[:24]
        artifact_root = repositories.root / "expert_source_artifacts" / artifact_key
        raw_path = artifact_root / "raw.bin"
        canonical_path = artifact_root / "canonical.txt"
        _immutable_copy(raw_path, raw)
        _immutable_copy(canonical_path, canonical_bytes)
        source_id = f"expert-canonical-{artifact_key}"
        source_version = "1"
        evidence_store = KnowledgeEvidenceStore(repositories.root)
        source_document = evidence_store.ingest_source(
            canonical_path,
            source_id,
            source_version,
            title=f"Expert source for {expert_id}",
            source_role=source_role,
            source_type=source_document_type,
        )
        attribution_evidence_identity = {
            "source_id": source_id,
            "source_version": source_version,
            "start_offset": 0,
            "end_offset": len(canonical_text),
            "text_hash": content_hash({"text": canonical_text}),
            "purpose": "expert-attribution-source",
        }
        attribution_evidence = EvidenceSpan(
            evidence_id=f"evidence-{content_hash(attribution_evidence_identity)[:24]}",
            source_id=source_id,
            source_version=source_version,
            content_sha256=source_document.content_sha256,
            start_offset=0,
            end_offset=len(canonical_text),
            text=canonical_text,
            locator="entire expert source",
        )
        evidence_store.add_evidence(attribution_evidence)
        attribution_identity = {
            "expert_id": expert_id,
            "source_id": source_id,
            "source_version": source_version,
            "evidence_refs": (attribution_evidence.evidence_id,),
            "medium": source_type,
            "attribution_basis": attribution_basis,
            "attribution_status": attribution_status,
        }
        attribution_id = f"expert-attribution-{content_hash(attribution_identity)[:24]}"
        attribution_payload = {"attribution_id": attribution_id, **attribution_identity}
        attribution = ExpertAttributionRecord(
            **attribution_payload,
            content_hash=content_hash(attribution_payload),
        )
        _put_reuse(repositories.expert_attributions, attribution_id, attribution)
        source_identity = {
            "expert_id": expert_id,
            "source_relationship": source_relationship,
            "source_type": source_type,
            "domain": domain,
            "input_kind": input_kind,
            "raw_sha256": raw_sha256,
            "raw_artifact_path": raw_path.relative_to(repositories.root).as_posix(),
            "canonical_sha256": canonical_sha256,
            "source_id": source_id,
            "source_version": source_version,
            "source_hash": content_hash(source_document),
            "attribution_id": attribution_id,
            "attribution_hash": attribution.content_hash,
            "attribution_status": attribution_status,
            "page_ranges": pages,
        }
        expert_source_id = f"expert-source-{content_hash(source_identity)[:24]}"
        source_payload = {"expert_source_id": expert_source_id, **source_identity}
        record = ExpertSourceRecord(
            **source_payload,
            content_hash=content_hash(source_payload),
        )
        _put_reuse(repositories.expert_sources, expert_source_id, record)
        return record


def validate_expert_source(
    record: ExpertSourceRecord,
    repositories: KnowledgeRepositories,
) -> ExpertSourceRecord:
    stored = repositories.expert_sources.get(record.expert_source_id)
    if stored != record:
        raise ValueError("expert source differs from repository record")
    raw_path = repositories.root.joinpath(*Path(record.raw_artifact_path).parts)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError("expert source raw artifact is unavailable or unsafe")
    if not raw_path.resolve().is_relative_to(repositories.root.resolve()):
        raise ValueError("expert source raw artifact escapes knowledge root")
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != record.raw_sha256:
        raise ValueError("expert source raw artifact hash mismatch")
    source = repositories.evidence_store.get_source(record.source_id, record.source_version)
    repositories.evidence_store.verify_source_integrity(source)
    if content_hash(source) != record.source_hash or source.content_sha256 != record.canonical_sha256:
        raise ValueError("expert source canonical SourceDocument binding is invalid")
    attribution = repositories.expert_attributions.get(record.attribution_id)
    if attribution.content_hash != record.attribution_hash:
        raise ValueError("expert source attribution hash mismatch")
    if (
        attribution.expert_id != record.expert_id
        or attribution.source_id != record.source_id
        or attribution.source_version != record.source_version
        or attribution.attribution_status != record.attribution_status
    ):
        raise ValueError("expert source attribution binding is invalid")
    return record


__all__ = [
    "ExpertSourceIngestionService",
    "make_expert_profile",
    "validate_expert_source",
]
