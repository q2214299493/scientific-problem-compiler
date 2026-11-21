"""Ingest and validate supplementary sources against the main representation."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from ...models import (
    CanonicalTextArtifact,
    LiteratureSupplementarySource,
    RawLiteratureArtifact,
    SourceDocument,
    SourceRole,
    SourceType,
)
from ...repositories import EvidenceStore, KnowledgeEvidenceStore, KnowledgeRepositories
from ...serialization import content_hash
from ..literature_text import LiteratureTextExtractor
from ..representation import LiteratureRepresentationSelector
from .docx import DocxSupplementaryTextExtractor
from .pdf import PypdfLiteratureTextExtractor


def ingest_literature_supplementary_pdf(
    pdf_path: Path,
    literature_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: KnowledgeEvidenceStore,
    *,
    extractor: LiteratureTextExtractor | None = None,
) -> LiteratureSupplementarySource:
    """Store an independent SI PDF without replacing the selected main article."""
    return _ingest_literature_supplementary(
        pdf_path,
        literature_id,
        repositories,
        evidence_store,
        extractor=extractor or PypdfLiteratureTextExtractor(),
        allow_docx=False,
    )


def ingest_literature_supplementary_docx(
    docx_path: Path,
    literature_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: KnowledgeEvidenceStore,
) -> LiteratureSupplementarySource:
    """Store an original SI DOCX with a deterministic non-paginated text view."""
    return _ingest_literature_supplementary(
        docx_path,
        literature_id,
        repositories,
        evidence_store,
        extractor=DocxSupplementaryTextExtractor(),
        allow_docx=True,
    )


def _ingest_literature_supplementary(
    source_path: Path,
    literature_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: KnowledgeEvidenceStore,
    *,
    extractor: LiteratureTextExtractor,
    allow_docx: bool,
) -> LiteratureSupplementarySource:
    if allow_docx:
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("raw DOCX path must be a regular non-symlink file")
        with source_path.open("rb") as handle:
            if handle.read(4) != b"PK\x03\x04":
                raise ValueError("supplementary DOCX must retain its original DOCX format")
    if evidence_store.root.resolve() != repositories.evidence_store.root.resolve():
        raise ValueError("supplementary sources must use this knowledge repository's evidence store")
    document = repositories.literature_documents.get(literature_id)
    selection = repositories.literature_representation_selections.resolve_current(
        literature_id
    )
    main = LiteratureRepresentationSelector.resolve_selection(
        selection, repositories, evidence_store
    )
    raw = repositories.raw_literature_artifacts.put(
        source_path, literature_id, allow_docx=allow_docx
    )
    if raw.artifact_id == main.raw_artifact_id:
        raise ValueError("supplementary source must differ from the selected main article")
    raw_path = repositories.root.joinpath(*PurePosixPath(raw.stored_path).parts)
    extraction = extractor.extract(raw, raw_path)
    if extraction.requires_ocr or extraction.text is None:
        raise ValueError("supplementary source has no extractable canonical text")
    canonical = repositories.canonical_text_artifacts.put(
        extraction.text,
        literature_id=literature_id,
        raw_artifact=raw,
        parser_id=extractor.parser_id,
        parser_version=extractor.parser_version,
        parser_config_hash=extractor.parser_config_hash,
        page_count=extraction.total_pages,
        blocks=extraction.blocks,
    )
    canonical_path = repositories.root.joinpath(
        *PurePosixPath(canonical.stored_path).parts
    )
    source = evidence_store.ingest_source(
        canonical_path,
        f"source-supplement-{literature_id}",
        f"canonical-{canonical.canonical_text_id.removeprefix('canonical-text-')}",
        f"{document.title} [Supporting Information]",
        source_role=SourceRole.LITERATURE_AUTHOR,
        source_type=SourceType.SUPPORTING_INFORMATION,
    )
    identity = {
        "literature_id": literature_id,
        "literature_hash": document.content_hash,
        "main_selection_id": selection.selection_id,
        "main_selection_hash": selection.content_hash,
        "raw_artifact_id": raw.artifact_id,
        "raw_artifact_hash": raw.content_hash,
        "canonical_text_id": canonical.canonical_text_id,
        "canonical_text_hash": canonical.content_hash,
        "source_id": source.source_id,
        "source_version": source.version,
        "source_sha256": source.content_sha256,
    }
    supplement_id = f"literature-supplement-{content_hash(identity)[:24]}"
    payload = {"supplement_id": supplement_id, **identity}
    supplement = LiteratureSupplementarySource(
        **payload, content_hash=content_hash(payload)
    )
    repositories.literature_supplementary_sources.put(supplement_id, supplement)
    return supplement


def validate_literature_supplementary_source(
    supplement: LiteratureSupplementarySource,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
) -> tuple[RawLiteratureArtifact, CanonicalTextArtifact, SourceDocument]:
    """Verify raw/canonical/source binding for an independent SI source."""
    document = repositories.literature_documents.get(supplement.literature_id)
    selection = repositories.literature_representation_selections.get(
        supplement.main_selection_id
    )
    main = LiteratureRepresentationSelector.resolve_selection(
        selection, repositories, evidence_store
    )
    raw = repositories.raw_literature_artifacts.get(supplement.raw_artifact_id)
    canonical = repositories.canonical_text_artifacts.get(
        supplement.canonical_text_id
    )
    source = evidence_store.get_source(
        supplement.source_id, supplement.source_version
    )
    evidence_store.verify_source_integrity(source)
    if (
        document.content_hash != supplement.literature_hash
        or selection.literature_id != supplement.literature_id
        or selection.content_hash != supplement.main_selection_hash
        or main.literature_id != supplement.literature_id
        or main.raw_artifact_id == raw.artifact_id
        or main.source_id == source.source_id
        or raw.literature_id != supplement.literature_id
        or raw.content_hash != supplement.raw_artifact_hash
        or canonical.literature_id != supplement.literature_id
        or canonical.raw_artifact_id != raw.artifact_id
        or canonical.raw_artifact_hash != raw.content_hash
        or canonical.content_hash != supplement.canonical_text_hash
        or source.source_id != f"source-supplement-{supplement.literature_id}"
        or source.version
        != f"canonical-{canonical.canonical_text_id.removeprefix('canonical-text-')}"
        or source.content_sha256 != canonical.text_sha256
        or source.content_sha256 != supplement.source_sha256
        or source.source_role != SourceRole.LITERATURE_AUTHOR
        or source.source_type != SourceType.SUPPORTING_INFORMATION
    ):
        raise ValueError("supplementary raw/canonical/source binding is invalid")
    return raw, canonical, source
