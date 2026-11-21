"""Ingest a local literature PDF and bind the canonical source."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ...models import (
    CanonicalHTMLTextArtifact,
    CanonicalTextArtifact,
    HTMLLiteratureIngestionRecord,
    RawHTMLLiteratureArtifact,
    SourceDocument,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    LiteratureDocument,
    LiteratureIngestionOutcome,
    LiteratureIngestionRecord,
    LiteratureIngestionStatus,
    RawLiteratureArtifact,
    literature_identity_id,
)
from ...repositories import EvidenceStore, KnowledgeRepositories
from ...serialization import content_hash
from ..representation import build_representation_reference, validate_literature_representation
from ..literature_text import (
    HTMLTextExtraction,
    HTMLTextExtractor,
    LiteratureExtractionError,
    LiteratureMetadata,
    LiteratureTextExtraction,
    LiteratureTextExtractor,
)
from .pdf import PypdfLiteratureTextExtractor


class LiteratureIngestionService:
    def __init__(
        self,
        extractor: LiteratureTextExtractor | None = None,
    ) -> None:
        self.extractor = extractor or PypdfLiteratureTextExtractor()

    def ingest(
        self,
        pdf_path: Path,
        metadata: LiteratureMetadata | Mapping[str, Any],
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureIngestionOutcome:
        metadata_record = (
            metadata
            if isinstance(metadata, LiteratureMetadata)
            else LiteratureMetadata.model_validate(metadata)
        )
        literature_id = literature_identity_id(
            title=metadata_record.title,
            authors=metadata_record.authors,
            year=metadata_record.year,
            doi=metadata_record.doi,
        )
        raw_artifact = repositories.raw_literature_artifacts.put(
            pdf_path, literature_id
        )
        raw_path = repositories.root.joinpath(
            *PurePosixPath(raw_artifact.stored_path).parts
        )
        try:
            extraction = self.extractor.extract(raw_artifact, raw_path)
        except LiteratureExtractionError:
            ingestion = self._make_ingestion(
                literature_id=literature_id,
                raw_artifact=raw_artifact,
                status=LiteratureIngestionStatus.FAILED,
                warnings=("PDF extraction failed",),
            )
            repositories.literature_ingestions.put(
                ingestion.ingestion_id, ingestion
            )
            return LiteratureIngestionOutcome(
                literature_id=literature_id,
                artifact_id=raw_artifact.artifact_id,
                ingestion_id=ingestion.ingestion_id,
                warnings=ingestion.warnings,
            )
        if extraction.requires_ocr:
            ingestion = self._make_ingestion(
                literature_id=literature_id,
                raw_artifact=raw_artifact,
                status=LiteratureIngestionStatus.REQUIRES_OCR,
                warnings=extraction.warnings,
                extraction=extraction,
            )
            repositories.literature_ingestions.put(
                ingestion.ingestion_id, ingestion
            )
            return LiteratureIngestionOutcome(
                literature_id=literature_id,
                artifact_id=raw_artifact.artifact_id,
                ingestion_id=ingestion.ingestion_id,
                warnings=ingestion.warnings,
                requires_ocr=True,
            )
        if extraction.text is None:
            raise ValueError("extractor returned no canonical text without requires_ocr")
        canonical = repositories.canonical_text_artifacts.put(
            extraction.text,
            literature_id=literature_id,
            raw_artifact=raw_artifact,
            parser_id=self.extractor.parser_id,
            parser_version=self.extractor.parser_version,
            parser_config_hash=self.extractor.parser_config_hash,
            page_count=len(extraction.page_texts),
            blocks=extraction.blocks,
        )
        canonical_path = repositories.root.joinpath(
            *PurePosixPath(canonical.stored_path).parts
        )
        source_id = f"source-{literature_id}"
        source_version = f"canonical-{canonical.canonical_text_id.removeprefix('canonical-text-')}"
        source = evidence_store.ingest_source(
            canonical_path,
            source_id,
            source_version,
            metadata_record.title,
            source_role="literature_author",
            source_type="literature_article",
        )
        ingestion = self._make_ingestion(
            literature_id=literature_id,
            raw_artifact=raw_artifact,
            canonical=canonical,
            source_id=source.source_id,
            source_version=source.version,
            status=LiteratureIngestionStatus.ACCEPTED,
            warnings=extraction.warnings,
            extraction=extraction,
        )
        repositories.literature_ingestions.put(ingestion.ingestion_id, ingestion)
        finalize_literature_document(
            metadata_record,
            literature_id,
            raw_artifact,
            canonical,
            source.source_id,
            source.version,
            repositories,
        )
        return LiteratureIngestionOutcome(
            literature_id=literature_id,
            artifact_id=raw_artifact.artifact_id,
            canonical_text_id=canonical.canonical_text_id,
            ingestion_id=ingestion.ingestion_id,
            source_id=source.source_id,
            source_version=source.version,
            warnings=ingestion.warnings,
            requires_ocr=False,
        )

    def _make_ingestion(
        self,
        *,
        literature_id: str,
        raw_artifact: RawLiteratureArtifact,
        status: LiteratureIngestionStatus,
        warnings: tuple[str, ...],
        extraction: LiteratureTextExtraction | None = None,
        canonical: CanonicalTextArtifact | None = None,
        source_id: str | None = None,
        source_version: str | None = None,
    ) -> LiteratureIngestionRecord:
        identity = {
            "literature_id": literature_id,
            "raw_artifact_id": raw_artifact.artifact_id,
            "raw_artifact_hash": raw_artifact.content_hash,
            "canonical_text_id": (
                canonical.canonical_text_id if canonical is not None else None
            ),
            "canonical_text_hash": (
                canonical.content_hash if canonical is not None else None
            ),
            "source_id": source_id,
            "source_version": source_version,
            "parser_id": self.extractor.parser_id,
            "parser_version": self.extractor.parser_version,
            "parser_config_hash": self.extractor.parser_config_hash,
            "ingestion_status": status,
            "total_pages": extraction.total_pages if extraction is not None else 0,
            "pages_with_text": (
                extraction.pages_with_text if extraction is not None else 0
            ),
            "pages_without_text": (
                extraction.pages_without_text if extraction is not None else 0
            ),
            "text_coverage_ratio": (
                extraction.text_coverage_ratio if extraction is not None else 0.0
            ),
            "warnings": warnings,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        ingestion_id = f"literature-ingestion-{content_hash(identity)[:24]}"
        payload = {"ingestion_id": ingestion_id, **identity}
        return LiteratureIngestionRecord(
            **payload,
            content_hash=content_hash(payload),
        )


def finalize_literature_document(
    metadata: LiteratureMetadata,
    literature_id: str,
    raw_artifact: RawLiteratureArtifact | RawHTMLLiteratureArtifact,
    canonical: CanonicalTextArtifact | CanonicalHTMLTextArtifact,
    source_id: str,
    source_version: str,
    repositories: KnowledgeRepositories,
) -> LiteratureDocument:
    try:
        return repositories.literature_documents.get(literature_id)
    except FileNotFoundError:
        pass
    identity = {
        **metadata.model_dump(mode="json", exclude_none=True),
        "literature_id": literature_id,
        "raw_artifact_ref": raw_artifact.stored_path,
        "canonical_text_ref": canonical.stored_path,
        "source_id": source_id,
        "source_version": source_version,
    }
    document = LiteratureDocument(
        **identity,
        content_hash=content_hash(identity),
    )
    repositories.literature_documents.put(literature_id, document)
    return document


def _html_ingestion_record(
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


def ingest_extracted_html(
    *,
    body: bytes,
    source_url: str,
    media_type: str,
    extraction: HTMLTextExtraction,
    extractor: HTMLTextExtractor,
    metadata: LiteratureMetadata,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
) -> LiteratureRepresentationReference:
    """Persist the fetched bytes, their canonical blocks, and the verified source binding."""
    literature_id = literature_identity_id(
        title=metadata.title,
        authors=metadata.authors,
        year=metadata.year,
        doi=metadata.doi,
    )
    raw = repositories.raw_html_literature_artifacts.put(
        body,
        source_url=source_url,
        literature_id=literature_id,
        media_type=media_type,
    )
    canonical = repositories.canonical_html_text_artifacts.put(
        extraction.canonical_text,
        raw_artifact=raw,
        extractor_id=extractor.extractor_id,
        extractor_version=extractor.extractor_version,
        extractor_config_hash=extractor.extractor_config_hash,
        blocks=extraction.blocks,
    )
    canonical_path = repositories.root.joinpath(
        *Path(canonical.stored_path).parts
    )
    source = evidence_store.ingest_source(
        canonical_path,
        f"source-{literature_id}",
        f"html-{canonical.canonical_text_id.removeprefix('canonical-html-text-')}",
        metadata.title,
        source_role="literature_author",
        source_type="literature_article",
    )
    ingestion = _html_ingestion_record(raw, canonical, source)
    repositories.html_literature_ingestions.put(
        ingestion.ingestion_id, ingestion
    )
    finalize_literature_document(
        metadata, literature_id, raw, canonical, source.source_id, source.version, repositories
    )
    representation = build_representation_reference(
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
    return representation
