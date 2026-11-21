"""Verify and select literature representations against persisted source records."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    CanonicalTextArtifact,
    LiteratureIngestionRecord,
    LiteratureIngestionStatus,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    LiteratureRepresentationSelection,
    LiteratureRepresentationSelectionOutcome,
    RawLiteratureArtifact,
    SourceDocument,
    SourceRole,
    SourceType,
)
from ..repositories import EvidenceStore, KnowledgeRepositories
from ..serialization import content_hash


@dataclass(frozen=True)
class ValidatedLiteratureIngestionChain:
    ingestion: LiteratureIngestionRecord
    raw_artifact: RawLiteratureArtifact
    canonical_text: CanonicalTextArtifact
    source: SourceDocument


def validate_literature_ingestion_chain(
    ingestion: LiteratureIngestionRecord,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
) -> ValidatedLiteratureIngestionChain:
    if ingestion.ingestion_status != LiteratureIngestionStatus.ACCEPTED:
        raise ValueError("representation selection requires accepted ingestion")
    canonical = repositories.canonical_text_artifacts.get(
        ingestion.canonical_text_id or ""
    )
    raw = repositories.raw_literature_artifacts.get(ingestion.raw_artifact_id)
    if (
        raw.media_type != "application/pdf"
        or ingestion.raw_artifact_hash != raw.content_hash
        or ingestion.canonical_text_hash != canonical.content_hash
        or ingestion.literature_id != canonical.literature_id
        or ingestion.literature_id != raw.literature_id
        or canonical.raw_artifact_id != raw.artifact_id
        or canonical.raw_artifact_hash != raw.content_hash
        or ingestion.parser_id != canonical.parser_id
        or ingestion.parser_version != canonical.parser_version
        or ingestion.parser_config_hash != canonical.parser_config_hash
    ):
        raise ValueError("raw/canonical/ingestion binding is invalid")
    source = evidence_store.get_source(
        ingestion.source_id or "", ingestion.source_version or ""
    )
    evidence_store.verify_source_integrity(source)
    if (
        source.source_id != ingestion.source_id
        or source.version != ingestion.source_version
        or source.content_sha256 != canonical.text_sha256
        or source.source_role != SourceRole.LITERATURE_AUTHOR
        or source.source_type != SourceType.LITERATURE_ARTICLE
    ):
        raise ValueError("canonical/source binding is invalid")
    return ValidatedLiteratureIngestionChain(
        ingestion=ingestion,
        raw_artifact=raw,
        canonical_text=canonical,
        source=source,
    )


def validate_literature_representation(
    representation: LiteratureRepresentationReference,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
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
    source = evidence_store.get_source(
        representation.source_id, representation.source_version
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


class LiteratureRepresentationSelector:
    @staticmethod
    def _pdf_representation(
        ingestion: LiteratureIngestionRecord,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureRepresentationReference:
        chain = validate_literature_ingestion_chain(
            ingestion, repositories, evidence_store
        )
        identity = {
            "representation_kind": LiteratureRepresentationKind.PDF,
            "literature_id": ingestion.literature_id,
            "raw_artifact_id": chain.raw_artifact.artifact_id,
            "raw_artifact_hash": chain.raw_artifact.content_hash,
            "canonical_text_id": chain.canonical_text.canonical_text_id,
            "canonical_text_hash": chain.canonical_text.content_hash,
            "ingestion_id": ingestion.ingestion_id,
            "ingestion_hash": ingestion.content_hash,
            "source_id": chain.source.source_id,
            "source_version": chain.source.version,
        }
        representation_id = (
            f"literature-representation-{content_hash(identity)[:24]}"
        )
        payload = {"representation_id": representation_id, **identity}
        representation = LiteratureRepresentationReference(
            **payload, content_hash=content_hash(payload)
        )
        repositories.literature_representation_refs.put(
            representation.representation_id, representation
        )
        return representation

    @classmethod
    def resolve_reference(
        cls,
        representation_or_ingestion_id: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureRepresentationReference:
        try:
            representation = repositories.literature_representation_refs.get(
                representation_or_ingestion_id
            )
        except FileNotFoundError:
            ingestion = repositories.literature_ingestions.get(
                representation_or_ingestion_id
            )
            representation = cls._pdf_representation(
                ingestion, repositories, evidence_store
            )
        return validate_literature_representation(
            representation, repositories, evidence_store
        )

    @classmethod
    def resolve_selection(
        cls,
        selection: LiteratureRepresentationSelection,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureRepresentationReference:
        if selection.representation_id is not None:
            representation = cls.resolve_reference(
                selection.representation_id, repositories, evidence_store
            )
            if (
                selection.representation_hash != representation.content_hash
                or selection.literature_id != representation.literature_id
            ):
                raise ValueError("selection representation binding is invalid")
            return representation
        ingestion = repositories.literature_ingestions.get(
            selection.ingestion_id or ""
        )
        chain = validate_literature_ingestion_chain(
            ingestion, repositories, evidence_store
        )
        if (
            selection.literature_id != ingestion.literature_id
            or selection.ingestion_hash != ingestion.content_hash
            or selection.canonical_text_id != chain.canonical_text.canonical_text_id
            or selection.canonical_text_hash != chain.canonical_text.content_hash
            or selection.source_id != chain.source.source_id
            or selection.source_version != chain.source.version
        ):
            raise ValueError("selection binding is invalid")
        return cls._pdf_representation(ingestion, repositories, evidence_store)

    def select(
        self,
        literature_id: str,
        representation_id: str,
        selected_by: str,
        rationale: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureRepresentationSelectionOutcome:
        representation = self.resolve_reference(
            representation_id, repositories, evidence_store
        )
        if representation.literature_id != literature_id:
            raise ValueError("representation belongs to another literature work")
        repository = repositories.literature_representation_selections
        try:
            current = repository.resolve_current(literature_id)
        except FileNotFoundError:
            current = None
        identity = {
            "literature_id": literature_id,
            "representation_id": representation.representation_id,
            "representation_hash": representation.content_hash,
            "selected_by": selected_by,
            "rationale": rationale,
            "supersedes_selection_id": (
                current.selection_id if current is not None else None
            ),
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        selection_id = f"literature-selection-{content_hash(identity)[:24]}"
        payload = {"selection_id": selection_id, **identity}
        selection = LiteratureRepresentationSelection(
            **payload,
            content_hash=content_hash(payload),
        )
        repository.put(selection.selection_id, selection)
        if representation.representation_kind == LiteratureRepresentationKind.PDF:
            ingestion = repositories.literature_ingestions.get(
                representation.ingestion_id
            )
            total_pages = ingestion.total_pages
            pages_with_text = ingestion.pages_with_text
            pages_without_text = ingestion.pages_without_text
            text_coverage_ratio = ingestion.text_coverage_ratio
            warnings = ingestion.warnings
        else:
            ingestion = repositories.html_literature_ingestions.get(
                representation.ingestion_id
            )
            total_pages = None
            pages_with_text = None
            pages_without_text = None
            text_coverage_ratio = None
            warnings = ()
        return LiteratureRepresentationSelectionOutcome(
            selection=selection,
            ingestion_status=ingestion.ingestion_status,
            total_pages=total_pages,
            pages_with_text=pages_with_text,
            pages_without_text=pages_without_text,
            text_coverage_ratio=text_coverage_ratio,
            warnings=warnings,
        )
