from __future__ import annotations

from dataclasses import dataclass

from ..models import EvidenceSpan, SourceDocument
from ..repositories import (
    KnowledgeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
    source_documents_equivalent,
)


@dataclass(frozen=True)
class KnowledgeEvidenceMigrationResult:
    source_count: int
    evidence_count: int
    copied_source_count: int
    copied_evidence_count: int


def _literature_source_keys(
    repositories: KnowledgeRepositories,
) -> tuple[tuple[str, str], ...]:
    keys: set[tuple[str, str]] = set()
    for document in repositories.literature_documents.list():
        if document.source_id and document.source_version:
            keys.add((document.source_id, document.source_version))
    for ingestion in repositories.literature_ingestions.list():
        if ingestion.source_id and ingestion.source_version:
            keys.add((ingestion.source_id, ingestion.source_version))
    for ingestion in repositories.html_literature_ingestions.list():
        keys.add((ingestion.source_id, ingestion.source_version))
    for representation in repositories.literature_representation_refs.list():
        keys.add((representation.source_id, representation.source_version))
    return tuple(sorted(keys))


def _copy_source(
    source: SourceDocument,
    legacy_store: ProjectEvidenceStore,
    knowledge_store: KnowledgeEvidenceStore,
) -> bool:
    legacy_store.verify_source_integrity(source)
    try:
        existing = knowledge_store.get_source(source.source_id, source.version)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        knowledge_store.verify_source_integrity(existing)
        if not source_documents_equivalent(existing, source):
            raise ValueError(
                f"conflicting knowledge source identity: {source.source_id}--{source.version}"
            )
        return False
    copied = knowledge_store.import_source_record(
        source, legacy_store.read_source_bytes(source)
    )
    knowledge_store.verify_source_integrity(copied)
    return True


def _copy_evidence(
    evidence: EvidenceSpan,
    legacy_store: ProjectEvidenceStore,
    knowledge_store: KnowledgeEvidenceStore,
) -> bool:
    legacy_store.verify_evidence_integrity(evidence)
    try:
        existing = knowledge_store.get_evidence(evidence.evidence_id)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        knowledge_store.verify_evidence_integrity(existing)
        if existing != evidence:
            raise ValueError(
                f"conflicting knowledge EvidenceSpan identity: {evidence.evidence_id}"
            )
        return False
    knowledge_store.add_evidence(evidence)
    knowledge_store.verify_evidence_integrity(evidence)
    return True


def migrate_knowledge_evidence(
    repositories: KnowledgeRepositories,
    legacy_store: ProjectEvidenceStore,
    knowledge_store: KnowledgeEvidenceStore,
) -> KnowledgeEvidenceMigrationResult:
    """Copy legacy literature evidence into shared storage without deletion."""

    source_keys = _literature_source_keys(repositories)
    copied_sources = 0
    for source_id, source_version in source_keys:
        source = legacy_store.get_source(source_id, source_version)
        copied_sources += _copy_source(source, legacy_store, knowledge_store)

    relevant_keys = set(source_keys)
    evidence_records = tuple(
        evidence
        for evidence in legacy_store.list_evidence()
        if (evidence.source_id, evidence.source_version) in relevant_keys
    )
    copied_evidence = 0
    for evidence in evidence_records:
        copied_evidence += _copy_evidence(evidence, legacy_store, knowledge_store)

    return KnowledgeEvidenceMigrationResult(
        source_count=len(source_keys),
        evidence_count=len(evidence_records),
        copied_source_count=copied_sources,
        copied_evidence_count=copied_evidence,
    )
