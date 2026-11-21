from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import (
    DocumentStructureSelection,
    DomainProfile,
    EvidenceSpan,
    KnowledgeSnapshot,
    SourceDocument,
    SourceType,
)
from ..serialization import content_hash
from .evidence import CompositeEvidenceStore, EvidenceStore, KnowledgeEvidenceStore

if TYPE_CHECKING:
    from ..knowledge.trust import _ValidatedTrustedRead
    from .knowledge import KnowledgeRepositories


def create_snapshot(
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
    domain_profile: DomainProfile,
    *,
    _validated_read: _ValidatedTrustedRead | None = None,
) -> KnowledgeSnapshot:
    from ..knowledge.trust import TrustedKnowledgeValidator

    if isinstance(evidence_store, CompositeEvidenceStore):
        snapshot_evidence = evidence_store
    elif (
        isinstance(evidence_store, KnowledgeEvidenceStore)
        and evidence_store.root.resolve() == repositories.evidence_store.root.resolve()
    ):
        snapshot_evidence = evidence_store
    else:
        snapshot_evidence = CompositeEvidenceStore(
            repositories.evidence_store, evidence_store
        )
    if _validated_read is None:
        trusted = TrustedKnowledgeValidator(repositories, snapshot_evidence).validate()
    else:
        _validated_read.require_same_sources(repositories, snapshot_evidence)
        trusted = _validated_read.selection
    literature_source_keys = {
        (item.source_id, item.source_version)
        for item in repositories.literature_ingestions.list()
        if item.source_id is not None and item.source_version is not None
    }
    literature_source_keys.update(
        (item.source_id, item.source_version)
        for item in repositories.html_literature_ingestions.list()
    )
    # Uncurated supplementary evidence must not enter a trusted snapshot
    # merely because it lives in the shared EvidenceStore.
    literature_source_keys.update(
        (item.source_id, item.version)
        for item in repositories.evidence_store.list_sources()
        if item.source_type == SourceType.SUPPORTING_INFORMATION
    )
    trusted_evidence_ids = {
        record_id
        for (record_type, record_id) in trusted.trusted_records
        if record_type == "evidence_span"
    }
    trusted_evidence = {
        record_id: record
        for (record_type, record_id), record in trusted.trusted_records.items()
        if record_type == "evidence_span" and isinstance(record, EvidenceSpan)
    }
    trusted_source_keys = {
        (record.source_id, record.version)
        for (record_type, _), record in trusted.trusted_records.items()
        if record_type == "source_document"
        and isinstance(record, SourceDocument)
    }
    from ..knowledge.structure import (
        _verify_locator_against_validated_structure,
    )
    from ..knowledge.structure_selection import (
        resolve_current_structure_selection,
    )
    from ..knowledge.acquisition import validate_literature_representation

    trusted_representation_ids = {
        record_id
        for (record_type, record_id) in trusted.trusted_records
        if record_type == "literature_representation_ref"
    }
    trusted_structure_selections: list[DocumentStructureSelection] = []
    if _validated_read is not None:
        _validated_read.require_unchanged_structures()
    for representation_id in sorted(trusted_representation_ids):
        try:
            current = repositories.document_structure_selections.resolve_current(
                representation_id
            )
        except FileNotFoundError:
            continue
        cached = (
            _validated_read.structure_selections.get(current.selection_id)
            if _validated_read is not None else None
        )
        if cached is None:
            selection = resolve_current_structure_selection(
                representation_id, repositories, snapshot_evidence
            )
        else:
            if current != cached or cached.representation_id != representation_id:
                raise ValueError("validated structure selection changed")
            structure = repositories.document_structure_artifacts.get(cached.structure_id)
            if structure.content_hash != cached.structure_hash:
                raise ValueError("validated structure artifact changed")
            representation = repositories.literature_representation_refs.get(representation_id)
            validate_literature_representation(
                representation, repositories, snapshot_evidence
            )
            selection = cached
        trusted_structure_selections.append(selection)
    trusted_structures = tuple(
        repositories.document_structure_artifacts.get(selection.structure_id)
        for selection in trusted_structure_selections
    )
    trusted_structure_ids = {
        item.structure_id for item in trusted_structures
    }
    trusted_structure_by_id = {
        item.structure_id: item for item in trusted_structures
    }
    trusted_locators = tuple(
        _verify_locator_against_validated_structure(
            item, trusted_structure_by_id[item.structure_id], repositories, snapshot_evidence,
            verified_evidence=trusted_evidence[item.evidence_id],
        )
        for item in repositories.structured_evidence_locators.list()
        if item.structure_id in trusted_structure_ids
        and item.evidence_id in trusted_evidence_ids
    )
    # The trusted read rechecked cached structure files above. Evidence is
    # reverified by list_evidence() below before the snapshot can return.
    payload = {
        "domain_profile_hash": content_hash(domain_profile),
        "expert_case_hashes": {
            item.case_id: content_hash(item) for item in repositories.expert_cases.list()
        },
        "workflow_pattern_hashes": {
            item.pattern_id: content_hash(item) for item in repositories.workflow_patterns.list()
        },
        "capability_hashes": {
            item.capability_id: content_hash(item) for item in repositories.capabilities.list()
        },
        "evidence_span_hashes": {
            item.evidence_id: content_hash(item)
            for item in snapshot_evidence.list_evidence()
            if (item.source_id, item.source_version) not in literature_source_keys
            or item.evidence_id in trusted_evidence_ids
        },
        "evidence_source_versions": {
            f"{item.source_id}@{item.version}": item.content_sha256
            for item in snapshot_evidence.list_sources()
            if (item.source_id, item.version) not in literature_source_keys
            or (item.source_id, item.version) in trusted_source_keys
        },
        "literature_document_hashes": {
            item.literature_id: item.content_hash
            for item in trusted.literature_documents
        },
        "raw_literature_artifact_hashes": {
            record_id: record.content_hash
            for (record_type, record_id), record in trusted.trusted_records.items()
            if record_type == "raw_literature_artifact"
        },
        "canonical_text_artifact_hashes": {
            record_id: record.content_hash
            for (record_type, record_id), record in trusted.trusted_records.items()
            if record_type == "canonical_text_artifact"
        },
        "literature_ingestion_hashes": {
            record_id: record.content_hash
            for (record_type, record_id), record in trusted.trusted_records.items()
            if record_type == "literature_ingestion"
        },
        "literature_representation_selection_hashes": {
            record_id: record.content_hash
            for (record_type, record_id), record in trusted.trusted_records.items()
            if record_type == "literature_representation_selection"
        },
        "literature_representation_hashes": {
            record_id: record.content_hash
            for (record_type, record_id), record in trusted.trusted_records.items()
            if record_type == "literature_representation_ref"
        },
        "historical_evidence_authorization_hashes": {
            record_id: record.content_hash
            for (record_type, record_id), record in trusted.trusted_records.items()
            if record_type == "historical_evidence_authorization"
        },
        "expert_profile_hashes": {
            item.expert_id: item.content_hash
            for item in trusted.expert_profiles
        },
        "expert_opinion_hashes": {
            item.opinion_id: item.content_hash
            for item in trusted.expert_opinions
        },
        "expert_attribution_hashes": {
            item.attribution_id: item.content_hash
            for item in trusted.expert_attributions
        },
        "knowledge_relation_hashes": {
            item.relation_id: item.content_hash
            for item in trusted.relations
        },
        "curation_record_hashes": {
            item.curation_id: item.content_hash for item in trusted.curations
        },
        "document_structure_hashes": {
            item.structure_id: item.content_hash for item in trusted_structures
        },
        "document_structure_selection_hashes": {
            item.selection_id: item.content_hash
            for item in trusted_structure_selections
        },
        "structured_evidence_locator_hashes": {
            item.locator_id: item.content_hash for item in trusted_locators
        },
        "trusted_record_hashes": {
            f"{record_type}:{record_id}": (
                getattr(record, "content_hash", None) or content_hash(record)
            )
            for (record_type, record_id), record in trusted.trusted_records.items()
        }
        | {
            f"document_structure:{item.structure_id}": item.content_hash
            for item in trusted_structures
        }
        | {
            f"document_structure_selection:{item.selection_id}": item.content_hash
            for item in trusted_structure_selections
        }
        | {
            f"structured_evidence_locator:{item.locator_id}": item.content_hash
            for item in trusted_locators
        },
    }
    snapshot_identity = {
        key: value
        for key, value in payload.items()
        if value
        or key
        not in {
            "literature_document_hashes",
            "raw_literature_artifact_hashes",
            "canonical_text_artifact_hashes",
            "literature_ingestion_hashes",
            "literature_representation_selection_hashes",
            "literature_representation_hashes",
            "historical_evidence_authorization_hashes",
            "expert_profile_hashes",
            "expert_opinion_hashes",
            "expert_attribution_hashes",
            "knowledge_relation_hashes",
            "curation_record_hashes",
            "document_structure_hashes",
            "document_structure_selection_hashes",
            "structured_evidence_locator_hashes",
            "trusted_record_hashes",
        }
    }
    return KnowledgeSnapshot(
        snapshot_id=f"snapshot-{content_hash(snapshot_identity)[:24]}",
        **payload,
    )
