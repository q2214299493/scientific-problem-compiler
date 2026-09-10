from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Mapping

from pydantic import BaseModel

from ..models import CurationStatus, KnowledgeCurationRecord

if TYPE_CHECKING:
    from ..repositories import EvidenceStore, KnowledgeRepositories


K1F_RECORD_FIELDS = {
    "source_claim": ("source_claim_ids", "source_claims", "claim_id"),
    "method_fact": ("method_fact_ids", "method_facts", "fact_id"),
    "model_fact": ("model_fact_ids", "model_facts", "fact_id"),
    "reported_result": ("reported_result_ids", "reported_results", "result_id"),
    "knowledge_relation": (
        "knowledge_relation_ids",
        "relations",
        "relation_id",
    ),
}


class K1FScientificAuthorityStatus(StrEnum):
    NOT_K1F = "not_k1f"
    CURRENT = "current"
    STALE = "stale"


def _require_accepted_curation(
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord],
    record_type: str,
    record_id: str,
) -> None:
    curation = current_curations.get((record_type, record_id))
    if curation is None or curation.status != CurationStatus.ACCEPTED:
        raise ValueError(f"current K1F authority is not accepted: {record_type}:{record_id}")


def _validate_compilation_input(
    compilation,
    repositories: KnowledgeRepositories,
    store: EvidenceStore,
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord],
):
    from .ingestion import LiteratureRepresentationSelector
    from .structure import validate_document_structure
    from .structure_selection import (
        resolve_current_structure_selection,
        validate_structure_selection,
    )

    record = repositories.literature_knowledge_inputs.get(
        compilation.compilation_input_id
    )
    if (
        record.content_hash != compilation.compilation_input_hash
        or record.representation_selection_id != compilation.representation_selection_id
        or record.representation_selection_hash
        != compilation.representation_selection_hash
        or record.structure_selection_id != compilation.structure_selection_id
        or record.structure_selection_hash != compilation.structure_selection_hash
    ):
        raise ValueError("K1F compilation input binding is invalid")
    proposal = repositories.literature_knowledge_proposals.get(
        compilation.proposal_set_id
    )
    if (
        proposal.content_hash != compilation.proposal_set_hash
        or proposal.compilation_input_id != record.compilation_input_id
        or proposal.compilation_input_hash != record.content_hash
        or proposal.provider_id != compilation.provider_id
        or proposal.provider_version != compilation.provider_version
        or proposal.provider_config_hash != compilation.provider_config_hash
    ):
        raise ValueError("K1F compilation proposal binding is invalid")
    document = repositories.literature_documents.get(record.literature_id)
    selection = repositories.literature_representation_selections.get(
        record.representation_selection_id
    )
    representation = LiteratureRepresentationSelector.resolve_selection(
        selection, repositories, store
    )
    structure_selection = validate_structure_selection(
        record.structure_selection_id, repositories, store
    )
    structure = validate_document_structure(record.structure_id, repositories, store)
    actual = (
        document.content_hash,
        selection.content_hash,
        representation.content_hash,
        representation.representation_kind,
        representation.canonical_text_id,
        representation.canonical_text_hash,
        representation.source_id,
        representation.source_version,
        structure_selection.content_hash,
        structure.content_hash,
    )
    expected = (
        record.literature_hash,
        record.representation_selection_hash,
        record.representation_hash,
        record.representation_kind,
        record.canonical_text_id,
        record.canonical_text_hash,
        record.source_id,
        record.source_version,
        record.structure_selection_hash,
        record.structure_hash,
    )
    if actual != expected:
        raise ValueError("K1F compilation authority hash binding is invalid")
    current_representation = (
        repositories.literature_representation_selections.resolve_current(
            record.literature_id
        )
    )
    if current_representation.selection_id != selection.selection_id:
        return False, record
    current_structure = resolve_current_structure_selection(
        representation.representation_id, repositories, store
    )
    if current_structure.selection_id != structure_selection.selection_id:
        return False, record
    _require_accepted_curation(
        current_curations, "literature_document", document.literature_id
    )
    _require_accepted_curation(
        current_curations,
        "literature_representation_selection",
        selection.selection_id,
    )
    return True, record


def _record_grounding_requirements(
    record_type: str,
    record: BaseModel,
) -> tuple[set[str], set[str]]:
    if record_type == "source_claim":
        return set(record.source_quote_refs), set(record.evidence_refs)
    return set(), set(record.evidence_refs)


def validate_k1f_compilation_authority(
    compilation,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord],
) -> K1FScientificAuthorityStatus:
    from .literature_knowledge.validation import validate_grounding_record

    is_current, compilation_input = _validate_compilation_input(
        compilation, repositories, evidence_store, current_curations
    )
    groundings = tuple(
        repositories.literature_knowledge_groundings.get(grounding_id)
        for grounding_id in compilation.grounding_hashes
    )
    if {item.grounding_id: item.content_hash for item in groundings} != dict(
        compilation.grounding_hashes
    ):
        raise ValueError("K1F compilation grounding inventory is invalid")
    for grounding in groundings:
        validate_grounding_record(
            grounding,
            repositories,
            evidence_store,
            require_current=is_current,
        )
        if (
            grounding.literature_id != compilation_input.literature_id
            or grounding.representation_selection_id
            != compilation_input.representation_selection_id
            or grounding.structure_selection_id
            != compilation_input.structure_selection_id
        ):
            raise ValueError("K1F grounding belongs to another compilation authority")
    quote_ids = {item.quote_id for item in groundings}
    evidence_ids = {item.evidence_id for item in groundings}
    if quote_ids != set(compilation.source_quote_ids):
        raise ValueError("K1F compilation SourceQuote grounding coverage is incomplete")
    for record_type, (field_name, repository_name, identity_field) in K1F_RECORD_FIELDS.items():
        for record_id in getattr(compilation, field_name):
            record = getattr(repositories, repository_name).get(record_id)
            if getattr(record, identity_field) != record_id:
                raise ValueError("K1F scientific record identity binding is invalid")
            required_quotes, required_evidence = _record_grounding_requirements(
                record_type, record
            )
            if not required_quotes.issubset(quote_ids) or not required_evidence.issubset(
                evidence_ids
            ):
                raise ValueError(
                    f"K1F grounding coverage is incomplete for {record_type}:{record_id}"
                )
    return (
        K1FScientificAuthorityStatus.CURRENT
        if is_current
        else K1FScientificAuthorityStatus.STALE
    )


def classify_k1f_scientific_authority(
    record_type: str,
    record_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord],
) -> K1FScientificAuthorityStatus:
    specification = K1F_RECORD_FIELDS.get(record_type)
    if specification is None:
        return K1FScientificAuthorityStatus.NOT_K1F
    field_name = specification[0]
    compilations = tuple(
        compilation
        for compilation in repositories.literature_knowledge_compilations.list()
        if record_id in getattr(compilation, field_name)
    )
    if not compilations:
        return K1FScientificAuthorityStatus.NOT_K1F
    statuses = {
        validate_k1f_compilation_authority(
            compilation,
            repositories,
            evidence_store,
            current_curations,
        )
        for compilation in compilations
    }
    if K1FScientificAuthorityStatus.CURRENT in statuses:
        return K1FScientificAuthorityStatus.CURRENT
    return K1FScientificAuthorityStatus.STALE
