from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from ...immutable import FrozenDict
from ...models import (
    CurationStatus,
    DocumentContentRegion,
    EpistemicStatus,
    KnowledgePredicate,
    LiteratureRepresentationKind,
    NonBlankStr,
    ResultStatus,
    Sha256Str,
    StrictModel,
)
from ...serialization import content_hash


def _validate_content_bound(model: StrictModel, *, id_field: str, prefix: str) -> None:
    identity = model.model_dump(
        mode="json",
        exclude={id_field, "content_hash"},
        exclude_none=True,
    )
    expected_id = f"{prefix}-{content_hash(identity)[:24]}"
    if getattr(model, id_field) != expected_id:
        raise ValueError(f"{type(model).__name__} identity is not content-bound")
    payload = {id_field: expected_id, **identity}
    if getattr(model, "content_hash") != content_hash(payload):
        raise ValueError(f"{type(model).__name__} content_hash is invalid")


class LiteratureKnowledgeRecordType(StrEnum):
    SOURCE_CLAIM = "source_claim"
    METHOD_FACT = "method_fact"
    MODEL_FACT = "model_fact"
    REPORTED_RESULT = "reported_result"


class LiteratureKnowledgeViewMode(StrEnum):
    AUDIT = "audit"
    TRUSTED_CURRENT = "trusted_current"


class LiteratureKnowledgeCompilationInput(StrictModel):
    compilation_input_id: NonBlankStr
    literature_id: NonBlankStr
    literature_hash: Sha256Str
    representation_selection_id: NonBlankStr
    representation_selection_hash: Sha256Str
    representation_id: NonBlankStr
    representation_hash: Sha256Str
    representation_kind: LiteratureRepresentationKind
    canonical_text_id: NonBlankStr
    canonical_text_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    structure_selection_id: NonBlankStr
    structure_selection_hash: Sha256Str
    structure_id: NonBlankStr
    structure_hash: Sha256Str
    compiler_policy_id: NonBlankStr
    compiler_policy_version: NonBlankStr
    included_content_regions: tuple[DocumentContentRegion, ...]
    chunk_policy_id: NonBlankStr
    chunk_policy_version: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureKnowledgeCompilationInput:
        if self.included_content_regions != tuple(sorted(set(self.included_content_regions))):
            raise ValueError("included content regions must be unique and sorted")
        _validate_content_bound(
            self,
            id_field="compilation_input_id",
            prefix="literature-knowledge-input",
        )
        return self


class LiteratureKnowledgeChunk(StrictModel):
    chunk_id: NonBlankStr
    literature_id: NonBlankStr
    representation_id: NonBlankStr
    structure_id: NonBlankStr
    block_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    block_hashes: dict[NonBlankStr, Sha256Str] = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[NonBlankStr, ...] = ()
    content_region: DocumentContentRegion
    region_uncertain: bool = False
    canonical_start_offset: int = Field(ge=0)
    canonical_end_offset: int = Field(gt=0)
    text: NonBlankStr
    table_context: FrozenDict | None = None
    figure_context: FrozenDict | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureKnowledgeChunk:
        if self.canonical_end_offset <= self.canonical_start_offset:
            raise ValueError("literature knowledge chunk range is invalid")
        if set(self.block_refs) != set(self.block_hashes):
            raise ValueError("chunk block refs and hashes differ")
        if self.region_uncertain != (self.content_region == DocumentContentRegion.UNKNOWN):
            raise ValueError("region_uncertain must exactly describe UNKNOWN content")
        _validate_content_bound(self, id_field="chunk_id", prefix="literature-knowledge-chunk")
        return self


class LiteratureQuoteProposal(StrictModel):
    quote_key: NonBlankStr
    chunk_id: NonBlankStr
    block_id: NonBlankStr
    exact_text: NonBlankStr


class LiteratureClaimProposal(StrictModel):
    claim_key: NonBlankStr
    text: NonBlankStr
    claim_type: NonBlankStr
    quote_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    claim_strength: NonBlankStr
    epistemic_status: EpistemicStatus


class LiteratureMethodFactProposal(StrictModel):
    fact_key: NonBlankStr
    text: NonBlankStr
    claim_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    attributes: FrozenDict = FrozenDict()


class LiteratureModelFactProposal(StrictModel):
    fact_key: NonBlankStr
    text: NonBlankStr
    claim_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    attributes: FrozenDict = FrozenDict()


class LiteratureReportedResultProposal(StrictModel):
    result_key: NonBlankStr
    claim_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    quantity: NonBlankStr
    value: float = Field(allow_inf_nan=False)
    unit: NonBlankStr | None = None
    system_context: FrozenDict
    method_context: FrozenDict
    method_fact_keys: tuple[NonBlankStr, ...] = ()
    model_fact_keys: tuple[NonBlankStr, ...] = ()
    result_status: ResultStatus


class LiteratureRelationProposal(StrictModel):
    relation_key: NonBlankStr
    subject_type: LiteratureKnowledgeRecordType
    subject_key: NonBlankStr
    predicate: KnowledgePredicate
    object_type: LiteratureKnowledgeRecordType
    object_key: NonBlankStr
    rationale: NonBlankStr


class LiteratureKnowledgeLLMResponse(StrictModel):
    quote_proposals: tuple[LiteratureQuoteProposal, ...] = ()
    claim_proposals: tuple[LiteratureClaimProposal, ...] = ()
    method_fact_proposals: tuple[LiteratureMethodFactProposal, ...] = ()
    model_fact_proposals: tuple[LiteratureModelFactProposal, ...] = ()
    reported_result_proposals: tuple[LiteratureReportedResultProposal, ...] = ()
    relation_proposals: tuple[LiteratureRelationProposal, ...] = ()


class LiteratureKnowledgeProposalSet(StrictModel):
    proposal_set_id: NonBlankStr
    compilation_input_id: NonBlankStr
    compilation_input_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config_hash: Sha256Str
    quote_proposals: tuple[LiteratureQuoteProposal, ...] = ()
    claim_proposals: tuple[LiteratureClaimProposal, ...] = ()
    method_fact_proposals: tuple[LiteratureMethodFactProposal, ...] = ()
    model_fact_proposals: tuple[LiteratureModelFactProposal, ...] = ()
    reported_result_proposals: tuple[LiteratureReportedResultProposal, ...] = ()
    relation_proposals: tuple[LiteratureRelationProposal, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureKnowledgeProposalSet:
        for values, field in (
            (self.quote_proposals, "quote_key"),
            (self.claim_proposals, "claim_key"),
            (self.method_fact_proposals, "fact_key"),
            (self.model_fact_proposals, "fact_key"),
            (self.reported_result_proposals, "result_key"),
            (self.relation_proposals, "relation_key"),
        ):
            keys = tuple(getattr(item, field) for item in values)
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate proposal-local key: {field}")
        _validate_content_bound(
            self,
            id_field="proposal_set_id",
            prefix="literature-knowledge-proposals",
        )
        return self


class LiteratureKnowledgeGroundingRecord(StrictModel):
    grounding_id: NonBlankStr
    literature_id: NonBlankStr
    literature_hash: Sha256Str
    representation_selection_id: NonBlankStr
    representation_selection_hash: Sha256Str
    representation_id: NonBlankStr
    representation_hash: Sha256Str
    structure_selection_id: NonBlankStr
    structure_selection_hash: Sha256Str
    structure_id: NonBlankStr
    structure_hash: Sha256Str
    block_id: NonBlankStr
    block_hash: Sha256Str
    evidence_id: NonBlankStr
    evidence_hash: Sha256Str
    locator_id: NonBlankStr
    locator_hash: Sha256Str
    quote_id: NonBlankStr
    content_region: DocumentContentRegion
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureKnowledgeGroundingRecord:
        _validate_content_bound(
            self,
            id_field="grounding_id",
            prefix="literature-knowledge-grounding",
        )
        return self


class RejectedLiteratureKnowledgeProposal(StrictModel):
    proposal_ref: NonBlankStr
    proposal_kind: NonBlankStr
    rejection_code: NonBlankStr


class LiteratureKnowledgeCompilationRecord(StrictModel):
    compilation_id: NonBlankStr
    compilation_input_id: NonBlankStr
    compilation_input_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config_hash: Sha256Str
    proposal_set_id: NonBlankStr
    proposal_set_hash: Sha256Str
    representation_selection_id: NonBlankStr
    representation_selection_hash: Sha256Str
    structure_selection_id: NonBlankStr
    structure_selection_hash: Sha256Str
    source_quote_ids: tuple[NonBlankStr, ...] = ()
    source_claim_ids: tuple[NonBlankStr, ...] = ()
    method_fact_ids: tuple[NonBlankStr, ...] = ()
    model_fact_ids: tuple[NonBlankStr, ...] = ()
    reported_result_ids: tuple[NonBlankStr, ...] = ()
    knowledge_relation_ids: tuple[NonBlankStr, ...] = ()
    grounding_hashes: dict[NonBlankStr, Sha256Str] = Field(default_factory=dict)
    rejected_proposals: tuple[RejectedLiteratureKnowledgeProposal, ...] = ()
    compiler_id: NonBlankStr
    compiler_version: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureKnowledgeCompilationRecord:
        for field in (
            "source_quote_ids",
            "source_claim_ids",
            "method_fact_ids",
            "model_fact_ids",
            "reported_result_ids",
            "knowledge_relation_ids",
        ):
            values = getattr(self, field)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be unique and sorted")
        _validate_content_bound(
            self,
            id_field="compilation_id",
            prefix="literature-knowledge-compilation",
        )
        return self


class LiteratureScientificKnowledgeViewRecord(StrictModel):
    record_type: NonBlankStr
    record_id: NonBlankStr
    record_hash: Sha256Str
    curation_status: CurationStatus | None = None
    grounding_refs: tuple[NonBlankStr, ...] = ()
    content_regions: tuple[DocumentContentRegion, ...] = ()
    section_paths: tuple[tuple[NonBlankStr, ...], ...] = ()


class LiteratureScientificKnowledgeView(StrictModel):
    view_id: NonBlankStr
    literature_id: NonBlankStr
    view_mode: LiteratureKnowledgeViewMode
    records: tuple[LiteratureScientificKnowledgeViewRecord, ...]
    compilation_ids: tuple[NonBlankStr, ...]
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureScientificKnowledgeView:
        _validate_content_bound(self, id_field="view_id", prefix="literature-knowledge-view")
        return self
