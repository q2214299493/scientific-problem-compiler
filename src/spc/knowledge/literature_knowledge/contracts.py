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


class LiteratureKnowledgeProviderInvocation(StrictModel):
    invocation_id: NonBlankStr
    transport_id: NonBlankStr
    transport_version: NonBlankStr
    runtime_version: NonBlankStr
    selected_model: NonBlankStr | None = None
    invocation_config_hash: Sha256Str
    input_hash: Sha256Str
    output_hash: Sha256Str
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureKnowledgeProviderInvocation:
        _validate_content_bound(
            self,
            id_field="invocation_id",
            prefix="literature-knowledge-provider-invocation",
        )
        return self


class LiteratureKnowledgeBatchInvocation(StrictModel):
    batch_index: int = Field(ge=1)
    chunk_ids: tuple[NonBlankStr, ...] = Field(min_length=1)
    chunk_hashes: tuple[Sha256Str, ...] = Field(min_length=1)
    input_hash: Sha256Str
    output_hash: Sha256Str
    selected_model: NonBlankStr
    runtime_version: NonBlankStr | None = None
    provider_invocation: LiteratureKnowledgeProviderInvocation | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_binding(self) -> LiteratureKnowledgeBatchInvocation:
        if len(self.chunk_ids) != len(self.chunk_hashes):
            raise ValueError("batch chunk IDs and hashes must align")
        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("batch chunk IDs must be unique")
        if self.provider_invocation is not None:
            if (
                self.provider_invocation.input_hash != self.input_hash
                or self.provider_invocation.output_hash != self.output_hash
                or self.provider_invocation.selected_model != self.selected_model
                or self.provider_invocation.runtime_version != self.runtime_version
            ):
                raise ValueError("batch provider invocation binding is invalid")
        identity = self.model_dump(
            mode="json",
            exclude={"content_hash"},
            exclude_none=True,
        )
        if self.content_hash != content_hash(identity):
            raise ValueError("batch invocation content_hash is invalid")
        return self


class LiteratureKnowledgeProposalSet(StrictModel):
    proposal_set_id: NonBlankStr
    compilation_input_id: NonBlankStr
    compilation_input_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config_hash: Sha256Str
    provider_invocation: LiteratureKnowledgeProviderInvocation | None = None
    batch_invocations: tuple[LiteratureKnowledgeBatchInvocation, ...] | None = None
    quote_proposals: tuple[LiteratureQuoteProposal, ...] = ()
    claim_proposals: tuple[LiteratureClaimProposal, ...] = ()
    method_fact_proposals: tuple[LiteratureMethodFactProposal, ...] = ()
    model_fact_proposals: tuple[LiteratureModelFactProposal, ...] = ()
    reported_result_proposals: tuple[LiteratureReportedResultProposal, ...] = ()
    relation_proposals: tuple[LiteratureRelationProposal, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureKnowledgeProposalSet:
        batch_indices = tuple(
            item.batch_index for item in (self.batch_invocations or ())
        )
        if batch_indices != tuple(sorted(set(batch_indices))):
            raise ValueError("batch invocation indices must be unique and sorted")
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


class LiteratureKnowledgeSupportingQuoteView(StrictModel):
    quote_id: NonBlankStr
    evidence_id: NonBlankStr
    exact_text: NonBlankStr
    locator_id: NonBlankStr
    locator: NonBlankStr
    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[NonBlankStr, ...] = ()
    table_id: NonBlankStr | None = None
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    content_region: DocumentContentRegion
    region_uncertain: bool = False

    @model_validator(mode="after")
    def validate_uncertainty(self) -> LiteratureKnowledgeSupportingQuoteView:
        if self.region_uncertain != (
            self.content_region == DocumentContentRegion.UNKNOWN
        ):
            raise ValueError("quote-view uncertainty must describe UNKNOWN content")
        return self


class LiteratureScientificKnowledgeViewRecord(StrictModel):
    record_type: NonBlankStr
    record_id: NonBlankStr
    record_hash: Sha256Str
    scientific_statement: NonBlankStr
    result_value: float | None = Field(default=None, allow_inf_nan=False)
    result_unit: NonBlankStr | None = None
    curation_status: CurationStatus | None = None
    grounding_refs: tuple[NonBlankStr, ...] = ()
    supporting_quotes: tuple[LiteratureKnowledgeSupportingQuoteView, ...] = ()
    content_regions: tuple[DocumentContentRegion, ...] = ()
    section_paths: tuple[tuple[NonBlankStr, ...], ...] = ()
    region_uncertain: bool = False


class LiteratureScientificKnowledgeView(StrictModel):
    view_id: NonBlankStr
    literature_id: NonBlankStr
    view_mode: LiteratureKnowledgeViewMode
    records: tuple[LiteratureScientificKnowledgeViewRecord, ...]
    compilation_ids: tuple[NonBlankStr, ...]
    rejected_proposals: tuple[RejectedLiteratureKnowledgeProposal, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureScientificKnowledgeView:
        _validate_content_bound(self, id_field="view_id", prefix="literature-knowledge-view")
        return self
