from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from ...models import NonBlankStr, Sha256Str, StrictModel
from ...serialization import content_hash
from ..literature_knowledge.contracts import LiteratureKnowledgeProviderInvocation


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


class ExpertKnowledgeViewMode(StrEnum):
    AUDIT = "audit"
    TRUSTED = "trusted"


class ExpertSourcePage(StrictModel):
    page_number: int = Field(ge=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> ExpertSourcePage:
        if self.end_offset <= self.start_offset:
            raise ValueError("expert source page range is invalid")
        return self


class ExpertSourceRecord(StrictModel):
    expert_source_id: NonBlankStr
    expert_id: NonBlankStr
    source_relationship: NonBlankStr
    source_type: NonBlankStr
    domain: NonBlankStr
    input_kind: NonBlankStr
    raw_sha256: Sha256Str
    raw_artifact_path: NonBlankStr
    canonical_sha256: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    source_hash: Sha256Str
    attribution_id: NonBlankStr
    attribution_hash: Sha256Str
    attribution_status: NonBlankStr
    page_ranges: tuple[ExpertSourcePage, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExpertSourceRecord:
        _validate_content_bound(self, id_field="expert_source_id", prefix="expert-source")
        return self


class ExpertKnowledgeCompilationInput(StrictModel):
    compilation_input_id: NonBlankStr
    expert_id: NonBlankStr
    expert_profile_hash: Sha256Str
    expert_source_id: NonBlankStr
    expert_source_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    source_hash: Sha256Str
    attribution_id: NonBlankStr
    attribution_hash: Sha256Str
    domain: NonBlankStr
    chunk_policy_id: NonBlankStr
    chunk_policy_version: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExpertKnowledgeCompilationInput:
        _validate_content_bound(
            self,
            id_field="compilation_input_id",
            prefix="expert-knowledge-input",
        )
        return self


class ExpertKnowledgeChunk(StrictModel):
    chunk_id: NonBlankStr
    expert_source_id: NonBlankStr
    expert_id: NonBlankStr
    attribution_id: NonBlankStr
    attribution_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    source_type: NonBlankStr
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text: NonBlankStr
    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExpertKnowledgeChunk:
        if self.end_offset <= self.start_offset:
            raise ValueError("expert chunk range is invalid")
        _validate_content_bound(self, id_field="chunk_id", prefix="expert-knowledge-chunk")
        return self


class ExpertQuoteProposal(StrictModel):
    quote_key: NonBlankStr
    chunk_id: NonBlankStr
    exact_text: NonBlankStr


class ExpertOpinionProposal(StrictModel):
    opinion_key: NonBlankStr
    quote_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    normalized_text: NonBlankStr
    opinion_category: NonBlankStr
    authority_status: NonBlankStr
    expert_id: NonBlankStr
    topic: NonBlankStr
    scope: NonBlankStr
    rationale: NonBlankStr
    conditions: tuple[NonBlankStr, ...]


class ExpertCaseProposal(StrictModel):
    case_key: NonBlankStr
    opinion_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    original_wording: NonBlankStr
    latent_concern: NonBlankStr
    atomic_questions: tuple[NonBlankStr, ...] = Field(min_length=1)
    good_question_formulations: tuple[NonBlankStr, ...] = Field(min_length=1)
    wrong_formulations: tuple[NonBlankStr, ...] = Field(min_length=1)
    answerability_conditions: tuple[NonBlankStr, ...] = Field(min_length=1)
    required_evidence_types: tuple[NonBlankStr, ...] = Field(min_length=1)
    baseline_guidance: tuple[NonBlankStr, ...]
    common_misinterpretations: tuple[NonBlankStr, ...]
    resolution_pattern: NonBlankStr
    applicability: tuple[NonBlankStr, ...] = Field(min_length=1)


class ExpertKnowledgeLLMResponse(StrictModel):
    quote_proposals: tuple[ExpertQuoteProposal, ...] = ()
    opinion_proposals: tuple[ExpertOpinionProposal, ...] = ()
    case_proposals: tuple[ExpertCaseProposal, ...] = ()


class ExpertKnowledgeBatchInvocation(StrictModel):
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
    def validate_binding(self) -> ExpertKnowledgeBatchInvocation:
        if len(self.chunk_ids) != len(self.chunk_hashes):
            raise ValueError("expert batch chunk IDs and hashes must align")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("expert batch chunk IDs must be unique")
        if self.provider_invocation is not None and (
            self.provider_invocation.input_hash != self.input_hash
            or self.provider_invocation.output_hash != self.output_hash
            or self.provider_invocation.selected_model != self.selected_model
            or self.provider_invocation.runtime_version != self.runtime_version
        ):
            raise ValueError("expert batch invocation binding is invalid")
        identity = self.model_dump(mode="json", exclude={"content_hash"}, exclude_none=True)
        if self.content_hash != content_hash(identity):
            raise ValueError("expert batch invocation content_hash is invalid")
        return self


class ExpertKnowledgeProposalSet(StrictModel):
    proposal_set_id: NonBlankStr
    compilation_input_id: NonBlankStr
    compilation_input_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config_hash: Sha256Str
    batch_invocations: tuple[ExpertKnowledgeBatchInvocation, ...] = ()
    quote_proposals: tuple[ExpertQuoteProposal, ...] = ()
    opinion_proposals: tuple[ExpertOpinionProposal, ...] = ()
    case_proposals: tuple[ExpertCaseProposal, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExpertKnowledgeProposalSet:
        for records, key_field in (
            (self.quote_proposals, "quote_key"),
            (self.opinion_proposals, "opinion_key"),
            (self.case_proposals, "case_key"),
        ):
            keys = tuple(getattr(item, key_field) for item in records)
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate expert proposal-local key: {key_field}")
        _validate_content_bound(
            self,
            id_field="proposal_set_id",
            prefix="expert-knowledge-proposals",
        )
        return self


class ExpertKnowledgeGroundingRecord(StrictModel):
    grounding_id: NonBlankStr
    expert_source_id: NonBlankStr
    expert_source_hash: Sha256Str
    attribution_id: NonBlankStr
    attribution_hash: Sha256Str
    chunk_id: NonBlankStr
    chunk_hash: Sha256Str
    evidence_id: NonBlankStr
    evidence_hash: Sha256Str
    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExpertKnowledgeGroundingRecord:
        _validate_content_bound(
            self,
            id_field="grounding_id",
            prefix="expert-knowledge-grounding",
        )
        return self


class RejectedExpertKnowledgeProposal(StrictModel):
    proposal_ref: NonBlankStr
    proposal_kind: NonBlankStr
    rejection_code: NonBlankStr


class ExpertKnowledgeCompilationRecord(StrictModel):
    compilation_id: NonBlankStr
    compilation_input_id: NonBlankStr
    compilation_input_hash: Sha256Str
    expert_source_id: NonBlankStr
    expert_source_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    source_hash: Sha256Str
    attribution_id: NonBlankStr
    attribution_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config_hash: Sha256Str
    proposal_set_id: NonBlankStr
    proposal_set_hash: Sha256Str
    chunk_hashes: dict[NonBlankStr, Sha256Str]
    batch_invocations: tuple[ExpertKnowledgeBatchInvocation, ...] = ()
    expert_opinion_ids: tuple[NonBlankStr, ...] = ()
    expert_case_ids: tuple[NonBlankStr, ...] = ()
    grounding_hashes: dict[NonBlankStr, Sha256Str]
    rejected_proposals: tuple[RejectedExpertKnowledgeProposal, ...] = ()
    compiler_id: NonBlankStr
    compiler_version: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExpertKnowledgeCompilationRecord:
        for field_name in ("expert_opinion_ids", "expert_case_ids"):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be unique and sorted")
        _validate_content_bound(
            self,
            id_field="compilation_id",
            prefix="expert-knowledge-compilation",
        )
        return self
