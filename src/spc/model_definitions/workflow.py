from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .base import (
    KnowledgeEntityType,
    NonBlankStr,
    Sha256Str,
    StrictModel,
)
from .knowledge import CurationStatus


class ScientificProblemRunStatus(StrEnum):
    CREATED = "CREATED"
    BLOCKED_SOURCE_CURATION = "BLOCKED_SOURCE_CURATION"
    READY = "READY"
    CONTEXT_BUILT = "CONTEXT_BUILT"
    INTERPRETED = "INTERPRETED"
    PLANNED = "PLANNED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVISION_BLOCKED = "REVISION_BLOCKED"
    HIERARCHICAL_PLANNING_BLOCKED = "HIERARCHICAL_PLANNING_BLOCKED"
    EVIDENCE_RESOLUTION_BLOCKED = "EVIDENCE_RESOLUTION_BLOCKED"
    REQUIRES_REPLANNING = "REQUIRES_REPLANNING"
    EXPORTED = "EXPORTED"
    FAILED = "FAILED"


class ScientificRunBlockingItem(StrictModel):
    target_type: KnowledgeEntityType
    target_id: NonBlankStr
    current_status: CurationStatus | None = None
    required_status: CurationStatus = CurationStatus.ACCEPTED
    message: NonBlankStr


class ScientificRunArtifactBinding(StrictModel):
    artifact_type: KnowledgeEntityType
    artifact_id: NonBlankStr
    artifact_hash: Sha256Str
    relative_path: NonBlankStr


class PlanRevisionRound(StrictModel):
    round_index: int = Field(ge=0)
    parent_plan_id: NonBlankStr | None = None
    revision_input: ScientificRunArtifactBinding | None = None
    revision_record: ScientificRunArtifactBinding | None = None
    planning_proposal: ScientificRunArtifactBinding
    candidate_plan: ScientificRunArtifactBinding
    compilation_receipt: ScientificRunArtifactBinding
    validation_record: ScientificRunArtifactBinding
    approval_review_input: ScientificRunArtifactBinding | None = None
    approval_review_record: ScientificRunArtifactBinding | None = None
    approval_verdict: ScientificRunArtifactBinding | None = None
    approval_receipt: ScientificRunArtifactBinding | None = None
    gate: ScientificRunArtifactBinding | None = None
    trust_policy: ScientificRunArtifactBinding | None = None
    outcome: NonBlankStr = "awaiting_approval"

    @model_validator(mode="after")
    def validate_round_shape(self) -> PlanRevisionRound:
        revision_values = (self.revision_input, self.revision_record)
        if self.round_index == 0 and any(value is not None for value in revision_values):
            raise ValueError("initial planning round cannot contain revision artifacts")
        if self.round_index > 0 and any(value is None for value in revision_values):
            raise ValueError("revision rounds require revision input and record")
        approval_values = (
            self.approval_review_input,
            self.approval_review_record,
            self.approval_verdict,
            self.approval_receipt,
            self.gate,
            self.trust_policy,
        )
        if any(value is not None for value in approval_values) and any(
            value is None for value in approval_values
        ):
            raise ValueError("approval round artifacts must be supplied together")
        return self


class PlanRevisionChain(StrictModel):
    chain_id: NonBlankStr
    run_id: NonBlankStr
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    context_id: NonBlankStr
    context_hash: Sha256Str
    knowledge_snapshot_id: NonBlankStr
    knowledge_snapshot_hash: Sha256Str
    max_revisions: int = Field(ge=0, le=10)
    revisions_used: int = Field(ge=0)
    rounds: tuple[PlanRevisionRound, ...] = Field(min_length=1)
    termination_reason: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_rounds(self) -> PlanRevisionChain:
        from ..serialization import content_hash

        expected_chain_id = f"plan-revision-chain-{content_hash({'run_id': self.run_id})[:24]}"
        if self.chain_id != expected_chain_id:
            raise ValueError("PlanRevisionChain chain_id is not bound to run_id")
        indices = tuple(item.round_index for item in self.rounds)
        if indices != tuple(range(len(self.rounds))):
            raise ValueError("revision round indices must be contiguous")
        if self.revisions_used != len(self.rounds) - 1:
            raise ValueError("revisions_used does not match revision rounds")
        if self.revisions_used > self.max_revisions:
            raise ValueError("revision chain exceeds max_revisions")
        for index, item in enumerate(self.rounds):
            if index == 0 and item.parent_plan_id is not None:
                raise ValueError("initial planning round cannot have parent_plan_id")
            if index > 0 and item.parent_plan_id != self.rounds[index - 1].candidate_plan.artifact_id:
                raise ValueError("revision round parent_plan_id does not match prior round")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("PlanRevisionChain content_hash is invalid")
        return self


class ScientificRunProviderBinding(StrictModel):
    stage: KnowledgeEntityType
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    model_id: NonBlankStr | None = None
    configuration_hash: Sha256Str


class ScientificRunFailure(StrictModel):
    stage: KnowledgeEntityType
    category: NonBlankStr
    message: NonBlankStr
    retryable: bool = False


class ScientificProblemRun(StrictModel):
    """Content-bound, resumable audit record for the unified SPC workflow."""

    run_id: NonBlankStr
    original_request: NonBlankStr
    domain: NonBlankStr
    state_dir: NonBlankStr
    knowledge_dir: NonBlankStr
    knowledge_snapshot_id: NonBlankStr | None = None
    knowledge_snapshot_hash: Sha256Str | None = None
    retrieval_context_id: NonBlankStr | None = None
    retrieval_context_hash: Sha256Str | None = None
    context_id: NonBlankStr | None = None
    context_hash: Sha256Str | None = None
    evidence_packet_id: NonBlankStr | None = None
    evidence_packet_hash: Sha256Str | None = None
    planning_input_id: NonBlankStr | None = None
    planning_input_hash: Sha256Str | None = None
    planning_proposal_id: NonBlankStr | None = None
    planning_proposal_hash: Sha256Str | None = None
    candidate_plans: tuple[ScientificRunArtifactBinding, ...] = ()
    candidate_compilation_receipts: tuple[ScientificRunArtifactBinding, ...] = ()
    plan_validation_records: tuple[ScientificRunArtifactBinding, ...] = ()
    selected_candidate_id: NonBlankStr | None = None
    approval_review_id: NonBlankStr | None = None
    approval_review_hash: Sha256Str | None = None
    approval_verdict_id: NonBlankStr | None = None
    approval_verdict_hash: Sha256Str | None = None
    approval_receipt_id: NonBlankStr | None = None
    approval_receipt_hash: Sha256Str | None = None
    gate_id: NonBlankStr | None = None
    gate_hash: Sha256Str | None = None
    export_path: NonBlankStr | None = None
    export_hash: Sha256Str | None = None
    status: ScientificProblemRunStatus
    last_successful_status: ScientificProblemRunStatus | None = None
    blocking_items: tuple[ScientificRunBlockingItem, ...] = ()
    providers: tuple[ScientificRunProviderBinding, ...] = ()
    failure: ScientificRunFailure | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_bindings(self) -> ScientificProblemRun:
        from ..serialization import content_hash

        identity = {
            "original_request": self.original_request,
            "domain": self.domain,
            "state_dir": self.state_dir,
            "knowledge_dir": self.knowledge_dir,
        }
        expected_id = f"scientific-run-{content_hash(identity)[:24]}"
        if self.run_id != expected_id:
            raise ValueError("ScientificProblemRun run_id is not content-bound")
        for id_field, hash_field in (
            ("knowledge_snapshot_id", "knowledge_snapshot_hash"),
            ("retrieval_context_id", "retrieval_context_hash"),
            ("context_id", "context_hash"),
            ("evidence_packet_id", "evidence_packet_hash"),
            ("planning_input_id", "planning_input_hash"),
            ("planning_proposal_id", "planning_proposal_hash"),
            ("approval_review_id", "approval_review_hash"),
            ("approval_verdict_id", "approval_verdict_hash"),
            ("approval_receipt_id", "approval_receipt_hash"),
            ("gate_id", "gate_hash"),
            ("export_path", "export_hash"),
        ):
            if (getattr(self, id_field) is None) != (getattr(self, hash_field) is None):
                raise ValueError(f"{id_field} and {hash_field} must be supplied together")
        if self.status == ScientificProblemRunStatus.BLOCKED_SOURCE_CURATION and not self.blocking_items:
            raise ValueError("BLOCKED_SOURCE_CURATION requires actionable blocking_items")
        if self.status == ScientificProblemRunStatus.FAILED and self.failure is None:
            raise ValueError("FAILED run requires failure details")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("ScientificProblemRun content_hash is invalid")
        return self
