from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ..immutable import FrozenDict

from .base import NonBlankStr, Sha256Str, StrictModel
from .evidence import (
    ComparisonConstraint,
    ConflictSet,
    EvidenceAssessment,
    EvidenceGap,
    MethodFact,
    ModelFact,
    ReportedObservation,
    ReportedResult,
    SourceClaim,
    SourceQuote,
)
from .knowledge import ExpertCase
from .literature import LiteratureWorkflowPattern
from .scientific import ScientificCapability, ScientificQuestionPlan


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    APPROVE_WITH_CONDITIONS = "approve_with_conditions"
    NEEDS_HUMAN_CHOICE = "needs_human_choice"
    REQUEST_REVISION = "request_revision"
    REJECT = "reject"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ApprovalMode(StrEnum):
    INDEPENDENT_REQUIRED = "independent_required"
    LEGACY_MANUAL_ALLOWED = "legacy_manual_allowed"


class RequiredFix(StrictModel):
    fix_id: str
    description: NonBlankStr
    blocking: bool = True


class FixResolution(StrictModel):
    fix_id: str
    resolved: bool
    resolution: NonBlankStr
    evidence_refs: tuple[str, ...] = ()


class HumanDecisionResolution(StrictModel):
    decision_id: str
    resolved: bool
    selected_option: NonBlankStr
    rationale: NonBlankStr


class ApprovalScores(StrictModel):
    intent_fidelity: int = Field(ge=0, le=5)
    evidence_grounding: int = Field(ge=0, le=5)
    model_observable_alignment: int = Field(ge=0, le=5)
    method_consistency: int = Field(ge=0, le=5)
    dag_executability: int = Field(ge=0, le=5)
    falsifiability: int = Field(ge=0, le=5)
    scientific_scope_adequacy: int = Field(ge=0, le=5)


class ApprovalVerdict(StrictModel):
    verdict_id: str
    candidate_id: str
    candidate_version: str
    candidate_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scores: ApprovalScores
    hard_red_flags: tuple[NonBlankStr, ...] = ()
    required_fixes: tuple[RequiredFix, ...] = ()
    fix_resolutions: tuple[FixResolution, ...] = ()
    human_decisions_required: tuple[str, ...] = ()
    human_decision_resolutions: tuple[HumanDecisionResolution, ...] = ()
    decision: ApprovalDecision
    approver_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlanValidationRecord(StrictModel):
    validation_id: str
    plan_id: str
    plan_version: str
    plan_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain: str
    domain_pack_version: str
    valid: bool
    issue_codes: tuple[str, ...] = ()
    validator_version: str = "2.1.0"


class ProjectTrustPolicy(StrictModel):
    approval_mode: ApprovalMode
    policy_version: NonBlankStr


class GateVerdict(StrictModel):
    gate_id: str
    candidate_id: str
    candidate_version: str
    candidate_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_verdict_id: str
    approval_verdict_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_validation_id: str
    plan_validation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    independent_approval_receipt_id: str | None = None
    independent_approval_receipt_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    trust_policy_version: str | None = None
    trust_policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    plan_compilation_receipt_id: str | None = None
    plan_compilation_receipt_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    passed: bool
    reasons: tuple[NonBlankStr, ...] = ()

    @model_validator(mode="after")
    def validate_optional_receipt_binding(self) -> GateVerdict:
        if (self.independent_approval_receipt_id is None) != (
            self.independent_approval_receipt_hash is None
        ):
            raise ValueError("independent approval receipt ID and hash must be supplied together")
        if (self.trust_policy_version is None) != (self.trust_policy_hash is None):
            raise ValueError("trust policy version and hash must be supplied together")
        if (self.plan_compilation_receipt_id is None) != (
            self.plan_compilation_receipt_hash is None
        ):
            raise ValueError("plan compilation receipt ID and hash must be supplied together")
        return self


class ApprovalRedFlagSeverity(StrEnum):
    WARNING = "warning"
    BLOCKING = "blocking"


class ApprovalDimensionScore(StrictModel):
    score: int = Field(ge=0, le=5)
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    claim_refs: tuple[NonBlankStr, ...] = ()
    task_refs: tuple[NonBlankStr, ...] = ()
    capability_refs: tuple[NonBlankStr, ...] = ()


class ApprovalReviewScores(StrictModel):
    intent_fidelity: ApprovalDimensionScore
    evidence_grounding: ApprovalDimensionScore
    model_observable_alignment: ApprovalDimensionScore
    method_consistency: ApprovalDimensionScore
    dag_executability: ApprovalDimensionScore
    falsifiability: ApprovalDimensionScore
    scientific_scope_adequacy: ApprovalDimensionScore


class ApprovalHardRedFlag(StrictModel):
    code: NonBlankStr
    severity: ApprovalRedFlagSeverity
    description: NonBlankStr
    plan_path: NonBlankStr | None = None
    evidence_refs: tuple[NonBlankStr, ...] = ()
    claim_refs: tuple[NonBlankStr, ...] = ()
    task_refs: tuple[NonBlankStr, ...] = ()
    capability_refs: tuple[NonBlankStr, ...] = ()


class ApprovalLLMResponse(StrictModel):
    scores: ApprovalReviewScores
    decision_recommendation: ApprovalDecision
    summary: NonBlankStr
    evidence_basis: tuple[NonBlankStr, ...] = ()
    hard_red_flags: tuple[ApprovalHardRedFlag, ...] = ()
    required_fixes: tuple[RequiredFix, ...] = ()
    unresolved_human_decisions: tuple[NonBlankStr, ...] = ()


class RevisionIssueStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_HUMAN_DECISION = "needs_human_decision"


class RevisionIssueAssessment(StrictModel):
    feedback_id: NonBlankStr
    status: RevisionIssueStatus
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    claim_refs: tuple[NonBlankStr, ...] = ()
    task_refs: tuple[NonBlankStr, ...] = ()


class RevisionApprovalLLMResponse(StrictModel):
    contract_version: Literal["1.0.0"] = "1.0.0"
    review: ApprovalLLMResponse
    issue_assessments: tuple[RevisionIssueAssessment, ...] = Field(min_length=1)


class ApprovalReviewInput(StrictModel):
    review_input_id: NonBlankStr
    original_request: NonBlankStr
    domain: NonBlankStr
    domain_pack_version: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    evidence_packet_id: NonBlankStr
    evidence_packet_hash: Sha256Str
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    candidate_plan: ScientificQuestionPlan
    candidate_plan_hash: Sha256Str
    plan_validation_record: PlanValidationRecord
    plan_validation_hash: Sha256Str
    source_quotes: tuple[SourceQuote, ...] = ()
    source_claims: tuple[SourceClaim, ...] = ()
    reported_results: tuple[ReportedResult, ...] = ()
    reported_observations: tuple[ReportedObservation, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    method_facts: tuple[MethodFact, ...] = ()
    model_facts: tuple[ModelFact, ...] = ()
    evidence_assessments: tuple[EvidenceAssessment, ...] = ()
    conflict_sets: tuple[ConflictSet, ...] = ()
    comparison_constraints: tuple[ComparisonConstraint, ...] = ()
    evidence_gaps: tuple[EvidenceGap, ...] = ()
    expert_cases: tuple[ExpertCase, ...] = ()
    workflow_patterns: tuple[LiteratureWorkflowPattern, ...] = ()
    scientific_capabilities: tuple[ScientificCapability, ...] = ()
    allowed_evidence_ids: tuple[NonBlankStr, ...] = ()
    allowed_claim_ids: tuple[NonBlankStr, ...] = ()
    allowed_task_ids: tuple[NonBlankStr, ...] = ()
    allowed_capability_ids: tuple[NonBlankStr, ...] = ()
    provenance_manifest: FrozenDict
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_allowlists(self) -> ApprovalReviewInput:
        from ..serialization import content_hash

        for field_name in (
            "allowed_evidence_ids",
            "allowed_claim_ids",
            "allowed_task_ids",
            "allowed_capability_ids",
        ):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must contain unique IDs")
        if self.candidate_plan_hash != content_hash(self.candidate_plan):
            raise ValueError("candidate_plan_hash does not match candidate_plan")
        if self.plan_validation_hash != content_hash(self.plan_validation_record):
            raise ValueError("plan_validation_hash does not match PlanValidationRecord")
        if (
            self.plan_validation_record.plan_id != self.candidate_plan.plan_id
            or self.plan_validation_record.plan_version != self.candidate_plan.version
            or self.plan_validation_record.plan_content_hash
            != self.candidate_plan_hash
        ):
            raise ValueError("PlanValidationRecord does not bind candidate_plan")
        if (
            self.domain != self.candidate_plan.domain
            or self.domain_pack_version != self.candidate_plan.domain_pack_version
        ):
            raise ValueError("review domain does not match candidate_plan")
        if set(self.allowed_claim_ids) != {
            claim.claim_id for claim in self.source_claims
        }:
            raise ValueError("allowed_claim_ids must match SourceClaim records")
        if set(self.allowed_task_ids) != {
            task.task_id for task in self.candidate_plan.tasks
        }:
            raise ValueError("allowed_task_ids must match candidate tasks")
        if set(self.allowed_capability_ids) != {
            capability.capability_id for capability in self.scientific_capabilities
        }:
            raise ValueError(
                "allowed_capability_ids must match ScientificCapability records"
            )
        referenced_evidence = {
            evidence_id
            for collection in (
                self.source_claims,
                self.reported_results,
                self.reported_observations,
                self.method_facts,
                self.model_facts,
                self.comparison_constraints,
                self.evidence_gaps,
            )
            for record in collection
            for evidence_id in record.evidence_refs
        } | {quote.evidence_ref for quote in self.source_quotes} | {
            reference.evidence_id for reference in self.candidate_plan.evidence_refs
        }
        if not referenced_evidence.issubset(set(self.allowed_evidence_ids)):
            raise ValueError("review content contains evidence outside allowed_evidence_ids")
        identity = self.model_dump(
            mode="json",
            exclude={"review_input_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"approval-review-input-{content_hash(identity)[:24]}"
        if self.review_input_id != expected_id:
            raise ValueError("ApprovalReviewInput review_input_id is not content-bound")
        payload = {"review_input_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ApprovalReviewInput content_hash is invalid")
        return self


class ApprovalReviewRecord(StrictModel):
    review_id: NonBlankStr
    review_input_id: NonBlankStr
    review_input_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config: FrozenDict = Field(default_factory=FrozenDict)
    response: ApprovalLLMResponse | RevisionApprovalLLMResponse
    policy_decision: ApprovalDecision
    policy_reasons: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ApprovalReviewRecord:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"review_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"approval-review-{content_hash(identity)[:24]}"
        if self.review_id != expected_id:
            raise ValueError("ApprovalReviewRecord review_id is not content-bound")
        payload = {"review_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ApprovalReviewRecord content_hash is invalid")
        return self


class IndependentApprovalReceipt(StrictModel):
    receipt_id: NonBlankStr
    review_id: NonBlankStr
    review_hash: Sha256Str
    review_input_id: NonBlankStr
    review_input_hash: Sha256Str
    verdict_id: NonBlankStr
    verdict_hash: Sha256Str
    candidate_id: NonBlankStr
    candidate_version: NonBlankStr
    candidate_hash: Sha256Str
    approver_id: NonBlankStr
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> IndependentApprovalReceipt:
        from ..serialization import content_hash

        expected_verdict_id = (
            "approval-verdict-"
            + content_hash(
                {
                    "review_hash": self.review_hash,
                    "candidate_hash": self.candidate_hash,
                    "approver_id": self.approver_id,
                }
            )[:24]
        )
        if self.verdict_id != expected_verdict_id:
            raise ValueError("receipt verdict_id is not bound to the independent review")
        identity = self.model_dump(
            mode="json", exclude={"receipt_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"independent-approval-receipt-{content_hash(identity)[:24]}"
        if self.receipt_id != expected_id:
            raise ValueError("IndependentApprovalReceipt receipt_id is not content-bound")
        payload = {"receipt_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("IndependentApprovalReceipt content_hash is invalid")
        return self
