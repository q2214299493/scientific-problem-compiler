from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from .immutable import FrozenDict, deep_freeze, deep_thaw


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> StrictModel:
        del deep
        data = self.model_dump(mode="python")
        data.update(update or {})
        return type(self).model_validate(data)


class EvidenceClassification(StrEnum):
    EVIDENCE = "evidence"
    ASSUMPTION = "assumption"
    DEFAULT = "default"
    PROPOSED_DEVIATION = "proposed_deviation"
    UNKNOWN = "unknown"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    APPROVE_WITH_CONDITIONS = "approve_with_conditions"
    NEEDS_HUMAN_CHOICE = "needs_human_choice"
    REQUEST_REVISION = "request_revision"
    REJECT = "reject"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class SourceDocument(StrictModel):
    source_id: str
    version: str
    title: str
    media_type: str = "text/plain"
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stored_path: str
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    read_only: bool = True


class EvidenceSpan(StrictModel):
    evidence_id: str
    source_id: str
    source_version: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text: str
    locator: str | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> EvidenceSpan:
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class EvidenceReference(StrictModel):
    evidence_id: str
    source_id: str
    source_version: str


class GroundedStatement(StrictModel):
    statement_id: str
    text: str
    classification: EvidenceClassification
    evidence_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_grounding(self) -> GroundedStatement:
        if self.classification == EvidenceClassification.EVIDENCE and not self.evidence_refs:
            raise ValueError("evidence-classified statements require evidence_refs")
        return self


class Hypothesis(StrictModel):
    primary: GroundedStatement
    null: GroundedStatement


class ScientificQuestion(StrictModel):
    question_id: str
    text: str
    evidence_refs: tuple[str, ...] = ()


class ModelDefinition(StrictModel):
    model_id: str
    description: GroundedStatement
    parameters: tuple[GroundedStatement, ...] = ()


class ObservableDefinition(StrictModel):
    observable_id: str
    description: GroundedStatement
    unit: str | None = None


class ComparisonBaseline(StrictModel):
    baseline_id: str
    description: GroundedStatement


class AcceptanceCriterion(StrictModel):
    criterion_id: str
    statement: str
    observable_id: str


class FalsificationCriterion(StrictModel):
    criterion_id: str
    statement: str
    observable_id: str


class AssumptionRecord(StrictModel):
    assumption_id: str
    statement: str
    impact: str


class UnknownRecord(StrictModel):
    unknown_id: str
    statement: str
    resolution: str


class ProposedDeviation(StrictModel):
    deviation_id: str
    statement: str
    baseline_ref: str
    rationale: str
    evidence_refs: tuple[str, ...] = ()


class IntentFingerprint(StrictModel):
    fingerprint_id: str
    objective: str
    constraints: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = ()


class SystemFingerprint(StrictModel):
    fingerprint_id: str
    attributes: FrozenDict
    evidence_refs: tuple[str, ...] = ()


class MethodFingerprint(StrictModel):
    fingerprint_id: str
    attributes: FrozenDict
    evidence_refs: tuple[str, ...] = ()
    proposed_deviation_refs: tuple[str, ...] = ()


class FingerprintDifference(StrictModel):
    field: str
    left: Any = None
    right: Any = None
    disclosed_deviation: bool = False

    @model_validator(mode="after")
    def freeze_values(self) -> FingerprintDifference:
        object.__setattr__(self, "left", deep_freeze(self.left))
        object.__setattr__(self, "right", deep_freeze(self.right))
        return self

    @field_serializer("left", "right")
    def serialize_frozen_values(self, value: Any) -> Any:
        return deep_thaw(value)


class RequiredHumanDecision(StrictModel):
    decision_id: str
    question: str
    options: tuple[str, ...]
    required_before: str


class DAGTask(StrictModel):
    task_id: str
    scientific_objective: str
    capability_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    falsification_relevance: str
    evidence_refs: tuple[str, ...] = ()
    release_gates: tuple[str, ...] = ()
    failure_policy: str
    provenance_requirements: tuple[str, ...] = ()
    cost_estimate: str = "unknown"
    runnable: bool = False


class ScientificQuestionPlan(StrictModel):
    plan_id: str
    version: str
    domain: str
    domain_pack_version: str
    original_question: str
    original_comment_id: str | None = None
    latent_concern: str
    atomic_questions: tuple[ScientificQuestion, ...]
    hypothesis: Hypothesis
    model: ModelDefinition
    observables: tuple[ObservableDefinition, ...]
    comparison_baselines: tuple[ComparisonBaseline, ...]
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    falsification_criteria: tuple[FalsificationCriterion, ...]
    intent_fingerprint: IntentFingerprint
    system_fingerprint: SystemFingerprint
    method_fingerprint: MethodFingerprint
    fingerprint_differences: tuple[FingerprintDifference, ...] = ()
    evidence_refs: tuple[EvidenceReference, ...] = ()
    assumptions: tuple[AssumptionRecord, ...] = ()
    defaults: tuple[GroundedStatement, ...] = ()
    unknowns: tuple[UnknownRecord, ...] = ()
    proposed_deviations: tuple[ProposedDeviation, ...] = ()
    scientific_capability_ids: tuple[str, ...]
    tasks: tuple[DAGTask, ...]
    distinguishing_axis: str | None = None
    cost_tier: str
    risks: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    required_human_decisions: tuple[RequiredHumanDecision, ...] = ()
    source_query_manifest: tuple[str, ...] = ()
    target_agent_capability_requirements: tuple[str, ...] = ()
    wave_id: str = "wave-1"
    follow_up_of: str | None = None
    source_proposal: str | None = None


class RequiredFix(StrictModel):
    fix_id: str
    description: str
    blocking: bool = True


class FixResolution(StrictModel):
    fix_id: str
    resolved: bool
    resolution: str
    evidence_refs: tuple[str, ...] = ()


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
    hard_red_flags: tuple[str, ...] = ()
    required_fixes: tuple[RequiredFix, ...] = ()
    fix_resolutions: tuple[FixResolution, ...] = ()
    human_decisions_required: tuple[str, ...] = ()
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
    validator_version: str = "1.1.0"


class GateVerdict(StrictModel):
    gate_id: str
    candidate_id: str
    candidate_version: str
    candidate_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_verdict_id: str
    approval_verdict_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_validation_id: str
    plan_validation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool
    reasons: tuple[str, ...] = ()


class ScientificCapability(StrictModel):
    capability_id: str
    domain: str = "base"
    scientific_goal: str
    required_inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    dag_expansion: tuple[str, ...] = ()
    validators: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    failure_branches: tuple[str, ...] = ()


class ExpertCase(StrictModel):
    case_id: str
    domain: str
    vague_request: str
    translated_questions: tuple[str, ...]
    positive: bool
    rationale: str
    evidence_refs: tuple[str, ...] = ()


class LiteratureWorkflowPattern(StrictModel):
    pattern_id: str
    domain: str
    trigger: str
    workflow_capabilities: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


class DomainProfile(StrictModel):
    domain_id: str
    version: str
    name: str
    terminology: dict[str, str] = Field(default_factory=dict)
    ontology: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    validators: tuple[str, ...] = ()


class AgentCapability(StrictModel):
    capability_id: str
    version: str
    supports_scientific_capability_ids: tuple[str, ...]
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)


class AgentCapabilityCatalog(StrictModel):
    agent_id: str
    version: str
    capabilities: tuple[AgentCapability, ...]


class CapabilityBinding(StrictModel):
    scientific_capability_id: str
    target_capability_id: str | None = None
    status: str
    reason: str | None = None


class ExecutionPolicy(StrictModel):
    mode: str = "planning_only"
    runnable: bool = False
    prohibited_actions: tuple[str, ...] = (
        "generate_execution_inputs",
        "submit_jobs",
        "execute_scientific_software",
    )


class AgentHandoffPackage(StrictModel):
    export_id: str
    target_agent: str
    source_plan_id: str
    source_plan_version: str
    source_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_bindings: tuple[CapabilityBinding, ...]
    execution_policy: ExecutionPolicy


class ExportManifest(StrictModel):
    export_id: str
    target_agent: str
    source_plan_id: str
    source_plan_version: str
    source_plan_hash: str
    files: tuple[str, ...]
