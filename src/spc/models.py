from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    model_validator,
)

from .immutable import FrozenDict, deep_freeze, deep_thaw


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("text must not be blank")
    return value


NonBlankStr = Annotated[
    str,
    StringConstraints(min_length=1),
    AfterValidator(_require_non_blank),
]


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
    title: NonBlankStr
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
    text: NonBlankStr
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
    text: NonBlankStr
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
    text: NonBlankStr
    evidence_refs: tuple[str, ...] = ()


class ModelDefinition(StrictModel):
    model_id: str
    description: GroundedStatement
    parameters: tuple[GroundedStatement, ...] = ()


class ObservableDefinition(StrictModel):
    observable_id: str
    description: GroundedStatement
    unit: NonBlankStr | None = None


class ComparisonBaseline(StrictModel):
    baseline_id: str
    description: GroundedStatement


class AcceptanceCriterion(StrictModel):
    criterion_id: str
    statement: NonBlankStr
    observable_id: str


class FalsificationCriterion(StrictModel):
    criterion_id: str
    statement: NonBlankStr
    observable_id: str


class AssumptionRecord(StrictModel):
    assumption_id: str
    statement: NonBlankStr
    impact: NonBlankStr


class UnknownRecord(StrictModel):
    unknown_id: str
    statement: NonBlankStr
    resolution: NonBlankStr


class ProposedDeviation(StrictModel):
    deviation_id: str
    statement: NonBlankStr
    baseline_ref: str
    rationale: NonBlankStr
    evidence_refs: tuple[str, ...] = ()


class IntentFingerprint(StrictModel):
    fingerprint_id: str
    objective: NonBlankStr
    constraints: tuple[NonBlankStr, ...] = ()
    requested_outputs: tuple[NonBlankStr, ...] = ()


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
    field: NonBlankStr
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
    question: NonBlankStr
    options: tuple[NonBlankStr, ...]
    required_before: str


class DAGTask(StrictModel):
    task_id: str
    scientific_objective: NonBlankStr
    capability_id: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: tuple[NonBlankStr, ...] = ()
    depends_on: tuple[str, ...] = ()
    success_criteria: tuple[NonBlankStr, ...] = ()
    falsification_relevance: NonBlankStr
    evidence_refs: tuple[str, ...] = ()
    release_gates: tuple[str, ...] = ()
    failure_policy: NonBlankStr
    provenance_requirements: tuple[NonBlankStr, ...] = ()
    cost_estimate: NonBlankStr = "unknown"
    runnable: bool = False


class ScientificQuestionPlan(StrictModel):
    plan_id: str
    version: str
    domain: str
    domain_pack_version: str
    original_question: NonBlankStr
    original_comment_id: str | None = None
    latent_concern: NonBlankStr
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
    distinguishing_axis: NonBlankStr | None = None
    cost_tier: str
    risks: tuple[NonBlankStr, ...] = ()
    limitations: tuple[NonBlankStr, ...] = ()
    required_human_decisions: tuple[RequiredHumanDecision, ...] = ()
    source_query_manifest: tuple[NonBlankStr, ...] = ()
    target_agent_capability_requirements: tuple[str, ...] = ()
    wave_id: str = "wave-1"
    follow_up_of: str | None = None
    source_proposal: str | None = None

    @model_validator(mode="after")
    def validate_required_content_and_unique_ids(self) -> ScientificQuestionPlan:
        required_strings = {
            "plan_id": self.plan_id,
            "version": self.version,
            "domain": self.domain,
            "domain_pack_version": self.domain_pack_version,
            "original_question": self.original_question,
            "latent_concern": self.latent_concern,
            "cost_tier": self.cost_tier,
            "wave_id": self.wave_id,
        }
        for field, value in required_strings.items():
            if not value.strip():
                raise ValueError(f"{field} must not be blank")
        if self.original_comment_id is not None and not self.original_comment_id.strip():
            raise ValueError("original_comment_id must not be blank when provided")
        required_collections = {
            "atomic_questions": self.atomic_questions,
            "observables": self.observables,
            "comparison_baselines": self.comparison_baselines,
            "acceptance_criteria": self.acceptance_criteria,
            "falsification_criteria": self.falsification_criteria,
            "evidence_refs": self.evidence_refs,
            "scientific_capability_ids": self.scientific_capability_ids,
            "tasks": self.tasks,
        }
        for field, values in required_collections.items():
            if not values:
                raise ValueError(f"{field} must not be empty")
        id_entries = [
            ("plan", self.plan_id),
            *(("question", item.question_id) for item in self.atomic_questions),
            ("hypothesis statement", self.hypothesis.primary.statement_id),
            ("hypothesis statement", self.hypothesis.null.statement_id),
            ("model", self.model.model_id),
            ("model statement", self.model.description.statement_id),
            *(("model parameter", item.statement_id) for item in self.model.parameters),
            *(("observable", item.observable_id) for item in self.observables),
            *(("observable statement", item.description.statement_id) for item in self.observables),
            *(("baseline", item.baseline_id) for item in self.comparison_baselines),
            *(("baseline statement", item.description.statement_id) for item in self.comparison_baselines),
            *(("acceptance criterion", item.criterion_id) for item in self.acceptance_criteria),
            *(("falsification criterion", item.criterion_id) for item in self.falsification_criteria),
            ("intent fingerprint", self.intent_fingerprint.fingerprint_id),
            ("system fingerprint", self.system_fingerprint.fingerprint_id),
            ("method fingerprint", self.method_fingerprint.fingerprint_id),
            *(("evidence", item.evidence_id) for item in self.evidence_refs),
            *(("assumption", item.assumption_id) for item in self.assumptions),
            *(("default statement", item.statement_id) for item in self.defaults),
            *(("unknown", item.unknown_id) for item in self.unknowns),
            *(("proposed deviation", item.deviation_id) for item in self.proposed_deviations),
            *(("task", item.task_id) for item in self.tasks),
            *(("human decision", item.decision_id) for item in self.required_human_decisions),
        ]
        blank_ids = [category for category, identifier in id_entries if not identifier.strip()]
        if blank_ids:
            raise ValueError(f"entity IDs must not be blank: {', '.join(blank_ids)}")
        counts: dict[str, int] = {}
        for _, identifier in id_entries:
            counts[identifier] = counts.get(identifier, 0) + 1
        duplicates = sorted(identifier for identifier, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"entity IDs must be globally unique: {', '.join(duplicates)}")
        if len(set(self.scientific_capability_ids)) != len(self.scientific_capability_ids):
            raise ValueError("scientific_capability_ids must be unique")
        return self


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
    validator_version: str = "1.2.1"


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
    reasons: tuple[NonBlankStr, ...] = ()


class ScientificCapability(StrictModel):
    capability_id: str
    domain: str = "base"
    scientific_goal: NonBlankStr
    required_inputs: tuple[NonBlankStr, ...] = ()
    outputs: tuple[NonBlankStr, ...] = ()
    dag_expansion: tuple[NonBlankStr, ...] = ()
    validators: tuple[str, ...] = ()
    limitations: tuple[NonBlankStr, ...] = ()
    failure_branches: tuple[NonBlankStr, ...] = ()


class ExpertCase(StrictModel):
    case_id: str
    domain: str
    vague_request: NonBlankStr
    translated_questions: tuple[NonBlankStr, ...]
    positive: bool
    rationale: NonBlankStr
    evidence_refs: tuple[str, ...] = ()


class LiteratureWorkflowPattern(StrictModel):
    pattern_id: str
    domain: str
    trigger: NonBlankStr
    workflow_capabilities: tuple[str, ...]
    limitations: tuple[NonBlankStr, ...] = ()
    evidence_refs: tuple[str, ...] = ()


class DomainProfile(StrictModel):
    domain_id: str
    version: str
    name: NonBlankStr
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
    reason: NonBlankStr | None = None


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
