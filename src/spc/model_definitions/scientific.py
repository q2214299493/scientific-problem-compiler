from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_serializer, model_validator

from ..immutable import FrozenDict, deep_freeze, deep_thaw

from .base import NonBlankStr, Sha256Str, StrictModel
from .evidence import EvidenceReference, GroundedStatement


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


PROHIBITED_EXECUTION_PAYLOAD_KEYS = frozenset(
    {
        "command",
        "commands",
        "script",
        "executable",
        "submit_command",
        "scheduler_command",
        "shell_command",
    }
)


def prohibited_execution_payload_paths(value: Any, path: str = "payload") -> tuple[str, ...]:
    """Return paths to command-bearing keys nested in descriptive payloads."""

    matches: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).strip().casefold() in PROHIBITED_EXECUTION_PAYLOAD_KEYS:
                matches.append(child_path)
            matches.extend(prohibited_execution_payload_paths(child, child_path))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, child in enumerate(value):
            matches.extend(
                prohibited_execution_payload_paths(child, f"{path}[{index}]")
            )
    return tuple(matches)


class ScientificTaskExecutionContext(StrictModel):
    context_id: NonBlankStr
    source_plan_id: NonBlankStr
    source_plan_hash: Sha256Str
    task_id: NonBlankStr
    scientific_objective: NonBlankStr
    task_inputs: FrozenDict = Field(default_factory=FrozenDict)
    hypothesis: Hypothesis
    model: ModelDefinition
    observables: tuple[ObservableDefinition, ...]
    comparison_baselines: tuple[ComparisonBaseline, ...]
    intent_fingerprint_ref: NonBlankStr
    system_fingerprint_ref: NonBlankStr
    method_fingerprint_ref: NonBlankStr
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    falsification_criteria: tuple[FalsificationCriterion, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    success_criteria: tuple[NonBlankStr, ...]
    provenance_requirements: tuple[NonBlankStr, ...]
    depends_on_task_ids: tuple[NonBlankStr, ...] = ()
    source_query_manifest: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_safety(self) -> ScientificTaskExecutionContext:
        from ..serialization import content_hash

        unsafe_paths = prohibited_execution_payload_paths(
            self.task_inputs, "task_inputs"
        )
        if unsafe_paths:
            raise ValueError(
                "ScientificTaskExecutionContext contains executable payload fields: "
                + ", ".join(unsafe_paths)
            )
        identity = self.model_dump(
            mode="json", exclude={"context_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"task-execution-context-{content_hash(identity)[:24]}"
        if self.context_id != expected_id:
            raise ValueError("ScientificTaskExecutionContext context_id is not content-bound")
        payload = {"context_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ScientificTaskExecutionContext content_hash is invalid")
        return self


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


class DomainProfile(StrictModel):
    domain_id: str
    version: str
    name: NonBlankStr
    terminology: dict[str, str] = Field(default_factory=dict)
    ontology: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    aliases: dict[NonBlankStr, tuple[NonBlankStr, ...]] = Field(default_factory=dict)
    synonyms: dict[NonBlankStr, tuple[NonBlankStr, ...]] = Field(default_factory=dict)
    ontology_relationships: dict[NonBlankStr, tuple[NonBlankStr, ...]] = Field(
        default_factory=dict
    )
    validators: tuple[str, ...] = ()
