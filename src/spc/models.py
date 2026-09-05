from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    AliasChoices,
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
Sha256Str = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


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


class RetrievalSourceType(StrEnum):
    EVIDENCE_SPAN = "evidence_span"
    EXPERT_CASE = "expert_case"
    WORKFLOW_PATTERN = "workflow_pattern"
    SCIENTIFIC_CAPABILITY = "scientific_capability"


class EpistemicStatus(StrEnum):
    SOURCE_REPORTED = "source_reported"
    SOURCE_HYPOTHESIS = "source_hypothesis"
    SOURCE_INTERPRETATION = "source_interpretation"
    BACKGROUND_STATEMENT = "background_statement"
    METHOD_STATEMENT = "method_statement"
    MODEL_STATEMENT = "model_statement"
    REPORTED_RESULT = "reported_result"
    UNRESOLVED = "unresolved"


class ResultStatus(StrEnum):
    LITERATURE_REPORTED = "literature_reported"
    EXPERIMENTAL_REPORTED = "experimental_reported"
    COMPUTED_REPORTED = "computed_reported"
    PREDICTED_REPORTED = "predicted_reported"
    UNKNOWN_ORIGIN = "unknown_origin"


class EvidenceAssessmentStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"
    INCOMPARABLE = "incomparable"


class SourceRole(StrEnum):
    AUTHOR = "author"
    REVIEWER = "reviewer"
    LITERATURE_AUTHOR = "literature_author"
    INTERNAL_RESEARCHER = "internal_researcher"
    SYSTEM = "system"
    UNSPECIFIED = "unspecified"


class SourceType(StrEnum):
    MANUSCRIPT = "manuscript"
    SUPPORTING_INFORMATION = "supporting_information"
    REVIEWER_COMMENT = "reviewer_comment"
    AUTHOR_RESPONSE = "author_response"
    LITERATURE_ARTICLE = "literature_article"
    CALCULATION_ARCHIVE = "calculation_archive"
    INTERNAL_NOTE = "internal_note"
    UNSPECIFIED = "unspecified"


class SourceDocument(StrictModel):
    source_id: str
    version: str
    title: NonBlankStr
    media_type: str = "text/plain"
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stored_path: str
    source_role: SourceRole = SourceRole.UNSPECIFIED
    source_type: SourceType = SourceType.UNSPECIFIED
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    read_only: bool = True

    @model_validator(mode="after")
    def validate_provenance(self) -> SourceDocument:
        if self.source_type == SourceType.AUTHOR_RESPONSE and self.source_role != SourceRole.AUTHOR:
            raise ValueError("author_response source_type requires author source_role")
        if self.source_type == SourceType.REVIEWER_COMMENT and self.source_role != SourceRole.REVIEWER:
            raise ValueError("reviewer_comment source_type requires reviewer source_role")
        if self.source_type == SourceType.LITERATURE_ARTICLE and self.source_role == SourceRole.REVIEWER:
            raise ValueError("literature_article cannot use reviewer source_role")
        return self


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
    aliases: dict[NonBlankStr, tuple[NonBlankStr, ...]] = Field(default_factory=dict)
    synonyms: dict[NonBlankStr, tuple[NonBlankStr, ...]] = Field(default_factory=dict)
    ontology_relationships: dict[NonBlankStr, tuple[NonBlankStr, ...]] = Field(
        default_factory=dict
    )
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


class RetrievalQuery(StrictModel):
    query_id: NonBlankStr
    raw_request: NonBlankStr
    domain: NonBlankStr
    concepts: tuple[NonBlankStr, ...] = ()
    system_terms: tuple[NonBlankStr, ...] = ()
    method_terms: tuple[NonBlankStr, ...] = ()
    desired_observables: tuple[NonBlankStr, ...] = ()
    evidence_types: tuple[RetrievalSourceType, ...] = ()
    exclusions: tuple[NonBlankStr, ...] = ()

    @model_validator(mode="after")
    def validate_query_id(self) -> RetrievalQuery:
        from .serialization import content_hash

        expected = f"query-{content_hash(self.model_dump(mode='json', exclude={'query_id'}))[:24]}"
        if self.query_id != expected:
            raise ValueError("RetrievalQuery query_id is not content-bound")
        return self


class RetrievalHit(StrictModel):
    hit_id: NonBlankStr
    source_type: RetrievalSourceType
    record_id: NonBlankStr
    score: float = Field(ge=0, allow_inf_nan=False)
    matched_terms: tuple[NonBlankStr, ...] = Field(min_length=1)
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    retriever_version: NonBlankStr


class KnowledgeSnapshot(StrictModel):
    snapshot_id: NonBlankStr
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    domain_profile_hash: Sha256Str
    expert_case_hashes: dict[NonBlankStr, Sha256Str]
    workflow_pattern_hashes: dict[NonBlankStr, Sha256Str]
    capability_hashes: dict[NonBlankStr, Sha256Str]
    evidence_span_hashes: dict[NonBlankStr, Sha256Str]
    evidence_source_versions: dict[NonBlankStr, Sha256Str]

    @model_validator(mode="after")
    def validate_snapshot_id(self) -> KnowledgeSnapshot:
        from .serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"snapshot_id", "created_at"}
        )
        if self.snapshot_id != f"snapshot-{content_hash(identity)[:24]}":
            raise ValueError("KnowledgeSnapshot snapshot_id is not content-bound")
        return self


class RetrievalManifest(StrictModel):
    retrieval_id: NonBlankStr
    query_hash: Sha256Str
    domain_pack_id: NonBlankStr
    domain_pack_version: NonBlankStr
    knowledge_snapshot_id: NonBlankStr
    retriever_version: NonBlankStr
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result_ids: tuple[NonBlankStr, ...]
    result_hashes: tuple[Sha256Str, ...]

    @model_validator(mode="after")
    def validate_retrieval_id(self) -> RetrievalManifest:
        from .serialization import content_hash

        identity = self.model_dump(mode="json", exclude={"retrieval_id", "timestamp"})
        current_id = f"retrieval-{content_hash(identity)[:24]}"
        if self.retrieval_id != current_id:
            raise ValueError("RetrievalManifest retrieval_id is not content-bound")
        if len(self.result_ids) != len(self.result_hashes):
            raise ValueError("RetrievalManifest result IDs and hashes must have equal length")
        return self


def scientific_context_semantic_hash(value: Any) -> str:
    from .serialization import content_hash, to_primitive

    payload = to_primitive(value)
    if not isinstance(payload, dict):
        raise TypeError("ScientificContextPacket semantic identity requires an object")
    payload.pop("content_hash", None)
    retrieval_manifest = payload.get("retrieval_manifest")
    if isinstance(retrieval_manifest, dict):
        retrieval_manifest.pop("timestamp", None)
    knowledge_snapshot = payload.get("knowledge_snapshot")
    if isinstance(knowledge_snapshot, dict):
        knowledge_snapshot.pop("created_at", None)
    return content_hash(payload)


class ScientificContextPacket(StrictModel):
    context_id: NonBlankStr
    original_request: NonBlankStr
    domain: NonBlankStr
    retrieval_query: RetrievalQuery
    evidence_hits: tuple[RetrievalHit, ...] = ()
    expert_case_hits: tuple[RetrievalHit, ...] = ()
    workflow_pattern_hits: tuple[RetrievalHit, ...] = ()
    capability_hits: tuple[RetrievalHit, ...] = ()
    retrieved_statements: tuple[GroundedStatement, ...] = Field(
        default=(),
        validation_alias=AliasChoices("retrieved_statements", "known_facts"),
    )
    assumptions: tuple[AssumptionRecord, ...] = ()
    conflicting_evidence: tuple[NonBlankStr, ...] = ()
    unknowns: tuple[UnknownRecord, ...] = ()
    retrieval_manifest: RetrievalManifest
    knowledge_snapshot: KnowledgeSnapshot
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_retrieval_bindings(self) -> ScientificContextPacket:
        from .serialization import content_hash

        categorized = (
            (self.evidence_hits, RetrievalSourceType.EVIDENCE_SPAN),
            (self.expert_case_hits, RetrievalSourceType.EXPERT_CASE),
            (self.workflow_pattern_hits, RetrievalSourceType.WORKFLOW_PATTERN),
            (self.capability_hits, RetrievalSourceType.SCIENTIFIC_CAPABILITY),
        )
        for hits, expected_type in categorized:
            if any(hit.source_type != expected_type for hit in hits):
                raise ValueError(f"context hit category does not match {expected_type}")
        result_ids = tuple(hit.hit_id for hits, _ in categorized for hit in hits)
        if result_ids != self.retrieval_manifest.result_ids:
            raise ValueError("retrieval manifest result IDs do not match context hits")
        result_hashes = tuple(content_hash(hit) for hits, _ in categorized for hit in hits)
        if self.retrieval_manifest.result_hashes != result_hashes:
            raise ValueError("retrieval manifest result hashes do not match context hits")
        if self.retrieval_manifest.knowledge_snapshot_id != self.knowledge_snapshot.snapshot_id:
            raise ValueError("retrieval manifest does not bind the included knowledge snapshot")
        if self.retrieval_query.domain != self.domain:
            raise ValueError("retrieval query domain does not match context domain")
        if self.original_request != self.retrieval_query.raw_request:
            raise ValueError("context request does not match RetrievalQuery raw_request")
        if self.retrieval_manifest.query_hash != content_hash(self.retrieval_query):
            raise ValueError("retrieval manifest query hash does not match RetrievalQuery")
        if self.retrieval_manifest.domain_pack_id != self.domain:
            raise ValueError("retrieval manifest Domain Pack does not match context domain")
        if any(
            hit.retriever_version != self.retrieval_manifest.retriever_version
            for hits, _ in categorized
            for hit in hits
        ):
            raise ValueError("retrieval hit version does not match retrieval manifest")
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("retrieval result IDs must be unique")
        if self.context_id != f"context-{self.retrieval_manifest.retrieval_id.removeprefix('retrieval-')}":
            raise ValueError("context ID does not bind the retrieval manifest")
        expected_hash = scientific_context_semantic_hash(self)
        legacy_payload = self.model_dump(mode="json", exclude={"content_hash"})
        legacy_payload["retrieval_manifest"].pop("result_hashes", None)
        legacy_current_hash = content_hash(legacy_payload)
        legacy_payload["known_facts"] = legacy_payload.pop("retrieved_statements")
        legacy_known_facts_hash = content_hash(legacy_payload)
        if self.content_hash not in {
            expected_hash,
            legacy_current_hash,
            legacy_known_facts_hash,
        }:
            raise ValueError("ScientificContextPacket content hash is invalid")
        return self


class SourceQuote(StrictModel):
    quote_id: NonBlankStr
    evidence_ref: NonBlankStr
    relative_start_offset: int = Field(ge=0)
    relative_end_offset: int = Field(gt=0)
    text: NonBlankStr
    source_id: NonBlankStr
    source_version: NonBlankStr
    source_role: SourceRole
    source_type: SourceType

    @model_validator(mode="after")
    def validate_quote_id(self) -> SourceQuote:
        from .serialization import content_hash

        if self.relative_end_offset <= self.relative_start_offset:
            raise ValueError("SourceQuote end offset must be greater than start offset")
        identity = {
            "evidence_ref": self.evidence_ref,
            "relative_start_offset": self.relative_start_offset,
            "relative_end_offset": self.relative_end_offset,
            "text_hash": content_hash({"text": self.text}),
        }
        if self.quote_id != f"quote-{content_hash(identity)[:24]}":
            raise ValueError("SourceQuote quote_id is not content-bound")
        return self


class SourceClaim(StrictModel):
    claim_id: NonBlankStr
    text: NonBlankStr
    claim_type: NonBlankStr
    source_role: SourceRole
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    source_quote_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    claim_strength: NonBlankStr
    epistemic_status: EpistemicStatus

    @model_validator(mode="after")
    def validate_quote_bindings(self) -> SourceClaim:
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("SourceClaim evidence_refs must be unique")
        if len(set(self.source_quote_refs)) != len(self.source_quote_refs):
            raise ValueError("SourceClaim source_quote_refs must be unique")
        return self


class ResultContext(StrictModel):
    context_id: NonBlankStr
    system_context: FrozenDict
    method_context: FrozenDict
    method_fact_refs: tuple[NonBlankStr, ...] = ()
    model_fact_refs: tuple[NonBlankStr, ...] = ()

    @model_validator(mode="after")
    def validate_context_id(self) -> ResultContext:
        from .serialization import content_hash

        identity = self.model_dump(mode="json", exclude={"context_id"})
        if self.context_id != f"result-context-{content_hash(identity)[:24]}":
            raise ValueError("ResultContext context_id is not content-bound")
        return self


class ReportedResult(StrictModel):
    result_id: NonBlankStr
    quantity: NonBlankStr
    value: float = Field(allow_inf_nan=False)
    unit: NonBlankStr
    system_context: FrozenDict
    method_context: FrozenDict
    result_context: ResultContext | None = None
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    result_status: ResultStatus


class MethodFact(StrictModel):
    fact_id: NonBlankStr
    text: NonBlankStr
    attributes: FrozenDict
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    epistemic_status: EpistemicStatus = EpistemicStatus.METHOD_STATEMENT

    @model_validator(mode="after")
    def validate_status(self) -> MethodFact:
        if self.epistemic_status != EpistemicStatus.METHOD_STATEMENT:
            raise ValueError("MethodFact must remain a method_statement")
        return self


class ModelFact(StrictModel):
    fact_id: NonBlankStr
    text: NonBlankStr
    attributes: FrozenDict
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    epistemic_status: EpistemicStatus = EpistemicStatus.MODEL_STATEMENT

    @model_validator(mode="after")
    def validate_status(self) -> ModelFact:
        if self.epistemic_status != EpistemicStatus.MODEL_STATEMENT:
            raise ValueError("ModelFact must remain a model_statement")
        return self


class EvidenceAssessment(StrictModel):
    assessment_id: NonBlankStr
    claim_ref: NonBlankStr
    supporting_evidence_refs: tuple[NonBlankStr, ...] = ()
    contradicting_evidence_refs: tuple[NonBlankStr, ...] = ()
    assessment: EvidenceAssessmentStatus
    limitations: tuple[NonBlankStr, ...] = ()
    confidence_basis: NonBlankStr


class ConflictSet(StrictModel):
    conflict_id: NonBlankStr
    topic: NonBlankStr
    claim_refs: tuple[NonBlankStr, ...] = Field(min_length=2)
    conflict_type: NonBlankStr
    possible_causes: tuple[NonBlankStr, ...] = ()
    required_discrimination: tuple[NonBlankStr, ...] = Field(min_length=1)
    resolution_status: NonBlankStr


class ComparisonConstraint(StrictModel):
    constraint_id: NonBlankStr
    comparison_target: NonBlankStr
    must_match_fields: tuple[NonBlankStr, ...] = Field(min_length=1)
    may_vary_fields: tuple[NonBlankStr, ...] = ()
    disclosure_required_fields: tuple[NonBlankStr, ...] = ()
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)


class EvidenceGap(StrictModel):
    gap_id: NonBlankStr
    scientific_question: NonBlankStr
    missing_evidence: NonBlankStr
    why_it_matters: NonBlankStr
    blocking: bool
    candidate_capabilities: tuple[NonBlankStr, ...] = ()
    evidence_refs: tuple[NonBlankStr, ...] = ()


class InterpretationProposal(StrictModel):
    proposal_id: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    source_quotes: tuple[SourceQuote, ...] = ()
    source_claims: tuple[SourceClaim, ...] = ()
    reported_results: tuple[ReportedResult, ...] = ()
    method_facts: tuple[MethodFact, ...] = ()
    model_facts: tuple[ModelFact, ...] = ()
    evidence_assessments: tuple[EvidenceAssessment, ...] = ()
    conflict_sets: tuple[ConflictSet, ...] = ()
    comparison_constraints: tuple[ComparisonConstraint, ...] = ()
    evidence_gaps: tuple[EvidenceGap, ...] = ()
    unknowns: tuple[NonBlankStr, ...] = ()
    assumption_candidates: tuple[NonBlankStr, ...] = ()
    capability_candidates: tuple[NonBlankStr, ...] = ()


class ScientificEvidencePacket(StrictModel):
    packet_id: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    source_quotes: tuple[SourceQuote, ...] = ()
    source_claims: tuple[SourceClaim, ...] = ()
    reported_results: tuple[ReportedResult, ...] = ()
    method_facts: tuple[MethodFact, ...] = ()
    model_facts: tuple[ModelFact, ...] = ()
    evidence_assessments: tuple[EvidenceAssessment, ...] = ()
    conflict_sets: tuple[ConflictSet, ...] = ()
    comparison_constraints: tuple[ComparisonConstraint, ...] = ()
    evidence_gaps: tuple[EvidenceGap, ...] = ()
    unknowns: tuple[NonBlankStr, ...] = ()
    assumption_candidates: tuple[NonBlankStr, ...] = ()
    capability_candidates: tuple[NonBlankStr, ...] = ()
    provenance_manifest: FrozenDict
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ScientificEvidencePacket:
        from .serialization import content_hash

        identity = self.model_dump(mode="json", exclude={"packet_id", "content_hash"})
        legacy_identity = self.model_dump(mode="json", exclude={"packet_id", "content_hash"})
        legacy_identity.pop("source_quotes", None)
        for claim in legacy_identity["source_claims"]:
            claim.pop("source_quote_refs", None)
        for result in legacy_identity["reported_results"]:
            result.pop("result_context", None)
        valid_ids = {
            f"evidence-packet-{content_hash(identity)[:24]}",
            f"evidence-packet-{content_hash(legacy_identity)[:24]}",
        }
        if self.packet_id not in valid_ids:
            raise ValueError("ScientificEvidencePacket packet_id is not content-bound")
        expected_hash = content_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        )
        legacy_payload = {"packet_id": self.packet_id, **legacy_identity}
        if self.content_hash not in {expected_hash, content_hash(legacy_payload)}:
            raise ValueError("ScientificEvidencePacket content hash is invalid")
        return self


class PlanningStrategyClass(StrEnum):
    MINIMAL_DECISIVE_TEST = "minimal_decisive_test"
    MECHANISM_DISCRIMINATION = "mechanism_discrimination"
    ROBUSTNESS_SENSITIVITY = "robustness_sensitivity"
    MODEL_DISCRIMINATION = "model_discrimination"
    EVIDENCE_GAP_RESOLUTION = "evidence_gap_resolution"


class ScientificPlanningInput(StrictModel):
    planning_input_id: NonBlankStr
    original_request: NonBlankStr
    domain: NonBlankStr
    domain_pack_version: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    evidence_packet_id: NonBlankStr
    evidence_packet_hash: Sha256Str
    source_quotes: tuple[SourceQuote, ...] = ()
    source_claims: tuple[SourceClaim, ...] = ()
    reported_results: tuple[ReportedResult, ...] = ()
    method_facts: tuple[MethodFact, ...] = ()
    model_facts: tuple[ModelFact, ...] = ()
    evidence_assessments: tuple[EvidenceAssessment, ...] = ()
    conflict_sets: tuple[ConflictSet, ...] = ()
    comparison_constraints: tuple[ComparisonConstraint, ...] = ()
    evidence_gaps: tuple[EvidenceGap, ...] = ()
    unknowns: tuple[NonBlankStr, ...] = ()
    assumption_candidates: tuple[NonBlankStr, ...] = ()
    expert_cases: tuple[ExpertCase, ...] = ()
    workflow_patterns: tuple[LiteratureWorkflowPattern, ...] = ()
    scientific_capabilities: tuple[ScientificCapability, ...] = ()
    allowed_evidence_ids: tuple[NonBlankStr, ...] = ()
    allowed_claim_ids: tuple[NonBlankStr, ...] = ()
    allowed_capability_ids: tuple[NonBlankStr, ...] = ()
    required_human_decisions: tuple[RequiredHumanDecision, ...] = ()
    provenance_manifest: FrozenDict
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_allowlists(self) -> ScientificPlanningInput:
        from .serialization import content_hash

        for field_name in (
            "allowed_evidence_ids",
            "allowed_claim_ids",
            "allowed_capability_ids",
        ):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must contain unique IDs")
        if set(self.allowed_claim_ids) != {claim.claim_id for claim in self.source_claims}:
            raise ValueError("allowed_claim_ids must match the included SourceClaim records")
        if set(self.allowed_capability_ids) != {
            capability.capability_id for capability in self.scientific_capabilities
        }:
            raise ValueError(
                "allowed_capability_ids must match the included ScientificCapability records"
            )
        allowed_evidence = set(self.allowed_evidence_ids)
        referenced_evidence = {
            evidence_id
            for records in (
                self.source_quotes,
                self.source_claims,
                self.reported_results,
                self.method_facts,
                self.model_facts,
                self.comparison_constraints,
                self.evidence_gaps,
            )
            for record in records
            for evidence_id in (
                (record.evidence_ref,)
                if isinstance(record, SourceQuote)
                else record.evidence_refs
            )
        }
        if not referenced_evidence.issubset(allowed_evidence):
            raise ValueError("planning input contains evidence outside allowed_evidence_ids")
        identity = self.model_dump(
            mode="json", exclude={"planning_input_id", "content_hash"}
        )
        expected_id = f"planning-input-{content_hash(identity)[:24]}"
        if self.planning_input_id != expected_id:
            raise ValueError("ScientificPlanningInput planning_input_id is not content-bound")
        payload = {"planning_input_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ScientificPlanningInput content_hash is invalid")
        return self


class IntentInterpretation(StrictModel):
    target_claim: NonBlankStr
    latent_concern: NonBlankStr
    atomic_questions: tuple[NonBlankStr, ...] = Field(min_length=1)
    excluded_substitutions: tuple[NonBlankStr, ...] = ()
    decision_relevant_observables: tuple[NonBlankStr, ...] = Field(min_length=1)
    evidence_basis: tuple[NonBlankStr, ...] = Field(min_length=1)
    unresolved_points: tuple[NonBlankStr, ...] = ()


class AmbiguityAssessment(StrictModel):
    multiple_candidates_required: bool
    rationale: NonBlankStr
    scientifically_distinct_axes: tuple[NonBlankStr, ...] = ()

    @model_validator(mode="after")
    def validate_axes(self) -> AmbiguityAssessment:
        if self.multiple_candidates_required and len(self.scientifically_distinct_axes) < 2:
            raise ValueError("multiple candidates require at least two scientific axes")
        return self


class ObservableDraft(StrictModel):
    observable_key: NonBlankStr
    description: NonBlankStr
    unit: NonBlankStr | None = None
    evidence_refs: tuple[NonBlankStr, ...] = ()


class ComparisonBaselineDraft(StrictModel):
    baseline_key: NonBlankStr
    description: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()


class CriterionDraft(StrictModel):
    statement: NonBlankStr
    observable_key: NonBlankStr


class ProposedDeviationDraft(StrictModel):
    field: NonBlankStr
    statement: NonBlankStr
    baseline_ref: NonBlankStr
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()


class CandidateTaskDraft(StrictModel):
    task_key: NonBlankStr
    scientific_objective: NonBlankStr
    capability_id: NonBlankStr
    inputs: FrozenDict = Field(default_factory=FrozenDict)
    outputs: tuple[NonBlankStr, ...] = Field(min_length=1)
    depends_on: tuple[NonBlankStr, ...] = ()
    success_criteria: tuple[NonBlankStr, ...] = Field(min_length=1)
    falsification_relevance: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    release_gates: tuple[NonBlankStr, ...] = (
        "deterministic-plan-validation",
        "human-selection",
    )
    failure_policy: NonBlankStr = "stop and request review"
    provenance_requirements: tuple[NonBlankStr, ...] = (
        "planning input hash",
        "evidence references",
    )
    cost_estimate: NonBlankStr = "unknown"


class CandidatePlanDraft(StrictModel):
    candidate_key: NonBlankStr
    strategy_class: PlanningStrategyClass
    distinguishing_axis: NonBlankStr
    primary_hypothesis: NonBlankStr
    null_hypothesis: NonBlankStr
    model_definition: NonBlankStr
    observables: tuple[ObservableDraft, ...] = Field(min_length=1)
    comparison_baselines: tuple[ComparisonBaselineDraft, ...] = Field(min_length=1)
    acceptance_criteria: tuple[CriterionDraft, ...] = Field(min_length=1)
    falsification_criteria: tuple[CriterionDraft, ...] = Field(min_length=1)
    assumptions: tuple[NonBlankStr, ...] = ()
    unknowns: tuple[NonBlankStr, ...] = ()
    proposed_deviations: tuple[ProposedDeviationDraft, ...] = ()
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    claim_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    capability_ids: tuple[NonBlankStr, ...] = Field(min_length=1)
    task_drafts: tuple[CandidateTaskDraft, ...] = Field(min_length=1)
    cost_tier: NonBlankStr
    risks: tuple[NonBlankStr, ...] = ()
    limitations: tuple[NonBlankStr, ...] = ()
    human_decisions_required: tuple[NonBlankStr, ...] = ()


class PlanningProposalSet(StrictModel):
    proposal_id: NonBlankStr
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config: FrozenDict = Field(default_factory=FrozenDict)
    intent: IntentInterpretation
    ambiguity_assessment: AmbiguityAssessment
    candidates: tuple[CandidatePlanDraft, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_identity_and_candidates(self) -> PlanningProposalSet:
        from .serialization import content_hash

        if len({candidate.candidate_key for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("candidate keys must be unique")
        multiple = len(self.candidates) > 1
        if self.ambiguity_assessment.multiple_candidates_required != multiple:
            raise ValueError("ambiguity assessment does not match candidate count")
        identity = self.model_dump(mode="json", exclude={"proposal_id"})
        if self.proposal_id != f"planning-proposal-{content_hash(identity)[:24]}":
            raise ValueError("PlanningProposalSet proposal_id is not content-bound")
        return self
