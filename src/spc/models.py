from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from collections.abc import Mapping
import unicodedata
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


def _normalize_identity_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalize_doi(value: str) -> str:
    normalized = _normalize_identity_text(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix).strip()
            break
    return normalized


NonBlankStr = Annotated[
    str,
    StringConstraints(min_length=1),
    AfterValidator(_require_non_blank),
]
Sha256Str = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
KnowledgeEntityType = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]*$"),
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


class ApprovalMode(StrEnum):
    INDEPENDENT_REQUIRED = "independent_required"
    LEGACY_MANUAL_ALLOWED = "legacy_manual_allowed"


class CurationStatus(StrEnum):
    MACHINE_EXTRACTED = "machine_extracted"
    HUMAN_REVIEWED = "human_reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class LiteratureIngestionStatus(StrEnum):
    ACCEPTED = "accepted"
    REQUIRES_OCR = "requires_ocr"
    FAILED = "failed"


class AcquisitionInputKind(StrEnum):
    DOI = "doi"
    URL = "url"
    LOCAL_FILE = "local_file"


class AcquisitionStatus(StrEnum):
    DISCOVERED = "discovered"
    METADATA_RESOLVED = "metadata_resolved"
    FULLTEXT_FOUND = "fulltext_found"
    INGESTED = "ingested"
    METADATA_ONLY = "metadata_only"
    FULLTEXT_UNAVAILABLE = "fulltext_unavailable"
    REQUIRES_AUTHENTICATION = "requires_authentication"
    UNSUPPORTED_MEDIA = "unsupported_media"
    FAILED = "failed"


class CollectionSourceKind(StrEnum):
    PROJECT_URL = "project_url"
    COLLECTION_URL = "collection_url"
    LOCAL_MANIFEST = "local_manifest"


class CollectionImportStatus(StrEnum):
    DISCOVERED = "discovered"
    IMPORTING = "importing"
    COMPLETE = "complete"
    PARTIAL = "partial"
    REQUIRES_AUTHENTICATION = "requires_authentication"
    FAILED = "failed"


class CollectionResourceKind(StrEnum):
    DOI = "doi"
    ARTICLE_URL = "article_url"
    PDF_URL = "pdf_url"
    METADATA_RECORD = "metadata_record"


class FullTextAccessStatus(StrEnum):
    DISCOVERED = "discovered"
    ACCESSIBLE = "accessible"
    REQUIRES_AUTHENTICATION = "requires_authentication"
    UNAVAILABLE = "unavailable"


class FullTextSourceKind(StrEnum):
    LOCAL_FILE = "local_file"
    DIRECT_PDF = "direct_pdf"
    HTML_ARTICLE = "html_article"
    LANDING_PAGE_PDF = "landing_page_pdf"
    METADATA_LINK = "metadata_link"


class LiteratureRepresentationKind(StrEnum):
    PDF = "pdf"
    HTML = "html"


class MetadataValueOrigin(StrEnum):
    EXPLICIT = "explicit"
    RESOLVED = "resolved"
    EMBEDDED = "embedded"
    UNRESOLVED = "unresolved"


class KnowledgeViewMode(StrEnum):
    TRUSTED = "trusted"
    AUDIT = "audit"


class KnowledgePredicate(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    USES_METHOD = "uses_method"
    USES_MODEL = "uses_model"
    REPORTS_RESULT = "reports_result"
    DERIVED_FROM = "derived_from"
    COMMENTS_ON = "comments_on"
    SUPERSEDES = "supersedes"
    APPLIES_TO = "applies_to"
    REQUIRES_CAPABILITY = "requires_capability"


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
        from .serialization import content_hash

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


class LiteratureDocument(StrictModel):
    literature_id: NonBlankStr
    title: NonBlankStr
    authors: tuple[NonBlankStr, ...] = Field(min_length=1)
    year: int = Field(ge=1000, le=9999)
    journal: NonBlankStr | None = None
    doi: NonBlankStr | None = None
    url: NonBlankStr | None = None
    domain: NonBlankStr
    topics: tuple[NonBlankStr, ...] = ()
    keywords: tuple[NonBlankStr, ...] = ()
    raw_artifact_ref: NonBlankStr
    canonical_text_ref: NonBlankStr
    source_id: NonBlankStr
    source_version: NonBlankStr
    citation_refs: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureDocument:
        from .serialization import content_hash

        for field_name in ("authors", "topics", "keywords", "citation_refs"):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"LiteratureDocument {field_name} must be unique")
        expected_id = literature_identity_id(
            title=self.title,
            authors=self.authors,
            year=self.year,
            doi=self.doi,
        )
        if self.literature_id != expected_id:
            raise ValueError("LiteratureDocument literature_id is not content-bound")
        payload = self.model_dump(mode="json", exclude={"content_hash"}, exclude_none=True)
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureDocument content_hash is invalid")
        return self


def literature_identity_id(
    *,
    title: str,
    authors: tuple[str, ...],
    year: int,
    doi: str | None,
) -> str:
    from .serialization import content_hash

    if doi is not None:
        stable_identity = {"doi": _normalize_doi(doi)}
    else:
        stable_identity = {
            "title": _normalize_identity_text(title),
            "authors": tuple(_normalize_identity_text(item) for item in authors),
            "year": year,
        }
    return f"literature-{content_hash(stable_identity)[:24]}"


class LiteratureAcquisitionRequest(StrictModel):
    request_id: NonBlankStr
    original_input: NonBlankStr
    input_kind: AcquisitionInputKind
    requested_domain: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureAcquisitionRequest:
        from .serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"request_id", "content_hash"}
        )
        expected_id = f"literature-acquisition-request-{content_hash(identity)[:24]}"
        if self.request_id != expected_id:
            raise ValueError("LiteratureAcquisitionRequest request_id is not content-bound")
        payload = {"request_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureAcquisitionRequest content_hash is invalid")
        return self


class FullTextCandidate(StrictModel):
    candidate_id: NonBlankStr
    url: NonBlankStr | None = None
    local_path_ref: NonBlankStr | None = None
    media_type: NonBlankStr
    access_status: FullTextAccessStatus
    source_kind: FullTextSourceKind
    discovered_by: NonBlankStr = "unspecified"
    discovery_source_url: NonBlankStr | None = None
    priority: int = Field(ge=0)
    content_sha256: Sha256Str | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> FullTextCandidate:
        from .serialization import content_hash

        if (self.url is None) == (self.local_path_ref is None):
            raise ValueError("FullTextCandidate requires exactly one URL or local path")
        identity = self.model_dump(
            mode="json",
            exclude={"candidate_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"fulltext-candidate-{content_hash(identity)[:24]}"
        if self.candidate_id != expected_id:
            raise ValueError("FullTextCandidate candidate_id is not content-bound")
        payload = {"candidate_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("FullTextCandidate content_hash is invalid")
        return self


class ResolvedLiteratureResource(StrictModel):
    resource_id: NonBlankStr
    canonical_identifier: NonBlankStr
    doi: NonBlankStr | None = None
    title: NonBlankStr | None = None
    authors: tuple[NonBlankStr, ...] = ()
    year: int | None = Field(default=None, ge=1000, le=9999)
    journal: NonBlankStr | None = None
    landing_url: NonBlankStr | None = None
    metadata_source: NonBlankStr
    metadata_retrieval_refs: tuple[NonBlankStr, ...] = ()
    metadata_retrieval_hashes: tuple[Sha256Str, ...] = ()
    fulltext_candidates: tuple[FullTextCandidate, ...] = ()
    resolution_status: AcquisitionStatus
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ResolvedLiteratureResource:
        from .serialization import content_hash

        candidate_ids = [item.candidate_id for item in self.fulltext_candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("ResolvedLiteratureResource candidate IDs must be unique")
        if len(set(self.metadata_retrieval_refs)) != len(
            self.metadata_retrieval_refs
        ):
            raise ValueError("metadata retrieval refs must be unique")
        if len(self.metadata_retrieval_refs) != len(
            self.metadata_retrieval_hashes
        ):
            raise ValueError("metadata retrieval ID/hash bindings must align")
        if tuple(candidate_ids) != tuple(
            item.candidate_id
            for item in sorted(
                self.fulltext_candidates,
                key=lambda item: (item.priority, item.candidate_id),
            )
        ):
            raise ValueError("full-text candidates must be deterministically ordered")
        identity = self.model_dump(
            mode="json", exclude={"resource_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"resolved-literature-{content_hash(identity)[:24]}"
        if self.resource_id != expected_id:
            raise ValueError("ResolvedLiteratureResource resource_id is not content-bound")
        payload = {"resource_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ResolvedLiteratureResource content_hash is invalid")
        return self


class LiteratureAcquisitionRecord(StrictModel):
    acquisition_id: NonBlankStr
    request_id: NonBlankStr
    request_hash: Sha256Str
    resolver_id: NonBlankStr
    resolver_version: NonBlankStr
    resolved_resource_id: NonBlankStr
    resolved_resource_hash: Sha256Str
    selected_candidate_id: NonBlankStr | None = None
    selected_candidate_hash: Sha256Str | None = None
    metadata_merge_manifest_id: NonBlankStr | None = None
    metadata_merge_manifest_hash: Sha256Str | None = None
    attempt_refs: tuple[NonBlankStr, ...] = ()
    attempt_hashes: tuple[Sha256Str, ...] = ()
    resulting_literature_id: NonBlankStr | None = None
    resulting_ingestion_id: NonBlankStr | None = None
    resulting_canonical_text_id: NonBlankStr | None = None
    resulting_representation_id: NonBlankStr | None = None
    resulting_representation_hash: Sha256Str | None = None
    status: AcquisitionStatus
    warnings: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureAcquisitionRecord:
        from .serialization import content_hash

        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("LiteratureAcquisitionRecord warnings must be unique")
        if len(set(self.attempt_refs)) != len(self.attempt_refs):
            raise ValueError("LiteratureAcquisitionRecord attempt_refs must be unique")
        if len(self.attempt_refs) != len(self.attempt_hashes):
            raise ValueError("acquisition attempt ID/hash bindings must align")
        candidate_binding = (
            self.selected_candidate_id,
            self.selected_candidate_hash,
        )
        if any(value is None for value in candidate_binding) and any(
            value is not None for value in candidate_binding
        ):
            raise ValueError("selected full-text candidate binding must be complete")
        for name, binding in (
            (
                "metadata merge manifest",
                (
                    self.metadata_merge_manifest_id,
                    self.metadata_merge_manifest_hash,
                ),
            ),
            (
                "literature representation",
                (
                    self.resulting_representation_id,
                    self.resulting_representation_hash,
                ),
            ),
        ):
            if any(value is None for value in binding) and any(
                value is not None for value in binding
            ):
                raise ValueError(f"{name} binding must be complete")
        required_result_binding = (
            self.resulting_literature_id,
            self.resulting_ingestion_id,
        )
        result_binding = (*required_result_binding, self.resulting_canonical_text_id)
        if self.status == AcquisitionStatus.INGESTED and any(
            value is None for value in required_result_binding
        ):
            raise ValueError("ingested acquisition requires literature and ingestion bindings")
        if self.status != AcquisitionStatus.INGESTED and any(
            value is not None for value in result_binding
        ):
            raise ValueError("non-ingested acquisition cannot bind K1B results")
        identity = self.model_dump(
            mode="json",
            exclude={"acquisition_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"literature-acquisition-{content_hash(identity)[:24]}"
        if self.acquisition_id != expected_id:
            raise ValueError("LiteratureAcquisitionRecord acquisition_id is not content-bound")
        payload = {"acquisition_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureAcquisitionRecord content_hash is invalid")
        return self


class LiteratureAcquisitionOutcome(StrictModel):
    input_kind: AcquisitionInputKind
    canonical_identifier: NonBlankStr
    doi: NonBlankStr | None = None
    title: NonBlankStr | None = None
    acquisition_status: AcquisitionStatus
    fulltext_status: AcquisitionStatus
    literature_id: NonBlankStr | None = None
    ingestion_id: NonBlankStr | None = None
    canonical_text_id: NonBlankStr | None = None
    representation_id: NonBlankStr | None = None
    acquisition_id: NonBlankStr
    attempt_ids: tuple[NonBlankStr, ...] = ()
    warnings: tuple[NonBlankStr, ...] = ()


class CollectionScopePolicy(StrictModel):
    allowed_origins: tuple[NonBlankStr, ...]
    allowed_path_prefixes: tuple[NonBlankStr, ...]
    allow_external_literature_links: bool = False
    max_pages: int = Field(gt=0)
    max_depth: int = Field(ge=0)
    max_resources: int = Field(gt=0)
    pagination_policy: NonBlankStr = "explicit_next_only"
    content_types: tuple[NonBlankStr, ...] = (
        "text/html",
        "application/xhtml+xml",
    )
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_policy(self) -> CollectionScopePolicy:
        from urllib.parse import urlparse

        from .serialization import content_hash

        for field_name in (
            "allowed_origins",
            "allowed_path_prefixes",
            "content_types",
        ):
            values = getattr(self, field_name)
            if not values or len(set(values)) != len(values) or tuple(sorted(values)) != values:
                raise ValueError(f"CollectionScopePolicy {field_name} must be sorted and unique")
        for origin in self.allowed_origins:
            parsed = urlparse(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.path not in {"", "/"}
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("CollectionScopePolicy allowed_origins must be HTTP(S) origins")
        if any(not prefix.startswith("/") for prefix in self.allowed_path_prefixes):
            raise ValueError("CollectionScopePolicy path prefixes must be absolute URL paths")
        identity = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(identity):
            raise ValueError("CollectionScopePolicy content_hash is invalid")
        return self


class CollectionDefinition(StrictModel):
    collection_id: NonBlankStr
    source_kind: CollectionSourceKind
    entry_url: NonBlankStr
    name: NonBlankStr
    domain: NonBlankStr
    connector_id: NonBlankStr
    connector_version: NonBlankStr
    scope_policy: CollectionScopePolicy
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionDefinition:
        from .serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"collection_id", "content_hash"}
        )
        expected_id = f"literature-collection-{content_hash(identity)[:24]}"
        if self.collection_id != expected_id:
            raise ValueError("CollectionDefinition collection_id is not content-bound")
        payload = {"collection_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionDefinition content_hash is invalid")
        return self


class CollectionPageRecord(StrictModel):
    page_id: NonBlankStr
    collection_id: NonBlankStr
    requested_url: NonBlankStr
    resolved_url: NonBlankStr
    http_status: int = Field(ge=0, le=599)
    media_type: NonBlankStr
    response_sha256: Sha256Str
    connector_id: NonBlankStr
    connector_version: NonBlankStr
    discovered_resource_refs: tuple[NonBlankStr, ...] = ()
    pagination_refs: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionPageRecord:
        from .serialization import content_hash

        for field_name in ("discovered_resource_refs", "pagination_refs"):
            values = getattr(self, field_name)
            if len(set(values)) != len(values) or tuple(sorted(values)) != values:
                raise ValueError(f"CollectionPageRecord {field_name} must be sorted and unique")
        identity = self.model_dump(mode="json", exclude={"page_id", "content_hash"})
        expected_id = f"collection-page-{content_hash(identity)[:24]}"
        if self.page_id != expected_id:
            raise ValueError("CollectionPageRecord page_id is not content-bound")
        payload = {"page_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionPageRecord content_hash is invalid")
        return self


class DiscoveredCollectionResource(StrictModel):
    discovered_resource_id: NonBlankStr
    collection_id: NonBlankStr
    resource_kind: CollectionResourceKind
    normalized_identifier: NonBlankStr
    doi: NonBlankStr | None = None
    url: NonBlankStr | None = None
    media_type: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> DiscoveredCollectionResource:
        from .serialization import content_hash

        if self.resource_kind == CollectionResourceKind.DOI:
            if self.doi != self.normalized_identifier or self.url is not None:
                raise ValueError("DOI collection resource binding is invalid")
        elif self.url != self.normalized_identifier or self.doi is not None:
            raise ValueError("URL collection resource binding is invalid")
        stable_identity = {
            "collection_id": self.collection_id,
            "resource_kind": self.resource_kind,
            "normalized_identifier": self.normalized_identifier,
        }
        expected_id = f"collection-resource-{content_hash(stable_identity)[:24]}"
        if self.discovered_resource_id != expected_id:
            raise ValueError("DiscoveredCollectionResource ID is not content-bound")
        payload = self.model_dump(mode="json", exclude={"content_hash"}, exclude_none=True)
        if self.content_hash != content_hash(payload):
            raise ValueError("DiscoveredCollectionResource content_hash is invalid")
        return self


class CollectionResourceOccurrence(StrictModel):
    occurrence_id: NonBlankStr
    collection_id: NonBlankStr
    discovered_resource_id: NonBlankStr
    discovery_page_id: NonBlankStr
    discovery_index: int = Field(ge=0)
    original_identifier: NonBlankStr
    discovery_method: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionResourceOccurrence:
        from .serialization import content_hash

        identity = self.model_dump(mode="json", exclude={"occurrence_id", "content_hash"})
        expected_id = f"collection-occurrence-{content_hash(identity)[:24]}"
        if self.occurrence_id != expected_id:
            raise ValueError("CollectionResourceOccurrence ID is not content-bound")
        payload = {"occurrence_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionResourceOccurrence content_hash is invalid")
        return self


class CollectionSnapshot(StrictModel):
    snapshot_id: NonBlankStr
    collection_id: NonBlankStr
    connector_id: NonBlankStr
    connector_version: NonBlankStr
    page_record_hashes: FrozenDict = Field(default_factory=FrozenDict)
    discovered_resource_hashes: FrozenDict = Field(default_factory=FrozenDict)
    occurrence_hashes: FrozenDict = Field(default_factory=FrozenDict)
    visited_page_count: int = Field(ge=0)
    unique_resource_count: int = Field(ge=0)
    completeness: CollectionImportStatus
    completeness_reasons: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionSnapshot:
        from .serialization import content_hash

        for field_name in (
            "page_record_hashes",
            "discovered_resource_hashes",
            "occurrence_hashes",
        ):
            values = getattr(self, field_name)
            if any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for key, value in values.items()
            ):
                raise ValueError(f"CollectionSnapshot {field_name} is invalid")
        if self.visited_page_count != len(self.page_record_hashes):
            raise ValueError("CollectionSnapshot visited page count is inconsistent")
        if self.unique_resource_count != len(self.discovered_resource_hashes):
            raise ValueError("CollectionSnapshot resource count is inconsistent")
        if set(self.occurrence_hashes) and not self.discovered_resource_hashes:
            raise ValueError("CollectionSnapshot occurrences require resources")
        if len(set(self.completeness_reasons)) != len(self.completeness_reasons):
            raise ValueError("CollectionSnapshot completeness reasons must be unique")
        if self.completeness == CollectionImportStatus.COMPLETE and self.completeness_reasons:
            raise ValueError("complete CollectionSnapshot cannot have incompleteness reasons")
        if self.completeness != CollectionImportStatus.COMPLETE and not self.completeness_reasons:
            raise ValueError("incomplete CollectionSnapshot requires an explicit reason")
        identity = self.model_dump(mode="json", exclude={"snapshot_id", "content_hash"})
        expected_id = f"collection-snapshot-{content_hash(identity)[:24]}"
        if self.snapshot_id != expected_id:
            raise ValueError("CollectionSnapshot snapshot_id is not content-bound")
        payload = {"snapshot_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionSnapshot content_hash is invalid")
        return self


class CollectionDiscoveryResult(StrictModel):
    definition: CollectionDefinition
    pages: tuple[CollectionPageRecord, ...]
    resources: tuple[DiscoveredCollectionResource, ...]
    occurrences: tuple[CollectionResourceOccurrence, ...]
    snapshot: CollectionSnapshot

    @model_validator(mode="after")
    def validate_bindings(self) -> CollectionDiscoveryResult:
        if self.snapshot.collection_id != self.definition.collection_id:
            raise ValueError("collection discovery snapshot belongs to another collection")
        page_hashes = {item.page_id: item.content_hash for item in self.pages}
        resource_hashes = {
            item.discovered_resource_id: item.content_hash for item in self.resources
        }
        occurrence_hashes = {
            item.occurrence_id: item.content_hash for item in self.occurrences
        }
        if dict(self.snapshot.page_record_hashes) != page_hashes:
            raise ValueError("collection discovery page inventory is incomplete")
        if dict(self.snapshot.discovered_resource_hashes) != resource_hashes:
            raise ValueError("collection discovery resource inventory is incomplete")
        if dict(self.snapshot.occurrence_hashes) != occurrence_hashes:
            raise ValueError("collection discovery occurrence inventory is incomplete")
        known_pages = set(page_hashes)
        known_resources = set(resource_hashes)
        if any(
            item.collection_id != self.definition.collection_id
            or item.discovery_page_id not in known_pages
            or item.discovered_resource_id not in known_resources
            for item in self.occurrences
        ):
            raise ValueError("collection occurrence references an unknown page or resource")
        if any(
            item.collection_id != self.definition.collection_id
            or not set(item.discovered_resource_refs).issubset(known_resources)
            for item in self.pages
        ):
            raise ValueError("collection page references an unknown resource")
        return self


class CollectionAcquisitionLink(StrictModel):
    link_id: NonBlankStr
    collection_id: NonBlankStr
    snapshot_id: NonBlankStr
    discovered_resource_id: NonBlankStr
    discovered_resource_hash: Sha256Str
    acquisition_id: NonBlankStr
    acquisition_hash: Sha256Str
    resulting_literature_id: NonBlankStr | None = None
    acquisition_status: AcquisitionStatus
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionAcquisitionLink:
        from .serialization import content_hash

        if (
            self.acquisition_status == AcquisitionStatus.INGESTED
        ) != (self.resulting_literature_id is not None):
            raise ValueError("collection acquisition literature binding is inconsistent")
        identity = self.model_dump(mode="json", exclude={"link_id", "content_hash"}, exclude_none=True)
        expected_id = f"collection-acquisition-link-{content_hash(identity)[:24]}"
        if self.link_id != expected_id:
            raise ValueError("CollectionAcquisitionLink link_id is not content-bound")
        payload = {"link_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionAcquisitionLink content_hash is invalid")
        return self


class CollectionImportRecord(StrictModel):
    import_id: NonBlankStr
    collection_id: NonBlankStr
    snapshot_id: NonBlankStr
    snapshot_hash: Sha256Str
    acquisition_link_hashes: FrozenDict = Field(default_factory=FrozenDict)
    total_resources: int = Field(ge=0)
    ingested_count: int = Field(ge=0)
    metadata_only_count: int = Field(ge=0)
    auth_required_count: int = Field(ge=0)
    unavailable_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    status: CollectionImportStatus
    warnings: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionImportRecord:
        from .serialization import content_hash

        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for key, value in self.acquisition_link_hashes.items()
        ):
            raise ValueError("CollectionImportRecord acquisition link hashes are invalid")
        if self.total_resources != len(self.acquisition_link_hashes):
            raise ValueError("CollectionImportRecord total_resources is inconsistent")
        if self.total_resources != sum(
            (
                self.ingested_count,
                self.metadata_only_count,
                self.auth_required_count,
                self.unavailable_count,
                self.failed_count,
            )
        ):
            raise ValueError("CollectionImportRecord result counts are inconsistent")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("CollectionImportRecord warnings must be unique")
        identity = self.model_dump(mode="json", exclude={"import_id", "content_hash"})
        expected_id = f"collection-import-{content_hash(identity)[:24]}"
        if self.import_id != expected_id:
            raise ValueError("CollectionImportRecord import_id is not content-bound")
        payload = {"import_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionImportRecord content_hash is invalid")
        return self


class CollectionDiff(StrictModel):
    diff_id: NonBlankStr
    old_snapshot_id: NonBlankStr
    old_snapshot_hash: Sha256Str
    new_snapshot_id: NonBlankStr
    new_snapshot_hash: Sha256Str
    added_resource_ids: tuple[NonBlankStr, ...] = ()
    removed_resource_ids: tuple[NonBlankStr, ...] = ()
    unchanged_resource_ids: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionDiff:
        from .serialization import content_hash

        groups = (
            self.added_resource_ids,
            self.removed_resource_ids,
            self.unchanged_resource_ids,
        )
        if any(tuple(sorted(group)) != group or len(set(group)) != len(group) for group in groups):
            raise ValueError("CollectionDiff resource IDs must be sorted and unique")
        if any(set(left) & set(right) for index, left in enumerate(groups) for right in groups[index + 1 :]):
            raise ValueError("CollectionDiff resource groups must be disjoint")
        identity = self.model_dump(mode="json", exclude={"diff_id", "content_hash"})
        expected_id = f"collection-diff-{content_hash(identity)[:24]}"
        if self.diff_id != expected_id:
            raise ValueError("CollectionDiff diff_id is not content-bound")
        payload = {"diff_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionDiff content_hash is invalid")
        return self


class CollectionImportOutcome(StrictModel):
    collection_id: NonBlankStr
    snapshot_id: NonBlankStr
    completeness: CollectionImportStatus
    visited_pages: int = Field(ge=0)
    discovered_resources: int = Field(ge=0)
    unique_resources: int = Field(ge=0)
    ingested: int = Field(ge=0)
    metadata_only: int = Field(ge=0)
    requires_authentication: int = Field(ge=0)
    unavailable: int = Field(ge=0)
    failed: int = Field(ge=0)
    import_id: NonBlankStr
    warnings: tuple[NonBlankStr, ...] = ()


class MetadataRetrievalRecord(StrictModel):
    retrieval_id: NonBlankStr
    source_url: NonBlankStr
    response_sha256: Sha256Str
    response_byte_size: int = Field(gt=0)
    media_type: NonBlankStr
    resolver_id: NonBlankStr
    resolver_version: NonBlankStr
    response_payload_utf8: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> MetadataRetrievalRecord:
        import hashlib

        from .serialization import content_hash

        if self.response_payload_utf8 is not None:
            encoded = self.response_payload_utf8.encode("utf-8")
            if (
                len(encoded) != self.response_byte_size
                or hashlib.sha256(encoded).hexdigest() != self.response_sha256
            ):
                raise ValueError("metadata response payload does not match its hash")
        identity = self.model_dump(
            mode="json",
            exclude={"retrieval_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"metadata-retrieval-{content_hash(identity)[:24]}"
        if self.retrieval_id != expected_id:
            raise ValueError("MetadataRetrievalRecord retrieval_id is not content-bound")
        payload = {"retrieval_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("MetadataRetrievalRecord content_hash is invalid")
        return self


class MetadataAlternative(StrictModel):
    origin: MetadataValueOrigin
    value: NonBlankStr | int | tuple[NonBlankStr, ...]


class MetadataFieldDecision(StrictModel):
    field_name: NonBlankStr
    selected_origin: MetadataValueOrigin
    selected_value: NonBlankStr | int | tuple[NonBlankStr, ...] | None = None
    alternatives: tuple[MetadataAlternative, ...] = ()

    @model_validator(mode="after")
    def validate_decision(self) -> MetadataFieldDecision:
        if self.selected_origin == MetadataValueOrigin.UNRESOLVED:
            if self.selected_value is not None:
                raise ValueError("unresolved metadata cannot have a selected value")
        elif self.selected_value is None:
            raise ValueError("resolved metadata requires a selected value")
        origins = [item.origin for item in self.alternatives]
        if len(set(origins)) != len(origins):
            raise ValueError("metadata alternatives must have unique origins")
        return self


class MetadataMergeManifest(StrictModel):
    manifest_id: NonBlankStr
    decisions: tuple[MetadataFieldDecision, ...]
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> MetadataMergeManifest:
        from .serialization import content_hash

        fields = [item.field_name for item in self.decisions]
        if len(set(fields)) != len(fields) or fields != sorted(fields):
            raise ValueError("metadata decisions must be unique and sorted")
        identity = self.model_dump(
            mode="json", exclude={"manifest_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"metadata-merge-{content_hash(identity)[:24]}"
        if self.manifest_id != expected_id:
            raise ValueError("MetadataMergeManifest manifest_id is not content-bound")
        payload = {"manifest_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("MetadataMergeManifest content_hash is invalid")
        return self


class AcquisitionAttemptRecord(StrictModel):
    attempt_id: NonBlankStr
    request_id: NonBlankStr
    request_hash: Sha256Str
    candidate_id: NonBlankStr
    candidate_hash: Sha256Str
    attempt_index: int = Field(ge=0)
    status: AcquisitionStatus
    http_status: int | None = Field(default=None, ge=100, le=599)
    media_type: NonBlankStr
    failure_code: NonBlankStr | None = None
    raw_artifact_id: NonBlankStr | None = None
    raw_artifact_hash: Sha256Str | None = None
    canonical_text_id: NonBlankStr | None = None
    canonical_text_hash: Sha256Str | None = None
    ingestion_id: NonBlankStr | None = None
    ingestion_hash: Sha256Str | None = None
    representation_id: NonBlankStr | None = None
    representation_hash: Sha256Str | None = None
    metadata_merge_manifest_id: NonBlankStr | None = None
    metadata_merge_manifest_hash: Sha256Str | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> AcquisitionAttemptRecord:
        from .serialization import content_hash

        for name, binding in (
            ("raw artifact", (self.raw_artifact_id, self.raw_artifact_hash)),
            (
                "canonical text",
                (self.canonical_text_id, self.canonical_text_hash),
            ),
            ("ingestion", (self.ingestion_id, self.ingestion_hash)),
            (
                "representation",
                (self.representation_id, self.representation_hash),
            ),
            (
                "metadata merge manifest",
                (
                    self.metadata_merge_manifest_id,
                    self.metadata_merge_manifest_hash,
                ),
            ),
        ):
            if any(value is None for value in binding) and any(
                value is not None for value in binding
            ):
                raise ValueError(f"attempt {name} binding must be complete")
        if self.status == AcquisitionStatus.INGESTED:
            if any(
                value is None
                for value in (
                    self.raw_artifact_id,
                    self.ingestion_id,
                    self.representation_id,
                )
            ):
                raise ValueError("successful attempt requires artifact and representation")
            if self.failure_code is not None:
                raise ValueError("successful attempt cannot have a failure code")
        elif self.failure_code is None:
            raise ValueError("unsuccessful attempt requires a failure code")
        identity = self.model_dump(
            mode="json", exclude={"attempt_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"acquisition-attempt-{content_hash(identity)[:24]}"
        if self.attempt_id != expected_id:
            raise ValueError("AcquisitionAttemptRecord attempt_id is not content-bound")
        payload = {"attempt_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("AcquisitionAttemptRecord content_hash is invalid")
        return self


class RawLiteratureArtifact(StrictModel):
    artifact_id: NonBlankStr
    literature_id: NonBlankStr
    original_filename: NonBlankStr
    media_type: NonBlankStr = "application/pdf"
    byte_size: int = Field(gt=0)
    sha256: Sha256Str
    stored_path: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> RawLiteratureArtifact:
        from pathlib import PurePosixPath, PureWindowsPath

        from .serialization import content_hash

        if (
            PurePosixPath(self.original_filename).name != self.original_filename
            or PureWindowsPath(self.original_filename).name != self.original_filename
        ):
            raise ValueError("RawLiteratureArtifact original_filename must be a basename")
        if self.media_type != "application/pdf":
            raise ValueError("RawLiteratureArtifact must use application/pdf")
        identity = {"literature_id": self.literature_id, "sha256": self.sha256}
        expected_id = f"literature-artifact-{content_hash(identity)[:24]}"
        if self.artifact_id != expected_id:
            raise ValueError("RawLiteratureArtifact artifact_id is not content-bound")
        expected_path = f"literature_artifacts/{expected_id}/artifact.pdf"
        if self.stored_path != expected_path:
            raise ValueError("RawLiteratureArtifact stored_path is not canonical")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("RawLiteratureArtifact content_hash is invalid")
        return self


class CanonicalTextBlock(StrictModel):
    block_id: NonBlankStr
    page_number: int = Field(ge=1)
    block_type: NonBlankStr
    section_path: tuple[NonBlankStr, ...] = ()
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CanonicalTextBlock:
        from .serialization import content_hash

        if self.end_offset <= self.start_offset:
            raise ValueError("CanonicalTextBlock end_offset must exceed start_offset")
        identity = self.model_dump(mode="json", exclude={"block_id"})
        expected_id = f"canonical-block-{content_hash(identity)[:24]}"
        if self.block_id != expected_id:
            raise ValueError("CanonicalTextBlock block_id is not content-bound")
        return self


class CanonicalTextArtifact(StrictModel):
    canonical_text_id: NonBlankStr
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    parser_id: NonBlankStr
    parser_version: NonBlankStr
    parser_config_hash: Sha256Str
    text_sha256: Sha256Str
    stored_path: NonBlankStr
    character_count: int = Field(gt=0)
    page_count: int = Field(gt=0)
    blocks: tuple[CanonicalTextBlock, ...] = Field(min_length=1)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CanonicalTextArtifact:
        from .serialization import content_hash

        block_ids = [block.block_id for block in self.blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("CanonicalTextArtifact block IDs must be unique")
        ordering = [
            (block.page_number, block.start_offset, block.end_offset)
            for block in self.blocks
        ]
        if ordering != sorted(ordering):
            raise ValueError("CanonicalTextArtifact blocks must be deterministically ordered")
        if any(
            right.start_offset < left.end_offset
            for left, right in zip(self.blocks, self.blocks[1:])
        ):
            raise ValueError("CanonicalTextArtifact blocks must not overlap")
        if any(
            block.page_number > self.page_count
            or block.end_offset > self.character_count
            for block in self.blocks
        ):
            raise ValueError("CanonicalTextArtifact block bounds are invalid")
        identity = self.model_dump(
            mode="json",
            exclude={"canonical_text_id", "stored_path", "content_hash"},
        )
        expected_id = f"canonical-text-{content_hash(identity)[:24]}"
        if self.canonical_text_id != expected_id:
            raise ValueError("CanonicalTextArtifact canonical_text_id is not content-bound")
        expected_path = f"canonical_text/{expected_id}/content.txt"
        if self.stored_path != expected_path:
            raise ValueError("CanonicalTextArtifact stored_path is not canonical")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("CanonicalTextArtifact content_hash is invalid")
        return self


class RawHTMLLiteratureArtifact(StrictModel):
    artifact_id: NonBlankStr
    source_url: NonBlankStr
    literature_id: NonBlankStr
    media_type: NonBlankStr
    byte_size: int = Field(gt=0)
    sha256: Sha256Str
    stored_path: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> RawHTMLLiteratureArtifact:
        from .serialization import content_hash

        if self.media_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("RawHTMLLiteratureArtifact requires an HTML media type")
        identity = {
            "source_url": self.source_url,
            "literature_id": self.literature_id,
            "sha256": self.sha256,
        }
        expected_id = f"html-artifact-{content_hash(identity)[:24]}"
        if self.artifact_id != expected_id:
            raise ValueError("RawHTMLLiteratureArtifact artifact_id is not content-bound")
        expected_path = f"html_literature_artifacts/{expected_id}/artifact.html"
        if self.stored_path != expected_path:
            raise ValueError("RawHTMLLiteratureArtifact stored_path is not canonical")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("RawHTMLLiteratureArtifact content_hash is invalid")
        return self


class CanonicalHTMLTextArtifact(StrictModel):
    canonical_text_id: NonBlankStr
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    source_url: NonBlankStr
    extractor_id: NonBlankStr
    extractor_version: NonBlankStr
    extractor_config_hash: Sha256Str
    text_sha256: Sha256Str
    stored_path: NonBlankStr
    character_count: int = Field(gt=0)
    blocks: tuple[CanonicalTextBlock, ...] = Field(min_length=1)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CanonicalHTMLTextArtifact:
        from .serialization import content_hash

        block_ids = [block.block_id for block in self.blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("CanonicalHTMLTextArtifact block IDs must be unique")
        ordering = [
            (block.start_offset, block.end_offset) for block in self.blocks
        ]
        if ordering != sorted(ordering):
            raise ValueError("HTML canonical blocks must be ordered")
        if any(
            right.start_offset < left.end_offset
            for left, right in zip(self.blocks, self.blocks[1:])
        ):
            raise ValueError("HTML canonical blocks must not overlap")
        if any(block.end_offset > self.character_count for block in self.blocks):
            raise ValueError("HTML canonical block bounds are invalid")
        identity = self.model_dump(
            mode="json",
            exclude={"canonical_text_id", "stored_path", "content_hash"},
        )
        expected_id = f"canonical-html-text-{content_hash(identity)[:24]}"
        if self.canonical_text_id != expected_id:
            raise ValueError("CanonicalHTMLTextArtifact ID is not content-bound")
        expected_path = f"canonical_html_text/{expected_id}/content.txt"
        if self.stored_path != expected_path:
            raise ValueError("CanonicalHTMLTextArtifact stored_path is not canonical")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("CanonicalHTMLTextArtifact content_hash is invalid")
        return self


class HTMLLiteratureIngestionRecord(StrictModel):
    ingestion_id: NonBlankStr
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    canonical_text_id: NonBlankStr
    canonical_text_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    extractor_id: NonBlankStr
    extractor_version: NonBlankStr
    extractor_config_hash: Sha256Str
    ingestion_status: LiteratureIngestionStatus = LiteratureIngestionStatus.ACCEPTED
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> HTMLLiteratureIngestionRecord:
        from .serialization import content_hash

        if self.ingestion_status != LiteratureIngestionStatus.ACCEPTED:
            raise ValueError("HTML ingestion records represent successful extraction only")
        identity = self.model_dump(
            mode="json", exclude={"ingestion_id", "content_hash"}
        )
        expected_id = f"html-literature-ingestion-{content_hash(identity)[:24]}"
        if self.ingestion_id != expected_id:
            raise ValueError("HTMLLiteratureIngestionRecord ID is not content-bound")
        payload = {"ingestion_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("HTMLLiteratureIngestionRecord content_hash is invalid")
        return self


class LiteratureRepresentationReference(StrictModel):
    representation_id: NonBlankStr
    representation_kind: LiteratureRepresentationKind
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    canonical_text_id: NonBlankStr
    canonical_text_hash: Sha256Str
    ingestion_id: NonBlankStr
    ingestion_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureRepresentationReference:
        from .serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"representation_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"literature-representation-{content_hash(identity)[:24]}"
        if self.representation_id != expected_id:
            raise ValueError("LiteratureRepresentationReference ID is not content-bound")
        payload = {"representation_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureRepresentationReference content_hash is invalid")
        return self


class LiteratureIngestionRecord(StrictModel):
    ingestion_id: NonBlankStr
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    canonical_text_id: NonBlankStr | None = None
    canonical_text_hash: Sha256Str | None = None
    source_id: NonBlankStr | None = None
    source_version: NonBlankStr | None = None
    parser_id: NonBlankStr
    parser_version: NonBlankStr
    parser_config_hash: Sha256Str
    ingestion_status: LiteratureIngestionStatus
    total_pages: int = Field(ge=0)
    pages_with_text: int = Field(ge=0)
    pages_without_text: int = Field(ge=0)
    text_coverage_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    warnings: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureIngestionRecord:
        from .serialization import content_hash

        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("LiteratureIngestionRecord warnings must be unique")
        if self.pages_with_text + self.pages_without_text != self.total_pages:
            raise ValueError("literature ingestion page metrics are inconsistent")
        expected_coverage = (
            self.pages_with_text / self.total_pages if self.total_pages else 0.0
        )
        if abs(self.text_coverage_ratio - expected_coverage) > 1e-12:
            raise ValueError("literature ingestion text coverage ratio is invalid")
        canonical_binding = (
            self.canonical_text_id,
            self.canonical_text_hash,
            self.source_id,
            self.source_version,
        )
        if self.ingestion_status == LiteratureIngestionStatus.ACCEPTED:
            if any(value is None for value in canonical_binding):
                raise ValueError("accepted literature ingestion requires canonical/source binding")
            if self.pages_with_text == 0:
                raise ValueError("accepted literature ingestion requires extractable text")
        elif self.ingestion_status == LiteratureIngestionStatus.REQUIRES_OCR:
            if self.pages_with_text != 0:
                raise ValueError("requires_ocr ingestion cannot contain extractable text")
            if any(value is not None for value in canonical_binding):
                raise ValueError("non-accepted literature ingestion cannot bind canonical/source records")
        elif any(value is not None for value in canonical_binding):
            raise ValueError("non-accepted literature ingestion cannot bind canonical/source records")
        identity = self.model_dump(
            mode="json",
            exclude={"ingestion_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"literature-ingestion-{content_hash(identity)[:24]}"
        if self.ingestion_id != expected_id:
            raise ValueError("LiteratureIngestionRecord ingestion_id is not content-bound")
        payload = {"ingestion_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureIngestionRecord content_hash is invalid")
        return self


class LiteratureRepresentationSelection(StrictModel):
    selection_id: NonBlankStr
    literature_id: NonBlankStr
    representation_id: NonBlankStr | None = None
    representation_hash: Sha256Str | None = None
    ingestion_id: NonBlankStr | None = None
    ingestion_hash: Sha256Str | None = None
    canonical_text_id: NonBlankStr | None = None
    canonical_text_hash: Sha256Str | None = None
    source_id: NonBlankStr | None = None
    source_version: NonBlankStr | None = None
    selected_by: NonBlankStr
    rationale: NonBlankStr
    supersedes_selection_id: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureRepresentationSelection:
        from .serialization import content_hash

        representation_binding = (self.representation_id, self.representation_hash)
        legacy_binding = (
            self.ingestion_id,
            self.ingestion_hash,
            self.canonical_text_id,
            self.canonical_text_hash,
            self.source_id,
            self.source_version,
        )
        has_representation = all(value is not None for value in representation_binding)
        has_legacy = all(value is not None for value in legacy_binding)
        if has_representation == has_legacy or (
            any(value is not None for value in representation_binding)
            and not has_representation
        ) or (any(value is not None for value in legacy_binding) and not has_legacy):
            raise ValueError(
                "selection requires exactly one complete representation or legacy binding"
            )
        identity = self.model_dump(
            mode="json",
            exclude={"selection_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"literature-selection-{content_hash(identity)[:24]}"
        if self.selection_id != expected_id:
            raise ValueError(
                "LiteratureRepresentationSelection selection_id is not content-bound"
            )
        payload = {"selection_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError(
                "LiteratureRepresentationSelection content_hash is invalid"
            )
        return self


class LiteratureRepresentationSelectionOutcome(StrictModel):
    selection: LiteratureRepresentationSelection
    ingestion_status: LiteratureIngestionStatus
    total_pages: int | None = Field(default=None, ge=0)
    pages_with_text: int | None = Field(default=None, ge=0)
    pages_without_text: int | None = Field(default=None, ge=0)
    text_coverage_ratio: float | None = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    warnings: tuple[NonBlankStr, ...] = ()


class HistoricalLiteratureEvidenceAuthorization(StrictModel):
    authorization_id: NonBlankStr
    literature_id: NonBlankStr
    representation_id: NonBlankStr | None = None
    representation_hash: Sha256Str | None = None
    ingestion_id: NonBlankStr | None = None
    ingestion_hash: Sha256Str | None = None
    source_id: NonBlankStr
    source_version: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    authorized_by: NonBlankStr
    rationale: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> HistoricalLiteratureEvidenceAuthorization:
        from .serialization import content_hash

        representation_binding = (self.representation_id, self.representation_hash)
        legacy_binding = (self.ingestion_id, self.ingestion_hash)
        has_representation = all(value is not None for value in representation_binding)
        has_legacy = all(value is not None for value in legacy_binding)
        if has_representation == has_legacy or (
            any(value is not None for value in representation_binding)
            and not has_representation
        ) or (any(value is not None for value in legacy_binding) and not has_legacy):
            raise ValueError(
                "historical authorization requires one representation or legacy ingestion binding"
            )
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError(
                "HistoricalLiteratureEvidenceAuthorization evidence_refs must be unique"
            )
        identity = self.model_dump(
            mode="json",
            exclude={"authorization_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"historical-evidence-authorization-{content_hash(identity)[:24]}"
        if self.authorization_id != expected_id:
            raise ValueError(
                "HistoricalLiteratureEvidenceAuthorization authorization_id is not content-bound"
            )
        payload = {"authorization_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError(
                "HistoricalLiteratureEvidenceAuthorization content_hash is invalid"
            )
        return self


class LiteratureIngestionOutcome(StrictModel):
    literature_id: NonBlankStr
    artifact_id: NonBlankStr
    canonical_text_id: NonBlankStr | None = None
    ingestion_id: NonBlankStr
    selection_id: NonBlankStr | None = None
    source_id: NonBlankStr | None = None
    source_version: NonBlankStr | None = None
    warnings: tuple[NonBlankStr, ...] = ()
    requires_ocr: bool = False


class ExpertProfile(StrictModel):
    expert_id: NonBlankStr
    display_name: NonBlankStr
    affiliations: tuple[NonBlankStr, ...] = ()
    expertise_domains: tuple[NonBlankStr, ...] = ()
    expertise_topics: tuple[NonBlankStr, ...] = ()
    profile_source_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_content_hash(self) -> ExpertProfile:
        from .serialization import content_hash

        for field_name in (
            "affiliations",
            "expertise_domains",
            "expertise_topics",
            "profile_source_refs",
        ):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"ExpertProfile {field_name} must be unique")
        identity = self.model_dump(
            mode="json", exclude={"content_hash"}, exclude_none=True
        )
        if self.content_hash != content_hash(identity):
            raise ValueError("ExpertProfile content_hash is invalid")
        return self


class ExpertAttributionRecord(StrictModel):
    attribution_id: NonBlankStr
    expert_id: NonBlankStr
    source_id: NonBlankStr
    source_version: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    medium: NonBlankStr
    attribution_basis: NonBlankStr
    captured_at: datetime | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_provenance(self) -> ExpertAttributionRecord:
        from .serialization import content_hash

        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("ExpertAttributionRecord evidence_refs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"attribution_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"expert-attribution-{content_hash(identity)[:24]}"
        if self.attribution_id != expected_id:
            raise ValueError("ExpertAttributionRecord attribution_id is not content-bound")
        payload = {"attribution_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExpertAttributionRecord content_hash is invalid")
        return self


class ExpertOpinion(StrictModel):
    opinion_id: NonBlankStr
    expert_id: NonBlankStr
    domain: NonBlankStr
    topic: NonBlankStr
    statement: NonBlankStr
    opinion_type: NonBlankStr
    scope: NonBlankStr
    rationale: NonBlankStr
    conditions: tuple[NonBlankStr, ...] = ()
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    attribution_refs: tuple[NonBlankStr, ...] = ()
    related_claim_refs: tuple[NonBlankStr, ...] = ()
    related_workflow_refs: tuple[NonBlankStr, ...] = ()
    supersedes: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_provenance(self) -> ExpertOpinion:
        from .serialization import content_hash

        for field_name in (
            "conditions",
            "evidence_refs",
            "attribution_refs",
            "related_claim_refs",
            "related_workflow_refs",
        ):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"ExpertOpinion {field_name} must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"opinion_id", "content_hash"},
            exclude_none=True,
        )
        if not self.attribution_refs:
            identity.pop("attribution_refs")
        expected_id = f"expert-opinion-{content_hash(identity)[:24]}"
        if self.opinion_id != expected_id:
            raise ValueError("ExpertOpinion opinion_id is not content-bound")
        if self.supersedes == self.opinion_id:
            raise ValueError("ExpertOpinion cannot supersede itself")
        payload = {"opinion_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExpertOpinion content_hash is invalid")
        return self


class KnowledgeRelation(StrictModel):
    relation_id: NonBlankStr
    subject_type: KnowledgeEntityType
    subject_id: NonBlankStr
    predicate: KnowledgePredicate
    object_type: KnowledgeEntityType
    object_id: NonBlankStr
    domain: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    rationale: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_endpoints(self) -> KnowledgeRelation:
        from .serialization import content_hash

        if (self.subject_type, self.subject_id) == (
            self.object_type,
            self.object_id,
        ):
            raise ValueError("KnowledgeRelation endpoints must be distinct")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("KnowledgeRelation evidence_refs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"relation_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"knowledge-relation-{content_hash(identity)[:24]}"
        if self.relation_id != expected_id:
            raise ValueError("KnowledgeRelation relation_id is not content-bound")
        payload = {"relation_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("KnowledgeRelation content_hash is invalid")
        return self


class KnowledgeCurationRecord(StrictModel):
    curation_id: NonBlankStr
    target_type: KnowledgeEntityType
    target_id: NonBlankStr
    target_hash: Sha256Str
    status: CurationStatus
    curator_id: NonBlankStr
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    supersedes_curation_id: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> KnowledgeCurationRecord:
        from .serialization import content_hash

        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("KnowledgeCurationRecord evidence_refs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"curation_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"knowledge-curation-{content_hash(identity)[:24]}"
        if self.curation_id != expected_id:
            raise ValueError("KnowledgeCurationRecord curation_id is not content-bound")
        if self.supersedes_curation_id == self.curation_id:
            raise ValueError("KnowledgeCurationRecord cannot supersede itself")
        payload = {"curation_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("KnowledgeCurationRecord content_hash is invalid")
        return self


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
    capability_id: NonBlankStr
    version: NonBlankStr
    supports_scientific_capability_ids: tuple[NonBlankStr, ...]
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_supported_capabilities(self) -> AgentCapability:
        if len(set(self.supports_scientific_capability_ids)) != len(
            self.supports_scientific_capability_ids
        ):
            raise ValueError("supported scientific capability IDs must be unique")
        return self


class AgentCapabilityCatalog(StrictModel):
    agent_id: NonBlankStr
    version: NonBlankStr
    capabilities: tuple[AgentCapability, ...]

    @model_validator(mode="after")
    def validate_catalog_consistency(self) -> AgentCapabilityCatalog:
        capability_ids = [item.capability_id for item in self.capabilities]
        if len(set(capability_ids)) != len(capability_ids):
            raise ValueError("agent capability IDs must be unique")
        scientific_mappings: dict[str, set[str]] = {}
        for capability in self.capabilities:
            for scientific_id in capability.supports_scientific_capability_ids:
                scientific_mappings.setdefault(scientific_id, set()).add(
                    capability.capability_id
                )
        ambiguous = sorted(
            scientific_id
            for scientific_id, targets in scientific_mappings.items()
            if len(targets) > 1
        )
        if ambiguous:
            raise ValueError(
                "scientific capabilities have ambiguous executable mappings: "
                + ", ".join(ambiguous)
            )
        return self


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
    literature_document_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    raw_literature_artifact_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    canonical_text_artifact_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    literature_ingestion_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    literature_representation_selection_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    literature_representation_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    historical_evidence_authorization_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    expert_profile_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    expert_opinion_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    expert_attribution_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    knowledge_relation_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    curation_record_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    trusted_record_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_snapshot_id(self) -> KnowledgeSnapshot:
        from .serialization import content_hash

        identity = self.model_dump(
            mode="json",
            exclude={"snapshot_id", "created_at"},
            exclude_defaults=True,
        )
        if self.snapshot_id != f"snapshot-{content_hash(identity)[:24]}":
            raise ValueError("KnowledgeSnapshot snapshot_id is not content-bound")
        return self


class KnowledgeGraphNode(StrictModel):
    node_id: NonBlankStr
    record_type: KnowledgeEntityType
    record_id: NonBlankStr
    record_hash: Sha256Str
    curation_status: CurationStatus | None = None

    @model_validator(mode="after")
    def validate_node_id(self) -> KnowledgeGraphNode:
        from .serialization import content_hash

        identity = {"record_type": self.record_type, "record_id": self.record_id}
        if self.node_id != f"knowledge-node-{content_hash(identity)[:24]}":
            raise ValueError("KnowledgeGraphNode node_id is not content-bound")
        return self


class KnowledgeGraphEdge(StrictModel):
    relation_id: NonBlankStr
    relation_hash: Sha256Str
    subject_node_id: NonBlankStr
    predicate: KnowledgePredicate
    object_node_id: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    curation_status: CurationStatus | None = None


class KnowledgeGraph(StrictModel):
    graph_id: NonBlankStr
    view_mode: KnowledgeViewMode
    domain_filter: NonBlankStr | None = None
    topic_filter: NonBlankStr | None = None
    nodes: tuple[KnowledgeGraphNode, ...]
    edges: tuple[KnowledgeGraphEdge, ...]
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_references(self) -> KnowledgeGraph:
        from .serialization import content_hash

        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("KnowledgeGraph node IDs must be unique")
        relation_ids = [edge.relation_id for edge in self.edges]
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("KnowledgeGraph relation IDs must be unique")
        known_nodes = set(node_ids)
        for edge in self.edges:
            if (
                edge.subject_node_id not in known_nodes
                or edge.object_node_id not in known_nodes
            ):
                raise ValueError("KnowledgeGraph edge references an unknown node")
        if self.view_mode == KnowledgeViewMode.TRUSTED:
            for node in self.nodes:
                if (
                    node.record_type
                    in {"literature_document", "expert_opinion", "knowledge_relation"}
                    and node.curation_status != CurationStatus.ACCEPTED
                ):
                    raise ValueError("trusted KnowledgeGraph contains an unaccepted entity")
            if any(
                edge.curation_status != CurationStatus.ACCEPTED
                for edge in self.edges
            ):
                raise ValueError("trusted KnowledgeGraph contains an unaccepted relation")
        if tuple(sorted(self.nodes, key=lambda item: (item.record_type, item.record_id))) != self.nodes:
            raise ValueError("KnowledgeGraph nodes must be deterministically ordered")
        if tuple(sorted(self.edges, key=lambda item: item.relation_id)) != self.edges:
            raise ValueError("KnowledgeGraph edges must be deterministically ordered")
        identity = self.model_dump(
            mode="json", exclude={"graph_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"knowledge-graph-{content_hash(identity)[:24]}"
        if self.graph_id != expected_id:
            raise ValueError("KnowledgeGraph graph_id is not content-bound")
        payload = {"graph_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("KnowledgeGraph content_hash is invalid")
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
        if self.multiple_candidates_required and not self.scientifically_distinct_axes:
            raise ValueError("multiple candidates require at least one scientific axis")
        if len(set(self.scientifically_distinct_axes)) != len(
            self.scientifically_distinct_axes
        ):
            raise ValueError("scientifically distinct axes must be unique")
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
    distinguishing_value: NonBlankStr
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


class PlanningLLMResponse(StrictModel):
    intent: IntentInterpretation
    ambiguity_assessment: AmbiguityAssessment
    candidates: tuple[CandidatePlanDraft, ...] = Field(min_length=1, max_length=4)


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
        candidate_distinctions = tuple(
            (candidate.distinguishing_axis, candidate.distinguishing_value)
            for candidate in self.candidates
        )
        if len(set(candidate_distinctions)) != len(candidate_distinctions):
            raise ValueError("candidate distinguishing axis/value pairs must be unique")
        declared_axes = set(self.ambiguity_assessment.scientifically_distinct_axes)
        candidate_axes = {candidate.distinguishing_axis for candidate in self.candidates}
        if multiple and declared_axes != candidate_axes:
            raise ValueError("ambiguity assessment axes do not match candidate axes")
        identity = self.model_dump(mode="json", exclude={"proposal_id"})
        if self.proposal_id != f"planning-proposal-{content_hash(identity)[:24]}":
            raise ValueError("PlanningProposalSet proposal_id is not content-bound")
        return self


class PlanCompilationReceipt(StrictModel):
    receipt_id: NonBlankStr
    plan_id: NonBlankStr
    plan_hash: Sha256Str
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    planning_proposal_id: NonBlankStr
    planning_proposal_hash: Sha256Str
    materializer_version: NonBlankStr
    origin: NonBlankStr = "phase2c_grounded_compiler"
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> PlanCompilationReceipt:
        from .serialization import content_hash

        if self.origin != "phase2c_grounded_compiler":
            raise ValueError("PlanCompilationReceipt origin is invalid")
        identity = self.model_dump(
            mode="json", exclude={"receipt_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"plan-compilation-receipt-{content_hash(identity)[:24]}"
        if self.receipt_id != expected_id:
            raise ValueError("PlanCompilationReceipt receipt_id is not content-bound")
        payload = {"receipt_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("PlanCompilationReceipt content_hash is invalid")
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
        from .serialization import content_hash

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
    response: ApprovalLLMResponse
    policy_decision: ApprovalDecision
    policy_reasons: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ApprovalReviewRecord:
        from .serialization import content_hash

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
        from .serialization import content_hash

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


class SPCExportPackage(StrictModel):
    """Verified, immutable view of an SPC export package."""

    package_id: NonBlankStr
    export_manifest: ExportManifest
    export_manifest_hash: Sha256Str
    checksums_hash: Sha256Str
    source_plan: ScientificQuestionPlan
    source_plan_hash: Sha256Str
    handoff: AgentHandoffPackage
    gate: GateVerdict
    gate_hash: Sha256Str
    trust_policy: ProjectTrustPolicy
    trust_policy_hash: Sha256Str
    plan_compilation_receipt: PlanCompilationReceipt
    approval_receipt: IndependentApprovalReceipt
    capability_bindings: tuple[CapabilityBinding, ...]
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_bindings(self) -> SPCExportPackage:
        from .serialization import content_hash

        plan_hash = content_hash(self.source_plan)
        if self.source_plan_hash != plan_hash:
            raise ValueError("source_plan_hash does not match source_plan")
        plan_binding = (
            self.source_plan.plan_id,
            self.source_plan.version,
            plan_hash,
        )
        if (
            self.export_manifest.source_plan_id,
            self.export_manifest.source_plan_version,
            self.export_manifest.source_plan_hash,
        ) != plan_binding:
            raise ValueError("export_manifest does not bind source_plan")
        if (
            self.handoff.source_plan_id,
            self.handoff.source_plan_version,
            self.handoff.source_plan_hash,
        ) != plan_binding:
            raise ValueError("handoff does not bind source_plan")
        if (
            self.gate.candidate_id,
            self.gate.candidate_version,
            self.gate.candidate_content_hash,
        ) != plan_binding:
            raise ValueError("gate does not bind source_plan")
        if not self.gate.passed:
            raise ValueError("gate must have passed")
        if self.export_manifest_hash != content_hash(self.export_manifest):
            raise ValueError("export_manifest_hash does not match export_manifest")
        if self.gate_hash != content_hash(self.gate):
            raise ValueError("gate_hash does not match gate")
        if self.trust_policy_hash != content_hash(self.trust_policy):
            raise ValueError("trust_policy_hash does not match trust_policy")
        if self.trust_policy.approval_mode != ApprovalMode.INDEPENDENT_REQUIRED:
            raise ValueError("downstream import requires independent approval policy")
        if (
            self.gate.trust_policy_version,
            self.gate.trust_policy_hash,
        ) != (self.trust_policy.policy_version, self.trust_policy_hash):
            raise ValueError("gate does not bind trust_policy")
        if (
            self.gate.plan_compilation_receipt_id,
            self.gate.plan_compilation_receipt_hash,
        ) != (
            self.plan_compilation_receipt.receipt_id,
            self.plan_compilation_receipt.content_hash,
        ):
            raise ValueError("gate does not bind plan_compilation_receipt")
        if (
            self.plan_compilation_receipt.plan_id,
            self.plan_compilation_receipt.plan_hash,
        ) != (self.source_plan.plan_id, plan_hash):
            raise ValueError("plan_compilation_receipt does not bind source_plan")
        if (
            self.gate.independent_approval_receipt_id,
            self.gate.independent_approval_receipt_hash,
        ) != (self.approval_receipt.receipt_id, self.approval_receipt.content_hash):
            raise ValueError("gate does not bind approval_receipt")
        if (
            self.gate.approval_verdict_id,
            self.gate.approval_verdict_hash,
        ) != (
            self.approval_receipt.verdict_id,
            self.approval_receipt.verdict_hash,
        ):
            raise ValueError("gate and approval_receipt bind different verdicts")
        if (
            self.approval_receipt.candidate_id,
            self.approval_receipt.candidate_version,
            self.approval_receipt.candidate_hash,
        ) != plan_binding:
            raise ValueError("approval_receipt does not bind source_plan")
        if self.capability_bindings != self.handoff.capability_bindings:
            raise ValueError("capability_bindings do not match handoff")
        if (
            self.export_manifest.export_id,
            self.export_manifest.target_agent,
        ) != (self.handoff.export_id, self.handoff.target_agent):
            raise ValueError("export_manifest does not bind handoff")
        identity = self.model_dump(
            mode="json", exclude={"package_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"spc-export-package-{content_hash(identity)[:24]}"
        if self.package_id != expected_id:
            raise ValueError("SPCExportPackage package_id is not content-bound")
        payload = {"package_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("SPCExportPackage content_hash is invalid")
        return self


class ExecutionProposal(StrictModel):
    """Non-authorized, non-runnable proposal for a downstream agent."""

    proposal_id: NonBlankStr
    source_plan_id: NonBlankStr
    source_plan_hash: Sha256Str
    export_id: NonBlankStr
    export_manifest_hash: Sha256Str
    gate_id: NonBlankStr
    gate_hash: Sha256Str
    approval_receipt_id: NonBlankStr
    approval_receipt_hash: Sha256Str
    plan_compilation_receipt_id: NonBlankStr
    plan_compilation_receipt_hash: Sha256Str
    execution_context: ScientificTaskExecutionContext
    execution_context_hash: Sha256Str
    task_id: NonBlankStr
    depends_on_task_ids: tuple[NonBlankStr, ...] = ()
    scientific_capability_id: NonBlankStr
    executable_capability_id: NonBlankStr
    executable_capability_version: NonBlankStr
    agent_capability_catalog_version: NonBlankStr
    agent_capability_catalog_hash: Sha256Str
    target_agent: NonBlankStr
    target_environment: NonBlankStr
    required_inputs: FrozenDict = Field(default_factory=FrozenDict)
    expected_outputs: tuple[NonBlankStr, ...] = ()
    execution_assumptions: tuple[NonBlankStr, ...] = ()
    resource_requirements: FrozenDict = Field(default_factory=FrozenDict)
    output_reconciliation: FrozenDict = Field(default_factory=FrozenDict)
    validation_requirements: tuple[NonBlankStr, ...] = ()
    provenance_requirements: tuple[NonBlankStr, ...] = ()
    adapter_id: NonBlankStr
    adapter_version: NonBlankStr
    authorized: bool = False
    runnable: bool = False
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_safety(self) -> ExecutionProposal:
        from .serialization import content_hash

        if self.authorized or self.runnable:
            raise ValueError("ExecutionProposal must remain unauthorized and non-runnable")
        if self.execution_context_hash != self.execution_context.content_hash:
            raise ValueError("execution_context_hash does not match execution_context")
        if (
            self.execution_context.source_plan_id,
            self.execution_context.source_plan_hash,
            self.execution_context.task_id,
        ) != (self.source_plan_id, self.source_plan_hash, self.task_id):
            raise ValueError("execution_context does not bind the proposal task and plan")
        if self.depends_on_task_ids != self.execution_context.depends_on_task_ids:
            raise ValueError("proposal dependencies do not match execution_context")
        unsafe_paths = tuple(
            path
            for field_name in (
                "required_inputs",
                "resource_requirements",
                "output_reconciliation",
            )
            for path in prohibited_execution_payload_paths(
                getattr(self, field_name), field_name
            )
        )
        if unsafe_paths:
            raise ValueError(
                "ExecutionProposal contains executable payload fields: "
                + ", ".join(unsafe_paths)
            )
        identity = self.model_dump(
            mode="json", exclude={"proposal_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"execution-proposal-{content_hash(identity)[:24]}"
        if self.proposal_id != expected_id:
            raise ValueError("ExecutionProposal proposal_id is not content-bound")
        payload = {"proposal_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExecutionProposal content_hash is invalid")
        return self
