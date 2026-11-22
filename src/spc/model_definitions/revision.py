from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ..immutable import FrozenDict

from .approval import (
    ApprovalDecision,
    ApprovalReviewInput,
    ApprovalReviewRecord,
    ApprovalVerdict,
    IndependentApprovalReceipt,
    PlanValidationRecord,
)
from .base import NonBlankStr, Sha256Str, StrictModel
from .evidence import (
    ComparisonConstraint,
    ConflictSet,
    EvidenceGap,
    ReportedResult,
    SourceClaim,
)
from .planning import (
    CandidatePlanDraft,
    IntentInterpretation,
    PlanCompilationReceipt,
    PlanningProposalSet,
    ScientificPlanningInput,
)
from .scientific import ScientificQuestionPlan


class PlanChangeOperation(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    REORDERED = "reordered"


class PlanRevisionActualChange(StrictModel):
    path: NonBlankStr
    operation: PlanChangeOperation
    parent_value_hash: Sha256Str | None = None
    revised_value_hash: Sha256Str | None = None


class PlanRevisionFeedbackSource(StrEnum):
    DETERMINISTIC_VALIDATION = "deterministic_validation"
    APPROVAL_RED_FLAG = "approval_red_flag"
    REQUIRED_FIX = "required_fix"


class PlanRevisionDisposition(StrEnum):
    ADDRESSED = "addressed"
    UNRESOLVED = "unresolved"
    NOT_ACCEPTED = "not_accepted"


class PlanRevisionFeedback(StrictModel):
    feedback_id: NonBlankStr
    source: PlanRevisionFeedbackSource
    code: NonBlankStr
    description: NonBlankStr
    plan_path: NonBlankStr | None = None
    evidence_refs: tuple[NonBlankStr, ...] = ()
    claim_refs: tuple[NonBlankStr, ...] = ()
    task_refs: tuple[NonBlankStr, ...] = ()
    capability_refs: tuple[NonBlankStr, ...] = ()
    blocking: bool = True


class PlanRevisionConstraints(StrictModel):
    original_request: NonBlankStr
    domain: NonBlankStr
    domain_pack_version: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    evidence_packet_id: NonBlankStr
    evidence_packet_hash: Sha256Str
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    allowed_evidence_ids: tuple[NonBlankStr, ...]
    allowed_claim_ids: tuple[NonBlankStr, ...]
    allowed_capability_ids: tuple[NonBlankStr, ...]
    required_human_decision_ids: tuple[NonBlankStr, ...] = ()
    unresolved_conflict_ids: tuple[NonBlankStr, ...] = ()


def _candidate_task_ids_by_key(
    candidate: CandidatePlanDraft,
    plan: ScientificQuestionPlan,
) -> dict[str, str]:
    from ..planning.identity import derive_candidate_task_id

    task_ids = {
        task.task_key: derive_candidate_task_id(
            candidate.candidate_key,
            task.model_dump(mode="json"),
        )
        for task in candidate.task_drafts
    }
    if len(task_ids) != len(candidate.task_drafts):
        raise ValueError("revision candidate task keys must be unique")
    if set(task_ids.values()) != {task.task_id for task in plan.tasks}:
        raise ValueError("revision candidate task keys do not bind the plan tasks")
    return task_ids


def _feedback_task_key_from_revision_history(
    feedback: PlanRevisionFeedback,
    task_ref: str,
    review_input: object,
    *,
    visited_revision_inputs: frozenset[str] = frozenset(),
) -> str | None:
    revision_context = getattr(review_input, "revision_context", None)
    if revision_context is None:
        return None
    source_input = revision_context.revision_input
    if source_input.revision_input_id in visited_revision_inputs:
        raise ValueError("revision feedback task lineage is cyclic")
    visited = visited_revision_inputs | {source_input.revision_input_id}
    source_feedback = next(
        (
            item
            for item in source_input.feedback
            if item.feedback_id == feedback.feedback_id
        ),
        None,
    )
    if source_feedback is not None and source_feedback != feedback:
        raise ValueError("carried revision feedback identity was modified")

    task_key = None
    if source_feedback is not None:
        source_candidate = next(
            candidate
            for candidate in source_input.parent_proposal.candidates
            if candidate.candidate_key == source_input.parent_candidate_key
        )
        source_tasks = _candidate_task_ids_by_key(
            source_candidate,
            source_input.parent_plan,
        )
        task_key = next(
            (key for key, task_id in source_tasks.items() if task_id == task_ref),
            None,
        )
    if task_key is None:
        task_key = _feedback_task_key_from_revision_history(
            feedback,
            task_ref,
            source_input.approval_review_input,
            visited_revision_inputs=visited,
        )
    if task_key is None:
        return None

    revised_tasks = _candidate_task_ids_by_key(
        revision_context.revision_record.response.candidate,
        revision_context.revised_plan,
    )
    if task_key not in revised_tasks:
        raise ValueError(
            "carried revision feedback task has no verified child task_key mapping"
        )
    return task_key


class PlanRevisionInput(StrictModel):
    revision_input_id: NonBlankStr
    revision_index: int = Field(ge=1)
    planning_input: ScientificPlanningInput
    parent_proposal: PlanningProposalSet
    parent_candidate_key: NonBlankStr
    parent_plan: ScientificQuestionPlan
    parent_plan_hash: Sha256Str
    parent_compilation_receipt: PlanCompilationReceipt
    plan_validation_record: PlanValidationRecord
    plan_validation_hash: Sha256Str
    approval_review_input: ApprovalReviewInput | RevisionApprovalReviewInput
    approval_review_record: ApprovalReviewRecord
    approval_verdict: ApprovalVerdict
    approval_receipt: IndependentApprovalReceipt
    feedback: tuple[PlanRevisionFeedback, ...] = Field(min_length=1)
    constraints: PlanRevisionConstraints
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_bindings_and_feedback(self) -> PlanRevisionInput:
        from ..serialization import content_hash

        if self.parent_plan_hash != content_hash(self.parent_plan):
            raise ValueError("parent_plan_hash does not match parent_plan")
        if self.plan_validation_hash != content_hash(self.plan_validation_record):
            raise ValueError("plan_validation_hash does not match PlanValidationRecord")
        if (
            self.parent_proposal.planning_input_id != self.planning_input.planning_input_id
            or self.parent_proposal.planning_input_hash != self.planning_input.content_hash
        ):
            raise ValueError("parent_proposal does not bind planning_input")
        if self.parent_candidate_key not in {
            candidate.candidate_key for candidate in self.parent_proposal.candidates
        }:
            raise ValueError("parent_candidate_key is not present in parent_proposal")
        if (
            self.parent_compilation_receipt.plan_id != self.parent_plan.plan_id
            or self.parent_compilation_receipt.plan_hash != self.parent_plan_hash
            or self.parent_compilation_receipt.planning_input_id
            != self.planning_input.planning_input_id
            or self.parent_compilation_receipt.planning_input_hash
            != self.planning_input.content_hash
            or self.parent_compilation_receipt.planning_proposal_id
            != self.parent_proposal.proposal_id
            or self.parent_compilation_receipt.planning_proposal_hash
            != content_hash(self.parent_proposal)
        ):
            raise ValueError("parent compilation receipt does not bind revision sources")
        if (
            self.plan_validation_record.plan_id != self.parent_plan.plan_id
            or self.plan_validation_record.plan_version != self.parent_plan.version
            or self.plan_validation_record.plan_content_hash != self.parent_plan_hash
        ):
            raise ValueError("PlanValidationRecord does not bind parent_plan")
        if (
            self.approval_review_input.candidate_plan_hash != self.parent_plan_hash
            or self.approval_review_input.review_input_id
            != self.approval_review_record.review_input_id
            or self.approval_review_input.content_hash
            != self.approval_review_record.review_input_hash
            or self.approval_review_record.review_id != self.approval_receipt.review_id
            or self.approval_review_record.content_hash != self.approval_receipt.review_hash
            or self.approval_receipt.verdict_id != self.approval_verdict.verdict_id
            or self.approval_receipt.verdict_hash != content_hash(self.approval_verdict)
            or self.approval_receipt.candidate_id != self.parent_plan.plan_id
            or self.approval_receipt.candidate_version != self.parent_plan.version
            or self.approval_receipt.candidate_hash != self.parent_plan_hash
        ):
            raise ValueError("independent approval chain does not bind parent_plan")
        if self.approval_verdict.decision != ApprovalDecision.REQUEST_REVISION:
            raise ValueError("PlanRevisionInput requires a REQUEST_REVISION verdict")

        expected_constraints = {
            "original_request": self.planning_input.original_request,
            "domain": self.planning_input.domain,
            "domain_pack_version": self.planning_input.domain_pack_version,
            "context_id": self.planning_input.context_id,
            "context_hash": self.planning_input.context_hash,
            "evidence_packet_id": self.planning_input.evidence_packet_id,
            "evidence_packet_hash": self.planning_input.evidence_packet_hash,
            "planning_input_id": self.planning_input.planning_input_id,
            "planning_input_hash": self.planning_input.content_hash,
            "allowed_evidence_ids": self.planning_input.allowed_evidence_ids,
            "allowed_claim_ids": self.planning_input.allowed_claim_ids,
            "allowed_capability_ids": self.planning_input.allowed_capability_ids,
            "required_human_decision_ids": tuple(
                item.decision_id for item in self.planning_input.required_human_decisions
            ),
            "unresolved_conflict_ids": tuple(
                item.conflict_id
                for item in self.planning_input.conflict_sets
                if item.resolution_status == "unresolved"
            ),
        }
        if self.constraints.model_dump(mode="python") != expected_constraints:
            raise ValueError("revision constraints do not match trusted planning input")

        feedback_ids = tuple(item.feedback_id for item in self.feedback)
        if len(set(feedback_ids)) != len(feedback_ids):
            raise ValueError("revision feedback IDs must be unique")
        allowed_evidence = set(self.planning_input.allowed_evidence_ids)
        allowed_claims = set(self.planning_input.allowed_claim_ids)
        allowed_tasks = {task.task_id for task in self.parent_plan.tasks}
        parent_candidate = next(
            candidate
            for candidate in self.parent_proposal.candidates
            if candidate.candidate_key == self.parent_candidate_key
        )
        current_tasks_by_key = _candidate_task_ids_by_key(
            parent_candidate,
            self.parent_plan,
        )
        allowed_capabilities = set(self.planning_input.allowed_capability_ids)
        allowed_plan_roots = {
            "plan",
            "atomic_questions",
            "hypothesis",
            "model",
            "observables",
            "comparison_baselines",
            "acceptance_criteria",
            "falsification_criteria",
            "intent_fingerprint",
            "system_fingerprint",
            "method_fingerprint",
            "fingerprint_differences",
            "evidence_refs",
            "assumptions",
            "unknowns",
            "proposed_deviations",
            "scientific_capability_ids",
            "tasks",
            "limitations",
            "required_human_decisions",
            "source_query_manifest",
        }
        for item in self.feedback:
            if item.plan_path is not None:
                root = item.plan_path.split(".", 1)[0].split("[", 1)[0]
                if root not in allowed_plan_roots:
                    raise ValueError(
                        f"revision feedback has unknown plan path {item.plan_path}"
                    )
            if not set(item.evidence_refs).issubset(allowed_evidence):
                raise ValueError("revision feedback contains fabricated evidence reference")
            if not set(item.claim_refs).issubset(allowed_claims):
                raise ValueError("revision feedback contains fabricated claim reference")
            for task_ref in item.task_refs:
                if task_ref in allowed_tasks:
                    continue
                task_key = _feedback_task_key_from_revision_history(
                    item,
                    task_ref,
                    self.approval_review_input,
                )
                if task_key is None:
                    raise ValueError(
                        "revision feedback contains fabricated task reference"
                    )
                if task_key not in current_tasks_by_key:
                    raise ValueError(
                        "revision feedback task has no verified current task_key mapping"
                    )
            if not set(item.capability_refs).issubset(allowed_capabilities):
                raise ValueError("revision feedback contains fabricated capability reference")

        identity = self.model_dump(
            mode="json", exclude={"revision_input_id", "content_hash"}
        )
        expected_id = f"plan-revision-input-{content_hash(identity)[:24]}"
        if self.revision_input_id != expected_id:
            raise ValueError("PlanRevisionInput revision_input_id is not content-bound")
        payload = {"revision_input_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("PlanRevisionInput content_hash is invalid")
        return self


class PlanRevisionFeedbackResponse(StrictModel):
    feedback_id: NonBlankStr
    disposition: PlanRevisionDisposition
    changed_plan_paths: tuple[NonBlankStr, ...] = ()
    rationale: NonBlankStr

    @model_validator(mode="after")
    def validate_disposition(self) -> PlanRevisionFeedbackResponse:
        if self.disposition == PlanRevisionDisposition.ADDRESSED and not self.changed_plan_paths:
            raise ValueError("addressed feedback requires changed_plan_paths")
        return self


class PlanRevisionLLMResponse(StrictModel):
    intent: IntentInterpretation
    candidate: CandidatePlanDraft
    feedback_responses: tuple[PlanRevisionFeedbackResponse, ...] = Field(min_length=1)


class PlanRevisionRecord(StrictModel):
    revision_id: NonBlankStr
    revision_index: int = Field(ge=1)
    revision_input_id: NonBlankStr
    revision_input_hash: Sha256Str
    parent_plan_id: NonBlankStr
    parent_plan_hash: Sha256Str
    trigger_review_id: NonBlankStr
    trigger_review_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config: FrozenDict = Field(default_factory=FrozenDict)
    response: PlanRevisionLLMResponse
    planning_proposal_id: NonBlankStr
    planning_proposal_hash: Sha256Str
    revised_plan_id: NonBlankStr
    revised_plan_hash: Sha256Str
    validation_id: NonBlankStr
    validation_hash: Sha256Str
    substantive_change: bool
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> PlanRevisionRecord:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"revision_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"plan-revision-{content_hash(identity)[:24]}"
        if self.revision_id != expected_id:
            raise ValueError("PlanRevisionRecord revision_id is not content-bound")
        payload = {"revision_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("PlanRevisionRecord content_hash is invalid")
        return self


class RevisionApprovalContext(StrictModel):
    contract_version: Literal["1.0.0"] = "1.0.0"
    parent_plan: ScientificQuestionPlan
    parent_plan_hash: Sha256Str
    revised_plan: ScientificQuestionPlan
    revised_plan_hash: Sha256Str
    trigger_review: ApprovalReviewRecord
    trigger_review_hash: Sha256Str
    revision_input: PlanRevisionInput
    revision_input_hash: Sha256Str
    revision_record: PlanRevisionRecord
    revision_record_hash: Sha256Str
    tracked_feedback: tuple[PlanRevisionFeedback, ...] = Field(min_length=1)
    planner_responses: tuple[PlanRevisionFeedbackResponse, ...] = Field(min_length=1)
    actual_changes: tuple[PlanRevisionActualChange, ...] = Field(min_length=1)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_bindings(self) -> RevisionApprovalContext:
        from ..serialization import content_hash

        if self.parent_plan_hash != content_hash(self.parent_plan):
            raise ValueError("revision approval parent_plan_hash is invalid")
        if self.revised_plan_hash != content_hash(self.revised_plan):
            raise ValueError("revision approval revised_plan_hash is invalid")
        if self.trigger_review_hash != self.trigger_review.content_hash:
            raise ValueError("revision approval trigger_review_hash is invalid")
        if self.revision_input_hash != self.revision_input.content_hash:
            raise ValueError("revision approval revision_input_hash is invalid")
        if self.revision_record_hash != self.revision_record.content_hash:
            raise ValueError("revision approval revision_record_hash is invalid")
        bindings = {
            "revision_input.parent_plan": (
                self.revision_input.parent_plan_hash,
                self.parent_plan_hash,
            ),
            "revision_input.trigger_review_id": (
                self.revision_input.approval_review_record.review_id,
                self.trigger_review.review_id,
            ),
            "revision_input.trigger_review_hash": (
                self.revision_input.approval_review_record.content_hash,
                self.trigger_review_hash,
            ),
            "revision_record.input_id": (
                self.revision_record.revision_input_id,
                self.revision_input.revision_input_id,
            ),
            "revision_record.input_hash": (
                self.revision_record.revision_input_hash,
                self.revision_input_hash,
            ),
            "revision_record.parent_id": (
                self.revision_record.parent_plan_id,
                self.parent_plan.plan_id,
            ),
            "revision_record.parent_hash": (
                self.revision_record.parent_plan_hash,
                self.parent_plan_hash,
            ),
            "revision_record.revised_id": (
                self.revision_record.revised_plan_id,
                self.revised_plan.plan_id,
            ),
            "revision_record.revised_hash": (
                self.revision_record.revised_plan_hash,
                self.revised_plan_hash,
            ),
        }
        mismatches = tuple(
            name for name, (actual, expected) in bindings.items() if actual != expected
        )
        if mismatches:
            raise ValueError(
                "revision approval context has cross-object bindings: "
                + ", ".join(mismatches)
            )
        feedback_ids = tuple(item.feedback_id for item in self.tracked_feedback)
        if len(set(feedback_ids)) != len(feedback_ids):
            raise ValueError("revision approval tracked feedback IDs must be unique")
        response_ids = tuple(item.feedback_id for item in self.planner_responses)
        if set(response_ids) != set(feedback_ids) or len(response_ids) != len(
            feedback_ids
        ):
            raise ValueError("planner responses must cover tracked feedback exactly once")
        if self.revision_input.feedback != self.tracked_feedback:
            raise ValueError("tracked feedback must match the bound revision input")
        if self.revision_record.response.feedback_responses != self.planner_responses:
            raise ValueError("planner responses must match the bound revision record")
        payload = self.model_dump(
            mode="json", exclude={"content_hash"}, exclude_none=True
        )
        if self.content_hash != content_hash(payload):
            raise ValueError("RevisionApprovalContext content_hash is invalid")
        return self


class RevisionApprovalReviewInput(StrictModel):
    contract_version: Literal["1.0.0"] = "1.0.0"
    review_input_id: NonBlankStr
    base_review_input: ApprovalReviewInput
    revision_context: RevisionApprovalContext
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> RevisionApprovalReviewInput:
        from ..serialization import content_hash

        if (
            self.base_review_input.candidate_plan_hash
            != self.revision_context.revised_plan_hash
            or self.base_review_input.candidate_plan
            != self.revision_context.revised_plan
        ):
            raise ValueError("revision approval input does not bind the revised plan")
        identity = self.model_dump(
            mode="json",
            exclude={"review_input_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"revision-approval-input-{content_hash(identity)[:24]}"
        if self.review_input_id != expected_id:
            raise ValueError("RevisionApprovalReviewInput ID is not content-bound")
        payload = {"review_input_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("RevisionApprovalReviewInput content_hash is invalid")
        return self

    @property
    def candidate_plan(self) -> ScientificQuestionPlan:
        return self.base_review_input.candidate_plan

    @property
    def candidate_plan_hash(self) -> str:
        return self.base_review_input.candidate_plan_hash

    @property
    def plan_validation_record(self) -> PlanValidationRecord:
        return self.base_review_input.plan_validation_record

    @property
    def original_request(self) -> str:
        return self.base_review_input.original_request

    @property
    def source_claims(self) -> tuple[SourceClaim, ...]:
        return self.base_review_input.source_claims

    @property
    def reported_results(self) -> tuple[ReportedResult, ...]:
        return self.base_review_input.reported_results

    @property
    def comparison_constraints(self) -> tuple[ComparisonConstraint, ...]:
        return self.base_review_input.comparison_constraints

    @property
    def conflict_sets(self) -> tuple[ConflictSet, ...]:
        return self.base_review_input.conflict_sets

    @property
    def evidence_gaps(self) -> tuple[EvidenceGap, ...]:
        return self.base_review_input.evidence_gaps

    @property
    def allowed_evidence_ids(self) -> tuple[str, ...]:
        return self.base_review_input.allowed_evidence_ids

    @property
    def allowed_claim_ids(self) -> tuple[str, ...]:
        return self.base_review_input.allowed_claim_ids

    @property
    def allowed_task_ids(self) -> tuple[str, ...]:
        return self.base_review_input.allowed_task_ids

    @property
    def allowed_capability_ids(self) -> tuple[str, ...]:
        return self.base_review_input.allowed_capability_ids


def parse_approval_review_input(
    value: object,
) -> ApprovalReviewInput | RevisionApprovalReviewInput:
    if isinstance(value, (ApprovalReviewInput, RevisionApprovalReviewInput)):
        return value
    if isinstance(value, dict) and "revision_context" in value:
        return RevisionApprovalReviewInput.model_validate(value)
    return ApprovalReviewInput.model_validate(value)


PlanRevisionInput.model_rebuild()


class PlanRevisionAttemptStatus(StrEnum):
    COMPLETED = "completed"
    TERMINATED = "terminated"
    UNCERTAIN = "uncertain"


class PlanRevisionAttemptStart(StrictModel):
    attempt_id: NonBlankStr
    chain_id: NonBlankStr
    attempt_index: int = Field(ge=1)
    revision_input_id: NonBlankStr
    revision_input_hash: Sha256Str
    parent_plan_id: NonBlankStr
    parent_plan_hash: Sha256Str
    trigger_review_id: NonBlankStr
    trigger_review_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config_hash: Sha256Str
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> PlanRevisionAttemptStart:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"attempt_id", "content_hash"}
        )
        expected_id = f"plan-revision-attempt-{content_hash(identity)[:24]}"
        if self.attempt_id != expected_id:
            raise ValueError("PlanRevisionAttemptStart ID is not content-bound")
        payload = {"attempt_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("PlanRevisionAttemptStart content_hash is invalid")
        return self


class PlanRevisionAttemptOutcome(StrictModel):
    attempt_id: NonBlankStr
    attempt_start_hash: Sha256Str
    status: PlanRevisionAttemptStatus
    revision_record_id: NonBlankStr | None = None
    revision_record_hash: Sha256Str | None = None
    round_index: int | None = Field(default=None, ge=1)
    failure_category: NonBlankStr | None = None
    failure_message: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_outcome(self) -> PlanRevisionAttemptOutcome:
        from ..serialization import content_hash

        completed = self.status == PlanRevisionAttemptStatus.COMPLETED
        completion_values = (
            self.revision_record_id,
            self.revision_record_hash,
            self.round_index,
        )
        if completed != all(value is not None for value in completion_values):
            raise ValueError("completed revision attempt requires round bindings")
        if completed and any(
            value is not None for value in (self.failure_category, self.failure_message)
        ):
            raise ValueError("completed revision attempt cannot contain failure details")
        if not completed and (
            self.failure_category is None or self.failure_message is None
        ):
            raise ValueError("non-completed revision attempt requires failure details")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("PlanRevisionAttemptOutcome content_hash is invalid")
        return self
