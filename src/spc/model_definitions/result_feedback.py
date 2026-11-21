from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from ..immutable import FrozenDict

from .base import NonBlankStr, Sha256Str, StrictModel
from .evidence import ResultStatus
from .scientific import FingerprintDifference, MethodFingerprint, SystemFingerprint


class ResultEvidenceType(StrEnum):
    COMPUTED_RESULT = "computed_result"
    EXPERIMENTAL_RESULT = "experimental_result"
    DIAGNOSTIC_RESULT = "diagnostic_result"
    FAILED_EXECUTION = "failed_execution"
    PARTIAL_RESULT = "partial_result"
    NULL_OR_NEGATIVE_RESULT = "null_or_negative_result"


class ResultExecutionOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    NULL_OR_NEGATIVE = "null_or_negative"


class ResultContextComparisonStatus(StrEnum):
    MATCHED = "MATCHED"
    EXPLICIT_APPROVED_DEVIATION = "EXPLICIT_APPROVED_DEVIATION"
    UNDECLARED_SYSTEM_CHANGE = "UNDECLARED_SYSTEM_CHANGE"
    UNDECLARED_METHOD_CHANGE = "UNDECLARED_METHOD_CHANGE"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class ResultEvidenceIntakeStatus(StrEnum):
    RESULT_REQUIRES_CURATION = "RESULT_REQUIRES_CURATION"
    BLOCKED_CONTENT_BINDING = "BLOCKED_CONTENT_BINDING"
    BLOCKED_PLAN_BINDING = "BLOCKED_PLAN_BINDING"
    BLOCKED_CONTEXT_COMPATIBILITY = "BLOCKED_CONTEXT_COMPATIBILITY"
    BLOCKED_SCIENTIFIC_ADMISSIBILITY = "BLOCKED_SCIENTIFIC_ADMISSIBILITY"
    REJECTED = "REJECTED"
    ACCEPTED = "ACCEPTED"


class SuccessorPlanningStatus(StrEnum):
    FOLLOW_UP_PLAN_REQUIRED = "FOLLOW_UP_PLAN_REQUIRED"
    INSUFFICIENT_RESULT_EVIDENCE = "INSUFFICIENT_RESULT_EVIDENCE"
    RESULT_REQUIRES_CURATION = "RESULT_REQUIRES_CURATION"
    HUMAN_SCIENTIFIC_DECISION_REQUIRED = "HUMAN_SCIENTIFIC_DECISION_REQUIRED"
    RESEARCH_QUESTION_RESOLVED_FOR_CURRENT_SCOPE = (
        "RESEARCH_QUESTION_RESOLVED_FOR_CURRENT_SCOPE"
    )
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ResultArtifactManifestEntry(StrictModel):
    relative_path: NonBlankStr
    sha256: Sha256Str
    size_bytes: int = Field(ge=0)
    media_type: NonBlankStr = "application/octet-stream"


class ReportedObservable(StrictModel):
    observable_key: NonBlankStr
    quantity: NonBlankStr
    value: float | None = Field(default=None, allow_inf_nan=False)
    qualitative_value: NonBlankStr | None = None
    unit: NonBlankStr | None = None
    uncertainty: NonBlankStr | None = None
    result_status: ResultStatus

    @model_validator(mode="after")
    def validate_value_and_unit(self) -> ReportedObservable:
        if (self.value is None) == (self.qualitative_value is None):
            raise ValueError(
                "ReportedObservable requires exactly one numeric or qualitative value"
            )
        if self.value is not None and self.unit is None:
            raise ValueError("numeric ReportedObservable requires an explicit unit")
        return self


class ResultEvidenceSubmission(StrictModel):
    submission_id: NonBlankStr
    parent_run_id: NonBlankStr
    parent_plan_id: NonBlankStr
    parent_plan_version: NonBlankStr
    parent_plan_hash: Sha256Str
    task_id: NonBlankStr
    capability_id: NonBlankStr
    export_id: NonBlankStr | None = None
    export_manifest_hash: Sha256Str | None = None
    execution_proposal_id: NonBlankStr | None = None
    execution_proposal_hash: Sha256Str | None = None
    executor_id: NonBlankStr
    executor_version: NonBlankStr
    result_type: ResultEvidenceType
    system_fingerprint: SystemFingerprint
    method_fingerprint: MethodFingerprint
    declared_deviations: tuple[FingerprintDifference, ...] = ()
    source_artifact_manifest: tuple[ResultArtifactManifestEntry, ...] = Field(
        min_length=1
    )
    raw_artifact_checksums: FrozenDict
    reported_observables: tuple[ReportedObservable, ...] = ()
    uncertainty_metadata: FrozenDict = Field(default_factory=FrozenDict)
    convergence_metadata: FrozenDict = Field(default_factory=FrozenDict)
    execution_outcome: ResultExecutionOutcome
    untrusted_interpretation: NonBlankStr | None = None
    provenance_timestamp: datetime
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_outcome(self) -> ResultEvidenceSubmission:
        from ..serialization import content_hash

        if (
            self.provenance_timestamp.tzinfo is None
            or self.provenance_timestamp.utcoffset() is None
        ):
            raise ValueError("result provenance_timestamp must be timezone-aware")
        manifest_checksums = {
            item.relative_path: item.sha256 for item in self.source_artifact_manifest
        }
        if len(manifest_checksums) != len(self.source_artifact_manifest):
            raise ValueError("result artifact paths must be unique")
        if dict(self.raw_artifact_checksums) != manifest_checksums:
            raise ValueError("raw_artifact_checksums must match source_artifact_manifest")
        paired_bindings = (
            (self.export_id, self.export_manifest_hash),
            (self.execution_proposal_id, self.execution_proposal_hash),
        )
        if any((left is None) != (right is None) for left, right in paired_bindings):
            raise ValueError("result handoff/proposal IDs and hashes must be paired")
        outcome_by_type = {
            ResultEvidenceType.FAILED_EXECUTION: ResultExecutionOutcome.FAILED,
            ResultEvidenceType.PARTIAL_RESULT: ResultExecutionOutcome.PARTIAL,
            ResultEvidenceType.NULL_OR_NEGATIVE_RESULT: (
                ResultExecutionOutcome.NULL_OR_NEGATIVE
            ),
        }
        expected_outcome = outcome_by_type.get(self.result_type)
        if expected_outcome is not None and self.execution_outcome != expected_outcome:
            raise ValueError("result_type and execution_outcome are inconsistent")
        if (
            self.result_type
            in {ResultEvidenceType.COMPUTED_RESULT, ResultEvidenceType.EXPERIMENTAL_RESULT}
            and self.execution_outcome != ResultExecutionOutcome.COMPLETED
        ):
            raise ValueError("completed result type requires completed execution outcome")
        if self.result_type == ResultEvidenceType.FAILED_EXECUTION and self.reported_observables:
            raise ValueError("failed_execution cannot report normal scientific observables")
        identity = self.model_dump(
            mode="json", exclude={"submission_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"result-submission-{content_hash(identity)[:24]}"
        if self.submission_id != expected_id:
            raise ValueError("ResultEvidenceSubmission ID is not content-bound")
        payload = {"submission_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ResultEvidenceSubmission content_hash is invalid")
        return self


class ResultContextComparison(StrictModel):
    comparison_id: NonBlankStr
    submission_id: NonBlankStr
    submission_hash: Sha256Str
    parent_plan_id: NonBlankStr
    parent_plan_hash: Sha256Str
    status: ResultContextComparisonStatus
    differences: tuple[FingerprintDifference, ...] = ()
    approved_deviation_refs: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ResultContextComparison:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"comparison_id", "content_hash"}
        )
        expected_id = f"result-context-comparison-{content_hash(identity)[:24]}"
        if self.comparison_id != expected_id:
            raise ValueError("ResultContextComparison ID is not content-bound")
        payload = {"comparison_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ResultContextComparison content_hash is invalid")
        return self


class ResultIntakeIssue(StrictModel):
    code: NonBlankStr
    message: NonBlankStr
    blocking: bool = True


class ResultEvidenceAssessment(StrictModel):
    assessment_id: NonBlankStr
    submission_id: NonBlankStr
    submission_hash: Sha256Str
    parent_run_id: NonBlankStr
    parent_plan_id: NonBlankStr
    parent_plan_hash: Sha256Str
    task_id: NonBlankStr
    capability_id: NonBlankStr
    parent_authority_origin: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    parent_authority_ref: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    content_bound: bool
    plan_bound: bool
    context_compatible: bool
    scientifically_admissible: bool
    context_comparison: ResultContextComparison | None = None
    issues: tuple[ResultIntakeIssue, ...] = ()
    status: ResultEvidenceIntakeStatus
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ResultEvidenceAssessment:
        from ..serialization import content_hash

        if self.status == ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION and not all(
            (
                self.content_bound,
                self.plan_bound,
                self.context_compatible,
                self.scientifically_admissible,
            )
        ):
            raise ValueError("curation-ready result must pass all deterministic intake gates")
        identity = self.model_dump(
            mode="json", exclude={"assessment_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"result-assessment-{content_hash(identity)[:24]}"
        if self.assessment_id != expected_id:
            raise ValueError("ResultEvidenceAssessment ID is not content-bound")
        payload = {"assessment_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ResultEvidenceAssessment content_hash is invalid")
        return self


class ResultEvidenceMaterializationReceipt(StrictModel):
    receipt_id: NonBlankStr
    submission_id: NonBlankStr
    submission_hash: Sha256Str
    assessment_id: NonBlankStr
    assessment_hash: Sha256Str
    curation_id: NonBlankStr
    curation_hash: Sha256Str
    parent_run_id: NonBlankStr
    parent_plan_id: NonBlankStr
    parent_plan_hash: Sha256Str
    task_id: NonBlankStr
    parent_authority_origin: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    parent_authority_ref: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_id: NonBlankStr
    source_version: NonBlankStr
    accepted_evidence_ids: tuple[NonBlankStr, ...]
    reported_result_hashes: dict[NonBlankStr, Sha256Str] = Field(default_factory=dict)
    reported_observation_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    method_fact_hashes: dict[NonBlankStr, Sha256Str] = Field(default_factory=dict)
    model_fact_hashes: dict[NonBlankStr, Sha256Str] = Field(default_factory=dict)
    missing_observables: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ResultEvidenceMaterializationReceipt:
        from ..serialization import content_hash

        if len(set(self.accepted_evidence_ids)) != len(self.accepted_evidence_ids):
            raise ValueError("accepted evidence IDs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"receipt_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"result-materialization-{content_hash(identity)[:24]}"
        if self.receipt_id != expected_id:
            raise ValueError(
                "ResultEvidenceMaterializationReceipt ID is not content-bound"
            )
        payload = {"receipt_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError(
                "ResultEvidenceMaterializationReceipt content_hash is invalid"
            )
        return self


class SuccessorParentBinding(StrictModel):
    binding_id: NonBlankStr
    parent_run_id: NonBlankStr
    parent_plan_id: NonBlankStr
    parent_plan_version: NonBlankStr
    parent_plan_hash: Sha256Str
    result_submission_ids: tuple[NonBlankStr, ...] = Field(min_length=1)
    result_submission_hashes: tuple[Sha256Str, ...] = Field(min_length=1)
    materialization_receipt_ids: tuple[NonBlankStr, ...] = Field(min_length=1)
    materialization_receipt_hashes: tuple[Sha256Str, ...] = Field(min_length=1)
    accepted_evidence_ids: tuple[NonBlankStr, ...] = Field(min_length=1)
    successor_cycle_index: int = Field(ge=1)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> SuccessorParentBinding:
        from ..serialization import content_hash

        if len(self.result_submission_ids) != len(self.result_submission_hashes):
            raise ValueError("successor submission IDs and hashes must align")
        if len(self.materialization_receipt_ids) != len(
            self.materialization_receipt_hashes
        ):
            raise ValueError("successor receipt IDs and hashes must align")
        for values, label in (
            (self.result_submission_ids, "result submission IDs"),
            (self.materialization_receipt_ids, "materialization receipt IDs"),
            (self.accepted_evidence_ids, "accepted evidence IDs"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"successor {label} must be unique")
        identity = self.model_dump(
            mode="json", exclude={"binding_id", "content_hash"}
        )
        expected_id = f"successor-parent-binding-{content_hash(identity)[:24]}"
        if self.binding_id != expected_id:
            raise ValueError("SuccessorParentBinding ID is not content-bound")
        payload = {"binding_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("SuccessorParentBinding content_hash is invalid")
        return self


class ResearchUpdate(StrictModel):
    update_id: NonBlankStr
    parent_binding_id: NonBlankStr
    parent_binding_hash: Sha256Str
    previous_questions: tuple[NonBlankStr, ...]
    previous_hypotheses: tuple[NonBlankStr, ...]
    new_reported_observations: tuple[NonBlankStr, ...]
    affected_acceptance_criteria: tuple[NonBlankStr, ...] = ()
    affected_falsification_criteria: tuple[NonBlankStr, ...] = ()
    unresolved_criteria: tuple[NonBlankStr, ...] = ()
    new_conflicts: tuple[NonBlankStr, ...] = ()
    new_evidence_gaps: tuple[NonBlankStr, ...] = ()
    why_successor_plan_is_needed: NonBlankStr
    recommended_status: SuccessorPlanningStatus
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ResearchUpdate:
        from ..serialization import content_hash

        identity = self.model_dump(mode="json", exclude={"update_id", "content_hash"})
        expected_id = f"research-update-{content_hash(identity)[:24]}"
        if self.update_id != expected_id:
            raise ValueError("ResearchUpdate ID is not content-bound")
        payload = {"update_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ResearchUpdate content_hash is invalid")
        return self


class SuccessorPlanningInvocation(StrictModel):
    invocation_id: NonBlankStr
    parent_run_id: NonBlankStr
    parent_plan_id: NonBlankStr
    parent_plan_version: NonBlankStr
    parent_plan_hash: Sha256Str
    result_submission_ids: tuple[NonBlankStr, ...] = Field(min_length=1)
    result_submission_hashes: tuple[Sha256Str, ...] = Field(min_length=1)
    materialization_receipt_ids: tuple[NonBlankStr, ...] = Field(min_length=1)
    materialization_receipt_hashes: tuple[Sha256Str, ...] = Field(min_length=1)
    follow_up_request: NonBlankStr | None = None
    planning_provider_id: NonBlankStr
    planning_provider_version: NonBlankStr
    planning_provider_config_hash: Sha256Str
    # Legacy invocation fields remain readable, but new invocations keep approval and
    # human selection outside the planning identity.
    approval_provider_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    approval_provider_version: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    approval_provider_config_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    selected_candidate_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> SuccessorPlanningInvocation:
        from ..serialization import content_hash

        for identifiers, hashes, label in (
            (
                self.result_submission_ids,
                self.result_submission_hashes,
                "result submissions",
            ),
            (
                self.materialization_receipt_ids,
                self.materialization_receipt_hashes,
                "materialization receipts",
            ),
        ):
            if len(identifiers) != len(hashes):
                raise ValueError(f"successor invocation {label} must align")
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"successor invocation {label} must be unique")
        approval_fields = (
            self.approval_provider_id,
            self.approval_provider_version,
            self.approval_provider_config_hash,
        )
        if any(value is None for value in approval_fields) and any(
            value is not None for value in approval_fields
        ):
            raise ValueError("legacy successor approval provider fields must be complete")
        identity = self.model_dump(
            mode="json",
            exclude={"invocation_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"successor-invocation-{content_hash(identity)[:24]}"
        if self.invocation_id != expected_id:
            raise ValueError("SuccessorPlanningInvocation ID is not content-bound")
        payload = {"invocation_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("SuccessorPlanningInvocation content_hash is invalid")
        return self


class SuccessorCandidateSelection(StrictModel):
    selection_id: NonBlankStr
    invocation_id: NonBlankStr
    invocation_hash: Sha256Str
    proposal_id: NonBlankStr
    proposal_hash: Sha256Str
    candidate_id: NonBlankStr
    candidate_hash: Sha256Str
    selection_provenance: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> SuccessorCandidateSelection:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"selection_id", "content_hash"}
        )
        expected_id = f"successor-selection-{content_hash(identity)[:24]}"
        if self.selection_id != expected_id:
            raise ValueError("SuccessorCandidateSelection ID is not content-bound")
        payload = {"selection_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("SuccessorCandidateSelection content_hash is invalid")
        return self


class SuccessorPlanningCycle(StrictModel):
    cycle_id: NonBlankStr
    invocation_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    invocation_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    parent_binding_id: NonBlankStr
    parent_binding_hash: Sha256Str
    research_update_id: NonBlankStr
    research_update_hash: Sha256Str
    context_id: NonBlankStr
    context_hash: Sha256Str
    evidence_packet_id: NonBlankStr
    evidence_packet_hash: Sha256Str
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    planning_proposal_id: NonBlankStr | None = None
    planning_proposal_hash: Sha256Str | None = None
    candidate_plan_ids: tuple[NonBlankStr, ...] = ()
    candidate_plan_hashes: tuple[Sha256Str, ...] = ()
    selection_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    selection_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    selected_plan_id: NonBlankStr | None = None
    selected_plan_hash: Sha256Str | None = None
    approval_review_id: NonBlankStr | None = None
    approval_review_hash: Sha256Str | None = None
    approval_verdict_id: NonBlankStr | None = None
    approval_verdict_hash: Sha256Str | None = None
    approval_receipt_id: NonBlankStr | None = None
    approval_receipt_hash: Sha256Str | None = None
    gate_id: NonBlankStr | None = None
    gate_hash: Sha256Str | None = None
    status: SuccessorPlanningStatus
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> SuccessorPlanningCycle:
        from ..serialization import content_hash

        pairs = (
            (self.invocation_id, self.invocation_hash),
            (self.planning_proposal_id, self.planning_proposal_hash),
            (self.selection_id, self.selection_hash),
            (self.selected_plan_id, self.selected_plan_hash),
            (self.approval_review_id, self.approval_review_hash),
            (self.approval_verdict_id, self.approval_verdict_hash),
            (self.approval_receipt_id, self.approval_receipt_hash),
            (self.gate_id, self.gate_hash),
        )
        if any((left is None) != (right is None) for left, right in pairs):
            raise ValueError("successor artifact IDs and hashes must be paired")
        if len(self.candidate_plan_ids) != len(self.candidate_plan_hashes):
            raise ValueError("successor candidate IDs and hashes must align")
        if len(set(self.candidate_plan_ids)) != len(self.candidate_plan_ids):
            raise ValueError("successor candidate plan IDs must be unique")
        if (
            self.selected_plan_id is not None
            and self.selected_plan_id not in self.candidate_plan_ids
        ):
            raise ValueError("selected successor plan must be a candidate")
        if self.status in {
            SuccessorPlanningStatus.APPROVED,
            SuccessorPlanningStatus.REJECTED,
        } and any(
            value is None
            for value in (
                self.planning_proposal_id,
                self.selected_plan_id,
                self.approval_review_id,
                self.approval_verdict_id,
                self.approval_receipt_id,
                self.gate_id,
            )
        ):
            raise ValueError(
                "reviewed successor status requires complete plan and approval bindings"
            )
        identity = self.model_dump(
            mode="json", exclude={"cycle_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"successor-cycle-{content_hash(identity)[:24]}"
        if self.cycle_id != expected_id:
            raise ValueError("SuccessorPlanningCycle ID is not content-bound")
        payload = {"cycle_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("SuccessorPlanningCycle content_hash is invalid")
        return self
