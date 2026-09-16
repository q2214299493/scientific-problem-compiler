from __future__ import annotations

from collections.abc import Mapping

from ..models import (
    AmbiguityAssessment,
    ApprovalDecision,
    ApprovalRedFlagSeverity,
    ApprovalReviewInput,
    ApprovalReviewRecord,
    ApprovalVerdict,
    CandidatePlanDraft,
    IndependentApprovalReceipt,
    IntentInterpretation,
    PlanCompilationReceipt,
    PlanRevisionConstraints,
    PlanRevisionFeedback,
    PlanRevisionFeedbackSource,
    PlanRevisionInput,
    PlanRevisionLLMResponse,
    PlanRevisionDisposition,
    PlanValidationRecord,
    PlanningProposalSet,
    ScientificPlanningInput,
    ScientificQuestionPlan,
)
from ..serialization import content_hash
from ..validators import ValidationIssue, ValidationReport
from .mock_provider import build_proposal_set


NON_REVISABLE_FEEDBACK_CODES = frozenset(
    {
        "MISSING_CRITICAL_EVIDENCE",
        "MISSING_EVIDENCE_REF",
        "EVIDENCE_NOT_FOUND",
        "UNKNOWN_EVIDENCE_REF",
        "SOURCE_INTEGRITY_FAILURE",
        "FABRICATED_EVIDENCE_ID",
        "FABRICATED_CLAIM_ID",
        "FABRICATED_CAPABILITY_ID",
        "FABRICATED_HUMAN_DECISION_ID",
    }
)

_DRAFT_PATHS = {
    "atomic_questions": "intent.atomic_questions",
    "hypothesis": "candidate.hypotheses",
    "model": "candidate.model_definition",
    "observables": "candidate.observables",
    "comparison_baselines": "candidate.comparison_baselines",
    "acceptance_criteria": "candidate.acceptance_criteria",
    "falsification_criteria": "candidate.falsification_criteria",
    "evidence_refs": "candidate.evidence_refs",
    "assumptions": "candidate.assumptions",
    "unknowns": "candidate.unknowns",
    "proposed_deviations": "candidate.proposed_deviations",
    "scientific_capability_ids": "candidate.capability_ids",
    "tasks": "candidate.task_drafts",
    "limitations": "candidate.limitations",
    "required_human_decisions": "candidate.human_decisions_required",
    "source_query_manifest": "candidate.claim_refs",
    "intent_fingerprint": "intent",
    "system_fingerprint": "candidate.system_inputs",
    "method_fingerprint": "candidate.method_inputs",
    "fingerprint_differences": "candidate.proposed_deviations",
    "plan": "plan",
}


def automatic_revision_block_reason(
    plan: ScientificQuestionPlan,
    review: ApprovalReviewRecord,
    verdict: ApprovalVerdict,
    validation_record: PlanValidationRecord | None = None,
) -> str | None:
    if verdict.decision != ApprovalDecision.REQUEST_REVISION:
        return f"approval decision {verdict.decision.value} is not automatically revisable"
    if plan.required_human_decisions or review.response.unresolved_human_decisions:
        return "revision requires an unresolved human decision"
    codes = {
        flag.code
        for flag in review.response.hard_red_flags
        if flag.severity == ApprovalRedFlagSeverity.BLOCKING
    }
    if validation_record is not None:
        codes.update(validation_record.issue_codes)
    if codes & NON_REVISABLE_FEEDBACK_CODES:
        return "revision requires new or repaired evidence outside the fixed trusted context"
    return None


def _feedback_id(payload: Mapping[str, object]) -> str:
    return f"plan-revision-feedback-{content_hash(payload)[:24]}"


def build_plan_revision_input(
    *,
    revision_index: int,
    planning_input: ScientificPlanningInput,
    parent_proposal: PlanningProposalSet,
    parent_candidate_key: str,
    parent_plan: ScientificQuestionPlan,
    parent_compilation_receipt: PlanCompilationReceipt,
    plan_validation_record: PlanValidationRecord,
    validation_report: ValidationReport,
    approval_review_input: ApprovalReviewInput,
    approval_review_record: ApprovalReviewRecord,
    approval_verdict: ApprovalVerdict,
    approval_receipt: IndependentApprovalReceipt,
) -> PlanRevisionInput:
    feedback: list[PlanRevisionFeedback] = []
    for issue in validation_report.issues:
        payload = {
            "source": PlanRevisionFeedbackSource.DETERMINISTIC_VALIDATION,
            "code": issue.code,
            "description": issue.message,
            "plan_path": issue.path,
            "evidence_refs": (),
            "claim_refs": (),
            "task_refs": (),
            "capability_refs": (),
            "blocking": True,
        }
        feedback.append(
            PlanRevisionFeedback(feedback_id=_feedback_id(payload), **payload)
        )
    for flag in approval_review_record.response.hard_red_flags:
        payload = {
            "source": PlanRevisionFeedbackSource.APPROVAL_RED_FLAG,
            "code": flag.code,
            "description": flag.description,
            "plan_path": flag.plan_path,
            "evidence_refs": flag.evidence_refs,
            "claim_refs": flag.claim_refs,
            "task_refs": flag.task_refs,
            "capability_refs": flag.capability_refs,
            "blocking": flag.severity == ApprovalRedFlagSeverity.BLOCKING,
        }
        feedback.append(
            PlanRevisionFeedback(feedback_id=_feedback_id(payload), **payload)
        )
    for fix in approval_review_record.response.required_fixes:
        payload = {
            "source": PlanRevisionFeedbackSource.REQUIRED_FIX,
            "code": fix.fix_id,
            "description": fix.description,
            "plan_path": None,
            "evidence_refs": (),
            "claim_refs": (),
            "task_refs": (),
            "capability_refs": (),
            "blocking": fix.blocking,
        }
        feedback.append(
            PlanRevisionFeedback(feedback_id=_feedback_id(payload), **payload)
        )
    if not feedback:
        payload = {
            "source": PlanRevisionFeedbackSource.APPROVAL_RED_FLAG,
            "code": "PLAN_LEVEL_REVISION_REQUEST",
            "description": approval_review_record.response.summary,
            "plan_path": "plan",
            "evidence_refs": (),
            "claim_refs": (),
            "task_refs": (),
            "capability_refs": (),
            "blocking": True,
        }
        feedback.append(
            PlanRevisionFeedback(feedback_id=_feedback_id(payload), **payload)
        )

    constraints = PlanRevisionConstraints(
        original_request=planning_input.original_request,
        domain=planning_input.domain,
        domain_pack_version=planning_input.domain_pack_version,
        context_id=planning_input.context_id,
        context_hash=planning_input.context_hash,
        evidence_packet_id=planning_input.evidence_packet_id,
        evidence_packet_hash=planning_input.evidence_packet_hash,
        planning_input_id=planning_input.planning_input_id,
        planning_input_hash=planning_input.content_hash,
        allowed_evidence_ids=planning_input.allowed_evidence_ids,
        allowed_claim_ids=planning_input.allowed_claim_ids,
        allowed_capability_ids=planning_input.allowed_capability_ids,
        required_human_decision_ids=tuple(
            item.decision_id for item in planning_input.required_human_decisions
        ),
        unresolved_conflict_ids=tuple(
            item.conflict_id
            for item in planning_input.conflict_sets
            if item.resolution_status == "unresolved"
        ),
    )
    identity = {
        "revision_index": revision_index,
        "planning_input": planning_input,
        "parent_proposal": parent_proposal,
        "parent_candidate_key": parent_candidate_key,
        "parent_plan": parent_plan,
        "parent_plan_hash": content_hash(parent_plan),
        "parent_compilation_receipt": parent_compilation_receipt,
        "plan_validation_record": plan_validation_record,
        "plan_validation_hash": content_hash(plan_validation_record),
        "approval_review_input": approval_review_input,
        "approval_review_record": approval_review_record,
        "approval_verdict": approval_verdict,
        "approval_receipt": approval_receipt,
        "feedback": tuple(feedback),
        "constraints": constraints,
    }
    serialized_identity = PlanRevisionInput.model_construct(
        revision_input_id="pending",
        **identity,
        content_hash="0" * 64,
    ).model_dump(mode="json", exclude={"revision_input_id", "content_hash"})
    revision_input_id = (
        f"plan-revision-input-{content_hash(serialized_identity)[:24]}"
    )
    payload = {"revision_input_id": revision_input_id, **serialized_identity}
    return PlanRevisionInput(**payload, content_hash=content_hash(payload))


def _semantic_response_payload(
    intent: IntentInterpretation,
    candidate: CandidatePlanDraft,
) -> dict[str, object]:
    candidate_payload = candidate.model_dump(mode="json", exclude={"candidate_key"})
    return {"intent": intent.model_dump(mode="json"), "candidate": candidate_payload}


def _changed_roots(
    revision_input: PlanRevisionInput,
    response: PlanRevisionLLMResponse,
) -> set[str]:
    parent = next(
        candidate
        for candidate in revision_input.parent_proposal.candidates
        if candidate.candidate_key == revision_input.parent_candidate_key
    )
    changed: set[str] = set()
    if response.intent != revision_input.parent_proposal.intent:
        if response.intent.atomic_questions != revision_input.parent_proposal.intent.atomic_questions:
            changed.add("atomic_questions")
        changed.add("intent_fingerprint")
    field_roots = {
        "primary_hypothesis": "hypothesis",
        "null_hypothesis": "hypothesis",
        "model_definition": "model",
        "observables": "observables",
        "comparison_baselines": "comparison_baselines",
        "acceptance_criteria": "acceptance_criteria",
        "falsification_criteria": "falsification_criteria",
        "assumptions": "assumptions",
        "unknowns": "unknowns",
        "proposed_deviations": "proposed_deviations",
        "evidence_refs": "evidence_refs",
        "claim_refs": "source_query_manifest",
        "capability_ids": "scientific_capability_ids",
        "task_drafts": "tasks",
        "limitations": "limitations",
        "human_decisions_required": "required_human_decisions",
        "strategy_class": "method_fingerprint",
        "distinguishing_axis": "method_fingerprint",
        "distinguishing_value": "method_fingerprint",
    }
    for field_name, root in field_roots.items():
        if getattr(parent, field_name) != getattr(response.candidate, field_name):
            changed.add(root)
    return changed


def validate_plan_revision_response(
    response: PlanRevisionLLMResponse,
    revision_input: PlanRevisionInput,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    expected_feedback = {item.feedback_id for item in revision_input.feedback}
    response_ids = tuple(item.feedback_id for item in response.feedback_responses)
    if len(set(response_ids)) != len(response_ids):
        issues.append(
            ValidationIssue(
                code="DUPLICATE_REVISION_FEEDBACK_RESPONSE",
                message="revision response contains duplicate feedback IDs",
                path="feedback_responses",
            )
        )
    if set(response_ids) != expected_feedback:
        issues.append(
            ValidationIssue(
                code="REVISION_FEEDBACK_BINDING_MISMATCH",
                message="revision response must address every bound feedback item exactly once",
                path="feedback_responses",
            )
        )
    parent = next(
        candidate
        for candidate in revision_input.parent_proposal.candidates
        if candidate.candidate_key == revision_input.parent_candidate_key
    )
    substantive = _semantic_response_payload(
        response.intent, response.candidate
    ) != _semantic_response_payload(revision_input.parent_proposal.intent, parent)
    if not substantive:
        issues.append(
            ValidationIssue(
                code="NO_SUBSTANTIVE_PLAN_CHANGE",
                message="revision changes only identity or explanatory metadata",
            )
        )
    changed_roots = _changed_roots(revision_input, response)
    declared_roots: set[str] = set()
    feedback_by_id = {item.feedback_id: item for item in revision_input.feedback}
    for index, item in enumerate(response.feedback_responses):
        for path in item.changed_plan_paths:
            root = path.split(".", 1)[0].split("[", 1)[0]
            if root not in _DRAFT_PATHS:
                issues.append(
                    ValidationIssue(
                        code="UNKNOWN_REVISION_PLAN_PATH",
                        message=f"revision response names unknown plan path {path}",
                        path=f"feedback_responses[{index}].changed_plan_paths",
                    )
                )
                continue
            declared_roots.add(root)
            if root != "plan" and root not in changed_roots:
                issues.append(
                    ValidationIssue(
                        code="REVISION_RESPONSE_NOT_REFLECTED",
                        message=f"declared changed path {path} did not change scientifically",
                        path=f"feedback_responses[{index}].changed_plan_paths",
                    )
                )
        feedback = feedback_by_id.get(item.feedback_id)
        if (
            feedback is not None
            and item.disposition == PlanRevisionDisposition.ADDRESSED
            and feedback.plan_path not in {None, "plan"}
        ):
            expected_root = feedback.plan_path.split(".", 1)[0].split("[", 1)[0]
            if expected_root not in declared_roots or expected_root not in changed_roots:
                issues.append(
                    ValidationIssue(
                        code="FEEDBACK_NOT_ACTUALLY_ADDRESSED",
                        message=(
                            f"feedback {item.feedback_id} is marked addressed without changing "
                            f"{feedback.plan_path}"
                        ),
                        path=f"feedback_responses[{index}]",
                    )
                )
    sensitive_changes = changed_roots & {
        "comparison_baselines",
        "acceptance_criteria",
        "falsification_criteria",
        "system_fingerprint",
        "method_fingerprint",
        "proposed_deviations",
        "tasks",
    }
    undeclared = sensitive_changes - declared_roots
    if undeclared:
        issues.append(
            ValidationIssue(
                code="UNDISCLOSED_SCIENTIFIC_REVISION",
                message="scientifically sensitive changes were not disclosed: "
                + ", ".join(sorted(undeclared)),
                path="feedback_responses",
            )
        )
    return ValidationReport(valid=not issues, issues=tuple(issues))


def build_revision_proposal_set(
    revision_input: PlanRevisionInput,
    response: PlanRevisionLLMResponse,
    *,
    provider_id: str,
    provider_version: str,
    provider_config: dict[str, object],
) -> PlanningProposalSet:
    config = {
        **provider_config,
        "revision_input_id": revision_input.revision_input_id,
        "revision_input_hash": revision_input.content_hash,
        "revision_index": revision_input.revision_index,
        "revision_parent_plan_id": revision_input.parent_plan.plan_id,
        "revision_parent_plan_hash": revision_input.parent_plan_hash,
    }
    return build_proposal_set(
        revision_input.planning_input,
        provider_id=provider_id,
        provider_version=provider_version,
        provider_config=config,
        intent=response.intent,
        ambiguity_assessment=AmbiguityAssessment(
            multiple_candidates_required=False,
            rationale="Bounded revision of the explicitly selected candidate.",
            scientifically_distinct_axes=(),
        ),
        candidates=(response.candidate,),
    )
