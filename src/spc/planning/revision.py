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
    PlanChangeOperation,
    PlanRevisionActualChange,
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
    RevisionApprovalLLMResponse,
)
from ..plan_paths import scientific_value_hash, scientific_values_equal
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
_KEYED_DRAFT_COLLECTIONS = {
    "observables": "observable_key",
    "comparison_baselines": "baseline_key",
    "tasks": "task_key",
}


def automatic_revision_block_reason(
    plan: ScientificQuestionPlan,
    review: ApprovalReviewRecord,
    verdict: ApprovalVerdict,
    validation_record: PlanValidationRecord | None = None,
) -> str | None:
    if verdict.decision != ApprovalDecision.REQUEST_REVISION:
        return f"approval decision {verdict.decision.value} is not automatically revisable"
    approval_response = (
        review.response.review
        if isinstance(review.response, RevisionApprovalLLMResponse)
        else review.response
    )
    if plan.required_human_decisions or approval_response.unresolved_human_decisions:
        return "revision requires an unresolved human decision"
    codes = {
        flag.code
        for flag in approval_response.hard_red_flags
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
    carried_feedback: tuple[PlanRevisionFeedback, ...] = (),
) -> PlanRevisionInput:
    feedback: list[PlanRevisionFeedback] = list(carried_feedback)
    approval_response = (
        approval_review_record.response.review
        if isinstance(
            approval_review_record.response, RevisionApprovalLLMResponse
        )
        else approval_review_record.response
    )
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
    for flag in approval_response.hard_red_flags:
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
    for fix in approval_response.required_fixes:
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
    feedback = list({item.feedback_id: item for item in feedback}.values())
    if not feedback:
        payload = {
            "source": PlanRevisionFeedbackSource.APPROVAL_RED_FLAG,
            "code": "PLAN_LEVEL_REVISION_REQUEST",
            "description": approval_response.summary,
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


def _path_parts(path: str) -> tuple[tuple[str, int | None], ...]:
    from ..plan_paths import parse_plan_path

    return tuple((item.name, item.index) for item in parse_plan_path(path))


def _revision_path_values(
    revision_input: PlanRevisionInput,
    response: PlanRevisionLLMResponse,
    path: str,
) -> tuple[bool, object | None, bool, object | None, str | None]:
    if path == "plan":
        return True, revision_input.parent_plan, True, response.candidate, None
    parts = _path_parts(path)
    root, root_index = parts[0]
    if root not in _DRAFT_PATHS:
        return False, None, False, None, "unknown root"
    parent_candidate = next(
        candidate
        for candidate in revision_input.parent_proposal.candidates
        if candidate.candidate_key == revision_input.parent_candidate_key
    )
    if root == "atomic_questions":
        parent_value: object = revision_input.parent_proposal.intent.atomic_questions
        revised_value: object = response.intent.atomic_questions
    elif root == "intent_fingerprint":
        parent_value = revision_input.parent_proposal.intent
        revised_value = response.intent
    elif root == "hypothesis":
        parent_value = {
            "primary": parent_candidate.primary_hypothesis,
            "null": parent_candidate.null_hypothesis,
        }
        revised_value = {
            "primary": response.candidate.primary_hypothesis,
            "null": response.candidate.null_hypothesis,
        }
    elif root == "model":
        parent_value = {"description": parent_candidate.model_definition}
        revised_value = {"description": response.candidate.model_definition}
    elif root == "system_fingerprint":
        parent_value = parent_candidate.model_definition
        revised_value = response.candidate.model_definition
    elif root in {"method_fingerprint", "fingerprint_differences"}:
        parent_value = {
            "strategy_class": parent_candidate.strategy_class,
            "distinguishing_axis": parent_candidate.distinguishing_axis,
            "distinguishing_value": parent_candidate.distinguishing_value,
            "proposed_deviations": parent_candidate.proposed_deviations,
        }
        revised_value = {
            "strategy_class": response.candidate.strategy_class,
            "distinguishing_axis": response.candidate.distinguishing_axis,
            "distinguishing_value": response.candidate.distinguishing_value,
            "proposed_deviations": response.candidate.proposed_deviations,
        }
    else:
        draft_field = _DRAFT_PATHS[root].removeprefix("candidate.")
        parent_value = getattr(parent_candidate, draft_field)
        revised_value = getattr(response.candidate, draft_field)

    remaining = list(parts[1:])
    if root_index is not None:
        remaining.insert(0, ("", root_index))
    parent_exists = True
    revised_exists = True
    for position, (name, index) in enumerate(remaining):
        if name:
            for side, value in (("parent", parent_value), ("revised", revised_value)):
                exists = parent_exists if side == "parent" else revised_exists
                if not exists:
                    continue
                if isinstance(value, Mapping) and name in value:
                    child = value[name]
                elif hasattr(type(value), "model_fields") and name in type(value).model_fields:
                    child = getattr(value, name)
                elif name == "text" and isinstance(value, str):
                    child = value
                else:
                    if side == "parent":
                        parent_exists = False
                    else:
                        revised_exists = False
                    continue
                if side == "parent":
                    parent_value = child
                else:
                    revised_value = child
        if index is not None:
            parent_sequence = parent_value if isinstance(parent_value, (list, tuple)) else ()
            revised_sequence = revised_value if isinstance(revised_value, (list, tuple)) else ()
            parent_exists = parent_exists and index < len(parent_sequence)
            revised_exists = revised_exists and index < len(revised_sequence)
            old_item = parent_sequence[index] if parent_exists else None
            new_item = revised_sequence[index] if revised_exists else None
            if position == 0 and root in _KEYED_DRAFT_COLLECTIONS:
                key_name = _KEYED_DRAFT_COLLECTIONS[root]
                old_key = getattr(old_item, key_name, None)
                new_key = getattr(new_item, key_name, None)
                if parent_exists and revised_exists and old_key != new_key:
                    return (
                        parent_exists,
                        old_item,
                        revised_exists,
                        new_item,
                        "indexed objects do not share the same stable local key",
                    )
                if parent_exists and not revised_exists:
                    revised_keys = {getattr(item, key_name) for item in revised_sequence}
                    if old_key in revised_keys:
                        return True, old_item, False, None, "indexed deletion is ambiguous"
                if revised_exists and not parent_exists:
                    parent_keys = {getattr(item, key_name) for item in parent_sequence}
                    if new_key in parent_keys:
                        return False, None, True, new_item, "indexed addition is ambiguous"
            parent_value = old_item
            revised_value = new_item
    return parent_exists, parent_value, revised_exists, revised_value, None


def _declared_change_operation(
    revision_input: PlanRevisionInput,
    response: PlanRevisionLLMResponse,
    path: str,
) -> tuple[PlanChangeOperation | None, str | None]:
    try:
        parent_exists, old, revised_exists, new, ambiguity = _revision_path_values(
            revision_input, response, path
        )
    except ValueError as error:
        return None, str(error)
    if ambiguity is not None:
        return None, ambiguity
    if not parent_exists and not revised_exists:
        return None, "path exists in neither the parent nor revised plan"
    if parent_exists and not revised_exists:
        return PlanChangeOperation.REMOVED, None
    if revised_exists and not parent_exists:
        return PlanChangeOperation.ADDED, None
    if scientific_values_equal(old, new):
        return None, "path content did not change"
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        if len(old) == len(new) and sorted(
            scientific_value_hash(item) for item in old
        ) == sorted(scientific_value_hash(item) for item in new):
            return PlanChangeOperation.REORDERED, None
    return PlanChangeOperation.MODIFIED, None


def compute_revision_actual_changes(
    revision_input: PlanRevisionInput,
    response: PlanRevisionLLMResponse,
) -> tuple[PlanRevisionActualChange, ...]:
    changes: list[PlanRevisionActualChange] = []
    for root in sorted(_changed_roots(revision_input, response)):
        operation, error = _declared_change_operation(revision_input, response, root)
        if operation is None or error is not None:
            continue
        parent_exists, old, revised_exists, new, _ = _revision_path_values(
            revision_input, response, root
        )
        changes.append(
            PlanRevisionActualChange(
                path=root,
                operation=operation,
                parent_value_hash=(
                    scientific_value_hash(old) if parent_exists else None
                ),
                revised_value_hash=(
                    scientific_value_hash(new) if revised_exists else None
                ),
            )
        )
    return tuple(changes)


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
    path_claimants: dict[str, str] = {}
    feedback_by_id = {item.feedback_id: item for item in revision_input.feedback}
    for index, item in enumerate(response.feedback_responses):
        for path in item.changed_plan_paths:
            previous_claimant = path_claimants.get(path)
            if previous_claimant is not None and previous_claimant != item.feedback_id:
                issues.append(
                    ValidationIssue(
                        code="REVISION_CHANGE_PATH_REUSED",
                        message=(
                            f"changed plan path {path} is already claimed by feedback "
                            f"{previous_claimant}; each feedback response must identify "
                            "its own actual change"
                        ),
                        path=f"feedback_responses[{index}].changed_plan_paths",
                    )
                )
            else:
                path_claimants[path] = item.feedback_id
            try:
                root = _path_parts(path)[0][0]
            except ValueError:
                root = ""
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
            operation, change_error = _declared_change_operation(
                revision_input, response, path
            )
            if operation is None:
                issue_code = (
                    "UNKNOWN_REVISION_PLAN_PATH"
                    if change_error
                    and "neither the parent nor revised plan" in change_error
                    else "AMBIGUOUS_REVISION_OBJECT_MAPPING"
                    if change_error and "ambiguous" in change_error
                    else "REVISION_RESPONSE_NOT_REFLECTED"
                )
                issues.append(
                    ValidationIssue(
                        code=issue_code,
                        message=(
                            f"declared changed path {path} is not an exact actual change"
                            + (f": {change_error}" if change_error else "")
                        ),
                        path=f"feedback_responses[{index}].changed_plan_paths",
                    )
                )
        feedback = feedback_by_id.get(item.feedback_id)
        if (
            feedback is not None
            and item.disposition == PlanRevisionDisposition.ADDRESSED
            and not item.changed_plan_paths
        ):
            issues.append(
                ValidationIssue(
                    code="FEEDBACK_NOT_ACTUALLY_ADDRESSED",
                    message=(
                        f"feedback {item.feedback_id} is marked addressed without an "
                        "independently verifiable actual change"
                    ),
                    path=f"feedback_responses[{index}]",
                )
            )
        if (
            feedback is not None
            and item.disposition == PlanRevisionDisposition.ADDRESSED
            and feedback.plan_path not in {None, "plan"}
            and "[" in feedback.plan_path
            and all("[" not in path for path in item.changed_plan_paths)
        ):
            issues.append(
                ValidationIssue(
                    code="IMPRECISE_REVISION_RESPONSE",
                    message="specific-object feedback cannot be closed by a root-only change",
                    path=f"feedback_responses[{index}].changed_plan_paths",
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
