"""Independent approval chain, policy, and gate authority."""

from __future__ import annotations

from collections import Counter

from ..models import (
    ApprovalDecision,
    ApprovalMode,
    ApprovalReviewInput,
    ApprovalReviewRecord,
    ApprovalScores,
    ApprovalVerdict,
    GateVerdict,
    IndependentApprovalReceipt,
    PlanCompilationReceipt,
    PlanValidationRecord,
    ProjectTrustPolicy,
    RevisionApprovalLLMResponse,
    RevisionApprovalReviewInput,
    RevisionIssueStatus,
    ScientificQuestionPlan,
)
from ..serialization import content_hash
from .common import ValidationIssue, ValidationReport, content_binding_error, validation_report
from .plans import (
    validate_plan_compilation_receipt,
    validate_plan_validation_record,
    validation_record_matches_plan,
)


def validate_approval_boundary(
    plan: ScientificQuestionPlan, verdict: ApprovalVerdict
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    actual_hash = content_hash(plan)
    if verdict.candidate_id != plan.plan_id:
        issues.append(ValidationIssue(code="APPROVAL_ID_MISMATCH", message="verdict candidate ID does not match plan"))
    if verdict.candidate_version != plan.version:
        issues.append(ValidationIssue(code="APPROVAL_VERSION_MISMATCH", message="verdict version does not match plan"))
    if verdict.candidate_content_hash != actual_hash:
        issues.append(ValidationIssue(code="STALE_APPROVAL", message="verdict content hash does not match current plan"))
    return validation_report(issues)


def requires_independent_approval(plan: ScientificQuestionPlan) -> bool:
    """Return lineage evidence only; never use this helper as a security policy."""

    return plan.source_proposal is not None


def validate_approval_claim_evidence_coherence(
    plan: ScientificQuestionPlan,
    review_input: ApprovalReviewInput | RevisionApprovalReviewInput,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    claim_ids = {
        entry.removeprefix("claim:")
        for entry in plan.source_query_manifest
        if entry.startswith("claim:")
    }
    claims_by_id = {claim.claim_id: claim for claim in review_input.source_claims}
    missing_claims = claim_ids - set(claims_by_id)
    if missing_claims:
        issues.append(
            ValidationIssue(
                code="APPROVAL_CLAIM_BINDING_MISMATCH",
                message="candidate provenance names SourceClaim records absent from the approval input",
                path="source_query_manifest",
            )
        )
    required_evidence = {
        evidence_id
        for claim_id in claim_ids & set(claims_by_id)
        for evidence_id in claims_by_id[claim_id].evidence_refs
    }
    candidate_evidence = {reference.evidence_id for reference in plan.evidence_refs}
    if not required_evidence.issubset(candidate_evidence):
        issues.append(
            ValidationIssue(
                code="APPROVAL_CLAIM_EVIDENCE_MISMATCH",
                message="candidate evidence does not cover every SourceClaim named by candidate provenance",
                path="evidence_refs",
            )
        )
    return validation_report(issues)


def validate_independent_approval_chain(
    plan: ScientificQuestionPlan,
    verdict: ApprovalVerdict,
    review_input: ApprovalReviewInput | RevisionApprovalReviewInput,
    review: ApprovalReviewRecord,
    receipt: IndependentApprovalReceipt,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    plan_hash = content_hash(plan)
    requires_revision_review = any(
        entry.startswith("revision-input:")
        for entry in plan.source_query_manifest
    )
    if requires_revision_review and not isinstance(
        review_input, RevisionApprovalReviewInput
    ):
        issues.append(
            ValidationIssue(
                code="MISSING_REVISION_APPROVAL_CONTEXT",
                message="revised plan approval requires the versioned revision review contract",
            )
        )
    if isinstance(review_input, RevisionApprovalReviewInput):
        if review_input.revision_context.revised_plan_hash != plan_hash:
            issues.append(
                ValidationIssue(
                    code="REVISION_APPROVAL_PLAN_MISMATCH",
                    message="revision approval context does not bind the supplied plan",
                )
            )
        if not isinstance(review.response, RevisionApprovalLLMResponse):
            issues.append(
                ValidationIssue(
                    code="MISSING_REVISION_ISSUE_ASSESSMENTS",
                    message="revision review record lacks item-by-item issue assessments",
                )
            )
        else:
            expected_issue_ids = {
                item.feedback_id
                for item in review_input.revision_context.tracked_feedback
            }
            assessed_issue_ids = tuple(
                item.feedback_id for item in review.response.issue_assessments
            )
            if (
                len(set(assessed_issue_ids)) != len(assessed_issue_ids)
                or set(assessed_issue_ids) != expected_issue_ids
            ):
                issues.append(
                    ValidationIssue(
                        code="REVISION_ISSUE_ASSESSMENT_MISMATCH",
                        message=(
                            "revision review record must assess every tracked issue "
                            "exactly once"
                        ),
                    )
                )
            blocking_by_id = {
                item.feedback_id: item.blocking
                for item in review_input.revision_context.tracked_feedback
            }
            unresolved = tuple(
                item
                for item in review.response.issue_assessments
                if blocking_by_id.get(item.feedback_id, True)
                and item.status
                in {
                    RevisionIssueStatus.UNRESOLVED,
                    RevisionIssueStatus.NEEDS_HUMAN_DECISION,
                }
            )
            if verdict.decision in {
                ApprovalDecision.APPROVE,
                ApprovalDecision.APPROVE_WITH_CONDITIONS,
            } and unresolved:
                issues.append(
                    ValidationIssue(
                        code="UNRESOLVED_REVISION_ISSUE_APPROVED",
                        message="approval verdict cannot pass unresolved tracked issues",
                    )
                )

    review_input_prefix = (
        "revision-approval-input"
        if isinstance(review_input, RevisionApprovalReviewInput)
        else "approval-review-input"
    )
    if content_binding_error(review_input, id_field="review_input_id", prefix=review_input_prefix) is not None:
        issues.append(
            ValidationIssue(
                code="STALE_APPROVAL_REVIEW_INPUT",
                message="ApprovalReviewInput identity or content hash is stale",
            )
        )

    if content_binding_error(review, id_field="review_id", prefix="approval-review") is not None:
        issues.append(
            ValidationIssue(
                code="STALE_APPROVAL_REVIEW_RECORD",
                message="ApprovalReviewRecord identity or content hash is stale",
            )
        )

    if content_binding_error(receipt, id_field="receipt_id", prefix="independent-approval-receipt") is not None:
        issues.append(
            ValidationIssue(
                code="STALE_INDEPENDENT_APPROVAL_RECEIPT",
                message="IndependentApprovalReceipt identity or content hash is stale",
            )
        )

    if (
        review_input.candidate_plan.plan_id != plan.plan_id
        or review_input.candidate_plan.version != plan.version
        or review_input.candidate_plan_hash != plan_hash
        or content_hash(review_input.candidate_plan) != plan_hash
    ):
        issues.append(
            ValidationIssue(
                code="REVIEW_CANDIDATE_BINDING_MISMATCH",
                message="ApprovalReviewInput is not bound to the supplied candidate plan",
            )
        )
    if (review.review_input_id, review.review_input_hash) != (
        review_input.review_input_id,
        review_input.content_hash,
    ):
        issues.append(
            ValidationIssue(
                code="INDEPENDENT_REVIEW_INPUT_HASH_MISMATCH",
                message="ApprovalReviewRecord is not bound to the supplied ApprovalReviewInput",
            )
        )
    if (receipt.review_id, receipt.review_hash) != (
        review.review_id,
        review.content_hash,
    ):
        issues.append(
            ValidationIssue(
                code="INDEPENDENT_REVIEW_HASH_MISMATCH",
                message="receipt is not bound to the supplied ApprovalReviewRecord",
            )
        )
    if (receipt.review_input_id, receipt.review_input_hash) != (
        review_input.review_input_id,
        review_input.content_hash,
    ):
        issues.append(
            ValidationIssue(
                code="INDEPENDENT_REVIEW_INPUT_HASH_MISMATCH",
                message="receipt is not bound to the supplied ApprovalReviewInput",
            )
        )
    if (receipt.verdict_id, receipt.verdict_hash) != (
        verdict.verdict_id,
        content_hash(verdict),
    ):
        issues.append(
            ValidationIssue(
                code="INDEPENDENT_VERDICT_HASH_MISMATCH",
                message="receipt is not bound to the supplied ApprovalVerdict",
            )
        )
    if (receipt.candidate_id, receipt.candidate_version, receipt.candidate_hash) != (
        plan.plan_id,
        plan.version,
        plan_hash,
    ):
        issues.append(
            ValidationIssue(
                code="INDEPENDENT_RECEIPT_CANDIDATE_MISMATCH",
                message="receipt is not bound to the supplied candidate plan",
            )
        )
    if (receipt.provider_id, receipt.provider_version) != (
        review.provider_id,
        review.provider_version,
    ):
        issues.append(
            ValidationIssue(
                code="INDEPENDENT_PROVIDER_BINDING_MISMATCH",
                message="receipt provider binding does not match the review record",
            )
        )
    if receipt.approver_id != verdict.approver_id:
        issues.append(
            ValidationIssue(
                code="INDEPENDENT_APPROVER_BINDING_MISMATCH",
                message="receipt approver does not match the authoritative verdict",
            )
        )
    expected_verdict_id = (
        "approval-verdict-"
        + content_hash(
            {
                "review_hash": review.content_hash,
                "candidate_hash": plan_hash,
                "approver_id": verdict.approver_id,
            }
        )[:24]
    )
    base_review_response = (
        review.response.review
        if isinstance(review.response, RevisionApprovalLLMResponse)
        else review.response
    )
    detailed_scores = base_review_response.scores
    expected_scores = ApprovalScores(
        **{
            name: getattr(detailed_scores, name).score
            for name in ApprovalScores.model_fields
        }
    )
    expected_human_decisions = tuple(
        decision.decision_id for decision in plan.required_human_decisions
    )
    if (
        verdict.verdict_id != expected_verdict_id
        or verdict.decision != review.policy_decision
        or verdict.scores != expected_scores
        or verdict.hard_red_flags
        != tuple(flag.code for flag in base_review_response.hard_red_flags)
        or verdict.required_fixes != base_review_response.required_fixes
        or verdict.human_decisions_required != expected_human_decisions
    ):
        issues.append(
            ValidationIssue(
                code="INDEPENDENT_VERDICT_REVIEW_MISMATCH",
                message="ApprovalVerdict is not the deterministic materialization of the independent review",
            )
        )
    issues.extend(validate_approval_boundary(plan, verdict).issues)
    issues.extend(validate_approval_claim_evidence_coherence(plan, review_input).issues)
    return validation_report(issues)


def validate_conditional_approval(verdict: ApprovalVerdict) -> ValidationReport:
    issues: list[ValidationIssue] = []
    fix_counts = Counter(item.fix_id for item in verdict.required_fixes)
    for fix_id, count in fix_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_REQUIRED_FIX",
                    message=f"required fix is declared more than once: {fix_id}",
                )
            )
    resolution_counts = Counter(item.fix_id for item in verdict.fix_resolutions)
    for fix_id, count in resolution_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_FIX_RESOLUTION",
                    message=f"blocking fix has multiple resolutions: {fix_id}",
                )
            )
    known_fixes = {item.fix_id for item in verdict.required_fixes}
    for resolution in verdict.fix_resolutions:
        if resolution.fix_id not in known_fixes:
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_FIX_RESOLUTION",
                    message=f"resolution references unknown required fix: {resolution.fix_id}",
                )
            )
    if verdict.decision in {
        ApprovalDecision.APPROVE,
        ApprovalDecision.APPROVE_WITH_CONDITIONS,
    }:
        resolutions = {item.fix_id: item for item in verdict.fix_resolutions}
        for fix in verdict.required_fixes:
            resolution = resolutions.get(fix.fix_id)
            if fix.blocking and (
                resolution is None or not resolution.resolved or not resolution.resolution.strip()
            ):
                issues.append(
                    ValidationIssue(
                        code="UNRESOLVED_BLOCKING_FIX",
                        message=f"exportable approval has unresolved blocking fix: {fix.fix_id}",
                    )
                )
    return validation_report(issues)


def validate_approval_state(
    plan: ScientificQuestionPlan,
    verdict: ApprovalVerdict,
) -> ValidationReport:
    issues = list(validate_conditional_approval(verdict).issues)
    exportable = verdict.decision in {
        ApprovalDecision.APPROVE,
        ApprovalDecision.APPROVE_WITH_CONDITIONS,
    }
    if exportable and verdict.hard_red_flags:
        issues.append(
            ValidationIssue(
                code="HARD_RED_FLAGS_BLOCK_EXPORT",
                message="an approval with hard red flags cannot be exported",
            )
        )

    required_decision_counts = Counter(verdict.human_decisions_required)
    for decision_id, count in required_decision_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_REQUIRED_HUMAN_DECISION",
                    message=f"human decision is required more than once: {decision_id}",
                )
            )
    plan_decisions = {item.decision_id: item for item in plan.required_human_decisions}
    verdict_decisions = set(verdict.human_decisions_required)
    plan_decision_ids = set(plan_decisions)
    for decision_id in sorted(verdict_decisions - plan_decision_ids):
        issues.append(
            ValidationIssue(
                code="UNKNOWN_REQUIRED_HUMAN_DECISION",
                message=f"approval requires a human decision not declared by the plan: {decision_id}",
            )
        )
    for decision_id in sorted(plan_decision_ids - verdict_decisions):
        issues.append(
            ValidationIssue(
                code="HUMAN_DECISION_STATE_MISMATCH",
                message=f"approval omits a human decision required by the plan: {decision_id}",
            )
        )

    resolution_counts = Counter(
        item.decision_id for item in verdict.human_decision_resolutions
    )
    for decision_id, count in resolution_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_HUMAN_DECISION_RESOLUTION",
                    message=f"human decision has multiple resolutions: {decision_id}",
                )
            )
    resolutions = {
        item.decision_id: item for item in verdict.human_decision_resolutions
    }
    for decision_id in sorted(set(resolutions) - verdict_decisions):
        issues.append(
            ValidationIssue(
                code="UNKNOWN_HUMAN_DECISION_RESOLUTION",
                message=f"resolution references a human decision not required by the approval: {decision_id}",
            )
        )
    if exportable:
        for decision_id in sorted(plan_decision_ids | verdict_decisions):
            resolution = resolutions.get(decision_id)
            if (
                resolution is None
                or not resolution.resolved
                or not resolution.selected_option.strip()
                or not resolution.rationale.strip()
            ):
                issues.append(
                    ValidationIssue(
                        code="UNRESOLVED_HUMAN_DECISION",
                        message=f"exportable approval has unresolved human decision: {decision_id}",
                    )
                )
                continue
            planned = plan_decisions.get(decision_id)
            if planned is not None and resolution.selected_option not in planned.options:
                issues.append(
                    ValidationIssue(
                        code="INVALID_HUMAN_DECISION_OPTION",
                        message=f"selected option is not declared for human decision: {decision_id}",
                    )
                )
    if (
        verdict.decision == ApprovalDecision.APPROVE_WITH_CONDITIONS
        and not verdict.required_fixes
        and not verdict.human_decisions_required
    ):
        issues.append(
            ValidationIssue(
                code="CONDITIONAL_APPROVAL_WITHOUT_CONDITIONS",
                message="approve_with_conditions requires at least one fix or human decision",
            )
        )
    return validation_report(issues)


def gate_binds_plan_version(gate: GateVerdict, plan_binding: tuple[str, str, str]) -> bool:
    return (gate.candidate_id, gate.candidate_version, gate.candidate_content_hash) == plan_binding


def gate_binds_verdict(gate: GateVerdict, verdict: ApprovalVerdict) -> bool:
    return (gate.approval_verdict_id, gate.approval_verdict_hash) == (verdict.verdict_id, content_hash(verdict))


def gate_binds_validation(gate: GateVerdict, record: PlanValidationRecord) -> bool:
    return (gate.plan_validation_id, gate.plan_validation_hash) == (record.validation_id, content_hash(record))


def gate_binds_policy(gate: GateVerdict, policy: ProjectTrustPolicy) -> bool:
    return (gate.trust_policy_version, gate.trust_policy_hash) == (policy.policy_version, content_hash(policy))


def gate_binds_compilation_receipt(gate: GateVerdict, receipt: PlanCompilationReceipt) -> bool:
    return (gate.plan_compilation_receipt_id, gate.plan_compilation_receipt_hash) == (
        receipt.receipt_id, receipt.content_hash
    )


def gate_binds_approval_receipt(gate: GateVerdict, receipt: IndependentApprovalReceipt) -> bool:
    return (gate.independent_approval_receipt_id, gate.independent_approval_receipt_hash) == (
        receipt.receipt_id, receipt.content_hash
    )


def validate_approved_plan_authority(
    plan: ScientificQuestionPlan,
    validation_record: PlanValidationRecord,
    review_input: ApprovalReviewInput | RevisionApprovalReviewInput,
    review: ApprovalReviewRecord,
    verdict: ApprovalVerdict,
    receipt: IndependentApprovalReceipt,
    gate: GateVerdict,
    trust_policy: ProjectTrustPolicy,
    compilation_receipt: PlanCompilationReceipt,
    *,
    current_report: ValidationReport | None = None,
    require_passed: bool = True,
) -> ValidationReport:
    """Validate the shared approval authority boundary used by every loop."""

    issues: list[ValidationIssue] = []
    issues.extend(validate_plan_compilation_receipt(plan, compilation_receipt).issues)
    if current_report is None:
        if not validation_record_matches_plan(plan, validation_record):
            issues.append(
                ValidationIssue(
                    code="STALE_PLAN_VALIDATION",
                    message="PlanValidationRecord is not bound to the current plan",
                )
            )
        if not validation_record.valid:
            issues.append(
                ValidationIssue(
                    code="PLAN_VALIDATION_FAILED",
                    message="approved authority requires a valid plan validation record",
                )
            )
    else:
        issues.extend(
            validate_plan_validation_record(
                plan, validation_record, current_report
            ).issues
        )
    issues.extend(
        validate_independent_approval_chain(
            plan, verdict, review_input, review, receipt
        ).issues
    )
    issues.extend(validate_approval_state(plan, verdict).issues)
    if trust_policy.approval_mode != ApprovalMode.INDEPENDENT_REQUIRED:
        issues.append(
            ValidationIssue(
                code="INDEPENDENT_APPROVAL_POLICY_REQUIRED",
                message="approved Phase 2+ authority requires independent approval policy",
            )
        )
    if require_passed and verdict.decision not in {
        ApprovalDecision.APPROVE,
        ApprovalDecision.APPROVE_WITH_CONDITIONS,
    }:
        issues.append(
            ValidationIssue(
                code="APPROVAL_DECISION_NOT_EXPORTABLE",
                message="approval authority requires an exportable approval decision",
            )
        )
    if require_passed and not gate.passed:
        issues.append(
            ValidationIssue(
                code="PLAN_GATE_NOT_PASSED",
                message="approval authority requires a passed plan gate",
            )
        )
    gate_bindings = (
        gate_binds_plan_version(gate, (plan.plan_id, plan.version, content_hash(plan))),
        gate_binds_verdict(gate, verdict),
        gate_binds_validation(gate, validation_record),
        gate_binds_approval_receipt(gate, receipt),
        gate_binds_policy(gate, trust_policy),
        gate_binds_compilation_receipt(gate, compilation_receipt),
    )
    if not all(gate_bindings):
        issues.append(
            ValidationIssue(
                code="PLAN_GATE_AUTHORITY_BINDING_MISMATCH",
                message="Plan Gate does not bind the complete approval authority chain",
            )
        )
    return validation_report(issues)
