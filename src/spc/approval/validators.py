from __future__ import annotations

from collections import Counter

from ..models import (
    ApprovalLLMResponse,
    ApprovalReviewInput,
    RevisionApprovalLLMResponse,
    RevisionApprovalReviewInput,
)
from ..plan_paths import resolve_plan_path
from ..validators import ValidationIssue, ValidationReport


ApprovalInput = ApprovalReviewInput | RevisionApprovalReviewInput
ApprovalResponse = ApprovalLLMResponse | RevisionApprovalLLMResponse


def _plan_path_exists(review_input: ApprovalInput, path: str) -> bool:
    try:
        return resolve_plan_path(review_input.candidate_plan, path).exists
    except ValueError:
        return False


def validate_approval_response(
    response: ApprovalResponse,
    review_input: ApprovalInput,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    revision_response = (
        response if isinstance(response, RevisionApprovalLLMResponse) else None
    )
    base_response = revision_response.review if revision_response else response
    allowed_evidence = set(review_input.allowed_evidence_ids)
    allowed_claims = set(review_input.allowed_claim_ids)
    allowed_tasks = set(review_input.allowed_task_ids)
    allowed_capabilities = set(review_input.allowed_capability_ids)
    allowed_decisions = {
        decision.decision_id
        for decision in review_input.candidate_plan.required_human_decisions
    }

    if not set(base_response.evidence_basis).issubset(
        allowed_evidence | allowed_claims
    ):
        issues.append(
            ValidationIssue(
                code="FABRICATED_APPROVAL_EVIDENCE_BASIS",
                message="approval evidence_basis contains a non-allowlisted ID",
                path="evidence_basis",
            )
        )
    for dimension, score in base_response.scores:
        if not set(score.evidence_refs).issubset(allowed_evidence):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_APPROVAL_EVIDENCE_REF",
                    message=f"{dimension} score references non-allowlisted evidence",
                    path=f"scores.{dimension}.evidence_refs",
                )
            )
        if not set(score.claim_refs).issubset(allowed_claims):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_APPROVAL_CLAIM_REF",
                    message=f"{dimension} score references a non-allowlisted claim",
                    path=f"scores.{dimension}.claim_refs",
                )
            )
        if not set(score.task_refs).issubset(allowed_tasks):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_APPROVAL_TASK_REF",
                    message=f"{dimension} score references a non-allowlisted task",
                    path=f"scores.{dimension}.task_refs",
                )
            )
        if not set(score.capability_refs).issubset(allowed_capabilities):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_APPROVAL_CAPABILITY_REF",
                    message=f"{dimension} score references a non-allowlisted capability",
                    path=f"scores.{dimension}.capability_refs",
                )
            )
    for index, flag in enumerate(base_response.hard_red_flags):
        if flag.plan_path is not None and not _plan_path_exists(
            review_input, flag.plan_path
        ):
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_APPROVAL_PLAN_PATH",
                    message="hard red flag references an unknown candidate plan path",
                    path=f"hard_red_flags[{index}].plan_path",
                )
            )
        if not set(flag.evidence_refs).issubset(allowed_evidence):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_APPROVAL_EVIDENCE_REF",
                    message="hard red flag references non-allowlisted evidence",
                    path=f"hard_red_flags[{index}].evidence_refs",
                )
            )
        if not set(flag.claim_refs).issubset(allowed_claims):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_APPROVAL_CLAIM_REF",
                    message="hard red flag references a non-allowlisted claim",
                    path=f"hard_red_flags[{index}].claim_refs",
                )
            )
        if not set(flag.task_refs).issubset(allowed_tasks):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_APPROVAL_TASK_REF",
                    message="hard red flag references a non-allowlisted task",
                    path=f"hard_red_flags[{index}].task_refs",
                )
            )
        if not set(flag.capability_refs).issubset(allowed_capabilities):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_APPROVAL_CAPABILITY_REF",
                    message="hard red flag references a non-allowlisted capability",
                    path=f"hard_red_flags[{index}].capability_refs",
                )
            )
    unknown_decisions = (
        set(base_response.unresolved_human_decisions) - allowed_decisions
    )
    if unknown_decisions:
        issues.append(
            ValidationIssue(
                code="FABRICATED_APPROVAL_HUMAN_DECISION",
                message="approval response references an unknown human decision",
                path="unresolved_human_decisions",
            )
        )
    for field_name, identifiers in (
        ("hard_red_flags", (flag.code for flag in base_response.hard_red_flags)),
        ("required_fixes", (fix.fix_id for fix in base_response.required_fixes)),
        (
            "unresolved_human_decisions",
            base_response.unresolved_human_decisions,
        ),
    ):
        duplicates = tuple(
            identifier
            for identifier, count in Counter(identifiers).items()
            if count > 1
        )
        if duplicates:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_APPROVAL_RESPONSE_ID",
                    message=f"{field_name} contains duplicate identifiers",
                    path=field_name,
                )
            )

    if isinstance(review_input, RevisionApprovalReviewInput):
        if revision_response is None:
            issues.append(
                ValidationIssue(
                    code="MISSING_REVISION_ISSUE_ASSESSMENTS",
                    message="revision approval requires item-by-item issue assessments",
                    path="issue_assessments",
                )
            )
        else:
            expected_ids = {
                item.feedback_id
                for item in review_input.revision_context.tracked_feedback
            }
            assessment_ids = tuple(
                item.feedback_id for item in revision_response.issue_assessments
            )
            if len(set(assessment_ids)) != len(assessment_ids) or set(
                assessment_ids
            ) != expected_ids:
                issues.append(
                    ValidationIssue(
                        code="REVISION_ISSUE_ASSESSMENT_MISMATCH",
                        message=(
                            "revision approval must assess every tracked issue exactly once"
                        ),
                        path="issue_assessments",
                    )
                )
            for index, assessment in enumerate(
                revision_response.issue_assessments
            ):
                if not set(assessment.evidence_refs).issubset(allowed_evidence):
                    issues.append(
                        ValidationIssue(
                            code="FABRICATED_REVISION_ASSESSMENT_EVIDENCE_REF",
                            message="revision assessment uses non-allowlisted evidence",
                            path=f"issue_assessments[{index}].evidence_refs",
                        )
                    )
                if not set(assessment.claim_refs).issubset(allowed_claims):
                    issues.append(
                        ValidationIssue(
                            code="FABRICATED_REVISION_ASSESSMENT_CLAIM_REF",
                            message="revision assessment uses a non-allowlisted claim",
                            path=f"issue_assessments[{index}].claim_refs",
                        )
                    )
                if not set(assessment.task_refs).issubset(allowed_tasks):
                    issues.append(
                        ValidationIssue(
                            code="FABRICATED_REVISION_ASSESSMENT_TASK_REF",
                            message="revision assessment uses a non-allowlisted task",
                            path=f"issue_assessments[{index}].task_refs",
                        )
                    )
    elif revision_response is not None:
        issues.append(
            ValidationIssue(
                code="UNEXPECTED_REVISION_APPROVAL_RESPONSE",
                message="initial approval cannot use the revision approval contract",
            )
        )
    return ValidationReport(valid=not issues, issues=tuple(issues))


class ApprovalResponseError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues)
        super().__init__(f"ApprovalLLMResponse validation failed: {codes}")
