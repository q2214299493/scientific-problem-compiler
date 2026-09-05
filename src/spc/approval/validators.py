from __future__ import annotations

from collections import Counter

from ..models import ApprovalLLMResponse, ApprovalReviewInput
from ..validators import ValidationIssue, ValidationReport


def validate_approval_response(
    response: ApprovalLLMResponse,
    review_input: ApprovalReviewInput,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    allowed_evidence = set(review_input.allowed_evidence_ids)
    allowed_claims = set(review_input.allowed_claim_ids)
    allowed_decisions = {
        decision.decision_id
        for decision in review_input.candidate_plan.required_human_decisions
    }

    if not set(response.evidence_basis).issubset(allowed_evidence | allowed_claims):
        issues.append(
            ValidationIssue(
                code="FABRICATED_APPROVAL_EVIDENCE_BASIS",
                message="approval evidence_basis contains a non-allowlisted ID",
                path="evidence_basis",
            )
        )
    for dimension, score in response.scores:
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
    for index, flag in enumerate(response.hard_red_flags):
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
    unknown_decisions = set(response.unresolved_human_decisions) - allowed_decisions
    if unknown_decisions:
        issues.append(
            ValidationIssue(
                code="FABRICATED_APPROVAL_HUMAN_DECISION",
                message="approval response references an unknown human decision",
                path="unresolved_human_decisions",
            )
        )
    for field_name, identifiers in (
        ("hard_red_flags", (flag.code for flag in response.hard_red_flags)),
        ("required_fixes", (fix.fix_id for fix in response.required_fixes)),
        ("unresolved_human_decisions", response.unresolved_human_decisions),
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
    return ValidationReport(valid=not issues, issues=tuple(issues))


class ApprovalResponseError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues)
        super().__init__(f"ApprovalLLMResponse validation failed: {codes}")
