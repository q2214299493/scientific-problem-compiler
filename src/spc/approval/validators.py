from __future__ import annotations

from collections import Counter
import re

from ..models import ApprovalLLMResponse, ApprovalReviewInput
from ..validators import ValidationIssue, ValidationReport


_PLAN_PATH_PATTERN = re.compile(
    r"[a-zA-Z_][a-zA-Z0-9_]*(?:\[\d+\])?"
    r"(?:\.[a-zA-Z_][a-zA-Z0-9_]*(?:\[\d+\])?)*"
)


def _plan_path_exists(review_input: ApprovalReviewInput, path: str) -> bool:
    if _PLAN_PATH_PATTERN.fullmatch(path) is None:
        return False
    value: object = review_input.candidate_plan.model_dump(mode="python")
    for name, index_text in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)(?:\[(\d+)\])?", path):
        if not isinstance(value, dict) or name not in value:
            return False
        value = value[name]
        if index_text:
            if not isinstance(value, (list, tuple)):
                return False
            index = int(index_text)
            if index >= len(value):
                return False
            value = value[index]
    return True


def validate_approval_response(
    response: ApprovalLLMResponse,
    review_input: ApprovalReviewInput,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    allowed_evidence = set(review_input.allowed_evidence_ids)
    allowed_claims = set(review_input.allowed_claim_ids)
    allowed_tasks = set(review_input.allowed_task_ids)
    allowed_capabilities = set(review_input.allowed_capability_ids)
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
    for index, flag in enumerate(response.hard_red_flags):
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
