from __future__ import annotations

from ..models import PlanningProposalSet, PlanningStrategyClass, ScientificPlanningInput
from ..serialization import content_hash
from ..validators import ValidationIssue, ValidationReport


class PlanningProposalError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues)
        super().__init__(f"PlanningProposalSet validation failed: {codes}")


def validate_planning_proposal_set(
    proposal: PlanningProposalSet,
    planning_input: ScientificPlanningInput,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if (
        proposal.planning_input_id != planning_input.planning_input_id
        or proposal.planning_input_hash != planning_input.content_hash
    ):
        issues.append(
            ValidationIssue(
                code="PLANNING_INPUT_MISMATCH",
                message="proposal is not bound to the supplied ScientificPlanningInput",
            )
        )
    allowed_evidence = set(planning_input.allowed_evidence_ids)
    allowed_claims = set(planning_input.allowed_claim_ids)
    allowed_capabilities = set(planning_input.allowed_capability_ids)
    allowed_decisions = {
        decision.decision_id for decision in planning_input.required_human_decisions
    }
    intent_basis = set(proposal.intent.evidence_basis)
    if not intent_basis.issubset(allowed_evidence | allowed_claims):
        issues.append(
            ValidationIssue(
                code="FABRICATED_INTENT_REFERENCE",
                message="intent evidence basis contains a non-allowlisted reference",
                path="intent.evidence_basis",
            )
        )
    for index, candidate in enumerate(proposal.candidates):
        prefix = f"candidates[{index}]"
        if not set(candidate.evidence_refs).issubset(allowed_evidence):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_EVIDENCE_ID",
                    message="candidate contains a non-allowlisted evidence ID",
                    path=f"{prefix}.evidence_refs",
                )
            )
        if not set(candidate.claim_refs).issubset(allowed_claims):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_CLAIM_ID",
                    message="candidate contains a non-allowlisted SourceClaim ID",
                    path=f"{prefix}.claim_refs",
                )
            )
        if not set(candidate.capability_ids).issubset(allowed_capabilities):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_CAPABILITY_ID",
                    message="candidate contains a non-allowlisted capability ID",
                    path=f"{prefix}.capability_ids",
                )
            )
        if not set(candidate.human_decisions_required).issubset(allowed_decisions):
            issues.append(
                ValidationIssue(
                    code="FABRICATED_HUMAN_DECISION_ID",
                    message="candidate contains an unknown required human decision",
                    path=f"{prefix}.human_decisions_required",
                )
            )
        task_keys = {task.task_key for task in candidate.task_drafts}
        if len(task_keys) != len(candidate.task_drafts):
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_TASK_KEY",
                    message="candidate task keys must be unique",
                    path=f"{prefix}.task_drafts",
                )
            )
        for task_index, task in enumerate(candidate.task_drafts):
            task_path = f"{prefix}.task_drafts[{task_index}]"
            if task.capability_id not in allowed_capabilities:
                issues.append(
                    ValidationIssue(
                        code="FABRICATED_CAPABILITY_ID",
                        message="task uses a non-allowlisted capability ID",
                        path=f"{task_path}.capability_id",
                    )
                )
            if task.capability_id not in candidate.capability_ids:
                issues.append(
                    ValidationIssue(
                        code="UNDECLARED_CAPABILITY",
                        message="task capability is not declared by its candidate",
                        path=f"{task_path}.capability_id",
                    )
                )
            if not set(task.evidence_refs).issubset(allowed_evidence):
                issues.append(
                    ValidationIssue(
                        code="FABRICATED_EVIDENCE_ID",
                        message="task contains a non-allowlisted evidence ID",
                        path=f"{task_path}.evidence_refs",
                    )
                )
            for dependency in task.depends_on:
                if dependency not in task_keys:
                    issues.append(
                        ValidationIssue(
                            code="UNKNOWN_TASK_DRAFT_DEPENDENCY",
                            message=f"task depends on unknown draft key {dependency}",
                            path=f"{task_path}.depends_on",
                        )
                    )

    for conflict in planning_input.conflict_sets:
        if conflict.resolution_status != "unresolved":
            continue
        conflict_claims = set(conflict.claim_refs)
        retained = any(
            candidate.strategy_class
            in {
                PlanningStrategyClass.MECHANISM_DISCRIMINATION,
                PlanningStrategyClass.MODEL_DISCRIMINATION,
            }
            and conflict_claims.issubset(set(candidate.claim_refs))
            for candidate in proposal.candidates
        )
        if not retained:
            issues.append(
                ValidationIssue(
                    code="UNRESOLVED_CONFLICT_DROPPED",
                    message=f"unresolved conflict {conflict.conflict_id} is not retained for discrimination",
                )
            )

    for gap in planning_input.evidence_gaps:
        if not gap.blocking:
            continue
        gap_capabilities = set(gap.candidate_capabilities) & allowed_capabilities
        addressed = any(
            task.capability_id in gap_capabilities
            or gap.gap_id in dict(task.inputs).get("evidence_gap_ids", ())
            for candidate in proposal.candidates
            for task in candidate.task_drafts
        )
        decision_id = f"decision-{content_hash({'gap_id': gap.gap_id})[:24]}"
        propagated = any(
            decision_id in candidate.human_decisions_required
            or any(gap.gap_id in limitation for limitation in candidate.limitations)
            for candidate in proposal.candidates
        )
        if not addressed and not propagated:
            issues.append(
                ValidationIssue(
                    code="BLOCKING_EVIDENCE_GAP_DROPPED",
                    message=f"blocking evidence gap {gap.gap_id} is neither addressed nor propagated",
                )
            )
    return ValidationReport(valid=not issues, issues=tuple(issues))
