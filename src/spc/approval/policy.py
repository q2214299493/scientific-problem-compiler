from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    ApprovalDecision,
    ApprovalLLMResponse,
    ApprovalRedFlagSeverity,
    ApprovalReviewInput,
    RevisionApprovalLLMResponse,
    RevisionApprovalReviewInput,
    RevisionIssueStatus,
)


@dataclass(frozen=True)
class ApprovalPolicyResult:
    decision: ApprovalDecision
    reasons: tuple[str, ...] = ()


class ApprovalPolicy:
    """Deterministic hard-failure policy; scores never override these rules."""

    def apply(
        self,
        review_input: ApprovalReviewInput | RevisionApprovalReviewInput,
        response: ApprovalLLMResponse | RevisionApprovalLLMResponse,
    ) -> ApprovalPolicyResult:
        revision_response = (
            response if isinstance(response, RevisionApprovalLLMResponse) else None
        )
        base_response = revision_response.review if revision_response else response
        decision = base_response.decision_recommendation
        reasons: list[str] = []
        approvable = {
            ApprovalDecision.APPROVE,
            ApprovalDecision.APPROVE_WITH_CONDITIONS,
        }
        blocking_flags = tuple(
            flag
            for flag in base_response.hard_red_flags
            if flag.severity == ApprovalRedFlagSeverity.BLOCKING
        )
        critical_evidence_codes = {
            "MISSING_CRITICAL_EVIDENCE",
            "UNSUPPORTED_FACTUAL_PROMOTION",
            "SOURCE_HYPOTHESIS_TREATED_AS_FACT",
            "REVIEWER_STATEMENT_TREATED_AS_FACT",
        }
        if decision in approvable and not review_input.plan_validation_record.valid:
            decision = ApprovalDecision.REQUEST_REVISION
            reasons.append("deterministic plan validation failed")
        if decision in approvable and blocking_flags:
            if any(flag.code in critical_evidence_codes for flag in blocking_flags):
                decision = ApprovalDecision.INSUFFICIENT_EVIDENCE
            else:
                decision = ApprovalDecision.REQUEST_REVISION
            reasons.append("blocking hard red flag forbids approval")

        unresolved_decisions = {
            decision.decision_id
            for decision in review_input.candidate_plan.required_human_decisions
        } | set(base_response.unresolved_human_decisions)
        if revision_response is not None:
            blocking_by_id = {
                item.feedback_id: item.blocking
                for item in review_input.revision_context.tracked_feedback
            }
            unresolved_revision = tuple(
                item
                for item in revision_response.issue_assessments
                if blocking_by_id.get(item.feedback_id, True)
                and item.status
                in {
                    RevisionIssueStatus.UNRESOLVED,
                    RevisionIssueStatus.NEEDS_HUMAN_DECISION,
                }
            )
            if any(
                item.status == RevisionIssueStatus.NEEDS_HUMAN_DECISION
                for item in unresolved_revision
            ):
                decision = ApprovalDecision.NEEDS_HUMAN_CHOICE
                reasons.append("tracked revision issue requires a human decision")
            elif unresolved_revision and decision in approvable:
                decision = ApprovalDecision.REQUEST_REVISION
                reasons.append("tracked blocking revision issue remains unresolved")
        if decision == ApprovalDecision.APPROVE and unresolved_decisions:
            decision = ApprovalDecision.NEEDS_HUMAN_CHOICE
            reasons.append("required human decisions remain unresolved")
        if decision == ApprovalDecision.APPROVE and base_response.required_fixes:
            decision = ApprovalDecision.APPROVE_WITH_CONDITIONS
            reasons.append("required fixes prevent plain approval")
        if (
            decision == ApprovalDecision.APPROVE_WITH_CONDITIONS
            and not base_response.required_fixes
            and not unresolved_decisions
        ):
            decision = ApprovalDecision.REQUEST_REVISION
            reasons.append("conditional approval has no declared condition")
        return ApprovalPolicyResult(decision=decision, reasons=tuple(reasons))
