from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    ApprovalDecision,
    ApprovalLLMResponse,
    ApprovalRedFlagSeverity,
    ApprovalReviewInput,
)


@dataclass(frozen=True)
class ApprovalPolicyResult:
    decision: ApprovalDecision
    reasons: tuple[str, ...] = ()


class ApprovalPolicy:
    """Deterministic hard-failure policy; scores never override these rules."""

    def apply(
        self,
        review_input: ApprovalReviewInput,
        response: ApprovalLLMResponse,
    ) -> ApprovalPolicyResult:
        decision = response.decision_recommendation
        reasons: list[str] = []
        approvable = {
            ApprovalDecision.APPROVE,
            ApprovalDecision.APPROVE_WITH_CONDITIONS,
        }
        blocking_flags = tuple(
            flag
            for flag in response.hard_red_flags
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
        } | set(response.unresolved_human_decisions)
        if decision == ApprovalDecision.APPROVE and unresolved_decisions:
            decision = ApprovalDecision.NEEDS_HUMAN_CHOICE
            reasons.append("required human decisions remain unresolved")
        if decision == ApprovalDecision.APPROVE and response.required_fixes:
            decision = ApprovalDecision.APPROVE_WITH_CONDITIONS
            reasons.append("required fixes prevent plain approval")
        if (
            decision == ApprovalDecision.APPROVE_WITH_CONDITIONS
            and not response.required_fixes
            and not unresolved_decisions
        ):
            decision = ApprovalDecision.REQUEST_REVISION
            reasons.append("conditional approval has no declared condition")
        return ApprovalPolicyResult(decision=decision, reasons=tuple(reasons))
