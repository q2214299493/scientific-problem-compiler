from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..models import (
    ApprovalScores,
    ApprovalReviewInput,
    ApprovalReviewRecord,
    ApprovalVerdict,
    FixResolution,
    GateVerdict,
    HumanDecisionResolution,
    IndependentApprovalReceipt,
    PlanValidationRecord,
    RequiredFix,
    ScientificCapability,
    ScientificQuestionPlan,
)
from ..serialization import content_hash
from ..validators import (
    EvidenceSpanRepository,
    ValidationReport,
    requires_independent_approval,
    validate_independent_approval_chain,
    validate_question_plan,
)


@dataclass(frozen=True)
class ScientificPlanApprover:
    """Authoritative verdict binder; it never returns or modifies a candidate plan."""

    approver_id: str

    def bind_verdict(
        self,
        plan: ScientificQuestionPlan,
        *,
        verdict_id: str,
        scores: ApprovalScores,
        decision: str,
        hard_red_flags: tuple[str, ...] = (),
        required_fixes: tuple[RequiredFix, ...] = (),
        fix_resolutions: tuple[FixResolution, ...] = (),
        human_decisions_required: tuple[str, ...] = (),
        human_decision_resolutions: tuple[HumanDecisionResolution, ...] = (),
    ) -> ApprovalVerdict:
        return ApprovalVerdict(
            verdict_id=verdict_id,
            candidate_id=plan.plan_id,
            candidate_version=plan.version,
            candidate_content_hash=content_hash(plan),
            scores=scores,
            hard_red_flags=hard_red_flags,
            required_fixes=required_fixes,
            fix_resolutions=fix_resolutions,
            human_decisions_required=human_decisions_required,
            human_decision_resolutions=human_decision_resolutions,
            decision=decision,
            approver_id=self.approver_id,
        )

    def deterministic_precheck(
        self,
        plan: ScientificQuestionPlan,
        capabilities: Iterable[ScientificCapability] = (),
        evidence_repository: EvidenceSpanRepository | None = None,
    ) -> ValidationReport:
        return validate_question_plan(plan, capabilities, evidence_repository)


def bind_gate_verdict(
    plan: ScientificQuestionPlan,
    verdict: ApprovalVerdict,
    validation_record: PlanValidationRecord,
    *,
    gate_id: str,
    passed: bool,
    reasons: tuple[str, ...] = (),
    review_input: ApprovalReviewInput | None = None,
    review: ApprovalReviewRecord | None = None,
    receipt: IndependentApprovalReceipt | None = None,
) -> GateVerdict:
    if passed and requires_independent_approval(plan):
        if review_input is None or review is None or receipt is None:
            raise ValueError(
                "MISSING_INDEPENDENT_APPROVAL: Phase 2D gate requires review input, review record, and receipt"
            )
        chain = validate_independent_approval_chain(
            plan, verdict, review_input, review, receipt
        )
        if not chain.valid:
            codes = ", ".join(issue.code for issue in chain.issues)
            raise ValueError(f"INVALID_INDEPENDENT_APPROVAL_CHAIN: {codes}")
    return GateVerdict(
        gate_id=gate_id,
        candidate_id=plan.plan_id,
        candidate_version=plan.version,
        candidate_content_hash=content_hash(plan),
        approval_verdict_id=verdict.verdict_id,
        approval_verdict_hash=content_hash(verdict),
        plan_validation_id=validation_record.validation_id,
        plan_validation_hash=content_hash(validation_record),
        independent_approval_receipt_id=(
            receipt.receipt_id if receipt is not None else None
        ),
        independent_approval_receipt_hash=(
            receipt.content_hash if receipt is not None else None
        ),
        passed=passed,
        reasons=reasons,
    )
