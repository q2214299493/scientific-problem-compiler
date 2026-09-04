from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import ApprovalScores, ApprovalVerdict, RequiredFix, ScientificQuestionPlan
from .serialization import content_hash
from .validators import ValidationReport, validate_question_plan


class ApprovalPolicy(Protocol):
    def review(self, plan: ScientificQuestionPlan, report: ValidationReport) -> tuple[ApprovalScores, tuple[str, ...], tuple[RequiredFix, ...], str]: ...


@dataclass(frozen=True)
class ScientificPlanApprover:
    """Independent review boundary; emits a verdict and never returns a modified plan."""

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
        human_decisions_required: tuple[str, ...] = (),
    ) -> ApprovalVerdict:
        return ApprovalVerdict(
            verdict_id=verdict_id,
            candidate_id=plan.plan_id,
            candidate_version=plan.version,
            candidate_content_hash=content_hash(plan),
            scores=scores,
            hard_red_flags=hard_red_flags,
            required_fixes=required_fixes,
            human_decisions_required=human_decisions_required,
            decision=decision,
            approver_id=self.approver_id,
        )

    def deterministic_precheck(self, plan: ScientificQuestionPlan) -> ValidationReport:
        return validate_question_plan(plan)
