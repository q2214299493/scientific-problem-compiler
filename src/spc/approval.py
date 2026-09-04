from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .models import (
    ApprovalScores,
    ApprovalVerdict,
    FixResolution,
    GateVerdict,
    PlanValidationRecord,
    RequiredFix,
    ScientificCapability,
    ScientificQuestionPlan,
)
from .serialization import content_hash
from .validators import EvidenceSpanRepository, ValidationReport, validate_question_plan


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
        fix_resolutions: tuple[FixResolution, ...] = (),
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
            fix_resolutions=fix_resolutions,
            human_decisions_required=human_decisions_required,
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
) -> GateVerdict:
    return GateVerdict(
        gate_id=gate_id,
        candidate_id=plan.plan_id,
        candidate_version=plan.version,
        candidate_content_hash=content_hash(plan),
        approval_verdict_id=verdict.verdict_id,
        approval_verdict_hash=content_hash(verdict),
        plan_validation_id=validation_record.validation_id,
        plan_validation_hash=content_hash(validation_record),
        passed=passed,
        reasons=reasons,
    )
