from __future__ import annotations

import pytest
from pydantic import ValidationError

from spc.approval import ScientificPlanApprover
from spc.models import ApprovalScores, ApprovalVerdict
from spc.validators import validate_approval_boundary


def scores() -> ApprovalScores:
    return ApprovalScores(
        intent_fidelity=5,
        evidence_grounding=5,
        model_observable_alignment=5,
        method_consistency=5,
        dag_executability=5,
        falsifiability=5,
        scientific_scope_adequacy=5,
    )


def test_approver_output_cannot_modify_candidate(make_plan) -> None:
    plan = make_plan()
    verdict = ScientificPlanApprover("independent-reviewer").bind_verdict(
        plan, verdict_id="verdict-1", scores=scores(), decision="approve"
    )
    payload = verdict.model_dump(mode="json")
    payload["candidate"] = plan.model_dump(mode="json")
    with pytest.raises(ValidationError):
        ApprovalVerdict.model_validate(payload)


def test_candidate_change_invalidates_prior_verdict(make_plan) -> None:
    original = make_plan()
    verdict = ScientificPlanApprover("independent-reviewer").bind_verdict(
        original, verdict_id="verdict-1", scores=scores(), decision="approve"
    )
    changed = original.model_copy(update={"latent_concern": "changed concern"})
    report = validate_approval_boundary(changed, verdict)
    assert "STALE_APPROVAL" in {item.code for item in report.issues}
