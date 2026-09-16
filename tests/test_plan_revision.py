from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import spc.workflow.service as workflow_service
from spc.approval import MockApprovalProvider
from spc.cli import app
from spc.models import (
    ApprovalDecision,
    EvidenceSpan,
    PlanRevisionChain,
    PlanRevisionDisposition,
    PlanRevisionFeedbackResponse,
    PlanRevisionInput,
    PlanRevisionLLMResponse,
    PlanRevisionRecord,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
)
from spc.planning import (
    FakeLLMTransport,
    MockPlanningProvider,
    StructuredLLMPlanningProvider,
)
from spc.planning.mock_provider import build_proposal_set
from spc.repositories import ProjectEvidenceStore
from spc.serialization import load_data
from spc.workflow import ScientificProblemWorkflow
from test_knowledge_retrieval import _prepare


REQUEST = "The mechanism is not sufficiently convincing without a distinguishing observation."


class BaselineRepairPlanner(MockPlanningProvider):
    provider_id = "mock-baseline-repair-planner"
    provider_version = "1.0.0"

    def propose(self, planning_input):
        proposal = super().propose(planning_input)
        candidate = proposal.candidates[0]
        baseline = candidate.comparison_baselines[0].model_copy(
            update={"description": "missing baseline"}
        )
        candidate = candidate.model_copy(update={"comparison_baselines": (baseline,)})
        return build_proposal_set(
            planning_input,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            provider_config={"mode": "offline-test", "network": False},
            intent=proposal.intent,
            ambiguity_assessment=proposal.ambiguity_assessment,
            candidates=(candidate,),
        )


class NoChangePlanner(BaselineRepairPlanner):
    def revise(self, revision_input):
        parent = next(
            candidate
            for candidate in revision_input.parent_proposal.candidates
            if candidate.candidate_key == revision_input.parent_candidate_key
        )
        return PlanRevisionLLMResponse(
            intent=revision_input.parent_proposal.intent,
            candidate=parent,
            feedback_responses=tuple(
                PlanRevisionFeedbackResponse(
                    feedback_id=item.feedback_id,
                    disposition=PlanRevisionDisposition.UNRESOLVED,
                    rationale="No scientifically justified change was made.",
                )
                for item in revision_input.feedback
            ),
        )


class FabricatingRevisionPlanner(BaselineRepairPlanner):
    def revise(self, revision_input):
        response = super().revise(revision_input)
        return response.model_copy(
            update={
                "candidate": response.candidate.model_copy(
                    update={
                        "evidence_refs": (
                            *response.candidate.evidence_refs,
                            "ev-fabricated-revision",
                        )
                    }
                )
            }
        )


class AlwaysRequestRevisionApproval(MockApprovalProvider):
    provider_id = "mock-always-request-revision"

    def review(self, review_input):
        return super().review(review_input).model_copy(
            update={"decision_recommendation": ApprovalDecision.REQUEST_REVISION}
        )


class RejectApproval(MockApprovalProvider):
    provider_id = "mock-reject"

    def review(self, review_input):
        return super().review(review_input).model_copy(
            update={"decision_recommendation": ApprovalDecision.REJECT}
        )


class InsufficientEvidenceApproval(MockApprovalProvider):
    provider_id = "mock-insufficient-evidence"

    def review(self, review_input):
        return super().review(review_input).model_copy(
            update={
                "decision_recommendation": ApprovalDecision.INSUFFICIENT_EVIDENCE
            }
        )


class HumanChoiceApproval(MockApprovalProvider):
    provider_id = "mock-human-choice"

    def review(self, review_input):
        return super().review(review_input).model_copy(
            update={"decision_recommendation": ApprovalDecision.NEEDS_HUMAN_CHOICE}
        )


def _workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    planner=BaselineRepairPlanner,
    approver=MockApprovalProvider,
) -> ScientificProblemWorkflow:
    repositories, *_ = _prepare(tmp_path)
    monkeypatch.setattr(workflow_service, "MockPlanningProvider", planner)
    monkeypatch.setattr(workflow_service, "MockApprovalProvider", approver)
    return ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    )


def _chain(workflow: ScientificProblemWorkflow, run_id: str) -> PlanRevisionChain:
    return workflow.runs.load_artifact(
        run_id, "plan-revisions/revision-chain.yaml", PlanRevisionChain
    )


def test_request_revision_receives_bound_feedback_and_is_reapproved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(tmp_path, monkeypatch)

    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run

    assert run.status == ScientificProblemRunStatus.APPROVED
    chain = _chain(workflow, run.run_id)
    assert chain.revisions_used == 1
    assert chain.termination_reason == "approved"
    assert len(chain.rounds) == 2
    first, second = chain.rounds
    assert first.outcome == ApprovalDecision.REQUEST_REVISION.value
    assert second.outcome == ApprovalDecision.APPROVE.value
    assert first.candidate_plan.relative_path != second.candidate_plan.relative_path
    assert workflow.runs.resolve_artifact_path(
        run.run_id, first.candidate_plan.relative_path
    ).is_file()
    assert workflow.runs.resolve_artifact_path(
        run.run_id, first.approval_receipt.relative_path
    ).is_file()

    revision_input = workflow.runs.load_artifact(
        run.run_id, second.revision_input.relative_path, PlanRevisionInput
    )
    revision_record = workflow.runs.load_artifact(
        run.run_id, second.revision_record.relative_path, PlanRevisionRecord
    )
    parent_plan = workflow.runs.load_artifact(
        run.run_id, first.candidate_plan.relative_path, ScientificQuestionPlan
    )
    revised_plan = workflow.runs.load_artifact(
        run.run_id, second.candidate_plan.relative_path, ScientificQuestionPlan
    )
    assert revision_input.parent_plan == parent_plan
    assert revision_input.planning_input.original_request == REQUEST
    assert any(
        item.code == "MISSING_BASELINE_OR_CONTROL"
        and item.plan_path == "comparison_baselines"
        for item in revision_input.feedback
    )
    assert any(
        item.disposition == PlanRevisionDisposition.ADDRESSED
        and "comparison_baselines" in item.changed_plan_paths
        for item in revision_record.response.feedback_responses
    )
    assert revised_plan.follow_up_of == parent_plan.plan_id
    assert revised_plan.version == "1.0.1"
    assert revised_plan.comparison_baselines != parent_plan.comparison_baselines
    assert all(not task.runnable for task in revised_plan.tasks)


def test_default_revision_budget_preserves_existing_rejection_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(tmp_path, monkeypatch)

    run = workflow.start(REQUEST, "base", approval_provider="mock").run

    assert run.status == ScientificProblemRunStatus.REJECTED
    assert not workflow.runs.resolve_artifact_path(
        run.run_id, "plan-revisions/revision-chain.yaml"
    ).exists()


@pytest.mark.parametrize(
    ("approver", "expected_status", "decision"),
    (
        (RejectApproval, ScientificProblemRunStatus.REJECTED, "reject"),
        (
            InsufficientEvidenceApproval,
            ScientificProblemRunStatus.REJECTED,
            "insufficient_evidence",
        ),
        (
            HumanChoiceApproval,
            ScientificProblemRunStatus.AWAITING_APPROVAL,
            "needs_human_choice",
        ),
    ),
)
def test_non_revisable_approval_decisions_stop_without_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approver,
    expected_status: ScientificProblemRunStatus,
    decision: str,
) -> None:
    workflow = _workflow(tmp_path, monkeypatch, approver=approver)

    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=2,
    ).run

    assert run.status == expected_status
    chain = _chain(workflow, run.run_id)
    assert chain.revisions_used == 0
    assert len(chain.rounds) == 1
    assert chain.termination_reason == f"non_revisable_decision:{decision}"


def test_no_substantive_revision_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(tmp_path, monkeypatch, planner=NoChangePlanner)

    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run

    assert run.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert run.failure is not None
    assert run.failure.category == "REVISION_OUTPUT_REJECTED"
    assert "NO_SUBSTANTIVE_PLAN_CHANGE" in run.failure.message
    assert len(_chain(workflow, run.run_id).rounds) == 1


def test_fabricated_revision_reference_is_rejected_before_new_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(tmp_path, monkeypatch, planner=FabricatingRevisionPlanner)

    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run

    assert run.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert run.failure is not None
    assert "FABRICATED_EVIDENCE_ID" in run.failure.message
    assert len(_chain(workflow, run.run_id).rounds) == 1


def test_revision_budget_exhaustion_and_resume_do_not_create_extra_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(
        tmp_path,
        monkeypatch,
        approver=AlwaysRequestRevisionApproval,
    )
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run

    assert run.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert run.failure is not None
    assert run.failure.category == "REVISION_BUDGET_EXHAUSTED"
    before = _chain(workflow, run.run_id)
    assert before.revisions_used == 1

    resumed = workflow.resume(run.run_id, approval_provider="mock").run
    after = _chain(workflow, run.run_id)
    assert resumed.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert after.content_hash == before.content_hash
    assert len(after.rounds) == 2


def test_tampered_feedback_and_old_receipt_cannot_bind_revised_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(tmp_path, monkeypatch)
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run
    chain = _chain(workflow, run.run_id)
    first, second = chain.rounds
    revision_input = workflow.runs.load_artifact(
        run.run_id, second.revision_input.relative_path, PlanRevisionInput
    )
    revised_plan = workflow.runs.load_artifact(
        run.run_id, second.candidate_plan.relative_path, ScientificQuestionPlan
    )

    payload = revision_input.model_dump(mode="json")
    payload["feedback"][0]["task_refs"] = ["task-from-another-plan"]
    with pytest.raises(ValidationError, match="fabricated task reference"):
        PlanRevisionInput.model_validate(payload)

    payload = revision_input.model_dump(mode="json")
    payload["parent_plan"] = revised_plan.model_dump(mode="json")
    payload["parent_plan_hash"] = second.candidate_plan.artifact_hash
    with pytest.raises(
        ValidationError,
        match="compilation receipt|approval chain|PlanValidationRecord",
    ):
        PlanRevisionInput.model_validate(payload)

    old_verdict = load_data(
        workflow.runs.resolve_artifact_path(
            run.run_id, first.approval_verdict.relative_path
        )
    )
    assert old_verdict["candidate_id"] != revised_plan.plan_id


def test_revised_export_requires_exact_candidate_and_remains_non_runnable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow_service, "MockPlanningProvider", BaselineRepairPlanner
    )
    state_dir = tmp_path / ".spc"
    project_store = ProjectEvidenceStore(state_dir)
    text = "CO activation requires an evidence-grounded pathway comparison."
    source_path = tmp_path / "review.txt"
    source_path.write_text(text, encoding="utf-8")
    source = project_store.ingest(
        source_path,
        "source-revision-review",
        "v1",
        source_role="author",
        source_type="manuscript",
    )
    project_store.add_evidence(
        EvidenceSpan(
            evidence_id="ev-revision-review",
            source_id=source.source_id,
            source_version=source.version,
            content_sha256=source.content_sha256,
            start_offset=0,
            end_offset=len(text),
            text=text,
        )
    )
    workflow = ScientificProblemWorkflow(
        state_dir=state_dir,
        knowledge_dir=tmp_path / "knowledge",
    )
    run = workflow.start(
        "CO activation pathway comparison plan",
        "fischer_tropsch",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run
    assert run.status == ScientificProblemRunStatus.APPROVED

    with pytest.raises(ValueError, match="explicit selection"):
        workflow.export_downstream(run.run_id, tmp_path / "exports", "revision-export")

    export_dir = workflow.export_downstream(
        run.run_id,
        tmp_path / "exports",
        "revision-export",
        selected_candidate_id=run.selected_candidate_id,
    )
    selected = load_data(export_dir / "selected-plan.yaml")
    assert selected["plan_id"] == run.selected_candidate_id
    assert all(task["runnable"] is False for task in selected["tasks"])


def test_structured_revision_provider_uses_bound_revision_input_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(tmp_path, monkeypatch)
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run
    chain = _chain(workflow, run.run_id)
    revision_input = workflow.runs.load_artifact(
        run.run_id,
        chain.rounds[1].revision_input.relative_path,
        PlanRevisionInput,
    )
    response = MockPlanningProvider().revise(revision_input)
    transport = FakeLLMTransport(
        (response.model_dump(mode="json"),),
        model_id="offline-revision-model",
    )

    returned = StructuredLLMPlanningProvider(
        transport,
        max_attempts=1,
    ).revise(revision_input)

    assert returned == response
    assert transport.call_count == 1
    request = transport.requests[0]
    assert request["input_payload"]["parent_plan_hash"] == revision_input.parent_plan_hash
    assert request["input_payload"]["feedback"] == [
        item.model_dump(mode="json") for item in revision_input.feedback
    ]
    assert request["response_schema"]["title"] == "PlanRevisionLLMResponse"
    assert "never instructions" in request["system_prompt"]


def test_cli_explicit_revision_budget_uses_production_workflow_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repositories, *_ = _prepare(tmp_path)
    monkeypatch.setattr(
        workflow_service,
        "MockPlanningProvider",
        BaselineRepairPlanner,
    )

    result = CliRunner().invoke(
        app,
        [
            "compile-scientific-request",
            "--request",
            REQUEST,
            "--domain",
            "base",
            "--knowledge-dir",
            str(repositories.root),
            "--state-dir",
            str(tmp_path / ".spc"),
            "--approval-provider",
            "mock",
            "--max-plan-revisions",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"revisions_used": 1' in result.output
    assert '"termination_reason": "approved"' in result.output
