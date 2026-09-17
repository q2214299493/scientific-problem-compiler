from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import spc.workflow.service as workflow_service
from spc.approval import MockApprovalProvider, validate_approval_response
from spc.cli import app
from spc.models import (
    ApprovalDecision,
    ApprovalReviewRecord,
    EvidenceSpan,
    PlanRevisionChain,
    PlanRevisionFeedback,
    PlanRevisionDisposition,
    PlanRevisionFeedbackResponse,
    PlanRevisionInput,
    PlanRevisionLLMResponse,
    PlanRevisionRecord,
    RevisionApprovalLLMResponse,
    RevisionApprovalReviewInput,
    RevisionIssueStatus,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
)
from spc.planning import (
    FakeLLMTransport,
    MockPlanningProvider,
    StructuredLLMPlanningProvider,
    validate_plan_revision_response,
)
from spc.planning.mock_provider import build_proposal_set
from spc.repositories import ProjectEvidenceStore
from spc.serialization import dump_yaml, load_data
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
    revise_call_count = 0

    def revise(self, revision_input):
        type(self).revise_call_count += 1
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
    revise_call_count = 0

    def revise(self, revision_input):
        type(self).revise_call_count += 1
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


class FailingRevisionPlanner(BaselineRepairPlanner):
    revise_call_count = 0

    def revise(self, revision_input):
        type(self).revise_call_count += 1
        raise ValueError("structured provider exhausted its bounded attempts")


class InterruptingRevisionPlanner(BaselineRepairPlanner):
    revise_call_count = 0

    def revise(self, revision_input):
        type(self).revise_call_count += 1
        raise KeyboardInterrupt("simulated process interruption after provider start")


class CountingApproval(MockApprovalProvider):
    review_call_count = 0

    def review(self, review_input):
        type(self).review_call_count += 1
        return super().review(review_input)


class UnresolvedRevisionApproval(MockApprovalProvider):
    def review(self, review_input):
        response = super().review(review_input)
        if not isinstance(response, RevisionApprovalLLMResponse):
            return response
        return response.model_copy(
            update={
                "review": response.review.model_copy(
                    update={"decision_recommendation": ApprovalDecision.APPROVE}
                ),
                "issue_assessments": tuple(
                    item.model_copy(update={"status": RevisionIssueStatus.UNRESOLVED})
                    for item in response.issue_assessments
                ),
            }
        )


class NotApplicableRevisionApproval(MockApprovalProvider):
    def review(self, review_input):
        response = super().review(review_input)
        if not isinstance(response, RevisionApprovalLLMResponse):
            return response
        return response.model_copy(
            update={
                "review": response.review.model_copy(
                    update={"decision_recommendation": ApprovalDecision.APPROVE}
                ),
                "issue_assessments": tuple(
                    item.model_copy(
                        update={
                            "status": RevisionIssueStatus.NOT_APPLICABLE,
                            "rationale": (
                                "Independent evidence review shows the prior criticism "
                                "does not apply to the revised comparison."
                            ),
                        }
                    )
                    for item in response.issue_assessments
                ),
            }
        )


class TwoRoundPlanner(BaselineRepairPlanner):
    received_feedback_ids: list[tuple[str, ...]] = []

    def revise(self, revision_input):
        type(self).received_feedback_ids.append(
            tuple(item.feedback_id for item in revision_input.feedback)
        )
        if revision_input.revision_index == 1:
            return super().revise(revision_input)
        parent = next(
            candidate
            for candidate in revision_input.parent_proposal.candidates
            if candidate.candidate_key == revision_input.parent_candidate_key
        )
        candidate = parent.model_copy(
            update={
                "limitations": (
                    *parent.limitations,
                    "Retain the prior issue for an independent second review.",
                )
            }
        )
        return PlanRevisionLLMResponse(
            intent=revision_input.parent_proposal.intent,
            candidate=candidate,
            feedback_responses=tuple(
                PlanRevisionFeedbackResponse(
                    feedback_id=item.feedback_id,
                    disposition=PlanRevisionDisposition.ADDRESSED,
                    changed_plan_paths=("limitations",),
                    rationale="Added an explicit limitation for independent reassessment.",
                )
                for item in revision_input.feedback
            ),
        )


class TwoRoundApproval(MockApprovalProvider):
    def review(self, review_input):
        response = super().review(review_input)
        if not isinstance(response, RevisionApprovalLLMResponse):
            return response
        if review_input.revision_context.revision_input.revision_index == 1:
            return response.model_copy(
                update={
                    "review": response.review.model_copy(
                        update={
                            "decision_recommendation": ApprovalDecision.REQUEST_REVISION
                        }
                    ),
                    "issue_assessments": tuple(
                        item.model_copy(
                            update={"status": RevisionIssueStatus.UNRESOLVED}
                        )
                        for item in response.issue_assessments
                    ),
                }
            )
        return response


class AlwaysRequestRevisionApproval(MockApprovalProvider):
    provider_id = "mock-always-request-revision"

    def review(self, review_input):
        response = super().review(review_input)
        if isinstance(response, RevisionApprovalLLMResponse):
            return response.model_copy(
                update={
                    "review": response.review.model_copy(
                        update={
                            "decision_recommendation": ApprovalDecision.REQUEST_REVISION
                        }
                    )
                }
            )
        return response.model_copy(
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


def test_revised_approval_input_tracks_parent_feedback_and_response(
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
    second = _chain(workflow, run.run_id).rounds[1]
    review_input = load_data(
        workflow.runs.resolve_artifact_path(
            run.run_id,
            second.approval_review_input.relative_path,
        )
    )

    assert review_input["revision_context"]["parent_plan"]["plan_id"] == second.parent_plan_id
    assert review_input["revision_context"]["tracked_feedback"]
    assert review_input["revision_context"]["planner_responses"]
    assert review_input["revision_context"]["actual_changes"]


def test_revision_approval_context_rejects_tampered_parent_and_response(
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
    second = _chain(workflow, run.run_id).rounds[1]
    review_input = workflow._load_approval_input_binding(
        run.run_id,
        second.approval_review_input,
    )
    assert isinstance(review_input, RevisionApprovalReviewInput)

    response_payload = review_input.model_dump(mode="json")
    response_payload["revision_context"]["planner_responses"][0][
        "rationale"
    ] = "tampered planner response from another review"
    with pytest.raises(ValidationError, match="planner responses must match"):
        RevisionApprovalReviewInput.model_validate(response_payload)

    parent_payload = review_input.model_dump(mode="json")
    parent_payload["revision_context"]["parent_plan_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="parent_plan_hash"):
        RevisionApprovalReviewInput.model_validate(parent_payload)


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


@pytest.mark.parametrize(
    "planner",
    (NoChangePlanner, FabricatingRevisionPlanner, FailingRevisionPlanner),
)
def test_terminal_revision_failure_resume_never_calls_planner_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner,
) -> None:
    planner.revise_call_count = 0
    workflow = _workflow(tmp_path, monkeypatch, planner=planner)
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run
    assert run.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert planner.revise_call_count == 1

    for _ in range(3):
        run = workflow.resume(run.run_id, approval_provider="mock").run

    assert run.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert planner.revise_call_count == 1


def test_terminal_revision_resume_calls_neither_planner_nor_approver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    NoChangePlanner.revise_call_count = 0
    CountingApproval.review_call_count = 0
    workflow = _workflow(
        tmp_path,
        monkeypatch,
        planner=NoChangePlanner,
        approver=CountingApproval,
    )
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run
    planner_calls = NoChangePlanner.revise_call_count
    approval_calls = CountingApproval.review_call_count

    for _ in range(3):
        run = workflow.resume(run.run_id, approval_provider="mock").run

    assert run.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert NoChangePlanner.revise_call_count == planner_calls == 1
    assert CountingApproval.review_call_count == approval_calls == 1


def test_terminal_revision_resume_fails_closed_on_attempt_tampering(
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
    attempt_path = workflow.runs.resolve_artifact_path(
        run.run_id,
        "plan-revisions/attempts/attempt-1/start.yaml",
    )
    payload = load_data(attempt_path)
    payload["provider_version"] = "tampered-provider-version"
    dump_yaml(attempt_path, payload)

    resumed = workflow.resume(run.run_id, approval_provider="mock").run

    assert resumed.status == ScientificProblemRunStatus.FAILED
    assert resumed.failure.stage == "plan_revision_integrity"
    assert resumed.failure.category == "ValidationError"
    assert resumed.failure.retryable is False


def test_uncertain_started_attempt_is_not_retried_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    InterruptingRevisionPlanner.revise_call_count = 0
    workflow = _workflow(tmp_path, monkeypatch, planner=InterruptingRevisionPlanner)

    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        workflow.start(
            REQUEST,
            "base",
            approval_provider="mock",
            max_plan_revisions=1,
        )
    run = workflow.runs.list()[0]
    resumed = workflow.resume(run.run_id, approval_provider="mock").run

    assert resumed.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert resumed.failure.category == "REVISION_ATTEMPT_UNCERTAIN"
    assert InterruptingRevisionPlanner.revise_call_count == 1


def test_revision_attempt_claim_failure_does_not_call_planner_or_gain_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    BaselineRepairPlanner.revise_call_count = 0

    class CountingBaselinePlanner(BaselineRepairPlanner):
        revise_call_count = 0

        def revise(self, revision_input):
            type(self).revise_call_count += 1
            return super().revise(revision_input)

    workflow = _workflow(tmp_path, monkeypatch, planner=CountingBaselinePlanner)
    original_write = workflow.runs.write_exclusive_artifact

    def fail_attempt_claim(run_id, relative_path, artifact_type, artifact_id, value):
        if artifact_type == "plan_revision_attempt_start":
            raise OSError("simulated failure before provider invocation")
        return original_write(run_id, relative_path, artifact_type, artifact_id, value)

    monkeypatch.setattr(workflow.runs, "write_exclusive_artifact", fail_attempt_claim)
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run
    before = _chain(workflow, run.run_id)

    assert run.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert CountingBaselinePlanner.revise_call_count == 0
    assert before.revisions_used == 0

    resumed = workflow.resume(run.run_id, approval_provider="mock").run
    after = _chain(workflow, run.run_id)

    assert resumed.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert CountingBaselinePlanner.revise_call_count == 0
    assert after.content_hash == before.content_hash


def test_revision_round_waiting_for_approval_resumes_without_replanning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    BaselineRepairPlanner.revise_call_count = 0

    class CountingBaselinePlanner(BaselineRepairPlanner):
        revise_call_count = 0

        def revise(self, revision_input):
            type(self).revise_call_count += 1
            return super().revise(revision_input)

    workflow = _workflow(tmp_path, monkeypatch, planner=CountingBaselinePlanner)
    original_approve = workflow._approve

    def interrupt_before_revision_approval(run, *args, **kwargs):
        if kwargs.get("revision_approval") is not None:
            return run
        return original_approve(run, *args, **kwargs)

    monkeypatch.setattr(workflow, "_approve", interrupt_before_revision_approval)
    awaiting = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run
    assert awaiting.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    assert CountingBaselinePlanner.revise_call_count == 1
    chain_before = _chain(workflow, awaiting.run_id)
    assert chain_before.revisions_used == 1
    assert chain_before.rounds[-1].approval_receipt is None

    monkeypatch.setattr(workflow, "_approve", original_approve)
    resumed = workflow.resume(awaiting.run_id, approval_provider="mock").run
    chain_after = _chain(workflow, awaiting.run_id)

    assert resumed.status == ScientificProblemRunStatus.APPROVED
    assert CountingBaselinePlanner.revise_call_count == 1
    assert chain_after.revisions_used == chain_before.revisions_used
    assert len(chain_after.rounds) == len(chain_before.rounds)


def test_concurrent_resume_cannot_claim_duplicate_revision_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowPlanner(BaselineRepairPlanner):
        revise_call_count = 0

        def revise(self, revision_input):
            type(self).revise_call_count += 1
            started.set()
            assert release.wait(timeout=10)
            return super().revise(revision_input)

    workflow = _workflow(tmp_path, monkeypatch, planner=SlowPlanner)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            workflow.start,
            REQUEST,
            "base",
            approval_provider="mock",
            max_plan_revisions=1,
        )
        assert started.wait(timeout=10)
        run_id = workflow.runs.list()[0].run_id
        second = pool.submit(
            workflow.resume,
            run_id,
            approval_provider="mock",
        )
        with pytest.raises(RuntimeError, match="already being resumed"):
            second.result(timeout=10)
        release.set()
        result = first.result(timeout=20)

    assert result.run.status == ScientificProblemRunStatus.APPROVED
    assert SlowPlanner.revise_call_count == 1


def test_unresolved_revision_assessment_cannot_pass_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(
        tmp_path,
        monkeypatch,
        approver=UnresolvedRevisionApproval,
    )
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run

    assert run.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert run.failure.category == "REVISION_BUDGET_EXHAUSTED"
    assert run.gate_id is not None
    chain = _chain(workflow, run.run_id)
    assert chain.rounds[-1].outcome == ApprovalDecision.REQUEST_REVISION.value


def test_independent_reviewer_can_mark_prior_criticism_not_applicable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(
        tmp_path,
        monkeypatch,
        approver=NotApplicableRevisionApproval,
    )
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run

    assert run.status == ScientificProblemRunStatus.APPROVED
    second = _chain(workflow, run.run_id).rounds[-1]
    review_input = workflow._load_approval_input_binding(
        run.run_id,
        second.approval_review_input,
    )
    review = workflow.runs.load_artifact(
        run.run_id,
        second.approval_review_record.relative_path,
        ApprovalReviewRecord,
    )
    assert isinstance(review_input, RevisionApprovalReviewInput)
    assert isinstance(review.response, RevisionApprovalLLMResponse)
    assert all(
        item.status == RevisionIssueStatus.NOT_APPLICABLE
        and "does not apply" in item.rationale
        for item in review.response.issue_assessments
    )


def test_revision_approval_must_assess_every_tracked_blocking_issue(
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
    second = _chain(workflow, run.run_id).rounds[-1]
    review_input = workflow._load_approval_input_binding(
        run.run_id,
        second.approval_review_input,
    )
    review = workflow.runs.load_artifact(
        run.run_id,
        second.approval_review_record.relative_path,
        ApprovalReviewRecord,
    )
    assert isinstance(review_input, RevisionApprovalReviewInput)
    assert isinstance(review.response, RevisionApprovalLLMResponse)
    report = validate_approval_response(
        review.response.review,
        review_input,
    )

    assert not report.valid
    assert "MISSING_REVISION_ISSUE_ASSESSMENTS" in {
        issue.code for issue in report.issues
    }


def test_unresolved_feedback_is_carried_across_multiple_revision_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    TwoRoundPlanner.received_feedback_ids = []
    workflow = _workflow(
        tmp_path,
        monkeypatch,
        planner=TwoRoundPlanner,
        approver=TwoRoundApproval,
    )
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=2,
    ).run

    assert run.status == ScientificProblemRunStatus.APPROVED
    chain = _chain(workflow, run.run_id)
    assert chain.revisions_used == 2
    assert len(TwoRoundPlanner.received_feedback_ids) == 2
    assert set(TwoRoundPlanner.received_feedback_ids[0]).issubset(
        set(TwoRoundPlanner.received_feedback_ids[1])
    )
    third_input = workflow.runs.load_artifact(
        run.run_id,
        chain.rounds[2].revision_input.relative_path,
        PlanRevisionInput,
    )
    assert set(TwoRoundPlanner.received_feedback_ids[0]).issubset(
        {item.feedback_id for item in third_input.feedback}
    )

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


def test_revision_response_rejects_nonexistent_precise_changed_path(
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
    second = _chain(workflow, run.run_id).rounds[1]
    revision_input = workflow.runs.load_artifact(
        run.run_id,
        second.revision_input.relative_path,
        PlanRevisionInput,
    )
    revision_record = workflow.runs.load_artifact(
        run.run_id,
        second.revision_record.relative_path,
        PlanRevisionRecord,
    )
    first_response = revision_record.response.feedback_responses[0].model_copy(
        update={
            "changed_plan_paths": (
                "comparison_baselines[999].description",
            )
        }
    )
    response = revision_record.response.model_copy(
        update={"feedback_responses": (first_response,)}
    )

    report = validate_plan_revision_response(response, revision_input)

    assert not report.valid
    assert "UNKNOWN_REVISION_PLAN_PATH" in {issue.code for issue in report.issues}


def test_revision_response_rejects_change_to_a_claimed_as_change_to_b(
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
    second = _chain(workflow, run.run_id).rounds[1]
    revision_input = workflow.runs.load_artifact(
        run.run_id,
        second.revision_input.relative_path,
        PlanRevisionInput,
    )
    response = MockPlanningProvider().revise(revision_input)
    feedback_response = response.feedback_responses[0].model_copy(
        update={"changed_plan_paths": ("observables[0].description",)}
    )

    report = validate_plan_revision_response(
        response.model_copy(update={"feedback_responses": (feedback_response,)}),
        revision_input,
    )

    assert not report.valid
    assert "REVISION_RESPONSE_NOT_REFLECTED" in {
        issue.code for issue in report.issues
    }


def test_revision_feedback_cannot_borrow_another_feedback_change_path(
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
    second = _chain(workflow, run.run_id).rounds[1]
    revision_input = workflow.runs.load_artifact(
        run.run_id,
        second.revision_input.relative_path,
        PlanRevisionInput,
    )
    response = MockPlanningProvider().revise(revision_input)
    original_feedback = revision_input.feedback[0]
    extra_feedback = PlanRevisionFeedback(
        feedback_id="feedback-independent-second-issue",
        source=original_feedback.source,
        code="SECOND_INDEPENDENT_ISSUE",
        description="A separate issue requires a separate disclosed change.",
        blocking=True,
        plan_path="falsification_criteria[0].statement",
    )
    expanded_input = PlanRevisionInput.model_construct(
        **{
            **revision_input.__dict__,
            "feedback": (*revision_input.feedback, extra_feedback),
        }
    )
    first_response = response.feedback_responses[0]
    second_response = PlanRevisionFeedbackResponse(
        feedback_id=extra_feedback.feedback_id,
        disposition=PlanRevisionDisposition.ADDRESSED,
        changed_plan_paths=first_response.changed_plan_paths,
        rationale="Improperly reuses the first issue's change.",
    )

    report = validate_plan_revision_response(
        response.model_copy(
            update={
                "feedback_responses": (
                    first_response,
                    second_response,
                )
            }
        ),
        expanded_input,
    )

    assert not report.valid
    assert "REVISION_CHANGE_PATH_REUSED" in {issue.code for issue in report.issues}


def test_revision_response_accepts_exact_add_delete_reorder_and_cross_field_paths(
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
    second = _chain(workflow, run.run_id).rounds[1]
    revision_input = workflow.runs.load_artifact(
        run.run_id,
        second.revision_input.relative_path,
        PlanRevisionInput,
    )
    parent = next(
        candidate
        for candidate in revision_input.parent_proposal.candidates
        if candidate.candidate_key == revision_input.parent_candidate_key
    )
    feedback_id = revision_input.feedback[0].feedback_id

    added_baseline = parent.comparison_baselines[0].model_copy(
        update={
            "baseline_key": "baseline-added",
            "description": "A newly disclosed independent control.",
        }
    )
    additions = PlanRevisionLLMResponse(
        intent=revision_input.parent_proposal.intent,
        candidate=parent.model_copy(
            update={
                "comparison_baselines": (
                    *parent.comparison_baselines,
                    added_baseline,
                )
            }
        ),
        feedback_responses=(
            PlanRevisionFeedbackResponse(
                feedback_id=feedback_id,
                disposition=PlanRevisionDisposition.ADDRESSED,
                changed_plan_paths=("comparison_baselines[1]",),
                rationale="Added an explicit control for independent review.",
            ),
        ),
    )
    assert validate_plan_revision_response(additions, revision_input).valid

    deletion = PlanRevisionLLMResponse(
        intent=revision_input.parent_proposal.intent,
        candidate=parent.model_copy(update={"limitations": ()}),
        feedback_responses=(
            PlanRevisionFeedbackResponse(
                feedback_id=feedback_id,
                disposition=PlanRevisionDisposition.ADDRESSED,
                changed_plan_paths=("limitations[0]",),
                rationale="Removed the obsolete risk statement while retaining review.",
            ),
        ),
    )
    assert validate_plan_revision_response(deletion, revision_input).valid

    cross_field = PlanRevisionLLMResponse(
        intent=revision_input.parent_proposal.intent,
        candidate=parent.model_copy(
            update={
                "limitations": (
                    *parent.limitations,
                    "The new control is explicitly limited to this comparison.",
                )
            }
        ),
        feedback_responses=(
            PlanRevisionFeedbackResponse(
                feedback_id=feedback_id,
                disposition=PlanRevisionDisposition.ADDRESSED,
                changed_plan_paths=("limitations",),
                rationale="A cross-field limitation responds without claiming approval.",
            ),
        ),
    )
    assert validate_plan_revision_response(cross_field, revision_input).valid

    parent_with_two_limitations = parent.model_copy(
        update={
            "limitations": (
                *parent.limitations,
                "A second independent limitation.",
            )
        }
    )
    proposal_with_two_limitations = type(revision_input.parent_proposal).model_construct(
        **{
            **revision_input.parent_proposal.__dict__,
            "candidates": (parent_with_two_limitations,),
        }
    )
    reorder_input = PlanRevisionInput.model_construct(
        **{
            **revision_input.__dict__,
            "parent_proposal": proposal_with_two_limitations,
        }
    )
    reorder = PlanRevisionLLMResponse(
        intent=reorder_input.parent_proposal.intent,
        candidate=parent_with_two_limitations.model_copy(
            update={
                "limitations": tuple(reversed(parent_with_two_limitations.limitations))
            }
        ),
        feedback_responses=(
            PlanRevisionFeedbackResponse(
                feedback_id=feedback_id,
                disposition=PlanRevisionDisposition.ADDRESSED,
                changed_plan_paths=("limitations",),
                rationale="Disclosed a list reorder without inventing a scientific change.",
            ),
        ),
    )
    assert validate_plan_revision_response(reorder, reorder_input).valid


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
