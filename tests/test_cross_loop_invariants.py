from __future__ import annotations

from pathlib import Path
import shutil

import pytest

import spc.workflow.service as workflow_service
from spc.approval import MockApprovalProvider
from spc.models import (
    ApprovalDecision,
    AmbiguityAssessment,
    ReportedObservable,
    RequiredFix,
    ResultEvidenceType,
    ResultExecutionOutcome,
    ResultEvidenceIntakeStatus,
    ResultStatus,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
    RevisionApprovalLLMResponse,
)
from spc.planning import MockPlanningProvider
from spc.planning.mock_provider import build_proposal_set
from spc.result_feedback import (
    ResultEvidenceIntakeService,
    ResultFeedbackError,
    SuccessorPlanningService,
    result_feedback_status,
)
from spc.serialization import content_hash, dump_yaml, load_data
from spc.workflow import ScientificProblemWorkflow
from test_hierarchical_planning import (
    BaselineRepairHierarchicalProvider,
    CountingHierarchicalProvider,
    REQUEST as HIERARCHICAL_REQUEST,
    RetainAndHumanChoiceProvider,
)
from test_planning_evidence_resolution import InsufficientThenApproveProvider
from test_result_feedback import (
    _approved_parent,
    _intake_and_accept,
    _submission,
)


class CountingSuccessorPlanner(MockPlanningProvider):
    def __init__(self) -> None:
        self.call_count = 0

    def propose(self, planning_input):
        self.call_count += 1
        return super().propose(planning_input)


class CountingSuccessorApprover(MockApprovalProvider):
    def __init__(self) -> None:
        self.call_count = 0

    def review(self, review_input):
        self.call_count += 1
        return super().review(review_input)


class InterruptingSuccessorPlanner(MockPlanningProvider):
    def __init__(self) -> None:
        self.call_count = 0

    def propose(self, planning_input):
        self.call_count += 1
        raise KeyboardInterrupt("simulated successor interruption")


class TwoCandidateSuccessorPlanner(CountingSuccessorPlanner):
    def propose(self, planning_input):
        proposal = super().propose(planning_input)
        base = proposal.candidates[0]
        first = base.model_copy(
            update={
                "candidate_key": "mechanism-a",
                "distinguishing_axis": "competing mechanism",
                "distinguishing_value": "associative",
            }
        )
        second = base.model_copy(
            update={
                "candidate_key": "mechanism-b",
                "distinguishing_axis": "competing mechanism",
                "distinguishing_value": "dissociative",
            }
        )
        ambiguity = AmbiguityAssessment(
            multiple_candidates_required=True,
            rationale="Two scientifically distinct mechanisms require human selection.",
            scientifically_distinct_axes=("competing mechanism",),
        )
        return build_proposal_set(
            planning_input,
            provider_id=proposal.provider_id,
            provider_version=proposal.provider_version,
            provider_config=dict(proposal.provider_config),
            intent=proposal.intent,
            ambiguity_assessment=ambiguity,
            candidates=(first, second),
        )


class InterruptingSuccessorApprover(CountingSuccessorApprover):
    def review(self, review_input):
        self.call_count += 1
        raise KeyboardInterrupt("simulated approval interruption")


class RevisionThenEvidenceApproval(MockApprovalProvider):
    review_calls = 0

    def review(self, review_input):
        type(self).review_calls += 1
        response = super().review(review_input)
        if type(self).review_calls != 2:
            return response
        fix = RequiredFix(
            fix_id="fix-reported-method-source",
            description=(
                "Retrieve the literature-reported method and comparison condition "
                "DFT from a trusted source."
            ),
            blocking=True,
        )
        if isinstance(response, RevisionApprovalLLMResponse):
            return response.model_copy(
                update={
                    "review": response.review.model_copy(
                        update={
                            "decision_recommendation": (
                                ApprovalDecision.INSUFFICIENT_EVIDENCE
                            ),
                            "required_fixes": (fix,),
                        }
                    )
                }
            )
        return response.model_copy(
            update={
                "decision_recommendation": ApprovalDecision.INSUFFICIENT_EVIDENCE,
                "required_fixes": (fix,),
            }
        )


def test_same_successor_inputs_reuse_one_content_bound_cycle(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    planner = CountingSuccessorPlanner()
    approver = CountingSuccessorApprover()
    service = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    first = service.compile(
        parent_run_id=run.run_id,
        submission_ids=(submission.submission_id,),
        planning_provider=planner,
        approval_provider=approver,
    )
    second = service.compile(
        parent_run_id=run.run_id,
        submission_ids=(submission.submission_id,),
        planning_provider=planner,
        approval_provider=approver,
    )

    assert second.cycle_id == first.cycle_id
    assert planner.call_count == 1
    assert approver.call_count == 1


def test_successor_result_set_order_has_one_canonical_invocation(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    result1, root1, _ = _submission(tmp_path / "result-1", run, plan)
    result2, root2, _ = _submission(
        tmp_path / "result-2",
        run,
        plan,
        observables=(
            ReportedObservable(
                observable_key=plan.tasks[0].outputs[0],
                quantity="independent comparison observable",
                value=2.5,
                unit="eV",
                result_status=ResultStatus.COMPUTED_REPORTED,
            ),
        ),
    )
    _intake_and_accept(tmp_path, repositories, run, result1, root1)
    _intake_and_accept(tmp_path, repositories, run, result2, root2)
    planner = CountingSuccessorPlanner()
    approver = CountingSuccessorApprover()
    service = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    first = service.compile(
        parent_run_id=run.run_id,
        submission_ids=(result1.submission_id, result2.submission_id),
        planning_provider=planner,
        approval_provider=approver,
    )
    second = service.compile(
        parent_run_id=run.run_id,
        submission_ids=(result2.submission_id, result1.submission_id),
        planning_provider=planner,
        approval_provider=approver,
    )

    assert second.cycle_id == first.cycle_id
    assert planner.call_count == 1
    assert approver.call_count == 1


def test_human_candidate_selection_reuses_the_original_planning_output(
    tmp_path: Path,
) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    planner = TwoCandidateSuccessorPlanner()
    approver = CountingSuccessorApprover()
    service = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    pending = service.compile(
        parent_run_id=run.run_id,
        submission_ids=(submission.submission_id,),
        planning_provider=planner,
        approval_provider=approver,
    )
    assert pending.status.value == "HUMAN_SCIENTIFIC_DECISION_REQUIRED"
    selected_id = pending.candidate_plan_ids[0]
    original = workflow.runs.load_artifact(
        run.run_id,
        f"successor-planning/cycle-1/candidates/{selected_id}.yaml",
        ScientificQuestionPlan,
    )

    completed = service.compile(
        parent_run_id=run.run_id,
        submission_ids=(submission.submission_id,),
        planning_provider=planner,
        approval_provider=approver,
        selected_candidate_id=selected_id,
    )
    final = workflow.runs.load_artifact(
        run.run_id,
        f"successor-planning/cycle-1/candidates/{selected_id}.yaml",
        ScientificQuestionPlan,
    )

    assert completed.status.value == "APPROVED"
    assert completed.selected_plan_id == selected_id
    assert final == original
    assert all(not task.runnable for task in final.tasks)
    assert planner.call_count == 1
    assert approver.call_count == 1


def test_candidate_from_another_successor_invocation_is_rejected(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    planner = TwoCandidateSuccessorPlanner()
    service = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )
    first = service.compile(
        parent_run_id=run.run_id,
        submission_ids=(submission.submission_id,),
        follow_up_request="First successor question.",
        planning_provider=planner,
    )

    with pytest.raises(ResultFeedbackError, match="UNKNOWN_SUCCESSOR_CANDIDATE"):
        service.compile(
            parent_run_id=run.run_id,
            submission_ids=(submission.submission_id,),
            follow_up_request="Different successor question.",
            planning_provider=planner,
            selected_candidate_id=first.candidate_plan_ids[0],
        )


def test_uncertain_successor_provider_call_is_not_repeated(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    planner = InterruptingSuccessorPlanner()
    service = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    with pytest.raises(KeyboardInterrupt, match="successor interruption"):
        service.compile(
            parent_run_id=run.run_id,
            submission_ids=(submission.submission_id,),
            planning_provider=planner,
        )
    with pytest.raises(
        ResultFeedbackError, match="SUCCESSOR_PLANNING_OUTCOME_UNCERTAIN"
    ):
        service.compile(
            parent_run_id=run.run_id,
            submission_ids=(submission.submission_id,),
            planning_provider=planner,
        )

    assert planner.call_count == 1


def test_complete_planning_output_resumes_at_approval_without_replanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    planner = CountingSuccessorPlanner()
    approver = CountingSuccessorApprover()
    service = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )
    original_write = service.runs.write_exclusive_artifact

    def interrupt_before_approval(run_id, relative_path, *args, **kwargs):
        if relative_path.endswith("provider-calls/approval.yaml"):
            raise KeyboardInterrupt("crash before approval claim")
        return original_write(run_id, relative_path, *args, **kwargs)

    monkeypatch.setattr(service.runs, "write_exclusive_artifact", interrupt_before_approval)
    with pytest.raises(KeyboardInterrupt, match="before approval claim"):
        service.compile(
            parent_run_id=run.run_id,
            submission_ids=(submission.submission_id,),
            planning_provider=planner,
            approval_provider=approver,
        )
    monkeypatch.setattr(service.runs, "write_exclusive_artifact", original_write)

    cycle = service.compile(
        parent_run_id=run.run_id,
        submission_ids=(submission.submission_id,),
        planning_provider=planner,
        approval_provider=approver,
    )

    assert cycle.status.value == "APPROVED"
    assert planner.call_count == 1
    assert approver.call_count == 1


def test_approval_claim_without_complete_output_is_not_repeated(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    planner = CountingSuccessorPlanner()
    approver = InterruptingSuccessorApprover()
    service = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    with pytest.raises(KeyboardInterrupt, match="approval interruption"):
        service.compile(
            parent_run_id=run.run_id,
            submission_ids=(submission.submission_id,),
            planning_provider=planner,
            approval_provider=approver,
        )
    with pytest.raises(
        ResultFeedbackError, match="SUCCESSOR_APPROVAL_OUTCOME_UNCERTAIN"
    ):
        service.compile(
            parent_run_id=run.run_id,
            submission_ids=(submission.submission_id,),
            planning_provider=planner,
            approval_provider=approver,
        )

    assert planner.call_count == 1
    assert approver.call_count == 1


def test_complete_approval_output_rebuilds_cycle_without_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    planner = CountingSuccessorPlanner()
    approver = CountingSuccessorApprover()
    service = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )
    original_write = service.runs.write_immutable_artifact

    def interrupt_before_cycle(run_id, relative_path, *args, **kwargs):
        if relative_path.endswith("/cycle.yaml"):
            raise KeyboardInterrupt("crash before final cycle")
        return original_write(run_id, relative_path, *args, **kwargs)

    monkeypatch.setattr(service.runs, "write_immutable_artifact", interrupt_before_cycle)
    with pytest.raises(KeyboardInterrupt, match="before final cycle"):
        service.compile(
            parent_run_id=run.run_id,
            submission_ids=(submission.submission_id,),
            planning_provider=planner,
            approval_provider=approver,
        )
    monkeypatch.setattr(service.runs, "write_immutable_artifact", original_write)

    cycle = service.compile(
        parent_run_id=run.run_id,
        submission_ids=(submission.submission_id,),
        planning_provider=planner,
        approval_provider=approver,
    )

    assert cycle.status.value == "APPROVED"
    assert planner.call_count == 1
    assert approver.call_count == 1


def test_successor_status_uses_numeric_indexes_and_reports_incomplete_cycles(
    tmp_path: Path,
) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).compile(parent_run_id=run.run_id, submission_ids=(submission.submission_id,))
    successor_root = workflow.runs.resolve_artifact_path(
        run.run_id, "successor-planning"
    )
    (successor_root / "cycle-1").rename(successor_root / "cycle-10")
    incomplete = successor_root / "cycle-2" / "provider-calls"
    incomplete.mkdir(parents=True)
    shutil.copy2(
        successor_root / "cycle-10" / "invocation.yaml",
        incomplete.parent / "invocation.yaml",
    )
    shutil.copy2(
        successor_root / "cycle-10" / "parent-binding.yaml",
        incomplete.parent / "parent-binding.yaml",
    )
    shutil.copy2(
        successor_root / "cycle-10" / "provider-calls" / "planning.yaml",
        incomplete / "planning.yaml",
    )

    status = result_feedback_status(run.run_id, workflow.runs)["successor_planning"]

    assert status["latest_complete_cycle"] == 10
    assert status["latest_attempted_cycle"] == 10
    assert status["cycle"] == 10
    assert status["incomplete_cycles"] == [
        {"cycle": 2, "status": "SUCCESSOR_PLANNING_OUTCOME_UNCERTAIN"}
    ]
    assert status["triggering_result_ids"] == [submission.submission_id]


def test_completed_successor_cycle_is_not_reused_after_authority_tampering(
    tmp_path: Path,
) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    service = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )
    service.compile(
        parent_run_id=run.run_id,
        submission_ids=(submission.submission_id,),
    )
    policy_path = workflow.runs.resolve_artifact_path(
        run.run_id,
        "successor-planning/cycle-1/approval/project-trust-policy.yaml",
    )
    policy = load_data(policy_path)
    policy["approval_mode"] = "legacy_manual_allowed"
    dump_yaml(policy_path, policy)

    with pytest.raises(ResultFeedbackError, match="SUCCESSOR_CYCLE_INTEGRITY_INVALID"):
        service.compile(
            parent_run_id=run.run_id,
            submission_ids=(submission.submission_id,),
        )


def test_resume_does_not_reuse_approval_after_trust_policy_tampering(
    tmp_path: Path,
) -> None:
    _, workflow, run, _ = _approved_parent(tmp_path)
    policy_path = workflow.runs.resolve_artifact_path(
        run.run_id, "approval/project-trust-policy.yaml"
    )
    policy = load_data(policy_path)
    policy["approval_mode"] = "legacy_manual_allowed"
    dump_yaml(policy_path, policy)

    resumed = workflow.resume(run.run_id, approval_provider="none").run

    assert resumed.status != ScientificProblemRunStatus.APPROVED


@pytest.mark.parametrize(
    ("relative_path", "field", "replacement"),
    (
        ("approval/project-trust-policy.yaml", "approval_mode", "legacy_manual_allowed"),
        ("approval/plan-gate.yaml", "passed", False),
        ("validation-records", "valid", False),
        ("compilation-receipts", "origin", "fabricated-compiler"),
        ("approval/approval-review.yaml", "provider_version", "tampered-version"),
    ),
)
def test_each_approval_authority_component_fails_closed_on_resume(
    tmp_path: Path,
    relative_path: str,
    field: str,
    replacement: object,
) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    if relative_path == "validation-records":
        path = workflow.runs.resolve_artifact_path(
            run.run_id, run.plan_validation_records[0].relative_path
        )
    elif relative_path == "compilation-receipts":
        path = workflow.runs.resolve_artifact_path(
            run.run_id, run.candidate_compilation_receipts[0].relative_path
        )
    else:
        path = workflow.runs.resolve_artifact_path(run.run_id, relative_path)
    payload = load_data(path)
    payload[field] = replacement
    dump_yaml(path, payload)

    assessment = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).intake(submission, artifact_root=artifact_root)

    assert assessment.status == ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING


def test_hierarchy_then_revision_uses_revision_loop_without_rerunning_hierarchy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_knowledge_retrieval import _prepare

    knowledge, *_ = _prepare(tmp_path)
    monkeypatch.setattr(
        workflow_service, "MockPlanningProvider", BaselineRepairHierarchicalProvider
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=knowledge.root
    )

    run = workflow.start(
        HIERARCHICAL_REQUEST,
        "base",
        planning_strategy="hierarchical",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run
    chain = workflow._load_revision_chain(run.run_id)

    assert run.status == ScientificProblemRunStatus.APPROVED
    assert chain is not None and chain.revisions_used == 1
    assert workflow.runs.resolve_artifact_path(
        run.run_id, "hierarchical-planning/directions.yaml"
    ).is_file()
    assert len(
        tuple(
            workflow.runs.resolve_artifact_path(
                run.run_id, "hierarchical-planning"
            ).glob("directions.yaml")
        )
    ) == 1


def test_insufficient_evidence_replans_hierarchy_and_requires_fresh_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_knowledge_retrieval import _prepare

    knowledge, *_ = _prepare(tmp_path)
    CountingHierarchicalProvider.directions_calls = 0
    CountingHierarchicalProvider.triage_calls = 0
    CountingHierarchicalProvider.expansion_calls = 0
    InsufficientThenApproveProvider.review_calls = 0
    monkeypatch.setattr(
        workflow_service, "MockPlanningProvider", CountingHierarchicalProvider
    )
    monkeypatch.setattr(
        workflow_service, "MockApprovalProvider", InsufficientThenApproveProvider
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=knowledge.root
    )

    run = workflow.start(
        HIERARCHICAL_REQUEST,
        "base",
        planning_strategy="hierarchical",
        approval_provider="mock",
        max_evidence_resolution_cycles=1,
    ).run

    assert run.status == ScientificProblemRunStatus.APPROVED
    assert InsufficientThenApproveProvider.review_calls == 2
    assert CountingHierarchicalProvider.directions_calls == 2
    assert CountingHierarchicalProvider.triage_calls == 2
    assert CountingHierarchicalProvider.expansion_calls == 2


def test_evidence_replan_does_not_reset_global_revision_attempt_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_knowledge_retrieval import _prepare

    knowledge, *_ = _prepare(tmp_path)
    RevisionThenEvidenceApproval.review_calls = 0
    monkeypatch.setattr(
        workflow_service, "MockPlanningProvider", BaselineRepairHierarchicalProvider
    )
    monkeypatch.setattr(
        workflow_service, "MockApprovalProvider", RevisionThenEvidenceApproval
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=knowledge.root
    )

    run = workflow.start(
        HIERARCHICAL_REQUEST,
        "base",
        planning_strategy="hierarchical",
        approval_provider="mock",
        max_plan_revisions=1,
        max_evidence_resolution_cycles=1,
    ).run
    attempts = workflow._load_revision_attempts(run.run_id)

    assert run.status == ScientificProblemRunStatus.REVISION_BLOCKED
    assert run.failure is not None
    assert run.failure.category == "REVISION_BUDGET_EXHAUSTED"
    assert len(attempts) == 1
    assert RevisionThenEvidenceApproval.review_calls == 3


def test_human_choice_blocks_expansion_and_resume_does_not_cross_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_knowledge_retrieval import _prepare

    knowledge, *_ = _prepare(tmp_path)
    RetainAndHumanChoiceProvider.directions_calls = 0
    RetainAndHumanChoiceProvider.triage_calls = 0
    RetainAndHumanChoiceProvider.expansion_calls = 0
    monkeypatch.setattr(
        workflow_service, "MockPlanningProvider", RetainAndHumanChoiceProvider
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=knowledge.root
    )
    run = workflow.start(
        HIERARCHICAL_REQUEST,
        "base",
        planning_strategy="hierarchical",
    ).run
    before = (
        RetainAndHumanChoiceProvider.directions_calls,
        RetainAndHumanChoiceProvider.triage_calls,
        RetainAndHumanChoiceProvider.expansion_calls,
    )

    resumed = workflow.resume(run.run_id).run

    assert run.status == ScientificProblemRunStatus.HIERARCHICAL_PLANNING_BLOCKED
    assert resumed.status == ScientificProblemRunStatus.HIERARCHICAL_PLANNING_BLOCKED
    assert RetainAndHumanChoiceProvider.expansion_calls == 0
    assert (
        RetainAndHumanChoiceProvider.directions_calls,
        RetainAndHumanChoiceProvider.triage_calls,
        RetainAndHumanChoiceProvider.expansion_calls,
    ) == before


def test_negative_observation_is_visible_but_does_not_become_a_claim(
    tmp_path: Path,
) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    observable = ReportedObservable(
        observable_key=plan.tasks[0].outputs[0],
        quantity=plan.tasks[0].outputs[0],
        qualitative_value="not detected",
        result_status=ResultStatus.COMPUTED_REPORTED,
    )
    submission, artifact_root, _ = _submission(
        tmp_path,
        run,
        plan,
        result_type=ResultEvidenceType.NULL_OR_NEGATIVE_RESULT,
        outcome=ResultExecutionOutcome.NULL_OR_NEGATIVE,
        observables=(observable,),
        untrusted_interpretation="this proves mechanism A is false",
    )
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).compile(parent_run_id=run.run_id, submission_ids=(submission.submission_id,))

    planning_input = workflow.runs.load_artifact(
        run.run_id,
        "successor-planning/cycle-1/planning-input.yaml",
        ScientificPlanningInput,
    )
    packet = workflow.runs.load_artifact(
        run.run_id,
        "successor-planning/cycle-1/evidence-packet.yaml",
        ScientificEvidencePacket,
    )
    assert any(
        item.qualitative_value == "not detected"
        for item in planning_input.reported_observations
    )
    assert all(
        "this proves mechanism A is false" not in item.text
        for item in packet.source_claims
    )


def test_failed_execution_creates_no_scientific_result_or_observation(
    tmp_path: Path,
) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(
        tmp_path,
        run,
        plan,
        result_type=ResultEvidenceType.FAILED_EXECUTION,
        outcome=ResultExecutionOutcome.FAILED,
        observables=(),
    )
    _, receipt = _intake_and_accept(
        tmp_path, repositories, run, submission, artifact_root
    )

    assert not receipt.reported_result_hashes
    assert not receipt.reported_observation_hashes
    assert receipt.parent_plan_hash == content_hash(plan)
