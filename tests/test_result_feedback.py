from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from spc.approval import MockApprovalProvider
from spc.models import (
    ApprovalDecision,
    CurationStatus,
    FingerprintDifference,
    IndependentApprovalReceipt,
    ReportedObservable,
    ResearchUpdate,
    ResultArtifactManifestEntry,
    ResultContextComparisonStatus,
    ResultEvidenceIntakeStatus,
    ResultEvidenceType,
    ResultExecutionOutcome,
    ResultStatus,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
    SuccessorParentBinding,
)
from spc.result_feedback import (
    ResultFeedbackError,
    ResultEvidenceIntakeService,
    SuccessorPlanningService,
    build_result_evidence_submission,
    result_feedback_status,
)
from spc.serialization import content_hash, file_sha256
from spc.workflow import ScientificProblemWorkflow
from test_knowledge_retrieval import _prepare


REQUEST = (
    "The mechanism is not sufficiently convincing without a distinguishing "
    "observation."
)


class RejectSuccessorApproval(MockApprovalProvider):
    def review(self, review_input):
        return super().review(review_input).model_copy(
            update={"decision_recommendation": ApprovalDecision.REJECT}
        )


def _approved_parent(tmp_path: Path, *, domain: str = "base"):
    repositories, *_ = _prepare(tmp_path)
    state_dir = tmp_path / ".spc"
    workflow = ScientificProblemWorkflow(
        state_dir=state_dir,
        knowledge_dir=repositories.root,
    )
    run = workflow.start(REQUEST, domain, approval_provider="mock").run
    assert run.status == ScientificProblemRunStatus.APPROVED
    plan = workflow.runs.load_artifact(
        run.run_id,
        run.candidate_plans[0].relative_path,
        ScientificQuestionPlan,
    )
    return repositories, workflow, run, plan


def _submission(
    tmp_path: Path,
    run,
    plan: ScientificQuestionPlan,
    *,
    result_type: ResultEvidenceType = ResultEvidenceType.COMPUTED_RESULT,
    outcome: ResultExecutionOutcome = ResultExecutionOutcome.COMPLETED,
    observables: tuple[ReportedObservable, ...] | None = None,
    task_id: str | None = None,
    capability_id: str | None = None,
    parent_plan_id: str | None = None,
    parent_plan_hash: str | None = None,
    system_fingerprint=None,
    method_fingerprint=None,
    declared_deviations: tuple[FingerprintDifference, ...] = (),
    untrusted_interpretation: str | None = None,
):
    artifact_root = tmp_path / "downstream-result"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact = artifact_root / "result.txt"
    artifact.write_text("raw downstream result bytes\n", encoding="utf-8")
    task = plan.tasks[0]
    if observables is None:
        observables = (
            ReportedObservable(
                observable_key=task.outputs[0],
                quantity=task.outputs[0],
                value=1.25,
                unit="eV",
                uncertainty="not independently evaluated by SPC",
                result_status=ResultStatus.COMPUTED_REPORTED,
            ),
        )
    manifest = (
        ResultArtifactManifestEntry(
            relative_path="result.txt",
            sha256=file_sha256(artifact),
            size_bytes=artifact.stat().st_size,
            media_type="text/plain",
        ),
    )
    submission = build_result_evidence_submission(
        parent_run_id=run.run_id,
        parent_plan_id=parent_plan_id or plan.plan_id,
        parent_plan_version=plan.version,
        parent_plan_hash=parent_plan_hash or content_hash(plan),
        task_id=task_id or task.task_id,
        capability_id=capability_id or task.capability_id,
        executor_id="offline-fixture-executor",
        executor_version="1.0.0",
        result_type=result_type,
        system_fingerprint=system_fingerprint or plan.system_fingerprint,
        method_fingerprint=method_fingerprint or plan.method_fingerprint,
        declared_deviations=declared_deviations,
        source_artifact_manifest=manifest,
        raw_artifact_checksums={"result.txt": manifest[0].sha256},
        reported_observables=observables,
        uncertainty_metadata={"basis": "fixture declaration"},
        convergence_metadata={"converged": result_type != ResultEvidenceType.FAILED_EXECUTION},
        execution_outcome=outcome,
        untrusted_interpretation=untrusted_interpretation,
        provenance_timestamp=datetime(2026, 9, 22, tzinfo=timezone.utc),
    )
    return submission, artifact_root, artifact


def _intake_and_accept(tmp_path: Path, repositories, run, submission, artifact_root):
    service = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    )
    assessment = service.intake(submission, artifact_root=artifact_root)
    assert assessment.status == ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
    curation, receipt = service.curate(
        parent_run_id=run.run_id,
        submission_id=submission.submission_id,
        status=CurationStatus.ACCEPTED,
        curator_id="human-curator",
        rationale="Accept provenance-valid result as project evidence for successor planning.",
    )
    assert curation.status == CurationStatus.ACCEPTED
    assert receipt is not None
    return assessment, receipt


def test_approved_plan_task_bound_computed_result_requires_explicit_curation(
    tmp_path: Path,
) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    service = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    assessment = service.intake(submission, artifact_root=artifact_root)

    assert assessment.content_bound is True
    assert assessment.plan_bound is True
    assert assessment.context_compatible is True
    assert assessment.scientifically_admissible is True
    assert assessment.status == ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
    status = result_feedback_status(run.run_id, workflow.runs)
    assert status["result_evidence"]["accepted"] == 0
    assert status["result_evidence"]["pending_curation"] == 1
    assert status["result_evidence"]["rejected_or_blocked"] == 0

    _, receipt = service.curate(
        parent_run_id=run.run_id,
        submission_id=submission.submission_id,
        status=CurationStatus.ACCEPTED,
        curator_id="human-curator",
        rationale="Explicit project-result acceptance.",
    )
    assert receipt is not None
    assert receipt.reported_result_hashes
    assert receipt.accepted_evidence_ids


@pytest.mark.parametrize(
    ("overrides", "issue_code"),
    (
        ({"task_id": "fabricated-task"}, "FABRICATED_TASK_ID"),
        ({"capability_id": "fabricated-capability"}, "WRONG_CAPABILITY"),
        ({"parent_plan_hash": "0" * 64}, "PARENT_PLAN_BINDING_INVALID"),
        ({"parent_plan_id": "plan-from-another-run"}, "PARENT_PLAN_BINDING_INVALID"),
    ),
)
def test_cross_plan_stale_or_fabricated_bindings_fail_closed(
    tmp_path: Path, overrides: dict[str, str], issue_code: str
) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(
        tmp_path, run, plan, **overrides
    )
    assessment = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).intake(submission, artifact_root=artifact_root)

    assert assessment.status == ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING
    assert issue_code in {item.code for item in assessment.issues}


def test_tampered_raw_artifact_checksum_blocks_intake(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, artifact = _submission(tmp_path, run, plan)
    artifact.write_text("tampered after manifest\n", encoding="utf-8")

    assessment = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).intake(submission, artifact_root=artifact_root)

    assert assessment.status == ResultEvidenceIntakeStatus.BLOCKED_CONTENT_BINDING
    assert "RESULT_ARTIFACT_INTEGRITY_FAILURE" in {
        item.code for item in assessment.issues
    }


def test_cross_run_result_cannot_bind_a_plan_from_another_run(tmp_path: Path) -> None:
    repositories, workflow, first_run, first_plan = _approved_parent(tmp_path)
    second_run = workflow.start(
        "Which independent observation should be measured next?",
        "base",
        approval_provider="mock",
    ).run
    assert second_run.run_id != first_run.run_id
    submission, artifact_root, _ = _submission(
        tmp_path / "cross-run", second_run, first_plan
    )

    assessment = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).intake(submission, artifact_root=artifact_root)

    assert assessment.status == ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING
    assert "PARENT_PLAN_BINDING_INVALID" in {
        item.code for item in assessment.issues
    }


def test_blocked_intake_cannot_be_curated_as_accepted(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, artifact = _submission(tmp_path, run, plan)
    artifact.write_text("tampered after manifest\n", encoding="utf-8")
    service = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )
    service.intake(submission, artifact_root=artifact_root)

    with pytest.raises(ResultFeedbackError, match="RESULT_NOT_ADMISSIBLE_FOR_ACCEPTANCE"):
        service.curate(
            parent_run_id=run.run_id,
            submission_id=submission.submission_id,
            status=CurationStatus.ACCEPTED,
            curator_id="human-curator",
            rationale="This must not bypass deterministic intake failures.",
        )


def test_archived_result_tampering_before_curation_is_rejected(tmp_path: Path) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    service = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )
    assessment = service.intake(submission, artifact_root=artifact_root)
    assert assessment.status == ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
    archived = workflow.runs.resolve_artifact_path(
        run.run_id,
        "result-evidence/submissions/"
        f"{submission.submission_id}/raw/result.txt",
    )
    archived.chmod(0o644)
    archived.write_text("tampered before curation\n", encoding="utf-8")

    with pytest.raises(ResultFeedbackError, match="TAMPERED_ACCEPTED_RESULT_ARTIFACT"):
        service.curate(
            parent_run_id=run.run_id,
            submission_id=submission.submission_id,
            status=CurationStatus.ACCEPTED,
            curator_id="human-curator",
            rationale="Must be rejected after raw evidence tampering.",
        )


def test_undeclared_system_or_method_change_blocks_intake(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    changed_system = plan.system_fingerprint.model_copy(
        update={"attributes": {**dict(plan.system_fingerprint.attributes), "domain": "other"}}
    )
    submission, artifact_root, _ = _submission(
        tmp_path, run, plan, system_fingerprint=changed_system
    )

    assessment = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).intake(submission, artifact_root=artifact_root)

    assert assessment.status == ResultEvidenceIntakeStatus.BLOCKED_CONTEXT_COMPATIBILITY
    assert assessment.context_comparison is not None
    assert (
        assessment.context_comparison.status
        == ResultContextComparisonStatus.UNDECLARED_SYSTEM_CHANGE
    )


def test_explicit_plan_approved_deviation_is_preserved(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path)
    changed_system = plan.system_fingerprint.model_copy(
        update={"attributes": {**dict(plan.system_fingerprint.attributes), "domain": "other"}}
    )
    difference = FingerprintDifference(
        field="system.domain",
        left=plan.system_fingerprint.attributes["domain"],
        right="other",
        disclosed_deviation=True,
    )
    plan_with_approved_deviation = plan.model_copy(
        update={"fingerprint_differences": (difference,)}
    )
    submission, _, _ = _submission(
        tmp_path,
        run,
        plan,
        system_fingerprint=changed_system,
        declared_deviations=(difference,),
    )

    comparison = ResultEvidenceIntakeService._comparison(
        submission, plan_with_approved_deviation
    )

    assert (
        comparison.status
        == ResultContextComparisonStatus.EXPLICIT_APPROVED_DEVIATION
    )
    assert comparison.approved_deviation_refs


@pytest.mark.parametrize(
    ("approved_left", "approved_right", "actual_right"),
    (
        ("PBE", "RPBE", "SCAN"),
        ("LDA", "RPBE", "RPBE"),
    ),
)
def test_fingerprint_deviation_requires_exact_approved_values(
    tmp_path: Path,
    approved_left: str,
    approved_right: str,
    actual_right: str,
) -> None:
    _, _, run, plan = _approved_parent(tmp_path)
    base_attributes = {**dict(plan.method_fingerprint.attributes), "functional": "PBE"}
    expected_plan = plan.model_copy(
        update={
            "method_fingerprint": plan.method_fingerprint.model_copy(
                update={"attributes": base_attributes}
            ),
            "fingerprint_differences": (
                FingerprintDifference(
                    field="method_context.functional",
                    left=approved_left,
                    right=approved_right,
                    disclosed_deviation=True,
                ),
            ),
        }
    )
    actual = expected_plan.method_fingerprint.model_copy(
        update={"attributes": {**base_attributes, "functional": actual_right}}
    )
    declared = FingerprintDifference(
        field="method.functional",
        left="PBE",
        right=actual_right,
        disclosed_deviation=True,
    )
    submission, _, _ = _submission(
        tmp_path,
        run,
        expected_plan,
        method_fingerprint=actual,
        declared_deviations=(declared,),
    )

    comparison = ResultEvidenceIntakeService._comparison(submission, expected_plan)

    assert (
        comparison.status
        == ResultContextComparisonStatus.UNDECLARED_METHOD_CHANGE
    )


def test_one_wrong_value_blocks_multiple_declared_deviations(tmp_path: Path) -> None:
    _, _, run, plan = _approved_parent(tmp_path)
    base = {
        **dict(plan.method_fingerprint.attributes),
        "functional": "PBE",
        "dispersion": "none",
    }
    approved = (
        FingerprintDifference(
            field="method.functional",
            left="PBE",
            right="RPBE",
            disclosed_deviation=True,
        ),
        FingerprintDifference(
            field="method.dispersion",
            left="none",
            right="D3",
            disclosed_deviation=True,
        ),
    )
    expected_plan = plan.model_copy(
        update={
            "method_fingerprint": plan.method_fingerprint.model_copy(
                update={"attributes": base}
            ),
            "fingerprint_differences": approved,
        }
    )
    actual = expected_plan.method_fingerprint.model_copy(
        update={
            "attributes": {
                **base,
                "functional": "RPBE",
                "dispersion": "D4",
            }
        }
    )
    declared = (
        approved[0],
        FingerprintDifference(
            field="method.dispersion",
            left="none",
            right="D4",
            disclosed_deviation=True,
        ),
    )
    submission, _, _ = _submission(
        tmp_path,
        run,
        expected_plan,
        method_fingerprint=actual,
        declared_deviations=declared,
    )

    comparison = ResultEvidenceIntakeService._comparison(submission, expected_plan)

    assert (
        comparison.status
        == ResultContextComparisonStatus.UNDECLARED_METHOD_CHANGE
    )


def test_failed_execution_never_materializes_reported_result(tmp_path: Path) -> None:
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
    assert (
        receipt.parent_run_id,
        receipt.parent_plan_id,
        receipt.parent_plan_hash,
        receipt.task_id,
    ) == (run.run_id, plan.plan_id, content_hash(plan), plan.tasks[0].task_id)

    assert not receipt.reported_result_hashes
    assert not receipt.reported_observation_hashes
    assert receipt.accepted_evidence_ids
    assert receipt.missing_observables == plan.tasks[0].outputs


def test_partial_and_null_results_are_preserved_without_fabricating_missing_values(
    tmp_path: Path,
) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    task = plan.tasks[0]
    partial_observable = ReportedObservable(
        observable_key="partial-observable",
        quantity="partial-observable",
        value=0.0,
        unit="eV",
        result_status=ResultStatus.COMPUTED_REPORTED,
    )
    partial, partial_root, _ = _submission(
        tmp_path / "partial",
        run,
        plan,
        result_type=ResultEvidenceType.PARTIAL_RESULT,
        outcome=ResultExecutionOutcome.PARTIAL,
        observables=(partial_observable,),
    )
    _, partial_receipt = _intake_and_accept(
        tmp_path, repositories, run, partial, partial_root
    )
    assert task.outputs[0] in partial_receipt.missing_observables
    assert partial_receipt.reported_result_hashes

    negative_observable = ReportedObservable(
        observable_key=task.outputs[0],
        quantity=task.outputs[0],
        qualitative_value="not detected",
        result_status=ResultStatus.COMPUTED_REPORTED,
    )
    negative, negative_root, _ = _submission(
        tmp_path / "negative",
        run,
        plan,
        result_type=ResultEvidenceType.NULL_OR_NEGATIVE_RESULT,
        outcome=ResultExecutionOutcome.NULL_OR_NEGATIVE,
        observables=(negative_observable,),
        untrusted_interpretation="this proves mechanism A is false",
    )
    _, negative_receipt = _intake_and_accept(
        tmp_path, repositories, run, negative, negative_root
    )
    assert negative_receipt.accepted_evidence_ids
    assert not negative_receipt.reported_result_hashes
    assert negative_receipt.reported_observation_hashes

    SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).compile(
        parent_run_id=run.run_id,
        submission_ids=(negative.submission_id,),
        follow_up_request="Assess the reported absence without treating it as proof.",
    )
    successor_input = workflow.runs.load_artifact(
        run.run_id,
        "successor-planning/cycle-1/planning-input.yaml",
        ScientificPlanningInput,
    )
    observations = getattr(successor_input, "reported_observations", ())
    assert any(
        item.quantity == task.outputs[0]
        and item.qualitative_value == "not detected"
        for item in observations
    )
    research_update = workflow.runs.load_artifact(
        run.run_id,
        "successor-planning/cycle-1/research-update.yaml",
        ResearchUpdate,
    )
    assert f"{task.outputs[0]} = not detected" in (
        research_update.new_reported_observations
    )
    successor_packet = workflow.runs.load_artifact(
        run.run_id,
        "successor-planning/cycle-1/evidence-packet.yaml",
        ScientificEvidencePacket,
    )
    assert all(
        "this proves mechanism A is false" not in claim.text
        for claim in successor_packet.source_claims
    )


def test_accepted_result_builds_new_context_input_and_independently_approved_successor(
    tmp_path: Path,
) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(
        tmp_path,
        run,
        plan,
        untrusted_interpretation="This result proves mechanism A is true.",
    )
    _, receipt = _intake_and_accept(
        tmp_path, repositories, run, submission, artifact_root
    )

    cycle = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).compile(
        parent_run_id=run.run_id,
        submission_ids=(submission.submission_id,),
        follow_up_request="Plan one additional discriminating test without treating the result as a conclusion.",
    )

    assert cycle.status.value == "APPROVED"
    assert cycle.context_id != run.context_id
    assert cycle.planning_input_id != run.planning_input_id
    assert cycle.selected_plan_id != plan.plan_id
    assert cycle.approval_receipt_id != run.approval_receipt_id
    prefix = "successor-planning/cycle-1"
    successor_plan = workflow.runs.load_artifact(
        run.run_id,
        f"{prefix}/candidates/{cycle.selected_plan_id}.yaml",
        ScientificQuestionPlan,
    )
    assert successor_plan.follow_up_of == plan.plan_id
    assert all(not task.runnable for task in successor_plan.tasks)
    assert set(receipt.accepted_evidence_ids).issubset(
        {item.evidence_id for item in successor_plan.evidence_refs}
    )
    assert any(
        item == f"successor-parent-binding:{cycle.parent_binding_id}"
        for item in successor_plan.source_query_manifest
    )
    successor_context = workflow.runs.load_artifact(
        run.run_id, f"{prefix}/context.yaml", ScientificContextPacket
    )
    successor_packet = workflow.runs.load_artifact(
        run.run_id, f"{prefix}/evidence-packet.yaml", ScientificEvidencePacket
    )
    successor_input = workflow.runs.load_artifact(
        run.run_id, f"{prefix}/planning-input.yaml", ScientificPlanningInput
    )
    assert set(receipt.accepted_evidence_ids).issubset(
        successor_input.allowed_evidence_ids
    )
    assert successor_packet.context_id == successor_context.context_id
    assert successor_packet.reported_results
    assert successor_packet.source_claims == workflow.runs.load_artifact(
        run.run_id, "evidence-packet.yaml", ScientificEvidencePacket
    ).source_claims
    assert successor_packet.provenance_manifest["result_submission_ids"] == (
        submission.submission_id,
    )
    status = result_feedback_status(run.run_id, workflow.runs)
    assert status["result_evidence"]["accepted"] == 1
    assert status["successor_planning"]["latest_status"] == "APPROVED"


def test_approved_successor_is_exact_parent_of_second_result_cycle(
    tmp_path: Path,
) -> None:
    repositories, workflow, run, plan0 = _approved_parent(tmp_path)
    result0, root0, _ = _submission(tmp_path / "result-0", run, plan0)
    _intake_and_accept(tmp_path, repositories, run, result0, root0)
    cycle1 = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).compile(
        parent_run_id=run.run_id,
        submission_ids=(result0.submission_id,),
        follow_up_request="Plan the first bounded follow-up.",
    )
    assert cycle1.status.value == "APPROVED"
    plan1 = workflow.runs.load_artifact(
        run.run_id,
        f"successor-planning/cycle-1/candidates/{cycle1.selected_plan_id}.yaml",
        ScientificQuestionPlan,
    )

    result1, root1, _ = _submission(
        tmp_path / "result-1",
        run,
        plan1,
        observables=(
            ReportedObservable(
                observable_key=plan1.tasks[0].outputs[0],
                quantity="second-cycle discriminating observation",
                value=2.5,
                unit="eV",
                result_status=ResultStatus.COMPUTED_REPORTED,
            ),
        ),
    )
    assessment1, receipt1 = _intake_and_accept(
        tmp_path, repositories, run, result1, root1
    )
    assert assessment1.parent_plan_id == plan1.plan_id
    assert assessment1.parent_authority_origin == "successor_cycle"
    assert receipt1.parent_authority_origin == "successor_cycle"
    cycle2 = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).compile(
        parent_run_id=run.run_id,
        submission_ids=(result1.submission_id,),
        follow_up_request="Plan the second bounded follow-up.",
    )
    assert cycle2.status.value == "APPROVED"
    plan2 = workflow.runs.load_artifact(
        run.run_id,
        f"successor-planning/cycle-2/candidates/{cycle2.selected_plan_id}.yaml",
        ScientificQuestionPlan,
    )
    binding2 = workflow.runs.load_artifact(
        run.run_id,
        "successor-planning/cycle-2/parent-binding.yaml",
        SuccessorParentBinding,
    )
    assert binding2.parent_plan_id == plan1.plan_id
    assert binding2.parent_plan_hash == content_hash(plan1)
    assert plan2.follow_up_of == plan1.plan_id
    assert plan2.plan_id != plan1.plan_id != plan0.plan_id
    assert all(not task.runnable for task in plan2.tasks)
    approval1 = workflow.runs.load_artifact(
        run.run_id,
        "successor-planning/cycle-1/approval/independent-approval-receipt.yaml",
        IndependentApprovalReceipt,
    )
    approval2 = workflow.runs.load_artifact(
        run.run_id,
        "successor-planning/cycle-2/approval/independent-approval-receipt.yaml",
        IndependentApprovalReceipt,
    )
    assert approval2.receipt_id != approval1.receipt_id

    cross_cycle, cross_root, _ = _submission(
        tmp_path / "cross-cycle",
        run,
        plan1,
        parent_plan_hash=content_hash(plan2),
    )
    cross_assessment = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).intake(cross_cycle, artifact_root=cross_root)
    assert cross_assessment.status == ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING


def test_rejected_successor_cannot_be_result_parent(tmp_path: Path) -> None:
    repositories, workflow, run, plan0 = _approved_parent(tmp_path)
    result0, root0, _ = _submission(tmp_path / "result-0", run, plan0)
    _intake_and_accept(tmp_path, repositories, run, result0, root0)
    cycle = SuccessorPlanningService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).compile(
        parent_run_id=run.run_id,
        submission_ids=(result0.submission_id,),
        follow_up_request="Propose a plan that the independent reviewer rejects.",
        approval_provider=RejectSuccessorApproval(),
    )
    assert cycle.status.value == "REJECTED"
    rejected_plan = workflow.runs.load_artifact(
        run.run_id,
        f"successor-planning/cycle-1/candidates/{cycle.selected_plan_id}.yaml",
        ScientificQuestionPlan,
    )
    rejected_result, rejected_root, _ = _submission(
        tmp_path / "rejected-result", run, rejected_plan
    )

    assessment = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    ).intake(rejected_result, artifact_root=rejected_root)

    assert assessment.status == ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING
    assert "PARENT_SUCCESSOR_NOT_APPROVED" in {
        issue.code for issue in assessment.issues
    }


def test_tampered_accepted_raw_artifact_blocks_successor(tmp_path: Path) -> None:
    repositories, workflow, run, plan = _approved_parent(tmp_path)
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    _intake_and_accept(tmp_path, repositories, run, submission, artifact_root)
    archived = workflow.runs.resolve_artifact_path(
        run.run_id,
        "result-evidence/submissions/"
        f"{submission.submission_id}/raw/result.txt",
    )
    archived.chmod(0o644)
    archived.write_text("tampered accepted bytes\n", encoding="utf-8")

    with pytest.raises(ResultFeedbackError, match="TAMPERED_ACCEPTED_RESULT_ARTIFACT"):
        SuccessorPlanningService(
            state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
        ).compile(
            parent_run_id=run.run_id,
            submission_ids=(submission.submission_id,),
            follow_up_request="Plan a bounded follow-up.",
        )


def test_result_from_old_revision_round_is_not_rebound_to_latest_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_plan_revision import _chain, _workflow

    workflow = _workflow(tmp_path, monkeypatch)
    run = workflow.start(
        REQUEST,
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run
    chain = _chain(workflow, run.run_id)
    assert len(chain.rounds) == 2
    old_plan = workflow.runs.load_artifact(
        run.run_id,
        chain.rounds[0].candidate_plan.relative_path,
        ScientificQuestionPlan,
    )
    latest_plan = workflow.runs.load_artifact(
        run.run_id,
        chain.rounds[1].candidate_plan.relative_path,
        ScientificQuestionPlan,
    )
    assert old_plan.plan_id != latest_plan.plan_id

    old_submission, old_root, _ = _submission(
        tmp_path / "old-round", run, old_plan
    )
    old_assessment = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=workflow.knowledge_dir
    ).intake(old_submission, artifact_root=old_root)
    assert old_assessment.status == ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING
    assert "PARENT_GATE_INVALID" in {item.code for item in old_assessment.issues}

    latest_submission, latest_root, _ = _submission(
        tmp_path / "latest-round", run, latest_plan
    )
    latest_assessment = ResultEvidenceIntakeService(
        state_dir=tmp_path / ".spc", knowledge_dir=workflow.knowledge_dir
    ).intake(latest_submission, artifact_root=latest_root)
    assert (
        latest_assessment.status
        == ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
    )


def test_generic_base_domain_result_feedback_does_not_add_ft_logic(tmp_path: Path) -> None:
    repositories, _, run, plan = _approved_parent(tmp_path, domain="base")
    submission, artifact_root, _ = _submission(tmp_path, run, plan)
    assessment, _ = _intake_and_accept(
        tmp_path, repositories, run, submission, artifact_root
    )

    assert assessment.status == ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
    assert plan.domain == "base"
