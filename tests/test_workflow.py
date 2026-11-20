from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from spc.cli import app
from spc.knowledge.trust import TrustedKnowledgeValidator
from spc.models import (
    CurationStatus,
    EvidenceSpan,
    ScientificProblemRun,
    ScientificProblemRunStatus,
    ScientificPlanningInput,
    ScientificQuestionPlan,
)
from spc.repositories import ProjectEvidenceStore
from spc.serialization import content_hash, load_data
from spc.workflow import (
    ScientificProblemWorkflow,
    render_scientific_run_markdown,
)
from test_knowledge_retrieval import _prepare


REQUEST = "The mechanism is not sufficiently convincing without a distinguishing observation."


def test_workflow_service_compatibility_exports() -> None:
    from spc.workflow import service
    from spc.workflow.repository import ScientificProblemRunRepository as Repository
    from spc.workflow.service import ScientificProblemRunRepository

    assert ScientificProblemRunRepository is Repository
    assert "ScientificProblemRunRepository" in service.__all__
    for name in service.__all__:
        assert hasattr(service, name), name


def test_trusted_prepared_knowledge_dry_run_is_offline_and_bounded(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    )

    result = workflow.start(REQUEST, "base", dry_run=True)

    assert result.run.status == ScientificProblemRunStatus.CONTEXT_BUILT
    assert result.dry_run_report is not None
    assert result.dry_run_report["model_inference_performed"] is False
    assert result.dry_run_report["trusted_literature_count"] == 1
    assert result.dry_run_report["trusted_expert_case_count"] == 1
    assert result.dry_run_report["retrieval_hit_counts"]["literature"] > 0


def test_dry_run_reuses_its_verified_snapshot_for_status_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )
    original_validate = TrustedKnowledgeValidator.validate
    calls = 0

    def counted_validate(self: TrustedKnowledgeValidator):
        nonlocal calls
        calls += 1
        return original_validate(self)

    monkeypatch.setattr(TrustedKnowledgeValidator, "validate", counted_validate)
    result = workflow.start(REQUEST, "base", dry_run=True)
    assert result.run.status == ScientificProblemRunStatus.CONTEXT_BUILT
    assert result.dry_run_report is not None
    assert result.dry_run_report["trusted_literature_count"] == 1
    assert calls == 1


def test_chinese_fe110_barrier_request_reaches_grounded_nonrunnable_candidate(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / ".spc"
    store = ProjectEvidenceStore(state_dir)
    statement = (
        "A study defines CO adsorbed on Fe(110) and separated C and O on the "
        "same surface as endpoints for a dissociation barrier analysis."
    )
    source_path = tmp_path / "synthetic-method-note.txt"
    source_path.write_text(statement, encoding="utf-8")
    source = store.ingest(
        source_path,
        "synthetic-fe110-method-note",
        "v1",
        source_role="internal_researcher",
        source_type="internal_note",
    )
    store.add_evidence(
        EvidenceSpan(
            evidence_id="synthetic-fe110-evidence",
            source_id=source.source_id,
            source_version=source.version,
            content_sha256=source.content_sha256,
            start_offset=0,
            end_offset=len(statement),
            text=statement,
        )
    )
    workflow = ScientificProblemWorkflow(
        state_dir=state_dir, knowledge_dir=tmp_path / "knowledge"
    )
    run = workflow.start(
        "我要算一个Fe110表面CO解离的能垒 怎么做", "fischer_tropsch"
    ).run
    assert run.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    assert run.candidate_plans
    planning_input = workflow.runs.load_artifact(
        run.run_id, "planning-input.yaml", ScientificPlanningInput
    )
    assert planning_input.evidence_gaps
    for candidate in run.candidate_plans:
        plan = workflow.runs.load_artifact(
            run.run_id, candidate.relative_path, ScientificQuestionPlan
        )
        assert "Fe(110)" in plan.model.description.text
        assert "activation barrier" in plan.observables[0].description.text
        assert plan.tasks[0].capability_id == "ft_elementary_step_barrier_plan"
        assert plan.tasks[0].outputs == ("non_runnable_barrier_plan",)
        assert all(not task.runnable for task in plan.tasks)


def test_missing_curation_is_actionable_and_does_not_enter_context(tmp_path: Path) -> None:
    repositories, *_ = _prepare(
        tmp_path,
        claim_status=CurationStatus.MACHINE_EXTRACTED,
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    )

    result = workflow.start(REQUEST, "base", dry_run=True)

    assert result.run.status == ScientificProblemRunStatus.BLOCKED_SOURCE_CURATION
    assert result.run.context_id is None
    assert result.run.blocking_items
    assert any(
        item.target_type == "source_claim"
        and item.current_status == CurationStatus.MACHINE_EXTRACTED
        and "requires ACCEPTED curation" in item.message
        for item in result.run.blocking_items
    )


def test_unified_run_persists_stage_hashes_and_preserves_fingerprints(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    )

    run = workflow.start(REQUEST, "base").run

    assert run.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    assert run.knowledge_snapshot_id and run.knowledge_snapshot_hash
    assert run.context_id and run.context_hash
    assert run.evidence_packet_id and run.evidence_packet_hash
    assert run.planning_input_id and run.planning_input_hash
    assert run.planning_proposal_id and run.planning_proposal_hash
    assert run.candidate_plans and run.candidate_compilation_receipts
    plan = workflow.runs.load_artifact(
        run.run_id,
        run.candidate_plans[0].relative_path,
        ScientificQuestionPlan,
    )
    assert plan.intent_fingerprint.fingerprint_id
    assert plan.system_fingerprint.fingerprint_id
    assert plan.method_fingerprint.fingerprint_id
    assert all(not task.runnable for task in plan.tasks)
    context_text = workflow.runs.resolve_artifact_path(run.run_id, "context.yaml").read_text(
        encoding="utf-8"
    )
    assert "literature_knowledge_hits" in context_text
    assert "expert_case_hits" in context_text


def test_resume_reuses_valid_provider_outputs_and_rebuilds_stale_context(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    state_dir = tmp_path / ".spc"
    workflow = ScientificProblemWorkflow(state_dir=state_dir, knowledge_dir=repositories.root)
    first = workflow.start(REQUEST, "base").run
    first_candidate = first.candidate_plans[0].artifact_id

    repeated = workflow.resume(first.run_id).run
    assert repeated.context_hash == first.context_hash
    assert repeated.candidate_plans[0].artifact_id == first_candidate

    project_store = ProjectEvidenceStore(state_dir)
    source_path = tmp_path / "project-note.txt"
    source_path.write_text("Unrelated auditable project note.", encoding="utf-8")
    source = project_store.ingest(
        source_path,
        "source-new-project-note",
        "v1",
        source_role="internal_researcher",
        source_type="internal_note",
    )
    project_store.add_evidence(
        EvidenceSpan(
            evidence_id="ev-new-project-note",
            source_id=source.source_id,
            source_version=source.version,
            content_sha256=source.content_sha256,
            start_offset=0,
            end_offset=len("Unrelated auditable project note."),
            text="Unrelated auditable project note.",
        )
    )

    rebuilt = workflow.resume(first.run_id, dry_run=True).run
    assert rebuilt.knowledge_snapshot_id != first.knowledge_snapshot_id
    assert rebuilt.context_hash != first.context_hash


def test_provider_failure_keeps_last_successful_artifacts_recoverable(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    )

    failed = workflow.start(REQUEST, "base", planning_provider="llm").run

    assert failed.status == ScientificProblemRunStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.stage == "planning"
    assert failed.context_id is not None
    assert failed.evidence_packet_id is not None
    assert workflow.runs.resolve_artifact_path(failed.run_id, "context.yaml").is_file()
    assert workflow.runs.resolve_artifact_path(failed.run_id, "evidence-packet.yaml").is_file()

    resumed = workflow.resume(failed.run_id).run
    assert resumed.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    assert resumed.failure is None


def test_offline_approval_and_deterministic_report_preserve_provenance(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    )

    run = workflow.start(REQUEST, "base", approval_provider="mock").run
    report = render_scientific_run_markdown(run, workflow.runs)

    assert run.status == ScientificProblemRunStatus.APPROVED
    assert run.approval_receipt_id and run.gate_id
    assert REQUEST in report
    assert "Exact quote" in report
    assert "Expert cases" in report
    assert "IntentFingerprint" in report
    assert "SystemFingerprint" in report
    assert "MethodFingerprint" in report
    assert "Decision: `approve`" in report
    assert "Runnable tasks: False" in report
    plan = workflow.runs.load_artifact(
        run.run_id, run.candidate_plans[0].relative_path, ScientificQuestionPlan
    )
    assert f"- Scientific model: {plan.model.description.text}" in report
    assert "#### Model and observables" in report
    assert "#### Task handoff and acceptance checks" in report
    assert f"##### Step 1: `{plan.tasks[0].task_id}`" in report
    assert f"- Deliverables: {'; '.join(plan.tasks[0].outputs)}" in report
    assert f"- Success checks: {'; '.join(plan.tasks[0].success_criteria)}" in report
    assert f"- Failure action: {plan.tasks[0].failure_policy}" in report
    assert "- Runnable: False" in report

    resumed = workflow.resume(run.run_id).run
    assert resumed.status == ScientificProblemRunStatus.APPROVED
    assert resumed.approval_receipt_hash == run.approval_receipt_hash


def test_rejected_approval_blocks_downstream_export(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    )
    run = workflow.start(
        "The mechanism is not convincing; compare the DFT activation barrier.",
        "base",
        approval_provider="mock",
    ).run
    assert run.status == ScientificProblemRunStatus.REJECTED

    with pytest.raises(ValueError, match="independently APPROVED"):
        workflow.export_downstream(run.run_id, tmp_path / "exports", "blocked-export")


def test_approved_ft_run_reuses_immutable_non_runnable_downstream_export(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / ".spc"
    project_store = ProjectEvidenceStore(state_dir)
    text = "CO activation requires an evidence-grounded pathway comparison."
    source_path = tmp_path / "review.txt"
    source_path.write_text(text, encoding="utf-8")
    source = project_store.ingest(
        source_path,
        "source-review",
        "v1",
        source_role="author",
        source_type="manuscript",
    )
    project_store.add_evidence(
        EvidenceSpan(
            evidence_id="ev-review",
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
    ).run
    assert run.status == ScientificProblemRunStatus.APPROVED

    export_dir = workflow.export_downstream(
        run.run_id, tmp_path / "exports", "k1i-approved"
    )

    selected_plan = load_data(export_dir / "selected-plan.yaml")
    assert all(task["runnable"] is False for task in selected_plan["tasks"])
    for task_path in (export_dir / "tasks").glob("*.yaml"):
        assert load_data(task_path)["runnable"] is False


def test_scientific_problem_run_hash_is_fail_closed(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    run = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    ).start(REQUEST, "base", dry_run=True).run
    payload = run.model_dump(mode="json")
    payload["original_request"] = "tampered"

    with pytest.raises(ValidationError, match="run_id is not content-bound|content_hash is invalid"):
        ScientificProblemRun.model_validate(payload)


def test_unified_cli_dry_run_and_markdown_export(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    state_dir = tmp_path / ".spc"
    runner = CliRunner()

    result = runner.invoke(
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
            str(state_dir),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"model_inference_performed": false' in result.output
    run_id = f"scientific-run-{content_hash({'original_request': REQUEST, 'domain': 'base', 'state_dir': str(state_dir.resolve()), 'knowledge_dir': str(repositories.root.resolve())})[:24]}"
    report_path = tmp_path / "run-report.md"
    export_result = runner.invoke(
        app,
        [
            "export-scientific-run",
            "--run-id",
            run_id,
            "--state-dir",
            str(state_dir),
            "--format",
            "markdown",
            "--output",
            str(report_path),
        ],
    )
    assert export_result.exit_code == 0, export_result.output
    assert REQUEST in report_path.read_text(encoding="utf-8")


def test_legacy_entry_reuses_complete_outputs_with_explicit_approver(tmp_path, monkeypatch):
    from spc.approval import MockApprovalProvider
    from spc.planning import MockPlanningProvider
    from spc.retrieval import ScientificContextBuilder
    from spc.workflow import service

    knowledge, *_ = _prepare(tmp_path)
    assert service.ScientificProblemWorkflow is ScientificProblemWorkflow
    workflow = service.ScientificProblemWorkflow(state_dir=tmp_path / ".spc", knowledge_dir=knowledge.root)
    run = workflow.start(REQUEST, "base", approval_provider="mock").run
    root = workflow.runs.run_dir(run.run_id)
    artifacts = {path: path.read_bytes() for path in root.rglob("*.yaml")}

    def unexpected(*args, **kwargs):
        pytest.fail("a completed stage was invoked during resume")

    monkeypatch.setattr(ScientificContextBuilder, "build", unexpected)
    monkeypatch.setattr(MockPlanningProvider, "propose", unexpected)
    monkeypatch.setattr(MockApprovalProvider, "review", unexpected)
    resumed = workflow.resume(run.run_id, approval_provider="mock").run

    assert resumed == run
    assert artifacts == {path: path.read_bytes() for path in root.rglob("*.yaml")}
    assert service.scientific_run_status(resumed, workflow.runs)["current_stage"] == "APPROVED"
    assert service.render_scientific_run_markdown(resumed, workflow.runs) == render_scientific_run_markdown(run, workflow.runs)


@pytest.mark.parametrize("stage", ["planning", "approval"])
def test_resume_after_interrupted_direct_provider_keeps_claim(tmp_path, monkeypatch, stage):
    from spc.approval import MockApprovalProvider
    from spc.planning import MockPlanningProvider

    knowledge, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(state_dir=tmp_path / ".spc", knowledge_dir=knowledge.root)
    calls = 0

    def interrupt(*args):
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt

    provider, method = (MockPlanningProvider, "propose") if stage == "planning" else (MockApprovalProvider, "review")
    monkeypatch.setattr(provider, method, interrupt)
    with pytest.raises(KeyboardInterrupt):
        workflow.start(REQUEST, "base", approval_provider="mock")
    interrupted, = workflow.runs.list()
    resumed = workflow.resume(interrupted.run_id, approval_provider="mock").run

    assert resumed.status == ScientificProblemRunStatus.FAILED
    assert resumed.failure.stage == stage
    assert resumed.failure.category == "ProviderOutcomeUncertain"
    assert not resumed.failure.retryable
    assert calls == 1


def test_resume_validates_complete_approval_after_projection_write_interruption(tmp_path, monkeypatch):
    from spc.approval import MockApprovalProvider

    knowledge, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(state_dir=tmp_path / ".spc", knowledge_dir=knowledge.root)
    original_save = workflow.runs.save

    def interrupted(run):
        if run.status == ScientificProblemRunStatus.APPROVED:
            raise KeyboardInterrupt
        return original_save(run)

    with monkeypatch.context() as patch:
        patch.setattr(workflow.runs, "save", interrupted)
        with pytest.raises(KeyboardInterrupt):
            workflow.start(REQUEST, "base", approval_provider="mock")
    pending, = workflow.runs.list()
    assert pending.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    monkeypatch.setattr(MockApprovalProvider, "review", lambda *args: pytest.fail("review repeated"))
    resumed = workflow.resume(pending.run_id).run
    assert resumed.status == ScientificProblemRunStatus.APPROVED
    assert resumed.approval_receipt_id and resumed.gate_id


def test_knowledge_status_is_read_only_before_initialization(tmp_path):
    from spc.workflow.service import build_knowledge_status

    status = build_knowledge_status(tmp_path / "knowledge", tmp_path / ".spc")
    assert status["literature"]["trusted_documents"] == 0
    assert not tuple(tmp_path.iterdir())
