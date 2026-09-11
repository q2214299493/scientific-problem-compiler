from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from spc.cli import app
from spc.models import (
    CurationStatus,
    EvidenceSpan,
    ScientificProblemRun,
    ScientificProblemRunStatus,
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
