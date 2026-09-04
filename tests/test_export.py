from __future__ import annotations

import json

import pytest

from spc.adapters.ft_agent import FTAgentAdapter
from spc.approval import ScientificPlanApprover, bind_gate_verdict
from spc.domains import DomainPackLoader
from spc.export import ExportError, GenericExportService
from spc.models import ApprovalScores, EvidenceSpan, FixResolution, RequiredFix
from spc.repositories import ModelRepository
from spc.validators import build_plan_validation_record, validate_export, validate_question_plan


def approved_inputs(plan, evidence_repository, *, decision="approve", required_fixes=(), fix_resolutions=()):
    pack = DomainPackLoader().load(plan.domain)
    report = validate_question_plan(plan, pack.capabilities, evidence_repository)
    validation_record = build_plan_validation_record(plan, report, validation_id="validation-1")
    score = ApprovalScores(**{name: 5 for name in ApprovalScores.model_fields})
    verdict = ScientificPlanApprover("approver").bind_verdict(
        plan,
        verdict_id="verdict-1",
        scores=score,
        decision=decision,
        required_fixes=required_fixes,
        fix_resolutions=fix_resolutions,
    )
    gate = bind_gate_verdict(
        plan, verdict, validation_record, gate_id="gate-1", passed=True
    )
    return verdict, validation_record, gate


def run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate, **overrides):
    arguments = {
        "plan": plan,
        "verdict": verdict,
        "validation_record": validation_record,
        "gate": gate,
        "human_selected": True,
        "adapter": FTAgentAdapter(),
        "export_id": "x",
    }
    arguments.update(overrides)
    return GenericExportService(tmp_path / "exports", evidence_repository).export(**arguments)


def test_invalid_plan_cannot_export(tmp_path, make_plan, evidence_repository) -> None:
    plan = make_plan(task_overrides={"depends_on": ("task-1",)})
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    codes = {item.code for item in caught.value.report.issues}
    assert "DAG_CYCLE" in codes
    assert "PLAN_VALIDATION_FAILED" in codes


def test_fake_evidence_cannot_pass(tmp_path, make_plan) -> None:
    plan = make_plan()
    empty_repository = ModelRepository(tmp_path / "empty-evidence", EvidenceSpan)
    verdict, validation_record, gate = approved_inputs(plan, empty_repository)
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, empty_repository, verdict, validation_record, gate)
    assert "EVIDENCE_SPAN_NOT_FOUND" in {item.code for item in caught.value.report.issues}


def test_unresolved_conditional_approval_cannot_export(
    tmp_path, make_plan, evidence_repository
) -> None:
    plan = make_plan()
    blocking_fix = RequiredFix(fix_id="fix-1", description="Resolve the evidence gap")
    verdict, validation_record, gate = approved_inputs(
        plan,
        evidence_repository,
        decision="approve_with_conditions",
        required_fixes=(blocking_fix,),
    )
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    assert "UNRESOLVED_BLOCKING_FIX" in {item.code for item in caught.value.report.issues}


def test_resolved_conditional_approval_can_export(tmp_path, make_plan, evidence_repository) -> None:
    plan = make_plan()
    blocking_fix = RequiredFix(fix_id="fix-1", description="Resolve the evidence gap")
    resolution = FixResolution(
        fix_id="fix-1",
        resolved=True,
        resolution="The approver verified the corrected evidence linkage.",
        evidence_refs=("ev-1",),
    )
    verdict, validation_record, gate = approved_inputs(
        plan,
        evidence_repository,
        decision="approve_with_conditions",
        required_fixes=(blocking_fix,),
        fix_resolutions=(resolution,),
    )
    assert run_export(
        tmp_path, plan, evidence_repository, verdict, validation_record, gate
    ).is_dir()


def test_domain_version_mismatch_cannot_export(tmp_path, make_plan, evidence_repository) -> None:
    plan = make_plan().model_copy(update={"domain_pack_version": "9.9.9"})
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    assert "DOMAIN_PACK_VERSION_MISMATCH" in {item.code for item in caught.value.report.issues}


def test_target_domain_mismatch_cannot_export(tmp_path, make_plan, evidence_repository) -> None:
    plan = make_plan(
        domain="base",
        latent_concern="OER mechanism",
        question="Does the OER mechanism differ from the declared baseline?",
        capability_id="comparative_analysis",
    )
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    assert "TARGET_DOMAIN_MISMATCH" in {item.code for item in caught.value.report.issues}


def test_stale_plan_validation_record_cannot_export(
    tmp_path, make_plan, evidence_repository
) -> None:
    plan = make_plan()
    verdict, validation_record, _ = approved_inputs(plan, evidence_repository)
    validation_record = validation_record.model_copy(update={"plan_content_hash": "0" * 64})
    gate = bind_gate_verdict(
        plan, verdict, validation_record, gate_id="gate-1", passed=True
    )
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    assert "STALE_PLAN_VALIDATION" in {item.code for item in caught.value.report.issues}


def test_export_requires_passed_plan_gate(tmp_path, make_plan, evidence_repository) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    gate = gate.model_copy(update={"passed": False})
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    assert "PLAN_GATE_FAILED" in {item.code for item in caught.value.report.issues}


def test_export_requires_human_selection(tmp_path, make_plan, evidence_repository) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    with pytest.raises(ExportError) as caught:
        run_export(
            tmp_path,
            plan,
            evidence_repository,
            verdict,
            validation_record,
            gate,
            human_selected=False,
        )
    assert "HUMAN_SELECTION_REQUIRED" in {item.code for item in caught.value.report.issues}


@pytest.mark.parametrize(
    ("gate_update", "expected_code"),
    (
        ({"approval_verdict_id": "different-verdict"}, "GATE_APPROVAL_BINDING_MISMATCH"),
        ({"plan_validation_id": "different-validation"}, "GATE_VALIDATION_BINDING_MISMATCH"),
    ),
)
def test_gate_must_bind_approval_and_validation(
    tmp_path, make_plan, evidence_repository, gate_update, expected_code
) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    gate = gate.model_copy(update=gate_update)
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    assert expected_code in {item.code for item in caught.value.report.issues}


def test_missing_capability_mapping_is_unavailable(make_plan) -> None:
    plan = make_plan(capability_id="nonexistent_capability")
    binding = FTAgentAdapter().bind_capabilities(plan)[0]
    assert binding.status == "unavailable"
    assert binding.target_capability_id is None


def test_export_checksum_mismatch_is_detected(tmp_path, make_plan, evidence_repository) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    output = run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    selected = output / "selected-plan.yaml"
    selected.write_text(selected.read_text(encoding="utf-8") + "tampered: true\n", encoding="utf-8")
    report = validate_export(output)
    assert "CHECKSUM_MISMATCH" in {item.code for item in report.issues}


def test_unsafe_checksum_path_is_rejected(tmp_path, make_plan, evidence_repository) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    output = run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    checksums_path = output / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["../outside.txt"] = "0" * 64
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
    report = validate_export(output)
    assert "UNSAFE_CHECKSUM_PATH" in {item.code for item in report.issues}


def test_extra_export_file_is_detected(tmp_path, make_plan, evidence_repository) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    output = run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    (output / "extra.txt").write_text("unchecked", encoding="utf-8")
    report = validate_export(output)
    assert "EXTRA_EXPORT_FILE" in {item.code for item in report.issues}


def test_phase1_rejects_runnable_task(make_plan, evidence_repository) -> None:
    plan = make_plan(task_overrides={"runnable": True})
    report = validate_question_plan(plan, evidence_repository=evidence_repository)
    assert "PHASE1_RUNNABLE_TASK" in {item.code for item in report.issues}


def test_phase1_export_rejects_execution_payload(
    tmp_path, make_plan, evidence_repository
) -> None:
    plan = make_plan(task_overrides={"inputs": {"command": "submit scientific job"}})
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    assert "PHASE1_EXECUTION_PAYLOAD" in {item.code for item in caught.value.report.issues}


def test_staging_is_validated_before_atomic_rename(
    tmp_path, make_plan, evidence_repository, monkeypatch
) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    original_writer = GenericExportService._write_package

    def write_invalid_package(*args, **kwargs):
        original_writer(*args, **kwargs)
        root = args[0]
        (root / "unchecked.txt").write_text("invalid", encoding="utf-8")

    monkeypatch.setattr(
        GenericExportService,
        "_write_package",
        staticmethod(write_invalid_package),
    )
    with pytest.raises(ExportError) as caught:
        run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    assert "EXTRA_EXPORT_FILE" in {item.code for item in caught.value.report.issues}
    assert not (tmp_path / "exports" / "ft-agent" / "x").exists()


def test_valid_export_has_required_files_and_nonrunnable_tasks(
    tmp_path, make_plan, evidence_repository
) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    output = run_export(tmp_path, plan, evidence_repository, verdict, validation_record, gate)
    required = {
        "manifest.yaml",
        "selected-plan.yaml",
        "task-graph.yaml",
        "capability-bindings.yaml",
        "evidence-manifest.jsonl",
        "decisions.jsonl",
        "execution-policy.yaml",
        "checksums.json",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert (output / "approvals" / "plan-validation.yaml").is_file()
    task_text = (output / "tasks" / "task-1.yaml").read_text(encoding="utf-8")
    assert "runnable: false" in task_text
    assert validate_export(output).valid


def test_export_is_immutable_by_refusing_overwrite(
    tmp_path, make_plan, evidence_repository
) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    service = GenericExportService(tmp_path / "exports", evidence_repository)
    arguments = {
        "plan": plan,
        "verdict": verdict,
        "validation_record": validation_record,
        "gate": gate,
        "human_selected": True,
        "adapter": FTAgentAdapter(),
        "export_id": "x",
    }
    service.export(**arguments)
    with pytest.raises(FileExistsError):
        service.export(**arguments)
