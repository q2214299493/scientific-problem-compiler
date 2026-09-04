from __future__ import annotations

import pytest

from spc.adapters.ft_agent import FTAgentAdapter
from spc.approval import ScientificPlanApprover
from spc.export import ExportError, GenericExportService
from spc.models import ApprovalScores, GateVerdict
from spc.serialization import content_hash
from spc.validators import validate_export, validate_question_plan


def approved_inputs(plan):
    score = ApprovalScores(**{name: 5 for name in ApprovalScores.model_fields})
    verdict = ScientificPlanApprover("approver").bind_verdict(
        plan, verdict_id="verdict-1", scores=score, decision="approve"
    )
    gate = GateVerdict(
        gate_id="gate-1",
        candidate_id=plan.plan_id,
        candidate_version=plan.version,
        candidate_content_hash=content_hash(plan),
        passed=True,
    )
    return verdict, gate


def test_export_requires_passed_plan_gate(tmp_path, make_plan) -> None:
    plan = make_plan()
    verdict, gate = approved_inputs(plan)
    gate = gate.model_copy(update={"passed": False})
    with pytest.raises(ExportError) as caught:
        GenericExportService(tmp_path).export(
            plan=plan, verdict=verdict, gate=gate, human_selected=True, adapter=FTAgentAdapter(), export_id="x"
        )
    assert "PLAN_GATE_FAILED" in {item.code for item in caught.value.report.issues}


def test_export_requires_human_selection(tmp_path, make_plan) -> None:
    plan = make_plan()
    verdict, gate = approved_inputs(plan)
    with pytest.raises(ExportError) as caught:
        GenericExportService(tmp_path).export(
            plan=plan, verdict=verdict, gate=gate, human_selected=False, adapter=FTAgentAdapter(), export_id="x"
        )
    assert "HUMAN_SELECTION_REQUIRED" in {item.code for item in caught.value.report.issues}


def test_missing_capability_mapping_is_unavailable(make_plan) -> None:
    plan = make_plan(capability_id="nonexistent_capability")
    binding = FTAgentAdapter().bind_capabilities(plan)[0]
    assert binding.status == "unavailable"
    assert binding.target_capability_id is None


def test_export_checksum_mismatch_is_detected(tmp_path, make_plan) -> None:
    plan = make_plan()
    verdict, gate = approved_inputs(plan)
    output = GenericExportService(tmp_path).export(
        plan=plan, verdict=verdict, gate=gate, human_selected=True, adapter=FTAgentAdapter(), export_id="x"
    )
    selected = output / "selected-plan.yaml"
    selected.write_text(selected.read_text(encoding="utf-8") + "tampered: true\n", encoding="utf-8")
    report = validate_export(output)
    assert "CHECKSUM_MISMATCH" in {item.code for item in report.issues}


def test_phase1_rejects_runnable_task(make_plan) -> None:
    plan = make_plan(task_overrides={"runnable": True})
    report = validate_question_plan(plan)
    assert "PHASE1_RUNNABLE_TASK" in {item.code for item in report.issues}


def test_phase1_export_rejects_execution_payload(tmp_path, make_plan) -> None:
    plan = make_plan(task_overrides={"inputs": {"command": "submit scientific job"}})
    verdict, gate = approved_inputs(plan)
    with pytest.raises(ExportError) as caught:
        GenericExportService(tmp_path).export(
            plan=plan,
            verdict=verdict,
            gate=gate,
            human_selected=True,
            adapter=FTAgentAdapter(),
            export_id="x",
        )
    assert "PHASE1_EXECUTION_PAYLOAD" in {item.code for item in caught.value.report.issues}


def test_valid_export_has_required_files_and_nonrunnable_tasks(tmp_path, make_plan) -> None:
    plan = make_plan()
    verdict, gate = approved_inputs(plan)
    output = GenericExportService(tmp_path).export(
        plan=plan, verdict=verdict, gate=gate, human_selected=True, adapter=FTAgentAdapter(), export_id="x"
    )
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
    task_text = (output / "tasks" / "task-1.yaml").read_text(encoding="utf-8")
    assert "runnable: false" in task_text
    assert validate_export(output).valid


def test_export_is_immutable_by_refusing_overwrite(tmp_path, make_plan) -> None:
    plan = make_plan()
    verdict, gate = approved_inputs(plan)
    service = GenericExportService(tmp_path)
    service.export(plan=plan, verdict=verdict, gate=gate, human_selected=True, adapter=FTAgentAdapter(), export_id="x")
    with pytest.raises(FileExistsError):
        service.export(plan=plan, verdict=verdict, gate=gate, human_selected=True, adapter=FTAgentAdapter(), export_id="x")
