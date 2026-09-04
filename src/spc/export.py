from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Protocol

from .models import (
    AgentHandoffPackage,
    ApprovalVerdict,
    ExportManifest,
    GateVerdict,
    ScientificQuestionPlan,
)
from .serialization import content_hash, dump_json, dump_yaml, file_sha256, require_safe_path_component
from .validators import ValidationIssue, ValidationReport, validate_export, validate_handoff_package


class AgentAdapter(Protocol):
    target_agent: str

    def build_handoff(self, plan: ScientificQuestionPlan, export_id: str) -> AgentHandoffPackage: ...


class ExportError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("; ".join(f"{item.code}: {item.message}" for item in report.issues))


class GenericExportService:
    def __init__(self, exports_root: Path) -> None:
        self.exports_root = exports_root

    def export(
        self,
        *,
        plan: ScientificQuestionPlan,
        verdict: ApprovalVerdict,
        gate: GateVerdict,
        human_selected: bool,
        adapter: AgentAdapter,
        export_id: str,
    ) -> Path:
        require_safe_path_component(adapter.target_agent, field="target_agent")
        require_safe_path_component(export_id, field="export_id")
        destination = self.exports_root / adapter.target_agent / export_id
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite immutable export: {destination}")
        handoff = adapter.build_handoff(plan, export_id)
        report = validate_handoff_package(plan, verdict, gate, handoff, human_selected=human_selected)
        if not report.valid:
            raise ExportError(report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{export_id}-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "package"
            staging.mkdir()
            self._write_package(staging, plan, verdict, gate, handoff)
            staging.rename(destination)
        verification = validate_export(destination)
        if not verification.valid:
            raise ExportError(verification)
        return destination

    @staticmethod
    def _write_package(
        root: Path,
        plan: ScientificQuestionPlan,
        verdict: ApprovalVerdict,
        gate: GateVerdict,
        handoff: AgentHandoffPackage,
    ) -> None:
        plan_hash = content_hash(plan)
        for storage_id, field in (
            *((task.task_id, "task_id") for task in plan.tasks),
            (plan.intent_fingerprint.fingerprint_id, "intent fingerprint ID"),
            (plan.system_fingerprint.fingerprint_id, "system fingerprint ID"),
            (plan.method_fingerprint.fingerprint_id, "method fingerprint ID"),
        ):
            require_safe_path_component(storage_id, field=field)
        dump_yaml(root / "selected-plan.yaml", plan)
        dump_yaml(root / "handoff-package.yaml", handoff)
        dump_yaml(
            root / "task-graph.yaml",
            {"plan_id": plan.plan_id, "plan_hash": plan_hash, "tasks": [task.task_id for task in plan.tasks]},
        )
        bindings = {item.scientific_capability_id: item for item in handoff.capability_bindings}
        for task in plan.tasks:
            binding = bindings[task.capability_id]
            exported_task = {
                **task.model_dump(mode="json"),
                "source_plan_id": plan.plan_id,
                "source_plan_hash": plan_hash,
                "target_capability_binding": binding.model_dump(mode="json", exclude_none=True),
                "intent_fingerprint_ref": plan.intent_fingerprint.fingerprint_id,
                "system_fingerprint_ref": plan.system_fingerprint.fingerprint_id,
                "method_fingerprint_ref": plan.method_fingerprint.fingerprint_id,
                "execution_policy": handoff.execution_policy.model_dump(mode="json"),
                "runnable": False,
            }
            dump_yaml(root / "tasks" / f"{task.task_id}.yaml", exported_task)
        for fingerprint in (plan.intent_fingerprint, plan.system_fingerprint, plan.method_fingerprint):
            dump_yaml(root / "fingerprints" / f"{fingerprint.fingerprint_id}.yaml", fingerprint)
        dump_yaml(root / "approvals" / "plan-review.yaml", verdict)
        dump_yaml(root / "approvals" / "plan-gate.yaml", gate)
        dump_yaml(root / "capability-bindings.yaml", list(handoff.capability_bindings))
        dump_yaml(root / "execution-policy.yaml", handoff.execution_policy)
        evidence_lines = [json.dumps(item.model_dump(mode="json"), sort_keys=True) for item in plan.evidence_refs]
        (root / "evidence-manifest.jsonl").write_text(
            "\n".join(evidence_lines) + ("\n" if evidence_lines else ""), encoding="utf-8"
        )
        decision = {
            "decision": "human_selected",
            "plan_id": plan.plan_id,
            "plan_version": plan.version,
            "plan_hash": plan_hash,
        }
        (root / "decisions.jsonl").write_text(json.dumps(decision, sort_keys=True) + "\n", encoding="utf-8")
        package_files = tuple(
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        )
        manifest = ExportManifest(
            export_id=handoff.export_id,
            target_agent=handoff.target_agent,
            source_plan_id=plan.plan_id,
            source_plan_version=plan.version,
            source_plan_hash=plan_hash,
            files=package_files,
        )
        dump_yaml(root / "manifest.yaml", manifest)
        checksummed_files = tuple(
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "checksums.json"
        )
        dump_json(root / "checksums.json", {name: file_sha256(root / name) for name in checksummed_files})


def export_failure_report(error: Exception) -> ValidationReport:
    if isinstance(error, ExportError):
        return error.report
    return ValidationReport(valid=False, issues=(ValidationIssue(code="EXPORT_ERROR", message=str(error)),))
