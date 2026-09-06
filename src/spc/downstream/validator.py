from __future__ import annotations

from pathlib import Path

from ..models import (
    AgentHandoffPackage,
    ApprovalMode,
    CapabilityBinding,
    ExportManifest,
    GateVerdict,
    IndependentApprovalReceipt,
    PlanCompilationReceipt,
    ProjectTrustPolicy,
    SPCExportPackage,
    ScientificQuestionPlan,
)
from ..serialization import content_hash, file_sha256, load_data
from ..validators import ValidationIssue, ValidationReport, validate_export


class DownstreamImportError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("; ".join(f"{item.code}: {item.message}" for item in report.issues))


class DownstreamImportValidator:
    """Validates an on-disk export before exposing a typed package."""

    def import_package(
        self,
        export_dir: Path,
        *,
        expected_target_agent: str | None = None,
    ) -> SPCExportPackage:
        report = validate_export(export_dir)
        if not report.valid:
            raise DownstreamImportError(report)
        try:
            manifest = ExportManifest.model_validate(load_data(export_dir / "manifest.yaml"))
            plan = ScientificQuestionPlan.model_validate(
                load_data(export_dir / "selected-plan.yaml")
            )
            handoff = AgentHandoffPackage.model_validate(
                load_data(export_dir / "handoff-package.yaml")
            )
            gate = GateVerdict.model_validate(
                load_data(export_dir / "approvals" / "plan-gate.yaml")
            )
            trust_policy = ProjectTrustPolicy.model_validate(
                load_data(export_dir / "approvals" / "project-trust-policy.yaml")
            )
            if trust_policy.approval_mode != ApprovalMode.INDEPENDENT_REQUIRED:
                raise DownstreamImportError(
                    ValidationReport(
                        valid=False,
                        issues=(
                            ValidationIssue(
                                code="INDEPENDENT_APPROVAL_REQUIRED",
                                message=(
                                    "trusted downstream import does not accept "
                                    "legacy/manual approval"
                                ),
                            ),
                        ),
                    )
                )
            compilation_receipt = PlanCompilationReceipt.model_validate(
                load_data(
                    export_dir / "provenance" / "plan-compilation-receipt.yaml"
                )
            )
            approval_receipt = IndependentApprovalReceipt.model_validate(
                load_data(
                    export_dir
                    / "approvals"
                    / "independent-approval-receipt.yaml"
                )
            )
            bindings = tuple(
                CapabilityBinding.model_validate(item)
                for item in load_data(export_dir / "capability-bindings.yaml")
            )
        except (OSError, TypeError, ValueError) as error:
            raise DownstreamImportError(
                ValidationReport(
                    valid=False,
                    issues=(
                        ValidationIssue(
                            code="DOWNSTREAM_IMPORT_INVALID",
                            message=f"cannot load trusted export artifacts: {error}",
                        ),
                    ),
                )
            ) from error

        issues: list[ValidationIssue] = []
        if expected_target_agent is not None and manifest.target_agent != expected_target_agent:
            issues.append(
                ValidationIssue(
                    code="WRONG_TARGET_AGENT",
                    message=(
                        f"export targets {manifest.target_agent!r}, not "
                        f"{expected_target_agent!r}"
                    ),
                )
            )
        binding_ids = [item.scientific_capability_id for item in bindings]
        if len(set(binding_ids)) != len(binding_ids):
            issues.append(
                ValidationIssue(
                    code="INVALID_CAPABILITY_BINDING",
                    message="scientific capability bindings must be unique",
                )
            )
        binding_by_id = {item.scientific_capability_id: item for item in bindings}
        for task in plan.tasks:
            binding = binding_by_id.get(task.capability_id)
            if (
                binding is None
                or binding.status != "available"
                or binding.target_capability_id is None
            ):
                issues.append(
                    ValidationIssue(
                        code="MISSING_CAPABILITY_MAPPING",
                        message=f"task has no available capability binding: {task.task_id}",
                    )
                )
        if issues:
            raise DownstreamImportError(ValidationReport(valid=False, issues=tuple(issues)))

        identity = {
            "export_manifest": manifest,
            "export_manifest_hash": content_hash(manifest),
            "checksums_hash": file_sha256(export_dir / "checksums.json"),
            "source_plan": plan,
            "source_plan_hash": content_hash(plan),
            "handoff": handoff,
            "gate": gate,
            "gate_hash": content_hash(gate),
            "trust_policy": trust_policy,
            "trust_policy_hash": content_hash(trust_policy),
            "plan_compilation_receipt": compilation_receipt,
            "approval_receipt": approval_receipt,
            "capability_bindings": bindings,
        }
        package_id = f"spc-export-package-{content_hash(identity)[:24]}"
        payload = {"package_id": package_id, **identity}
        try:
            return SPCExportPackage(
                **payload,
                content_hash=content_hash(payload),
            )
        except ValueError as error:
            raise DownstreamImportError(
                ValidationReport(
                    valid=False,
                    issues=(
                        ValidationIssue(
                            code="DOWNSTREAM_TRUST_BINDING_INVALID",
                            message=str(error),
                        ),
                    ),
                )
            ) from error
