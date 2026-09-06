from __future__ import annotations

from ..models import AgentCapabilityCatalog, ExecutionProposal, SPCExportPackage
from ..serialization import content_hash
from .adapter import AgentExecutionAdapter
from .validator import DownstreamImportError
from ..validators import ValidationIssue, ValidationReport


class ExecutionProposalBuilder:
    """Builds non-executable proposals only from a verified package."""

    def build(
        self,
        package: SPCExportPackage,
        *,
        task_id: str,
        catalog: AgentCapabilityCatalog,
        adapter: AgentExecutionAdapter,
        target_environment: str,
    ) -> ExecutionProposal:
        try:
            package = SPCExportPackage.model_validate(package.model_dump(mode="python"))
        except ValueError as error:
            self._reject("DOWNSTREAM_PACKAGE_INVALID", str(error))

        if adapter.target_agent != package.export_manifest.target_agent:
            self._reject(
                "WRONG_TARGET_AGENT",
                f"adapter {adapter.target_agent!r} does not match export target",
            )
        if catalog.agent_id != package.export_manifest.target_agent:
            self._reject(
                "WRONG_TARGET_AGENT",
                f"catalog {catalog.agent_id!r} does not match export target",
            )
        if target_environment not in adapter.supported_environments:
            self._reject(
                "WRONG_TARGET_ENVIRONMENT",
                f"adapter does not support environment {target_environment!r}",
            )
        task = next(
            (item for item in package.source_plan.tasks if item.task_id == task_id),
            None,
        )
        if task is None:
            self._reject(
                "TASK_PLAN_MISMATCH",
                f"task {task_id!r} does not belong to the exported plan",
            )

        binding = next(
            (
                item
                for item in package.capability_bindings
                if item.scientific_capability_id == task.capability_id
            ),
            None,
        )
        mapped_id = adapter.map_capability(task.capability_id, catalog)
        if (
            binding is None
            or binding.status != "available"
            or binding.target_capability_id is None
            or mapped_id != binding.target_capability_id
        ):
            self._reject(
                "MISSING_CAPABILITY_MAPPING",
                f"no trusted executable mapping for {task.capability_id!r}",
            )
        executable = next(
            (
                item
                for item in catalog.capabilities
                if item.capability_id == mapped_id
                and task.capability_id in item.supports_scientific_capability_ids
            ),
            None,
        )
        if executable is None:
            self._reject(
                "MISSING_CAPABILITY_MAPPING",
                f"catalog does not support binding {task.capability_id!r} -> {mapped_id!r}",
            )

        resource_requirements = adapter.resource_requirements(
            executable.capability_id,
            target_environment,
        )
        identity = {
            "source_plan_id": package.source_plan.plan_id,
            "source_plan_hash": package.source_plan_hash,
            "export_id": package.export_manifest.export_id,
            "export_manifest_hash": package.export_manifest_hash,
            "gate_id": package.gate.gate_id,
            "gate_hash": package.gate_hash,
            "approval_receipt_id": package.approval_receipt.receipt_id,
            "approval_receipt_hash": package.approval_receipt.content_hash,
            "plan_compilation_receipt_id": package.plan_compilation_receipt.receipt_id,
            "plan_compilation_receipt_hash": package.plan_compilation_receipt.content_hash,
            "task_id": task.task_id,
            "scientific_capability_id": task.capability_id,
            "executable_capability_id": executable.capability_id,
            "executable_capability_version": executable.version,
            "agent_capability_catalog_version": catalog.version,
            "agent_capability_catalog_hash": content_hash(catalog),
            "target_agent": package.export_manifest.target_agent,
            "target_environment": target_environment,
            "required_inputs": task.inputs,
            "expected_outputs": task.outputs,
            "execution_assumptions": tuple(
                assumption.statement for assumption in package.source_plan.assumptions
            ),
            "resource_requirements": resource_requirements,
            "validation_requirements": (*task.success_criteria, *task.release_gates),
            "provenance_requirements": task.provenance_requirements,
            "adapter_id": adapter.adapter_id,
            "adapter_version": adapter.adapter_version,
            "authorized": False,
            "runnable": False,
        }
        proposal_id = f"execution-proposal-{content_hash(identity)[:24]}"
        payload = {"proposal_id": proposal_id, **identity}
        return ExecutionProposal(**payload, content_hash=content_hash(payload))

    @staticmethod
    def _reject(code: str, message: str) -> None:
        raise DownstreamImportError(
            ValidationReport(
                valid=False,
                issues=(ValidationIssue(code=code, message=message),),
            )
        )
