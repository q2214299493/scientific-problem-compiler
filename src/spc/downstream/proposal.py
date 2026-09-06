from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..models import (
    AgentCapabilityCatalog,
    ExecutionProposal,
    SPCExportPackage,
    ScientificTaskExecutionContext,
    prohibited_execution_payload_paths,
)
from ..serialization import content_hash
from .adapter import AgentExecutionAdapter
from .context import ExecutionContextError, ScientificTaskExecutionContextProjector
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

        try:
            catalog = AgentCapabilityCatalog.model_validate(
                catalog.model_dump(mode="python")
            )
        except ValueError as error:
            self._reject("INVALID_AGENT_CAPABILITY_CATALOG", str(error))

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
        try:
            execution_context = ScientificTaskExecutionContextProjector().project(
                package.source_plan,
                task.task_id,
            )
        except ExecutionContextError as error:
            self._reject(error.code, str(error))
        except ValueError as error:
            self._reject("EXECUTION_CONTEXT_INVALID", str(error))

        binding = next(
            (
                item
                for item in package.capability_bindings
                if item.scientific_capability_id == task.capability_id
            ),
            None,
        )
        catalog_hash = content_hash(catalog)
        mapped_id = adapter.map_capability(task.capability_id, catalog)
        repeated_mapping = adapter.map_capability(task.capability_id, catalog)
        if mapped_id != repeated_mapping:
            self._reject(
                "NONDETERMINISTIC_CAPABILITY_MAPPING",
                "adapter returned different capability mappings for identical input",
            )
        if content_hash(catalog) != catalog_hash:
            self._reject(
                "AGENT_CAPABILITY_CATALOG_MUTATED",
                "adapter modified the capability catalog",
            )
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

        required_fields = self._contract_fields(
            executable.input_contract,
            "requires",
            issue_code="AGENT_INPUT_CONTRACT_UNSATISFIED",
        )
        input_bindings = dict(adapter.input_bindings(executable.capability_id))
        repeated_input_bindings = dict(
            adapter.input_bindings(executable.capability_id)
        )
        if input_bindings != repeated_input_bindings:
            self._reject(
                "NONDETERMINISTIC_CAPABILITY_MAPPING",
                "adapter returned different input bindings for identical input",
            )
        projected_inputs: dict[str, Any] = {"task_inputs": execution_context.task_inputs}
        missing_inputs: list[str] = []
        for required_field in required_fields:
            context_path = input_bindings.get(required_field, required_field)
            available, value = self._context_value(execution_context, context_path)
            if not available or self._is_empty(value):
                missing_inputs.append(required_field)
            else:
                projected_inputs[required_field] = value
        if missing_inputs:
            self._reject(
                "AGENT_INPUT_CONTRACT_UNSATISFIED",
                "required capability inputs are absent from the execution context: "
                + ", ".join(missing_inputs),
            )

        declared_outputs = self._contract_fields(
            executable.output_contract,
            "produces",
            issue_code="AGENT_OUTPUT_CONTRACT_UNSATISFIED",
        )
        output_reconciliation = dict(
            adapter.reconcile_outputs(
                executable.capability_id,
                task.outputs,
                declared_outputs,
            )
        )
        repeated_output_reconciliation = dict(
            adapter.reconcile_outputs(
                executable.capability_id,
                task.outputs,
                declared_outputs,
            )
        )
        if output_reconciliation != repeated_output_reconciliation:
            self._reject(
                "NONDETERMINISTIC_CAPABILITY_MAPPING",
                "adapter returned different output mappings for identical input",
            )
        if set(output_reconciliation) != set(task.outputs) or any(
            target not in declared_outputs
            for target in output_reconciliation.values()
        ):
            self._reject(
                "AGENT_OUTPUT_CONTRACT_UNSATISFIED",
                "expected scientific outputs are not explicitly reconciled with "
                "the target capability output contract",
            )

        resource_requirements = adapter.resource_requirements(
            executable.capability_id,
            target_environment,
        )
        repeated_resource_requirements = adapter.resource_requirements(
            executable.capability_id,
            target_environment,
        )
        if dict(resource_requirements) != dict(repeated_resource_requirements):
            self._reject(
                "NONDETERMINISTIC_CAPABILITY_MAPPING",
                "adapter returned different resource requirements for identical input",
            )
        unsafe_paths = tuple(
            path
            for field_name, value in (
                ("required_inputs", projected_inputs),
                ("resource_requirements", resource_requirements),
                ("output_reconciliation", output_reconciliation),
            )
            for path in prohibited_execution_payload_paths(value, field_name)
        )
        if unsafe_paths:
            self._reject(
                "EXECUTABLE_PAYLOAD_FORBIDDEN",
                "Phase 3A payload contains executable fields: "
                + ", ".join(unsafe_paths),
            )
        if content_hash(catalog) != catalog_hash:
            self._reject(
                "AGENT_CAPABILITY_CATALOG_MUTATED",
                "adapter modified the capability catalog",
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
            "execution_context": execution_context,
            "execution_context_hash": execution_context.content_hash,
            "task_id": task.task_id,
            "depends_on_task_ids": task.depends_on,
            "scientific_capability_id": task.capability_id,
            "executable_capability_id": executable.capability_id,
            "executable_capability_version": executable.version,
            "agent_capability_catalog_version": catalog.version,
            "agent_capability_catalog_hash": content_hash(catalog),
            "target_agent": package.export_manifest.target_agent,
            "target_environment": target_environment,
            "required_inputs": projected_inputs,
            "expected_outputs": task.outputs,
            "execution_assumptions": tuple(
                assumption.statement for assumption in package.source_plan.assumptions
            ),
            "resource_requirements": resource_requirements,
            "output_reconciliation": output_reconciliation,
            "validation_requirements": (*task.success_criteria, *task.release_gates),
            "provenance_requirements": task.provenance_requirements,
            "adapter_id": adapter.adapter_id,
            "adapter_version": adapter.adapter_version,
            "authorized": False,
            "runnable": False,
        }
        proposal_id = f"execution-proposal-{content_hash(identity)[:24]}"
        payload = {"proposal_id": proposal_id, **identity}
        try:
            return ExecutionProposal(**payload, content_hash=content_hash(payload))
        except ValueError as error:
            self._reject("EXECUTION_PROPOSAL_INVALID", str(error))

    @classmethod
    def _contract_fields(
        cls,
        contract: Mapping[str, Any],
        field_name: str,
        *,
        issue_code: str,
    ) -> tuple[str, ...]:
        value = contract.get(field_name, ())
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            cls._reject(
                issue_code,
                f"capability contract {field_name!r} must be a list of nonblank fields",
            )
        if len(set(value)) != len(value):
            cls._reject(
                issue_code,
                f"capability contract {field_name!r} contains duplicates",
            )
        return tuple(value)

    @staticmethod
    def _context_value(
        context: ScientificTaskExecutionContext,
        path: str,
    ) -> tuple[bool, Any]:
        if path.startswith("task_inputs."):
            key = path.removeprefix("task_inputs.")
            return (key in context.task_inputs, context.task_inputs.get(key))
        if path in type(context).model_fields and path not in {
            "context_id",
            "content_hash",
            "source_plan_hash",
        }:
            return True, getattr(context, path)
        if path in context.task_inputs:
            return True, context.task_inputs[path]
        return False, None

    @staticmethod
    def _is_empty(value: Any) -> bool:
        if value is None or (isinstance(value, str) and not value.strip()):
            return True
        return isinstance(value, (Mapping, tuple, list, set, frozenset)) and not value

    @staticmethod
    def _reject(code: str, message: str) -> None:
        raise DownstreamImportError(
            ValidationReport(
                valid=False,
                issues=(ValidationIssue(code=code, message=message),),
            )
        )
