from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from ..immutable import FrozenDict

from .approval import (
    ApprovalMode,
    GateVerdict,
    IndependentApprovalReceipt,
    ProjectTrustPolicy,
)
from .base import NonBlankStr, Sha256Str, StrictModel
from .planning import PlanCompilationReceipt
from .scientific import (
    ScientificQuestionPlan,
    ScientificTaskExecutionContext,
    prohibited_execution_payload_paths,
)


class AgentCapability(StrictModel):
    capability_id: NonBlankStr
    version: NonBlankStr
    supports_scientific_capability_ids: tuple[NonBlankStr, ...]
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_supported_capabilities(self) -> AgentCapability:
        if len(set(self.supports_scientific_capability_ids)) != len(
            self.supports_scientific_capability_ids
        ):
            raise ValueError("supported scientific capability IDs must be unique")
        return self


class AgentCapabilityCatalog(StrictModel):
    agent_id: NonBlankStr
    version: NonBlankStr
    capabilities: tuple[AgentCapability, ...]

    @model_validator(mode="after")
    def validate_catalog_consistency(self) -> AgentCapabilityCatalog:
        capability_ids = [item.capability_id for item in self.capabilities]
        if len(set(capability_ids)) != len(capability_ids):
            raise ValueError("agent capability IDs must be unique")
        scientific_mappings: dict[str, set[str]] = {}
        for capability in self.capabilities:
            for scientific_id in capability.supports_scientific_capability_ids:
                scientific_mappings.setdefault(scientific_id, set()).add(
                    capability.capability_id
                )
        ambiguous = sorted(
            scientific_id
            for scientific_id, targets in scientific_mappings.items()
            if len(targets) > 1
        )
        if ambiguous:
            raise ValueError(
                "scientific capabilities have ambiguous executable mappings: "
                + ", ".join(ambiguous)
            )
        return self


class CapabilityBinding(StrictModel):
    scientific_capability_id: str
    target_capability_id: str | None = None
    status: str
    reason: NonBlankStr | None = None


class ExecutionPolicy(StrictModel):
    mode: str = "planning_only"
    runnable: bool = False
    prohibited_actions: tuple[str, ...] = (
        "generate_execution_inputs",
        "submit_jobs",
        "execute_scientific_software",
    )


class AgentHandoffPackage(StrictModel):
    export_id: str
    target_agent: str
    source_plan_id: str
    source_plan_version: str
    source_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_bindings: tuple[CapabilityBinding, ...]
    execution_policy: ExecutionPolicy


class ExportManifest(StrictModel):
    export_id: str
    target_agent: str
    source_plan_id: str
    source_plan_version: str
    source_plan_hash: str
    files: tuple[str, ...]


class SPCExportPackage(StrictModel):
    """Verified, immutable view of an SPC export package."""

    package_id: NonBlankStr
    export_manifest: ExportManifest
    export_manifest_hash: Sha256Str
    checksums_hash: Sha256Str
    source_plan: ScientificQuestionPlan
    source_plan_hash: Sha256Str
    handoff: AgentHandoffPackage
    gate: GateVerdict
    gate_hash: Sha256Str
    trust_policy: ProjectTrustPolicy
    trust_policy_hash: Sha256Str
    plan_compilation_receipt: PlanCompilationReceipt
    approval_receipt: IndependentApprovalReceipt
    capability_bindings: tuple[CapabilityBinding, ...]
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_bindings(self) -> SPCExportPackage:
        from ..serialization import content_hash

        plan_hash = content_hash(self.source_plan)
        if self.source_plan_hash != plan_hash:
            raise ValueError("source_plan_hash does not match source_plan")
        plan_binding = (
            self.source_plan.plan_id,
            self.source_plan.version,
            plan_hash,
        )
        if (
            self.export_manifest.source_plan_id,
            self.export_manifest.source_plan_version,
            self.export_manifest.source_plan_hash,
        ) != plan_binding:
            raise ValueError("export_manifest does not bind source_plan")
        if (
            self.handoff.source_plan_id,
            self.handoff.source_plan_version,
            self.handoff.source_plan_hash,
        ) != plan_binding:
            raise ValueError("handoff does not bind source_plan")
        if (
            self.gate.candidate_id,
            self.gate.candidate_version,
            self.gate.candidate_content_hash,
        ) != plan_binding:
            raise ValueError("gate does not bind source_plan")
        if not self.gate.passed:
            raise ValueError("gate must have passed")
        if self.export_manifest_hash != content_hash(self.export_manifest):
            raise ValueError("export_manifest_hash does not match export_manifest")
        if self.gate_hash != content_hash(self.gate):
            raise ValueError("gate_hash does not match gate")
        if self.trust_policy_hash != content_hash(self.trust_policy):
            raise ValueError("trust_policy_hash does not match trust_policy")
        if self.trust_policy.approval_mode != ApprovalMode.INDEPENDENT_REQUIRED:
            raise ValueError("downstream import requires independent approval policy")
        if (
            self.gate.trust_policy_version,
            self.gate.trust_policy_hash,
        ) != (self.trust_policy.policy_version, self.trust_policy_hash):
            raise ValueError("gate does not bind trust_policy")
        if (
            self.gate.plan_compilation_receipt_id,
            self.gate.plan_compilation_receipt_hash,
        ) != (
            self.plan_compilation_receipt.receipt_id,
            self.plan_compilation_receipt.content_hash,
        ):
            raise ValueError("gate does not bind plan_compilation_receipt")
        if (
            self.plan_compilation_receipt.plan_id,
            self.plan_compilation_receipt.plan_hash,
        ) != (self.source_plan.plan_id, plan_hash):
            raise ValueError("plan_compilation_receipt does not bind source_plan")
        if (
            self.gate.independent_approval_receipt_id,
            self.gate.independent_approval_receipt_hash,
        ) != (self.approval_receipt.receipt_id, self.approval_receipt.content_hash):
            raise ValueError("gate does not bind approval_receipt")
        if (
            self.gate.approval_verdict_id,
            self.gate.approval_verdict_hash,
        ) != (
            self.approval_receipt.verdict_id,
            self.approval_receipt.verdict_hash,
        ):
            raise ValueError("gate and approval_receipt bind different verdicts")
        if (
            self.approval_receipt.candidate_id,
            self.approval_receipt.candidate_version,
            self.approval_receipt.candidate_hash,
        ) != plan_binding:
            raise ValueError("approval_receipt does not bind source_plan")
        if self.capability_bindings != self.handoff.capability_bindings:
            raise ValueError("capability_bindings do not match handoff")
        if (
            self.export_manifest.export_id,
            self.export_manifest.target_agent,
        ) != (self.handoff.export_id, self.handoff.target_agent):
            raise ValueError("export_manifest does not bind handoff")
        identity = self.model_dump(
            mode="json", exclude={"package_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"spc-export-package-{content_hash(identity)[:24]}"
        if self.package_id != expected_id:
            raise ValueError("SPCExportPackage package_id is not content-bound")
        payload = {"package_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("SPCExportPackage content_hash is invalid")
        return self


class ExecutionProposal(StrictModel):
    """Non-authorized, non-runnable proposal for a downstream agent."""

    proposal_id: NonBlankStr
    source_plan_id: NonBlankStr
    source_plan_hash: Sha256Str
    export_id: NonBlankStr
    export_manifest_hash: Sha256Str
    gate_id: NonBlankStr
    gate_hash: Sha256Str
    approval_receipt_id: NonBlankStr
    approval_receipt_hash: Sha256Str
    plan_compilation_receipt_id: NonBlankStr
    plan_compilation_receipt_hash: Sha256Str
    execution_context: ScientificTaskExecutionContext
    execution_context_hash: Sha256Str
    task_id: NonBlankStr
    depends_on_task_ids: tuple[NonBlankStr, ...] = ()
    scientific_capability_id: NonBlankStr
    executable_capability_id: NonBlankStr
    executable_capability_version: NonBlankStr
    agent_capability_catalog_version: NonBlankStr
    agent_capability_catalog_hash: Sha256Str
    target_agent: NonBlankStr
    target_environment: NonBlankStr
    required_inputs: FrozenDict = Field(default_factory=FrozenDict)
    expected_outputs: tuple[NonBlankStr, ...] = ()
    execution_assumptions: tuple[NonBlankStr, ...] = ()
    resource_requirements: FrozenDict = Field(default_factory=FrozenDict)
    output_reconciliation: FrozenDict = Field(default_factory=FrozenDict)
    validation_requirements: tuple[NonBlankStr, ...] = ()
    provenance_requirements: tuple[NonBlankStr, ...] = ()
    adapter_id: NonBlankStr
    adapter_version: NonBlankStr
    authorized: bool = False
    runnable: bool = False
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_safety(self) -> ExecutionProposal:
        from ..serialization import content_hash

        if self.authorized or self.runnable:
            raise ValueError("ExecutionProposal must remain unauthorized and non-runnable")
        if self.execution_context_hash != self.execution_context.content_hash:
            raise ValueError("execution_context_hash does not match execution_context")
        if (
            self.execution_context.source_plan_id,
            self.execution_context.source_plan_hash,
            self.execution_context.task_id,
        ) != (self.source_plan_id, self.source_plan_hash, self.task_id):
            raise ValueError("execution_context does not bind the proposal task and plan")
        if self.depends_on_task_ids != self.execution_context.depends_on_task_ids:
            raise ValueError("proposal dependencies do not match execution_context")
        unsafe_paths = tuple(
            path
            for field_name in (
                "required_inputs",
                "resource_requirements",
                "output_reconciliation",
            )
            for path in prohibited_execution_payload_paths(
                getattr(self, field_name), field_name
            )
        )
        if unsafe_paths:
            raise ValueError(
                "ExecutionProposal contains executable payload fields: "
                + ", ".join(unsafe_paths)
            )
        identity = self.model_dump(
            mode="json", exclude={"proposal_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"execution-proposal-{content_hash(identity)[:24]}"
        if self.proposal_id != expected_id:
            raise ValueError("ExecutionProposal proposal_id is not content-bound")
        payload = {"proposal_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExecutionProposal content_hash is invalid")
        return self
