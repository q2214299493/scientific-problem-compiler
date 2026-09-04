from __future__ import annotations

from importlib.resources import as_file, files

from ..models import (
    AgentCapabilityCatalog,
    AgentHandoffPackage,
    CapabilityBinding,
    ExecutionPolicy,
    ScientificQuestionPlan,
)
from ..serialization import content_hash, load_data


class FTAgentAdapter:
    target_agent = "ft-agent"
    supported_domains = ("fischer_tropsch",)

    def __init__(self, catalog: AgentCapabilityCatalog | None = None) -> None:
        if catalog is None:
            resource = files("spc.adapters.catalogs").joinpath("ft-agent.yaml")
            with as_file(resource) as path:
                catalog = AgentCapabilityCatalog.model_validate(load_data(path))
        self.catalog = catalog

    def bind_capabilities(self, plan: ScientificQuestionPlan) -> tuple[CapabilityBinding, ...]:
        mapping = {
            scientific_id: item.capability_id
            for item in self.catalog.capabilities
            for scientific_id in item.supports_scientific_capability_ids
        }
        bindings: list[CapabilityBinding] = []
        for capability_id in plan.scientific_capability_ids:
            if capability_id in mapping:
                bindings.append(
                    CapabilityBinding(
                        scientific_capability_id=capability_id,
                        target_capability_id=mapping[capability_id],
                        status="available",
                    )
                )
            else:
                bindings.append(
                    CapabilityBinding(
                        scientific_capability_id=capability_id,
                        status="unavailable",
                        reason="capability is absent from the static FT Agent catalog",
                    )
                )
        return tuple(bindings)

    def build_handoff(self, plan: ScientificQuestionPlan, export_id: str) -> AgentHandoffPackage:
        return AgentHandoffPackage(
            export_id=export_id,
            target_agent=self.target_agent,
            source_plan_id=plan.plan_id,
            source_plan_version=plan.version,
            source_plan_hash=content_hash(plan),
            capability_bindings=self.bind_capabilities(plan),
            execution_policy=ExecutionPolicy(),
        )
