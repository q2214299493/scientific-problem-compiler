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
    adapter_id = "spc.ft-agent-execution-adapter"
    adapter_version = "1.0.0"
    supported_environments = ("ft-agent-staging",)

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

    def map_capability(
        self,
        scientific_capability_id: str,
        catalog: AgentCapabilityCatalog,
    ) -> str | None:
        """Resolve only mappings declared by the supplied agent catalog."""

        if catalog.agent_id != self.target_agent:
            return None
        matches = sorted(
            capability.capability_id
            for capability in catalog.capabilities
            if scientific_capability_id
            in capability.supports_scientific_capability_ids
        )
        return matches[0] if len(matches) == 1 else None

    def resource_requirements(
        self,
        executable_capability_id: str,
        target_environment: str,
    ) -> dict[str, str]:
        """Describe the handoff boundary without producing execution commands."""

        return {
            "capability": executable_capability_id,
            "environment": target_environment,
            "allocation": "not_authorized",
        }

    def input_bindings(self, executable_capability_id: str) -> dict[str, str]:
        bindings = {
            "ft.review_evidence": {
                "source_query_manifest": "source_query_manifest",
                "reaction_scope": "scientific_objective",
            },
            "ft.plan_pathway_comparison": {
                "hypotheses": "hypothesis",
                "model": "model",
                "observables": "observables",
                "baseline": "comparison_baselines",
            },
        }
        return bindings.get(executable_capability_id, {})

    def reconcile_outputs(
        self,
        executable_capability_id: str,
        expected_outputs: tuple[str, ...],
        declared_outputs: tuple[str, ...],
    ) -> dict[str, str]:
        target_by_capability = {
            "ft.review_evidence": "mechanistic_evidence_assessment",
            "ft.plan_pathway_comparison": "pathway_comparison_specification",
        }
        target = target_by_capability.get(executable_capability_id)
        if target is None or target not in declared_outputs:
            return {}
        compatible_outputs = {
            "evidence-grounded discrimination record",
            target,
        }
        if not set(expected_outputs).issubset(compatible_outputs):
            return {}
        return {expected: target for expected in expected_outputs}
