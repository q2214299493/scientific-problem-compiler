from __future__ import annotations

from typing import Protocol

from ..models import (
    DirectionExpansionLLMResponse,
    DirectionTriageLLMResponse,
    DirectionTriageRecord,
    PlanRevisionInput,
    PlanRevisionLLMResponse,
    PlanningProposalSet,
    ResearchDirectionLLMResponse,
    ResearchDirectionSet,
    ScientificPlanningInput,
)


class PlanningProvider(Protocol):
    provider_id: str
    provider_version: str

    def propose(self, planning_input: ScientificPlanningInput) -> PlanningProposalSet: ...

    def propose_directions(
        self, planning_input: ScientificPlanningInput
    ) -> ResearchDirectionLLMResponse: ...

    def triage_directions(
        self,
        planning_input: ScientificPlanningInput,
        directions: ResearchDirectionSet,
    ) -> DirectionTriageLLMResponse: ...

    def expand_directions(
        self,
        planning_input: ScientificPlanningInput,
        directions: ResearchDirectionSet,
        triage: DirectionTriageRecord,
    ) -> DirectionExpansionLLMResponse: ...

    def revise(self, revision_input: PlanRevisionInput) -> PlanRevisionLLMResponse: ...
