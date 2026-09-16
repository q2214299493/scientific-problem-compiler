from __future__ import annotations

from typing import Protocol

from ..models import (
    PlanRevisionInput,
    PlanRevisionLLMResponse,
    PlanningProposalSet,
    ScientificPlanningInput,
)


class PlanningProvider(Protocol):
    provider_id: str
    provider_version: str

    def propose(self, planning_input: ScientificPlanningInput) -> PlanningProposalSet: ...

    def revise(self, revision_input: PlanRevisionInput) -> PlanRevisionLLMResponse: ...
