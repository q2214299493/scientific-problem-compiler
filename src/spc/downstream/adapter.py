from __future__ import annotations

from typing import Any, Mapping, Protocol

from ..models import AgentCapabilityCatalog


class AgentExecutionAdapter(Protocol):
    """Maps catalogued capabilities without receiving or mutating an SPC plan."""

    adapter_id: str
    adapter_version: str
    target_agent: str
    supported_environments: tuple[str, ...]

    def map_capability(
        self,
        scientific_capability_id: str,
        catalog: AgentCapabilityCatalog,
    ) -> str | None: ...

    def resource_requirements(
        self,
        executable_capability_id: str,
        target_environment: str,
    ) -> Mapping[str, Any]: ...
