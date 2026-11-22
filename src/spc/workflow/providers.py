from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..approval import (
    MockApprovalProvider,
    StructuredLLMApprovalProvider,
)
from ..interpretation import MockInterpretationProvider
from ..models import (
    ScientificRunProviderBinding,
)
from ..planning import (
    HTTPJSONLLMTransport,
    MockPlanningProvider,
    StructuredLLMPlanningProvider,
)
from ..serialization import content_hash


@dataclass(frozen=True)
class ProviderSettings:
    llm_endpoint: str | None
    llm_model: str | None
    llm_api_key: str | None = field(repr=False)
    temperature: float
    max_attempts: int

    def planning(self, name: str):
        if name == "mock":
            return MockPlanningProvider()
        if name != "llm":
            raise ValueError("planning provider must be 'mock' or 'llm'")
        endpoint = self.llm_endpoint
        model = self.llm_model
        if endpoint is None or model is None:
            raise ValueError("planning provider llm requires --llm-endpoint and --llm-model")
        return StructuredLLMPlanningProvider(
            HTTPJSONLLMTransport(endpoint, model, api_key=self.llm_api_key),
            temperature=self.temperature,
            max_attempts=self.max_attempts,
        )

    def approval(self, provider_name: str):
        if provider_name == "mock":
            return MockApprovalProvider()
        elif provider_name == "llm":
            if self.llm_endpoint is None or self.llm_model is None:
                raise ValueError("approval provider llm requires --llm-endpoint and --llm-model")
            return StructuredLLMApprovalProvider(
                HTTPJSONLLMTransport(self.llm_endpoint, self.llm_model, api_key=self.llm_api_key),
                temperature=self.temperature,
                max_attempts=self.max_attempts,
            )
        else:
            raise ValueError("approval provider must be 'none', 'mock', or 'llm'")


def interpretation_provider():
    return MockInterpretationProvider()


def bind_provider(stage: str, provider: Any) -> ScientificRunProviderBinding:
    config = dict(getattr(provider, "provider_config", {}))
    model_id = getattr(getattr(provider, "transport", None), "model_id", None)
    for attribute in ("temperature", "max_attempts"):
        value = getattr(provider, attribute, None)
        if value is not None:
            config[attribute] = value
    public_config = {
        "provider_id": provider.provider_id,
        "provider_version": provider.provider_version,
        "model_id": model_id,
        "configuration": config,
    }
    return ScientificRunProviderBinding(
        stage=stage,
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        model_id=model_id,
        configuration_hash=content_hash(public_config),
    )


def replace_provider(
    providers: tuple[ScientificRunProviderBinding, ...],
    binding: ScientificRunProviderBinding,
) -> tuple[ScientificRunProviderBinding, ...]:
    return tuple(item for item in providers if item.stage != binding.stage) + (binding,)
