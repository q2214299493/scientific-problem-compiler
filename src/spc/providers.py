from __future__ import annotations

from typing import Protocol, Sequence

from .models import ScientificQuestionPlan


class Provider(Protocol):
    def compile(self, request: str, *, domain_context: str) -> Sequence[ScientificQuestionPlan]: ...


class MockProvider:
    """Offline provider that returns test-supplied plans without interpretation."""

    def __init__(self, plans: Sequence[ScientificQuestionPlan]) -> None:
        self._plans = tuple(plans)

    def compile(self, request: str, *, domain_context: str) -> Sequence[ScientificQuestionPlan]:
        del request, domain_context
        return self._plans
