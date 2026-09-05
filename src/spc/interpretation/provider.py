from __future__ import annotations

from typing import Protocol

from ..models import InterpretationProposal, ScientificContextPacket


class InterpretationProvider(Protocol):
    def interpret(self, context: ScientificContextPacket) -> InterpretationProposal: ...
