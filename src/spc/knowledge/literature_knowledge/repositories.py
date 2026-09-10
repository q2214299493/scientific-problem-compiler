from __future__ import annotations

from pathlib import Path

from ...repositories import IdentityBoundRepository
from .contracts import (
    LiteratureKnowledgeChunk,
    LiteratureKnowledgeCompilationInput,
    LiteratureKnowledgeCompilationRecord,
    LiteratureKnowledgeGroundingRecord,
    LiteratureKnowledgeProposalSet,
)


class LiteratureKnowledgeCompilationInputRepository(
    IdentityBoundRepository[LiteratureKnowledgeCompilationInput]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_knowledge_inputs",
            LiteratureKnowledgeCompilationInput,
            "compilation_input_id",
        )


class LiteratureKnowledgeChunkRepository(IdentityBoundRepository[LiteratureKnowledgeChunk]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_knowledge_chunks",
            LiteratureKnowledgeChunk,
            "chunk_id",
        )


class LiteratureKnowledgeProposalSetRepository(
    IdentityBoundRepository[LiteratureKnowledgeProposalSet]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_knowledge_proposals",
            LiteratureKnowledgeProposalSet,
            "proposal_set_id",
        )


class LiteratureKnowledgeGroundingRepository(
    IdentityBoundRepository[LiteratureKnowledgeGroundingRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_knowledge_groundings",
            LiteratureKnowledgeGroundingRecord,
            "grounding_id",
        )


class LiteratureKnowledgeCompilationRepository(
    IdentityBoundRepository[LiteratureKnowledgeCompilationRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_knowledge_compilations",
            LiteratureKnowledgeCompilationRecord,
            "compilation_id",
        )
