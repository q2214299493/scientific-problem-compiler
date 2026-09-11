from __future__ import annotations

from ...repositories import IdentityBoundRepository
from .contracts import (
    ExpertKnowledgeChunk,
    ExpertKnowledgeCompilationInput,
    ExpertKnowledgeCompilationRecord,
    ExpertKnowledgeGroundingRecord,
    ExpertKnowledgeProposalSet,
    ExpertSourceRecord,
)


class ExpertSourceRepository(IdentityBoundRepository[ExpertSourceRecord]):
    def __init__(self, knowledge_root):
        super().__init__(
            knowledge_root / "expert_sources",
            ExpertSourceRecord,
            "expert_source_id",
        )


class ExpertKnowledgeCompilationInputRepository(
    IdentityBoundRepository[ExpertKnowledgeCompilationInput]
):
    def __init__(self, knowledge_root):
        super().__init__(
            knowledge_root / "expert_knowledge_inputs",
            ExpertKnowledgeCompilationInput,
            "compilation_input_id",
        )


class ExpertKnowledgeChunkRepository(IdentityBoundRepository[ExpertKnowledgeChunk]):
    def __init__(self, knowledge_root):
        super().__init__(
            knowledge_root / "expert_knowledge_chunks",
            ExpertKnowledgeChunk,
            "chunk_id",
        )


class ExpertKnowledgeProposalSetRepository(
    IdentityBoundRepository[ExpertKnowledgeProposalSet]
):
    def __init__(self, knowledge_root):
        super().__init__(
            knowledge_root / "expert_knowledge_proposals",
            ExpertKnowledgeProposalSet,
            "proposal_set_id",
        )


class ExpertKnowledgeGroundingRepository(
    IdentityBoundRepository[ExpertKnowledgeGroundingRecord]
):
    def __init__(self, knowledge_root):
        super().__init__(
            knowledge_root / "expert_knowledge_groundings",
            ExpertKnowledgeGroundingRecord,
            "grounding_id",
        )


class ExpertKnowledgeCompilationRepository(
    IdentityBoundRepository[ExpertKnowledgeCompilationRecord]
):
    def __init__(self, knowledge_root):
        super().__init__(
            knowledge_root / "expert_knowledge_compilations",
            ExpertKnowledgeCompilationRecord,
            "compilation_id",
        )


__all__ = [
    "ExpertKnowledgeChunkRepository",
    "ExpertKnowledgeCompilationInputRepository",
    "ExpertKnowledgeCompilationRepository",
    "ExpertKnowledgeGroundingRepository",
    "ExpertKnowledgeProposalSetRepository",
    "ExpertSourceRepository",
]
