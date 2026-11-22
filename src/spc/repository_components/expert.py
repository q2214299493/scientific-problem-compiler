from __future__ import annotations

from pathlib import Path

from ..models import (
    ExpertAttributionRecord,
    ExpertCase,
    ExpertOpinion,
    ExpertProfile,
)

from .base import CatalogRepository, IdentityBoundRepository
from .evidence import KnowledgeEvidenceStore


class ExpertCaseRepository(CatalogRepository[ExpertCase]):
    _key_error = "expert case repository key must equal case_id"
    _stored_id_error = "stored expert case ID does not match repository key"
    _duplicate_error = "expert case repository contains duplicate record IDs"

    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root / "expert_cases", ExpertCase, "case_id")


class ExpertProfileRepository(IdentityBoundRepository[ExpertProfile]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "expert_profiles",
            ExpertProfile,
            "expert_id",
        )


class ExpertOpinionRepository(IdentityBoundRepository[ExpertOpinion]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "expert_opinions",
            ExpertOpinion,
            "opinion_id",
        )


class ExpertAttributionRepository(
    IdentityBoundRepository[ExpertAttributionRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "expert_attributions",
            ExpertAttributionRecord,
            "attribution_id",
        )


class ExpertRepositories:
    def __init__(
        self,
        root: Path,
        *,
        initialize: bool = True,
        evidence_store: KnowledgeEvidenceStore | None = None,
    ) -> None:
        from ..knowledge.expert_knowledge.repositories import (
            ExpertKnowledgeChunkRepository,
            ExpertKnowledgeCompilationInputRepository,
            ExpertKnowledgeCompilationRepository,
            ExpertKnowledgeGroundingRepository,
            ExpertKnowledgeProposalSetRepository,
            ExpertSourceRepository,
        )

        self.root = root
        self.evidence_store = (
            KnowledgeEvidenceStore(root, initialize=initialize) if evidence_store is None else evidence_store
        )
        self.expert_sources = ExpertSourceRepository(root)
        self.expert_cases = ExpertCaseRepository(root)
        self.expert_profiles = ExpertProfileRepository(root)
        self.expert_opinions = ExpertOpinionRepository(root)
        self.expert_attributions = ExpertAttributionRepository(root)
        self.expert_knowledge_inputs = ExpertKnowledgeCompilationInputRepository(root)
        self.expert_knowledge_chunks = ExpertKnowledgeChunkRepository(root)
        self.expert_knowledge_proposals = ExpertKnowledgeProposalSetRepository(root)
        self.expert_knowledge_groundings = ExpertKnowledgeGroundingRepository(root)
        self.expert_knowledge_compilations = ExpertKnowledgeCompilationRepository(root)
