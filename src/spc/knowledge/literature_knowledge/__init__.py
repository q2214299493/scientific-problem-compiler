"""Evidence-grounded, curation-gated literature knowledge compilation."""

from dataclasses import dataclass

from ...repositories import EvidenceStore, KnowledgeRepositories
from .context import (
    build_literature_knowledge_chunks,
    resolve_literature_knowledge_input,
    validate_literature_knowledge_chunk,
    validate_literature_knowledge_input,
)
from .contracts import (
    LiteratureClaimProposal,
    LiteratureKnowledgeChunk,
    LiteratureKnowledgeCompilationInput,
    LiteratureKnowledgeCompilationRecord,
    LiteratureKnowledgeGroundingRecord,
    LiteratureKnowledgeLLMResponse,
    LiteratureKnowledgeProposalSet,
    LiteratureKnowledgeRecordType,
    LiteratureKnowledgeViewMode,
    LiteratureMethodFactProposal,
    LiteratureModelFactProposal,
    LiteratureQuoteProposal,
    LiteratureRelationProposal,
    LiteratureReportedResultProposal,
    LiteratureScientificKnowledgeView,
    LiteratureScientificKnowledgeViewRecord,
    RejectedLiteratureKnowledgeProposal,
)
from .materializer import (
    LiteratureKnowledgeCompilationOutcome,
    LiteratureKnowledgeMaterializer,
)
from .provider import (
    LiteratureKnowledgeProvider,
    MockLiteratureKnowledgeProvider,
    StructuredLiteratureKnowledgeOutputError,
    StructuredLLMLiteratureKnowledgeProvider,
    build_literature_knowledge_proposal_set,
)
from .repositories import (
    LiteratureKnowledgeChunkRepository,
    LiteratureKnowledgeCompilationInputRepository,
    LiteratureKnowledgeCompilationRepository,
    LiteratureKnowledgeGroundingRepository,
    LiteratureKnowledgeProposalSetRepository,
)
from .validation import curate_knowledge_record, validate_grounding_record
from .view import LiteratureScientificKnowledgeViewBuilder


@dataclass(frozen=True)
class LiteratureKnowledgeCompilerOutcome:
    compilation_input: LiteratureKnowledgeCompilationInput
    chunks: tuple[LiteratureKnowledgeChunk, ...]
    proposal_set: LiteratureKnowledgeProposalSet
    materialized: LiteratureKnowledgeCompilationOutcome


class LiteratureKnowledgeCompiler:
    def compile(
        self,
        literature_id: str,
        provider: LiteratureKnowledgeProvider,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
        *,
        include_supplementary: bool = False,
    ) -> LiteratureKnowledgeCompilerOutcome:
        store = evidence_store or repositories.evidence_store
        compilation_input = resolve_literature_knowledge_input(
            literature_id,
            repositories,
            store,
            include_supplementary=include_supplementary,
        )
        chunks = build_literature_knowledge_chunks(compilation_input, repositories, store)
        proposal_set = provider.propose(compilation_input, chunks)
        materialized = LiteratureKnowledgeMaterializer().materialize(
            compilation_input,
            chunks,
            proposal_set,
            repositories,
            store,
        )
        return LiteratureKnowledgeCompilerOutcome(
            compilation_input=compilation_input,
            chunks=chunks,
            proposal_set=proposal_set,
            materialized=materialized,
        )


__all__ = [
    "LiteratureClaimProposal",
    "LiteratureKnowledgeChunk",
    "LiteratureKnowledgeChunkRepository",
    "LiteratureKnowledgeCompilationInput",
    "LiteratureKnowledgeCompilationInputRepository",
    "LiteratureKnowledgeCompilationOutcome",
    "LiteratureKnowledgeCompilationRecord",
    "LiteratureKnowledgeCompilationRepository",
    "LiteratureKnowledgeCompiler",
    "LiteratureKnowledgeCompilerOutcome",
    "LiteratureKnowledgeGroundingRecord",
    "LiteratureKnowledgeGroundingRepository",
    "LiteratureKnowledgeLLMResponse",
    "LiteratureKnowledgeMaterializer",
    "LiteratureKnowledgeProposalSet",
    "LiteratureKnowledgeProposalSetRepository",
    "LiteratureKnowledgeProvider",
    "LiteratureKnowledgeRecordType",
    "LiteratureKnowledgeViewMode",
    "LiteratureMethodFactProposal",
    "LiteratureModelFactProposal",
    "LiteratureQuoteProposal",
    "LiteratureRelationProposal",
    "LiteratureReportedResultProposal",
    "LiteratureScientificKnowledgeView",
    "LiteratureScientificKnowledgeViewBuilder",
    "LiteratureScientificKnowledgeViewRecord",
    "MockLiteratureKnowledgeProvider",
    "RejectedLiteratureKnowledgeProposal",
    "StructuredLiteratureKnowledgeOutputError",
    "StructuredLLMLiteratureKnowledgeProvider",
    "build_literature_knowledge_chunks",
    "build_literature_knowledge_proposal_set",
    "curate_knowledge_record",
    "resolve_literature_knowledge_input",
    "validate_literature_knowledge_chunk",
    "validate_grounding_record",
    "validate_literature_knowledge_input",
]
