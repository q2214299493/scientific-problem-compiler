"""Evidence-grounded, curation-gated expert knowledge compilation."""

from __future__ import annotations

from dataclasses import dataclass

from ...repositories import KnowledgeRepositories
from .context import build_expert_knowledge_chunks, resolve_expert_knowledge_input
from .contracts import (
    ExpertCaseProposal,
    ExpertKnowledgeBatchInvocation,
    ExpertKnowledgeChunk,
    ExpertKnowledgeCompilationInput,
    ExpertKnowledgeCompilationRecord,
    ExpertKnowledgeGroundingRecord,
    ExpertKnowledgeLLMResponse,
    ExpertKnowledgeProposalSet,
    ExpertKnowledgeViewMode,
    ExpertOpinionProposal,
    ExpertQuoteProposal,
    ExpertSourcePage,
    ExpertSourceRecord,
    RejectedExpertKnowledgeProposal,
)
from .materializer import (
    ExpertKnowledgeMaterializationOutcome,
    ExpertKnowledgeMaterializer,
)
from .provider import (
    DEFAULT_MAX_BATCH_TEXT_CHARACTERS,
    DEFAULT_MAX_CHUNKS_PER_BATCH,
    ExpertKnowledgeProvider,
    MockExpertKnowledgeProvider,
    StructuredLLMExpertKnowledgeProvider,
    partition_expert_knowledge_chunks,
)
from .source import ExpertSourceIngestionService, make_expert_profile, validate_expert_source
from .view import ExpertKnowledgeViewBuilder
from .wire import (
    ExpertCaseLLMWireProposal,
    ExpertKnowledgeLLMWireResponse,
    ExpertOpinionLLMWireProposal,
    ExpertQuoteLLMWireProposal,
    expert_knowledge_llm_wire_schema,
    expert_knowledge_wire_to_internal,
)


@dataclass(frozen=True)
class ExpertKnowledgeCompilerOutcome:
    compilation_input: ExpertKnowledgeCompilationInput
    chunks: tuple[ExpertKnowledgeChunk, ...]
    proposal_set: ExpertKnowledgeProposalSet
    materialized: ExpertKnowledgeMaterializationOutcome


class ExpertKnowledgeCompiler:
    def compile(
        self,
        expert_source_id: str,
        provider: ExpertKnowledgeProvider,
        repositories: KnowledgeRepositories,
    ) -> ExpertKnowledgeCompilerOutcome:
        compilation_input = resolve_expert_knowledge_input(expert_source_id, repositories)
        chunks = build_expert_knowledge_chunks(compilation_input, repositories)
        proposal_set = provider.propose(compilation_input, chunks)
        materialized = ExpertKnowledgeMaterializer().materialize(
            compilation_input,
            chunks,
            proposal_set,
            repositories,
        )
        return ExpertKnowledgeCompilerOutcome(
            compilation_input=compilation_input,
            chunks=chunks,
            proposal_set=proposal_set,
            materialized=materialized,
        )


__all__ = [
    "DEFAULT_MAX_BATCH_TEXT_CHARACTERS",
    "DEFAULT_MAX_CHUNKS_PER_BATCH",
    "ExpertCaseLLMWireProposal",
    "ExpertCaseProposal",
    "ExpertKnowledgeBatchInvocation",
    "ExpertKnowledgeChunk",
    "ExpertKnowledgeCompilationInput",
    "ExpertKnowledgeCompilationRecord",
    "ExpertKnowledgeCompiler",
    "ExpertKnowledgeCompilerOutcome",
    "ExpertKnowledgeGroundingRecord",
    "ExpertKnowledgeLLMResponse",
    "ExpertKnowledgeLLMWireResponse",
    "ExpertKnowledgeMaterializationOutcome",
    "ExpertKnowledgeMaterializer",
    "ExpertKnowledgeProposalSet",
    "ExpertKnowledgeProvider",
    "ExpertKnowledgeViewBuilder",
    "ExpertKnowledgeViewMode",
    "ExpertOpinionLLMWireProposal",
    "ExpertOpinionProposal",
    "ExpertQuoteLLMWireProposal",
    "ExpertQuoteProposal",
    "ExpertSourceIngestionService",
    "ExpertSourcePage",
    "ExpertSourceRecord",
    "MockExpertKnowledgeProvider",
    "RejectedExpertKnowledgeProposal",
    "StructuredLLMExpertKnowledgeProvider",
    "build_expert_knowledge_chunks",
    "expert_knowledge_llm_wire_schema",
    "expert_knowledge_wire_to_internal",
    "make_expert_profile",
    "partition_expert_knowledge_chunks",
    "resolve_expert_knowledge_input",
    "validate_expert_source",
]
