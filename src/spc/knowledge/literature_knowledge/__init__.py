"""Evidence-grounded, curation-gated literature knowledge compilation."""

from dataclasses import dataclass

from ...repositories import EvidenceStore, KnowledgeRepositories
from .codex_transport import (
    CodexCLIExecutionError,
    CodexCLILLMTransport,
    CodexCLIRuntime,
    CodexCLIUnavailableError,
)
from .context import (
    build_literature_knowledge_chunks,
    resolve_literature_knowledge_input,
    validate_literature_knowledge_chunk,
    validate_literature_knowledge_input,
)
from .contracts import (
    LiteratureClaimProposal,
    LiteratureKnowledgeBatchInvocation,
    LiteratureKnowledgeChunk,
    LiteratureKnowledgeCompilationInput,
    LiteratureKnowledgeCompilationRecord,
    LiteratureKnowledgeGroundingRecord,
    LiteratureKnowledgeLLMResponse,
    LiteratureKnowledgeProposalSet,
    LiteratureKnowledgeProviderInvocation,
    LiteratureKnowledgeRecordType,
    LiteratureKnowledgeSupportingQuoteView,
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
    DEFAULT_MAX_BATCH_TEXT_CHARACTERS,
    DEFAULT_MAX_CHUNKS_PER_BATCH,
    LiteratureKnowledgeProvider,
    MockLiteratureKnowledgeProvider,
    StructuredLiteratureKnowledgeOutputError,
    StructuredLLMLiteratureKnowledgeProvider,
    StructuredOutputDiagnostic,
    StructuredOutputFailureCategory,
    build_literature_knowledge_proposal_set,
    partition_literature_knowledge_chunks,
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
from .wire import (
    LiteratureClaimLLMWireProposal,
    LiteratureKnowledgeLLMWireEntry,
    LiteratureKnowledgeLLMWireResponse,
    LiteratureMethodFactLLMWireProposal,
    LiteratureModelFactLLMWireProposal,
    LiteratureQuoteLLMWireProposal,
    LiteratureRelationLLMWireProposal,
    LiteratureReportedResultLLMWireProposal,
    literature_knowledge_llm_wire_schema,
    literature_knowledge_wire_to_internal,
    validate_strict_structured_output_schema,
)


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
    "LiteratureClaimLLMWireProposal",
    "CodexCLIExecutionError",
    "CodexCLILLMTransport",
    "CodexCLIRuntime",
    "CodexCLIUnavailableError",
    "DEFAULT_MAX_BATCH_TEXT_CHARACTERS",
    "DEFAULT_MAX_CHUNKS_PER_BATCH",
    "LiteratureKnowledgeBatchInvocation",
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
    "LiteratureKnowledgeLLMWireEntry",
    "LiteratureKnowledgeLLMWireResponse",
    "LiteratureKnowledgeMaterializer",
    "LiteratureKnowledgeProposalSet",
    "LiteratureKnowledgeProviderInvocation",
    "LiteratureKnowledgeProposalSetRepository",
    "LiteratureKnowledgeProvider",
    "LiteratureKnowledgeRecordType",
    "LiteratureKnowledgeSupportingQuoteView",
    "LiteratureKnowledgeViewMode",
    "LiteratureMethodFactProposal",
    "LiteratureMethodFactLLMWireProposal",
    "LiteratureModelFactProposal",
    "LiteratureModelFactLLMWireProposal",
    "LiteratureQuoteProposal",
    "LiteratureQuoteLLMWireProposal",
    "LiteratureRelationProposal",
    "LiteratureRelationLLMWireProposal",
    "LiteratureReportedResultProposal",
    "LiteratureReportedResultLLMWireProposal",
    "LiteratureScientificKnowledgeView",
    "LiteratureScientificKnowledgeViewBuilder",
    "LiteratureScientificKnowledgeViewRecord",
    "MockLiteratureKnowledgeProvider",
    "RejectedLiteratureKnowledgeProposal",
    "StructuredLiteratureKnowledgeOutputError",
    "StructuredLLMLiteratureKnowledgeProvider",
    "StructuredOutputDiagnostic",
    "StructuredOutputFailureCategory",
    "build_literature_knowledge_chunks",
    "build_literature_knowledge_proposal_set",
    "curate_knowledge_record",
    "literature_knowledge_llm_wire_schema",
    "literature_knowledge_wire_to_internal",
    "partition_literature_knowledge_chunks",
    "resolve_literature_knowledge_input",
    "validate_literature_knowledge_chunk",
    "validate_grounding_record",
    "validate_literature_knowledge_input",
    "validate_strict_structured_output_schema",
]
