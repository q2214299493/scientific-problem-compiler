from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from .backends import (
    BackendCapability,
    BackendInvocationError,
    BackendRegistryError,
    BackendRunRecord,
    BackendRuntimeAvailability,
    BackendRuntimeArtifactManifest,
    BackendRuntimeIdentity,
    BackendUnavailableError,
    ExternalBackendDescriptor,
    ExternalDocumentElement,
    ExternalDocumentParseInput,
    ExternalDocumentParseProposal,
    ExternalDocumentStructureService,
    ExternalLiteratureRetrievalHit,
    ExternalLiteratureRetrievalQuery,
    ExternalLiteratureRetrievalResult,
    ExternalRetrievalResolution,
    ExternalRetrievalResolutionBatch,
    ExternalStructurePromotionRecord,
    ExternalStructureRebindingResult,
    ReboundDocumentElement,
    ScholarlyMetadataProposal,
    default_backend_registry,
    validate_backend_run,
)
from .adapters.ft_agent import FTAgentAdapter
from .approval import (
    ApprovalContextError,
    ApprovalContextResolver,
    ApprovalResponseError,
    ApprovalStructuredOutputError,
    IndependentApprovalService,
    MockApprovalProvider,
    ScientificPlanApprover,
    StructuredLLMApprovalProvider,
)
from .compiler import ScientificProblemCompiler
from .domains import DomainPackLoader
from .export import ExportError, GenericExportService
from .interpretation import (
    EvidencePacketIntegrityError,
    MockInterpretationProvider,
    ScientificEvidencePacketBuilder,
)
from .models import (
    AcquisitionAttemptRecord,
    AmbiguityAssessment,
    AgentCapabilityCatalog,
    AgentHandoffPackage,
    ApprovalMode,
    ApprovalScores,
    ApprovalDimensionScore,
    ApprovalHardRedFlag,
    ApprovalLLMResponse,
    ApprovalReviewInput,
    ApprovalReviewRecord,
    ApprovalReviewScores,
    ApprovalVerdict,
    CanonicalTextArtifact,
    CanonicalTextBlock,
    CanonicalHTMLTextArtifact,
    CollectionAcquisitionLink,
    CollectionDefinition,
    CollectionDiff,
    CollectionDiscoveryResult,
    CollectionImportOutcome,
    CollectionImportRecord,
    CollectionLiteratureMembership,
    CollectionPageArtifact,
    CollectionPageRecord,
    CollectionResourceImportResult,
    CollectionResourceOccurrence,
    CollectionScopePolicy,
    CollectionSnapshot,
    DiscoveredCollectionResource,
    DocumentStructureArtifact,
    DocumentStructureBlock,
    DocumentStructureSelection,
    CandidatePlanDraft,
    CandidateTaskDraft,
    ComparisonBaselineDraft,
    ComparisonConstraint,
    ConflictSet,
    CriterionDraft,
    CurationStatus,
    DAGTask,
    DomainProfile,
    EvidenceReference,
    EvidenceAssessment,
    EvidenceGap,
    EvidenceSpan,
    ExpertOpinion,
    ExpertProfile,
    ExecutionProposal,
    ExportManifest,
    FixResolution,
    GateVerdict,
    FullTextCandidate,
    FigureStructure,
    HistoricalLiteratureEvidenceAuthorization,
    HTMLLiteratureIngestionRecord,
    HumanDecisionResolution,
    IndependentApprovalReceipt,
    IntentFingerprint,
    IntentInterpretation,
    InterpretationProposal,
    KnowledgeCurationRecord,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    KnowledgeRelation,
    KnowledgeSnapshot,
    LiteratureAcquisitionOutcome,
    LiteratureAcquisitionRecord,
    LiteratureAcquisitionRequest,
    LiteratureDocument,
    LiteratureIngestionOutcome,
    LiteratureIngestionRecord,
    LiteratureRepresentationSelection,
    LiteratureRepresentationSelectionOutcome,
    LiteratureRepresentationReference,
    MetadataMergeManifest,
    MetadataRetrievalRecord,
    MethodFingerprint,
    MethodFact,
    ModelFact,
    ObservableDraft,
    PlanCompilationReceipt,
    PlanValidationRecord,
    PlanningLLMResponse,
    PlanningProposalSet,
    ProposedDeviationDraft,
    ProjectTrustPolicy,
    RawLiteratureArtifact,
    RawHTMLLiteratureArtifact,
    RequiredFix,
    RetrievalHit,
    RetrievalManifest,
    RetrievalQuery,
    ReportedResult,
    ResolvedLiteratureResource,
    ResultContext,
    ScientificCapability,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificQuestionPlan,
    ScientificTaskExecutionContext,
    SPCExportPackage,
    SourceClaim,
    SourceQuote,
    SourceDocument,
    StructuredEvidenceLocator,
    SystemFingerprint,
    TableCellStructure,
    TableStructure,
)
from .planning import (
    HTTPJSONLLMTransport,
    MockPlanningProvider,
    PlanningContextError,
    PlanningContextResolver,
    PlanningProposalError,
    StructuredLLMPlanningProvider,
    StructuredOutputError,
)
from .knowledge.ingestion import (
    LiteratureIngestionService,
    LiteratureRepresentationSelector,
)
from .knowledge.acquisition import LiteratureAcquisitionService
from .knowledge.collection import (
    CollectionImportService,
    make_collection_definition,
    make_collection_scope_policy,
)
from .knowledge.evidence_migration import migrate_knowledge_evidence
from .knowledge.literature_knowledge import (
    CodexCLIExecutionError,
    CodexCLILLMTransport,
    CodexCLIUnavailableError,
    DEFAULT_MAX_BATCH_TEXT_CHARACTERS,
    DEFAULT_MAX_CHUNKS_PER_BATCH,
    LiteratureClaimProposal,
    LiteratureKnowledgeChunk,
    LiteratureKnowledgeCompilationInput,
    LiteratureKnowledgeCompilationRecord,
    LiteratureKnowledgeCompiler,
    LiteratureKnowledgeGroundingRecord,
    LiteratureKnowledgeLLMResponse,
    LiteratureKnowledgeProposalSet,
    LiteratureKnowledgeSupportingQuoteView,
    LiteratureMethodFactProposal,
    LiteratureModelFactProposal,
    LiteratureQuoteProposal,
    LiteratureRelationProposal,
    LiteratureReportedResultProposal,
    LiteratureScientificKnowledgeView,
    LiteratureScientificKnowledgeViewBuilder,
    MockLiteratureKnowledgeProvider,
    StructuredLiteratureKnowledgeOutputError,
    StructuredLLMLiteratureKnowledgeProvider,
    curate_knowledge_record,
)
from .knowledge.structure import inspect_document_structure
from .knowledge.structure_selection import DocumentStructureSelector
from .providers import MockProvider
from .retrieval import ScientificContextBuilder
from .repositories import (
    CompositeEvidenceStore,
    KnowledgeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
    SourceEvidenceStore,
    initialize_state,
)
from .serialization import (
    dump_yaml,
    export_json_schemas,
    load_data,
    load_model,
    require_safe_path_component,
)
from .validators import (
    build_plan_validation_record,
    compare_method_fingerprints,
    validate_export,
    validate_question_plan,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Compile evidence-grounded scientific question plans without executing science.",
)


def _emit_report(report: object) -> None:
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


def _evidence_view(knowledge_dir: Path, state_dir: Path) -> CompositeEvidenceStore:
    return CompositeEvidenceStore(
        KnowledgeEvidenceStore(knowledge_dir),
        ProjectEvidenceStore(state_dir),
    )


@app.command()
def ingest(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    source_id: Annotated[str, typer.Option("--source-id")],
    version: Annotated[str, typer.Option("--version")],
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    title: Annotated[str | None, typer.Option("--title")] = None,
    source_role: Annotated[str, typer.Option("--source-role")] = "unspecified",
    source_type: Annotated[str, typer.Option("--source-type")] = "unspecified",
) -> None:
    """Copy a source into the versioned, read-only evidence store."""
    record = SourceEvidenceStore(state_dir).ingest(
        source,
        source_id,
        version,
        title,
        source_role=source_role,
        source_type=source_type,
    )
    typer.echo(record.model_dump_json(indent=2))


@app.command("ingest-literature")
def ingest_literature(
    paper: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    metadata: Annotated[Path, typer.Option("--metadata", exists=True, dir_okay=False, readable=True)],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
) -> None:
    """Persist a born-digital PDF and its canonical UTF-8 representation."""
    metadata_payload = load_data(metadata)
    if not isinstance(metadata_payload, dict):
        raise typer.BadParameter("--metadata must contain a mapping")
    outcome = LiteratureIngestionService().ingest(
        paper,
        metadata_payload,
        KnowledgeRepositories(knowledge_dir),
        KnowledgeEvidenceStore(knowledge_dir),
    )
    typer.echo(outcome.model_dump_json(indent=2))


@app.command("add-literature")
def add_literature(
    source: Annotated[str, typer.Argument()],
    domain: Annotated[str, typer.Option("--domain")],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    metadata: Annotated[
        Path | None,
        typer.Option("--metadata", exists=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Resolve a DOI, article URL, or local PDF without trusting its science."""
    metadata_payload = load_data(metadata) if metadata is not None else None
    if metadata_payload is not None and not isinstance(metadata_payload, dict):
        raise typer.BadParameter("--metadata must contain a mapping")
    outcome = LiteratureAcquisitionService().add(
        source,
        domain,
        KnowledgeRepositories(knowledge_dir),
        KnowledgeEvidenceStore(knowledge_dir),
        explicit_metadata=metadata_payload,
    )
    typer.echo(outcome.model_dump_json(indent=2))


@app.command("import-literature-collection")
def import_literature_collection(
    project_url: Annotated[str, typer.Argument()],
    domain: Annotated[str, typer.Option("--domain")],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    connector: Annotated[str, typer.Option("--connector")] = "generic-html",
    max_pages: Annotated[int, typer.Option("--max-pages", min=1)] = 100,
    max_resources: Annotated[int, typer.Option("--max-resources", min=1)] = 1000,
    max_depth: Annotated[int, typer.Option("--max-depth", min=0)] = 10,
    allowed_origin: Annotated[
        list[str] | None,
        typer.Option("--allowed-origin"),
    ] = None,
    allowed_path_prefix: Annotated[
        list[str] | None,
        typer.Option("--allowed-path-prefix"),
    ] = None,
    allow_external_literature_links: Annotated[
        bool,
        typer.Option("--allow-external-literature-links"),
    ] = False,
) -> None:
    """Discover a bounded static collection and acquire each unique resource."""
    if connector != "generic-html":
        raise typer.BadParameter("only the generic-html connector is available")
    policy = make_collection_scope_policy(
        project_url,
        max_pages=max_pages,
        max_depth=max_depth,
        max_resources=max_resources,
        allowed_origins=(tuple(allowed_origin) if allowed_origin else None),
        allowed_path_prefixes=(tuple(allowed_path_prefix) if allowed_path_prefix else None),
        allow_external_literature_links=allow_external_literature_links,
    )
    definition = make_collection_definition(
        project_url,
        domain,
        scope_policy=policy,
    )
    outcome = CollectionImportService().run(
        definition,
        KnowledgeRepositories(knowledge_dir),
        KnowledgeEvidenceStore(knowledge_dir),
    )
    typer.echo(outcome.model_dump_json(indent=2))


@app.command("select-literature-representation")
def select_literature_representation(
    literature_id: Annotated[str, typer.Option("--literature-id")],
    selected_by: Annotated[str, typer.Option("--selected-by")],
    rationale: Annotated[str, typer.Option("--rationale")],
    representation_id: Annotated[str | None, typer.Option("--representation-id")] = None,
    ingestion_id: Annotated[
        str | None,
        typer.Option(
            "--ingestion-id",
            help="Compatibility-only PDF ingestion selector.",
        ),
    ] = None,
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
) -> None:
    """Select one validated PDF/HTML representation without curating it."""
    if (representation_id is None) == (ingestion_id is None):
        raise typer.BadParameter("provide exactly one of --representation-id or compatibility --ingestion-id")
    outcome = LiteratureRepresentationSelector().select(
        literature_id,
        representation_id or ingestion_id or "",
        selected_by,
        rationale,
        KnowledgeRepositories(knowledge_dir),
        _evidence_view(knowledge_dir, state_dir),
    )
    typer.echo(outcome.model_dump_json(indent=2))


@app.command("migrate-knowledge-evidence")
def migrate_knowledge_evidence_command(
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
) -> None:
    """Copy legacy literature evidence into the shared knowledge evidence store."""
    result = migrate_knowledge_evidence(
        KnowledgeRepositories(knowledge_dir),
        ProjectEvidenceStore(state_dir),
        KnowledgeEvidenceStore(knowledge_dir),
    )
    typer.echo(json.dumps(asdict(result), indent=2, ensure_ascii=False))


@app.command("structure-literature")
def structure_literature(
    literature_id: Annotated[str, typer.Option("--literature-id")],
    representation_id: Annotated[str, typer.Option("--representation-id")],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    backend: Annotated[str, typer.Option("--backend")] = "builtin",
    promote_external: Annotated[
        bool,
        typer.Option(
            "--promote-external",
            help="Apply the deterministic selection policy after exact rebinding.",
        ),
    ] = False,
    docling_artifacts_path: Annotated[
        Path | None,
        typer.Option(
            "--docling-artifacts-path",
            help="Explicit local Docling model artifacts directory; network downloads are disabled.",
        ),
    ] = None,
) -> None:
    """Create an exact-offset document structure for one representation."""
    repositories = KnowledgeRepositories(knowledge_dir)
    evidence_store = KnowledgeEvidenceStore(knowledge_dir)
    backend_run_id = None
    external_proposal_id = None
    external_promotion_id = None
    external_promotion_policy_reason = None
    registry = default_backend_registry(docling_artifacts_path=docling_artifacts_path)
    if backend == "builtin":
        adapter = registry.resolve_capability(
            BackendCapability.BUILTIN_STRUCTURE,
            backend_id="builtin-document-parser",
        )
        result = adapter.structure(
            literature_id,
            representation_id,
            repositories,
            evidence_store,
        )
        selection = result.selection
    else:
        try:
            adapter = registry.resolve_capability(BackendCapability.DOCUMENT_PARSING, backend_id=backend)
        except (BackendRegistryError, ValueError) as error:
            raise typer.BadParameter(str(error)) from error
        if not hasattr(adapter, "parse"):
            raise typer.BadParameter(f"backend {backend!r} has no configured document parser")
        try:
            external = ExternalDocumentStructureService().structure(
                literature_id,
                representation_id,
                adapter,
                repositories,
                evidence_store,
                promote=promote_external,
            )
        except (BackendInvocationError, BackendUnavailableError) as error:
            raise typer.BadParameter(str(error)) from error
        result = external.structure
        selection = external.selection
        backend_run_id = external.run_record.run_id
        external_proposal_id = external.proposal.proposal_id
        external_promotion_id = external.promotion.promotion_id if external.promotion is not None else None
        external_promotion_policy_reason = external.promotion_policy_reason
    report = {
        "structure_id": result.artifact.structure_id,
        "structure_selection_id": (selection.selection_id if selection is not None else None),
        "authoritative": selection is not None,
        "backend_id": backend,
        "backend_run_id": backend_run_id,
        "external_proposal_id": external_proposal_id,
        "external_promotion_id": external_promotion_id,
        "external_promotion_policy_reason": external_promotion_policy_reason,
        "representation_id": result.artifact.representation_id,
        "pages": sum(block.block_type.value == "page" for block in result.blocks),
        "headings": sum(block.block_type.value == "heading" for block in result.blocks),
        "paragraphs": sum(block.block_type.value == "paragraph" for block in result.blocks),
        "tables": len(result.tables),
        "table_cells": len(result.table_cells),
        "figures": len(result.figures),
        "warnings": result.artifact.warnings,
    }
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@app.command("backends")
def list_backends(
    docling_artifacts_path: Annotated[
        Path | None,
        typer.Option("--docling-artifacts-path"),
    ] = None,
) -> None:
    """List registered backend adapters and current runtime availability."""
    registry = default_backend_registry(docling_artifacts_path=docling_artifacts_path)
    rows = []
    for descriptor in registry.list_descriptors():
        availability = registry.inspect_runtime(descriptor.backend_id)
        rows.append(
            {
                "backend_id": descriptor.backend_id,
                "capabilities": tuple(item.value for item in descriptor.capability_types),
                "available": availability.available,
                "backend_version": (availability.detected_version or descriptor.backend_version),
                "adapter_version": descriptor.adapter_version,
                "integration_mode": descriptor.integration_mode.value,
                "license_status": descriptor.license_status.value,
            }
        )
    typer.echo(json.dumps(rows, indent=2, ensure_ascii=False))


@app.command("backend-info")
def backend_info(
    backend_id: Annotated[str, typer.Argument()],
    docling_artifacts_path: Annotated[
        Path | None,
        typer.Option("--docling-artifacts-path"),
    ] = None,
) -> None:
    """Inspect one backend descriptor without importing optional engines."""
    registry = default_backend_registry(docling_artifacts_path=docling_artifacts_path)
    try:
        backend = registry.resolve(backend_id)
    except BackendRegistryError as error:
        raise typer.BadParameter(str(error)) from error
    availability = registry.inspect_runtime(backend_id)
    typer.echo(
        json.dumps(
            {
                "descriptor": backend.descriptor.model_dump(mode="json"),
                "runtime": availability.model_dump(mode="json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("inspect-backend-run")
def inspect_backend_run(
    run_id: Annotated[str, typer.Argument()],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
) -> None:
    """Inspect immutable backend invocation provenance."""
    validated = validate_backend_run(run_id, knowledge_dir)
    output = validated.output
    typer.echo(
        json.dumps(
            {
                "run": validated.run.model_dump(mode="json"),
                "descriptor": validated.descriptor.model_dump(mode="json"),
                "runtime_identity": validated.runtime_identity.model_dump(mode="json"),
                "runtime_artifact_manifest": (
                    validated.runtime_artifact_manifest.model_dump(mode="json")
                    if validated.runtime_artifact_manifest is not None
                    else None
                ),
                "output": output.model_dump(mode="json") if output is not None else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("check-codex-provider")
def check_codex_provider(
    codex_model: Annotated[str, typer.Option("--codex-model")],
    codex_executable: Annotated[str, typer.Option("--codex-executable")] = "codex",
) -> None:
    """Check Codex CLI K1F compatibility without running model inference."""
    try:
        runtime = CodexCLILLMTransport(
            codex_executable,
            model=codex_model,
        ).inspect_runtime()
    except (CodexCLIUnavailableError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(
        json.dumps(
            {
                "executable": runtime.executable,
                "cli_version": runtime.cli_version,
                "selected_model": runtime.selected_model,
                "authentication_status": runtime.authentication_status,
                "required_flags_accepted": True,
                "disabled_features": runtime.disabled_features,
                "model_inference_performed": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("extract-literature-knowledge")
def extract_literature_knowledge(
    literature_id: Annotated[str, typer.Option("--literature-id")],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    provider: Annotated[str, typer.Option("--provider")] = "mock",
    include_supplementary: Annotated[bool, typer.Option("--include-supplementary")] = False,
    llm_endpoint: Annotated[str | None, typer.Option("--llm-endpoint")] = None,
    llm_model: Annotated[str | None, typer.Option("--llm-model")] = None,
    llm_api_key_env: Annotated[str, typer.Option("--llm-api-key-env")] = "SPC_LLM_API_KEY",
    codex_executable: Annotated[str, typer.Option("--codex-executable")] = "codex",
    codex_model: Annotated[str | None, typer.Option("--codex-model")] = None,
    codex_timeout_seconds: Annotated[
        float,
        typer.Option("--codex-timeout-seconds"),
    ] = 300.0,
    codex_max_output_bytes: Annotated[
        int,
        typer.Option("--codex-max-output-bytes"),
    ] = 1_000_000,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    max_attempts: Annotated[int, typer.Option("--max-attempts")] = 1,
    max_chunks_per_batch: Annotated[
        int,
        typer.Option("--max-chunks-per-batch"),
    ] = DEFAULT_MAX_CHUNKS_PER_BATCH,
    max_batch_characters: Annotated[
        int,
        typer.Option("--max-batch-characters"),
    ] = DEFAULT_MAX_BATCH_TEXT_CHARACTERS,
    max_batches: Annotated[int | None, typer.Option("--max-batches")] = None,
    batch_failure_policy: Annotated[
        str,
        typer.Option("--batch-failure-policy"),
    ] = "stop",
    provider_output_dir: Annotated[
        Path | None,
        typer.Option("--provider-output-dir"),
    ] = None,
) -> None:
    """Compile untrusted, exactly grounded literature knowledge proposals."""
    if provider == "mock":
        selected_provider = MockLiteratureKnowledgeProvider()
        provider_notice = "offline test provider; output is not scientific paper interpretation"
    elif provider == "llm":
        if llm_endpoint is None or llm_model is None:
            raise typer.BadParameter(
                "--provider llm requires --llm-endpoint and --llm-model"
            )
        selected_provider = StructuredLLMLiteratureKnowledgeProvider(
            HTTPJSONLLMTransport(
                llm_endpoint,
                llm_model,
                api_key=os.getenv(llm_api_key_env),
            ),
            temperature=temperature,
            max_attempts=max_attempts,
            max_chunks_per_batch=max_chunks_per_batch,
            max_batch_text_characters=max_batch_characters,
            max_batches=max_batches,
            batch_failure_policy=batch_failure_policy,
            provider_output_dir=provider_output_dir,
        )
        provider_notice = "structured proposal provider; explicit human curation remains required"
    elif provider == "codex":
        if codex_model is None:
            raise typer.BadParameter(
                "--provider codex requires an explicit --codex-model"
            )
        transport = CodexCLILLMTransport(
            codex_executable,
            model=codex_model,
            timeout_seconds=codex_timeout_seconds,
            max_output_bytes=codex_max_output_bytes,
        )
        try:
            transport.inspect_runtime()
        except CodexCLIUnavailableError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(1) from error
        selected_provider = StructuredLLMLiteratureKnowledgeProvider(
            transport,
            temperature=temperature,
            max_attempts=max_attempts,
            max_chunks_per_batch=max_chunks_per_batch,
            max_batch_text_characters=max_batch_characters,
            max_batches=max_batches,
            batch_failure_policy=batch_failure_policy,
            provider_output_dir=provider_output_dir,
        )
        provider_notice = (
            "authenticated Codex CLI proposal provider; output is untrusted and explicit human "
            "curation remains required"
        )
    else:
        raise typer.BadParameter("--provider must be 'mock', 'llm', or 'codex'")
    repositories = KnowledgeRepositories(knowledge_dir)
    try:
        outcome = LiteratureKnowledgeCompiler().compile(
            literature_id,
            selected_provider,
            repositories,
            KnowledgeEvidenceStore(knowledge_dir),
            include_supplementary=include_supplementary,
        )
    except (
        CodexCLIExecutionError,
        CodexCLIUnavailableError,
        StructuredLiteratureKnowledgeOutputError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(
        json.dumps(
            {
                "compilation_id": outcome.materialized.record.compilation_id,
                "compilation_input_id": outcome.compilation_input.compilation_input_id,
                "proposal_set_id": outcome.proposal_set.proposal_set_id,
                "provider_id": outcome.proposal_set.provider_id,
                "provider_version": outcome.proposal_set.provider_version,
                "provider_config_hash": outcome.proposal_set.provider_config_hash,
                "provider_invocation": (
                    outcome.proposal_set.provider_invocation.model_dump(mode="json")
                    if outcome.proposal_set.provider_invocation is not None
                    else None
                ),
                "provider_notice": provider_notice,
                "chunk_count": len(outcome.chunks),
                "total_batch_count": getattr(selected_provider, "total_batch_count", 0),
                "processed_batch_count": len(outcome.proposal_set.batch_invocations or ()),
                "batch_invocations": [
                    item.model_dump(mode="json")
                    for item in (outcome.proposal_set.batch_invocations or ())
                ],
                "failed_batch_diagnostics": [
                    item.as_dict()
                    for item in getattr(selected_provider, "last_diagnostics", ())
                ],
                "region_uncertain_chunk_count": sum(chunk.region_uncertain for chunk in outcome.chunks),
                "source_quote_ids": outcome.materialized.record.source_quote_ids,
                "source_claim_ids": outcome.materialized.record.source_claim_ids,
                "method_fact_ids": outcome.materialized.record.method_fact_ids,
                "model_fact_ids": outcome.materialized.record.model_fact_ids,
                "reported_result_ids": outcome.materialized.record.reported_result_ids,
                "knowledge_relation_ids": outcome.materialized.record.knowledge_relation_ids,
                "rejected_proposals": [
                    item.model_dump(mode="json") for item in outcome.materialized.record.rejected_proposals
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("inspect-literature-knowledge")
def inspect_literature_knowledge(
    literature_id: Annotated[str, typer.Option("--literature-id")],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    view: Annotated[str, typer.Option("--view")] = "audit",
    record_type: Annotated[str | None, typer.Option("--record-type")] = None,
    curation_status: Annotated[str | None, typer.Option("--curation-status")] = None,
    content_region: Annotated[str | None, typer.Option("--content-region")] = None,
    section: Annotated[str | None, typer.Option("--section")] = None,
) -> None:
    """Inspect deterministic K1F records and their grounding provenance."""
    repositories = KnowledgeRepositories(knowledge_dir)
    result = LiteratureScientificKnowledgeViewBuilder().build(
        literature_id,
        repositories,
        KnowledgeEvidenceStore(knowledge_dir),
        view_mode=view,
        record_type=record_type,
        curation_status=curation_status,
        content_region=content_region,
        section=section,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("curate-knowledge")
def curate_knowledge(
    target_type: Annotated[str, typer.Option("--target-type")],
    target_id: Annotated[str, typer.Option("--target-id")],
    status: Annotated[CurationStatus, typer.Option("--status")],
    curator_id: Annotated[str, typer.Option("--curator-id")],
    rationale: Annotated[str, typer.Option("--rationale")],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
) -> None:
    """Append one immutable transition to the existing knowledge curation chain."""
    record = curate_knowledge_record(
        KnowledgeRepositories(knowledge_dir),
        target_type=target_type,
        target_id=target_id,
        status=status,
        curator_id=curator_id,
        rationale=rationale,
    )
    typer.echo(record.model_dump_json(indent=2))


@app.command("select-document-structure")
def select_document_structure(
    representation_id: Annotated[str, typer.Option("--representation-id")],
    structure_id: Annotated[str, typer.Option("--structure-id")],
    rationale: Annotated[str, typer.Option("--rationale")],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    allow_rollback: Annotated[
        bool,
        typer.Option(
            "--allow-rollback",
            help="Explicitly permit selecting an earlier structure from the immutable history.",
        ),
    ] = False,
) -> None:
    """Explicitly promote one stored document structure artifact."""
    repositories = KnowledgeRepositories(knowledge_dir)
    artifact = repositories.document_structure_artifacts.get(structure_id)
    selection = DocumentStructureSelector().select(
        artifact.literature_id,
        representation_id,
        structure_id,
        repositories,
        KnowledgeEvidenceStore(knowledge_dir),
        rationale=rationale,
        allow_rollback=allow_rollback,
    )
    typer.echo(selection.model_dump_json(indent=2))


@app.command("inspect-literature-structure")
def inspect_literature_structure(
    structure_id: Annotated[str, typer.Option("--structure-id")],
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    page: Annotated[int | None, typer.Option("--page", min=1)] = None,
    section: Annotated[
        list[str] | None,
        typer.Option("--section", help="Repeat for an exact section path."),
    ] = None,
    table_label: Annotated[str | None, typer.Option("--table-label")] = None,
    figure_label: Annotated[str | None, typer.Option("--figure-label")] = None,
) -> None:
    """Inspect a stored structure with deterministic exact-field filters."""
    result = inspect_document_structure(
        structure_id,
        KnowledgeRepositories(knowledge_dir),
        page=page,
        section_path=tuple(section) if section else None,
        table_label=table_label,
        figure_label=figure_label,
    )
    report = {
        "artifact": result["artifact"].model_dump(mode="json"),
        "blocks": [item.model_dump(mode="json") for item in result["blocks"]],
        "tables": [item.model_dump(mode="json") for item in result["tables"]],
        "table_cells": [item.model_dump(mode="json") for item in result["table_cells"]],
        "figures": [item.model_dump(mode="json") for item in result["figures"]],
    }
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@app.command()
def retrieve(
    request_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    domain: Annotated[str, typer.Option("--domain")],
    output: Annotated[Path, typer.Option("--output")],
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
) -> None:
    """Build an offline, evidence-grounded ScientificContextPacket."""
    packet = ScientificContextBuilder().build(
        request_file.read_text(encoding="utf-8"),
        domain,
        state_dir=state_dir,
        knowledge_dir=knowledge_dir,
    )
    dump_yaml(output, packet)
    typer.echo(str(output))


@app.command()
def interpret(
    context_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    provider: Annotated[str, typer.Option("--provider")] = "mock",
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
) -> None:
    """Build a validated ScientificEvidencePacket with an offline provider."""
    if provider != "mock":
        raise typer.BadParameter("Phase 2B only supports --provider mock")
    context = load_model(context_file, ScientificContextPacket)
    try:
        packet = ScientificEvidencePacketBuilder(MockInterpretationProvider()).build(
            context,
            _evidence_view(knowledge_dir, state_dir),
        )
    except EvidencePacketIntegrityError as error:
        _emit_report(error.report)
        raise typer.Exit(1) from error
    dump_yaml(output, packet)
    typer.echo(str(output))


@app.command()
def plan(
    context_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    evidence_packet_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    domain: Annotated[str, typer.Option("--domain")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    provider: Annotated[str, typer.Option("--provider")] = "mock",
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    llm_endpoint: Annotated[str | None, typer.Option("--llm-endpoint")] = None,
    llm_model: Annotated[str | None, typer.Option("--llm-model")] = None,
    llm_api_key_env: Annotated[str, typer.Option("--llm-api-key-env")] = "SPC_LLM_API_KEY",
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    max_attempts: Annotated[int, typer.Option("--max-attempts")] = 2,
) -> None:
    """Compile trusted context and evidence into validated candidate plans."""
    context = load_model(context_file, ScientificContextPacket)
    evidence_packet = load_model(evidence_packet_file, ScientificEvidencePacket)
    if context.domain != domain:
        raise typer.BadParameter("--domain must match the ScientificContextPacket domain")
    evidence_repository = _evidence_view(knowledge_dir, state_dir)
    try:
        planning_input = PlanningContextResolver().resolve(
            context,
            evidence_packet,
            KnowledgeRepositories(knowledge_dir),
            evidence_repository,
        )
        if provider == "mock":
            planning_provider = MockPlanningProvider()
        elif provider == "llm":
            if llm_endpoint is None or llm_model is None:
                raise typer.BadParameter("--provider llm requires --llm-endpoint and --llm-model")
            planning_provider = StructuredLLMPlanningProvider(
                HTTPJSONLLMTransport(
                    llm_endpoint,
                    llm_model,
                    api_key=os.getenv(llm_api_key_env),
                ),
                temperature=temperature,
                max_attempts=max_attempts,
            )
        else:
            raise typer.BadParameter("--provider must be 'mock' or 'llm'")
        result = ScientificProblemCompiler(
            planning_provider,
            evidence_repository=evidence_repository,
        ).compile(planning_input)
    except (PlanningContextError, PlanningProposalError, StructuredOutputError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error

    if result.proposal_set is None:
        raise RuntimeError("grounded planning did not return a PlanningProposalSet")
    trust_policy = ProjectTrustPolicy(
        approval_mode=ApprovalMode.INDEPENDENT_REQUIRED,
        policy_version="1.0.0",
    )
    plan_paths = tuple(output_dir / f"{candidate.plan_id}--{candidate.version}.yaml" for candidate in result.candidates)
    receipt_paths = tuple(
        output_dir / f"{receipt.plan_id}--compilation-receipt.yaml" for receipt in result.compilation_receipts
    )
    output_paths = (
        output_dir / "planning-input.yaml",
        output_dir / "planning-proposal.yaml",
        output_dir / "validation-reports.yaml",
        output_dir / "project-trust-policy.yaml",
        *plan_paths,
        *receipt_paths,
    )
    existing = tuple(path for path in output_paths if path.exists())
    if existing:
        raise typer.BadParameter("refusing to overwrite planning outputs: " + ", ".join(str(path) for path in existing))
    dump_yaml(output_paths[0], planning_input)
    dump_yaml(output_paths[1], result.proposal_set)
    dump_yaml(output_paths[3], trust_policy)
    for path, candidate in zip(plan_paths, result.candidates, strict=True):
        require_safe_path_component(candidate.plan_id, field="plan_id")
        dump_yaml(path, candidate)
    for path, receipt in zip(receipt_paths, result.compilation_receipts, strict=True):
        dump_yaml(path, receipt)
    validation_payload = {
        "valid": all(report.valid for report in result.reports),
        "reports": [report.model_dump(mode="json") for report in result.reports],
        "candidate_plan_ids": [candidate.plan_id for candidate in result.candidates],
        "compilation_receipt_ids": [receipt.receipt_id for receipt in result.compilation_receipts],
        "trust_policy": trust_policy.model_dump(mode="json"),
        "approved": False,
    }
    dump_yaml(output_paths[2], validation_payload)
    typer.echo(json.dumps(validation_payload, indent=2, ensure_ascii=False))
    if not validation_payload["valid"]:
        raise typer.Exit(1)


@app.command("compile")
def compile_command(
    request_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    mock_plan: Annotated[list[Path], typer.Option("--mock-plan", exists=True, dir_okay=False)],
    domain: Annotated[str, typer.Option("--domain")] = "base",
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
) -> None:
    """Compile with the offline MockProvider; phase 1 makes no external LLM calls."""
    request = request_file.read_text(encoding="utf-8")
    initialize_state(state_dir, domain=domain)
    plans = [load_model(path, ScientificQuestionPlan) for path in mock_plan]
    evidence_repository = SourceEvidenceStore(state_dir)
    result = ScientificProblemCompiler(
        MockProvider(plans),
        DomainPackLoader(),
        evidence_repository,
    ).compile(request, domain)
    output_dir = state_dir / "candidates"
    for plan in result.candidates:
        require_safe_path_component(plan.plan_id, field="plan_id")
        require_safe_path_component(plan.version, field="plan version")
        dump_yaml(output_dir / f"{plan.plan_id}--{plan.version}.yaml", plan)
    typer.echo(
        json.dumps(
            {
                "candidates": [plan.plan_id for plan in result.candidates],
                "valid": all(report.valid for report in result.reports),
                "reports": [report.model_dump(mode="json") for report in result.reports],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command()
def validate(
    target: Annotated[Path, typer.Argument(exists=True)],
    kind: Annotated[str, typer.Option("--kind", help="plan or export")] = "plan",
    domain: Annotated[str | None, typer.Option("--domain")] = None,
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    record_output: Annotated[Path | None, typer.Option("--record-output")] = None,
    validation_id: Annotated[str, typer.Option("--validation-id")] = "validation-1",
) -> None:
    """Run deterministic plan or export validators."""
    if kind == "export":
        report = validate_export(target)
    elif kind == "plan":
        plan = load_model(target, ScientificQuestionPlan)
        if domain is not None and domain != plan.domain:
            raise typer.BadParameter("--domain must match the plan domain")
        pack = DomainPackLoader().load(plan.domain)
        report = validate_question_plan(
            plan,
            pack.capabilities,
            _evidence_view(knowledge_dir, state_dir),
        )
        if record_output is not None:
            dump_yaml(
                record_output,
                build_plan_validation_record(plan, report, validation_id=validation_id),
            )
    else:
        raise typer.BadParameter("kind must be 'plan' or 'export'")
    _emit_report(report)
    if not report.valid:
        raise typer.Exit(1)


@app.command()
def review(
    context_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    evidence_packet_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    planning_input_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    candidate_plan_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    validation_record_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    provider: Annotated[str, typer.Option("--provider")] = "mock",
    verdict_output: Annotated[Path | None, typer.Option("--verdict-output")] = None,
    review_input_output: Annotated[Path | None, typer.Option("--review-input-output")] = None,
    receipt_output: Annotated[Path | None, typer.Option("--receipt-output")] = None,
    approver_id: Annotated[str, typer.Option("--approver-id")] = "independent-scientific-approver",
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    llm_endpoint: Annotated[str | None, typer.Option("--llm-endpoint")] = None,
    llm_model: Annotated[str | None, typer.Option("--llm-model")] = None,
    llm_api_key_env: Annotated[str, typer.Option("--llm-api-key-env")] = "SPC_LLM_API_KEY",
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    max_attempts: Annotated[int, typer.Option("--max-attempts")] = 2,
) -> None:
    """Independently review one candidate without modifying or gating it."""
    context = load_model(context_file, ScientificContextPacket)
    evidence_packet = load_model(evidence_packet_file, ScientificEvidencePacket)
    planning_input = load_model(planning_input_file, ScientificPlanningInput)
    candidate_plan = load_model(candidate_plan_file, ScientificQuestionPlan)
    validation_record = load_model(validation_record_file, PlanValidationRecord)
    evidence_repository = _evidence_view(knowledge_dir, state_dir)
    try:
        review_input = ApprovalContextResolver().resolve(
            context,
            evidence_packet,
            planning_input,
            candidate_plan,
            validation_record,
            KnowledgeRepositories(knowledge_dir),
            evidence_repository,
        )
        if provider == "mock":
            approval_provider = MockApprovalProvider()
        elif provider == "llm":
            if llm_endpoint is None or llm_model is None:
                raise typer.BadParameter("--provider llm requires --llm-endpoint and --llm-model")
            approval_provider = StructuredLLMApprovalProvider(
                HTTPJSONLLMTransport(
                    llm_endpoint,
                    llm_model,
                    api_key=os.getenv(llm_api_key_env),
                ),
                temperature=temperature,
                max_attempts=max_attempts,
            )
        else:
            raise typer.BadParameter("--provider must be 'mock' or 'llm'")
        result = IndependentApprovalService(
            approval_provider,
            approver_id=approver_id,
        ).review(review_input)
    except (
        ApprovalContextError,
        ApprovalResponseError,
        ApprovalStructuredOutputError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error

    final_verdict_output = verdict_output or output.with_name("approval-verdict.yaml")
    final_review_input_output = review_input_output or output.with_name("approval-review-input.yaml")
    final_receipt_output = receipt_output or output.with_name("independent-approval-receipt.yaml")
    existing = tuple(
        path
        for path in (
            output,
            final_verdict_output,
            final_review_input_output,
            final_receipt_output,
        )
        if path.exists()
    )
    if existing:
        raise typer.BadParameter("refusing to overwrite approval outputs: " + ", ".join(str(path) for path in existing))
    dump_yaml(output, result.review)
    dump_yaml(final_verdict_output, result.verdict)
    dump_yaml(final_review_input_output, review_input)
    dump_yaml(final_receipt_output, result.receipt)
    typer.echo(
        json.dumps(
            {
                "review_input_id": review_input.review_input_id,
                "review_id": result.review.review_id,
                "verdict_id": result.verdict.verdict_id,
                "decision": result.verdict.decision,
                "review_output": str(output),
                "verdict_output": str(final_verdict_output),
                "review_input_output": str(final_review_input_output),
                "receipt_id": result.receipt.receipt_id,
                "receipt_output": str(final_receipt_output),
                "plan_gate_passed": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command()
def approve(
    plan_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    decision: Annotated[str, typer.Option("--decision")],
    verdict_id: Annotated[str, typer.Option("--verdict-id")],
    approver_id: Annotated[str, typer.Option("--approver-id")],
    score: Annotated[int, typer.Option("--score", min=0, max=5)] = 3,
    required_fixes_file: Annotated[Path | None, typer.Option("--required-fixes", exists=True, dir_okay=False)] = None,
    fix_resolutions_file: Annotated[Path | None, typer.Option("--fix-resolutions", exists=True, dir_okay=False)] = None,
    human_decisions_required_file: Annotated[
        Path | None, typer.Option("--human-decisions-required", exists=True, dir_okay=False)
    ] = None,
    human_decision_resolutions_file: Annotated[
        Path | None, typer.Option("--human-decision-resolutions", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """LEGACY/manual verdict mode; it cannot satisfy Phase 2D independent approval."""
    plan = load_model(plan_file, ScientificQuestionPlan)
    scores = ApprovalScores(**{name: score for name in ApprovalScores.model_fields})
    required_fixes = tuple(
        RequiredFix.model_validate(item) for item in (load_data(required_fixes_file) if required_fixes_file else [])
    )
    fix_resolutions = tuple(
        FixResolution.model_validate(item) for item in (load_data(fix_resolutions_file) if fix_resolutions_file else [])
    )
    human_decisions_required = tuple(
        str(item) for item in (load_data(human_decisions_required_file) if human_decisions_required_file else [])
    )
    human_decision_resolutions = tuple(
        HumanDecisionResolution.model_validate(item)
        for item in (load_data(human_decision_resolutions_file) if human_decision_resolutions_file else [])
    )
    verdict = ScientificPlanApprover(approver_id).bind_verdict(
        plan,
        verdict_id=verdict_id,
        scores=scores,
        decision=decision,
        required_fixes=required_fixes,
        fix_resolutions=fix_resolutions,
        human_decisions_required=human_decisions_required,
        human_decision_resolutions=human_decision_resolutions,
    )
    dump_yaml(output, verdict)
    typer.echo(f"legacy/manual approval verdict: {output}")


@app.command()
def compare(
    left: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    right: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Compare method fingerprints and require disclosed differences."""
    report = compare_method_fingerprints(
        load_model(left, ScientificQuestionPlan), load_model(right, ScientificQuestionPlan)
    )
    _emit_report(report)
    if not report.valid:
        raise typer.Exit(1)


@app.command()
def export(
    plan_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    verdict_file: Annotated[Path, typer.Option("--verdict", exists=True, dir_okay=False)],
    validation_file: Annotated[Path, typer.Option("--validation-record", exists=True, dir_okay=False)],
    gate_file: Annotated[Path, typer.Option("--gate", exists=True, dir_okay=False)],
    export_id: Annotated[str, typer.Option("--export-id")],
    trust_policy_file: Annotated[Path, typer.Option("--trust-policy", exists=True, dir_okay=False)],
    compilation_receipt_file: Annotated[
        Path | None,
        typer.Option("--compilation-receipt", exists=True, dir_okay=False),
    ] = None,
    review_input_file: Annotated[
        Path | None,
        typer.Option("--review-input", exists=True, dir_okay=False),
    ] = None,
    review_record_file: Annotated[
        Path | None,
        typer.Option("--review-record", exists=True, dir_okay=False),
    ] = None,
    receipt_file: Annotated[
        Path | None,
        typer.Option("--approval-receipt", exists=True, dir_okay=False),
    ] = None,
    target: Annotated[str, typer.Option("--target")] = "ft-agent",
    exports_dir: Annotated[Path, typer.Option("--exports-dir")] = Path("exports"),
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    knowledge_dir: Annotated[Path, typer.Option("--knowledge-dir")] = Path("knowledge"),
    human_selected: Annotated[bool, typer.Option("--human-selected")] = False,
) -> None:
    """Create a planning-only immutable handoff package after both gates pass."""
    if target != "ft-agent":
        raise typer.BadParameter("phase 1 only provides the ft-agent adapter")
    plan = load_model(plan_file, ScientificQuestionPlan)
    verdict = load_model(verdict_file, ApprovalVerdict)
    validation_record = load_model(validation_file, PlanValidationRecord)
    gate = load_model(gate_file, GateVerdict)
    trust_policy = load_model(trust_policy_file, ProjectTrustPolicy)
    compilation_receipt = (
        load_model(compilation_receipt_file, PlanCompilationReceipt) if compilation_receipt_file is not None else None
    )
    review_input = load_model(review_input_file, ApprovalReviewInput) if review_input_file is not None else None
    review_record = load_model(review_record_file, ApprovalReviewRecord) if review_record_file is not None else None
    receipt = load_model(receipt_file, IndependentApprovalReceipt) if receipt_file is not None else None
    try:
        path = GenericExportService(
            exports_dir,
            _evidence_view(knowledge_dir, state_dir),
        ).export(
            plan=plan,
            verdict=verdict,
            validation_record=validation_record,
            gate=gate,
            human_selected=human_selected,
            adapter=FTAgentAdapter(),
            export_id=export_id,
            trust_policy=trust_policy,
            compilation_receipt=compilation_receipt,
            review_input=review_input,
            review=review_record,
            receipt=receipt,
        )
    except ExportError as error:
        _emit_report(error.report)
        raise typer.Exit(1) from error
    typer.echo(str(path))


@app.command("schema")
def schema_command(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("schemas"),
) -> None:
    """Export JSON Schemas for core contracts."""
    models = (
        ExternalBackendDescriptor,
        BackendRuntimeAvailability,
        BackendRuntimeArtifactManifest,
        BackendRuntimeIdentity,
        ExternalDocumentParseInput,
        ExternalDocumentElement,
        ExternalDocumentParseProposal,
        ReboundDocumentElement,
        ExternalStructureRebindingResult,
        ExternalLiteratureRetrievalQuery,
        ExternalLiteratureRetrievalHit,
        ExternalLiteratureRetrievalResult,
        ExternalRetrievalResolution,
        ExternalRetrievalResolutionBatch,
        ExternalStructurePromotionRecord,
        ScholarlyMetadataProposal,
        BackendRunRecord,
        AcquisitionAttemptRecord,
        LiteratureAcquisitionRequest,
        FullTextCandidate,
        ResolvedLiteratureResource,
        LiteratureAcquisitionRecord,
        LiteratureAcquisitionOutcome,
        MetadataRetrievalRecord,
        MetadataMergeManifest,
        RawHTMLLiteratureArtifact,
        CanonicalHTMLTextArtifact,
        HTMLLiteratureIngestionRecord,
        LiteratureRepresentationReference,
        DocumentStructureArtifact,
        DocumentStructureBlock,
        DocumentStructureSelection,
        TableStructure,
        TableCellStructure,
        FigureStructure,
        StructuredEvidenceLocator,
        CollectionScopePolicy,
        CollectionDefinition,
        CollectionPageArtifact,
        CollectionPageRecord,
        DiscoveredCollectionResource,
        CollectionResourceOccurrence,
        CollectionSnapshot,
        CollectionAcquisitionLink,
        CollectionResourceImportResult,
        CollectionLiteratureMembership,
        CollectionImportRecord,
        CollectionDiff,
        CollectionDiscoveryResult,
        CollectionImportOutcome,
        SourceDocument,
        EvidenceSpan,
        EvidenceReference,
        ScientificCapability,
        IntentFingerprint,
        SystemFingerprint,
        MethodFingerprint,
        DAGTask,
        ScientificQuestionPlan,
        ScientificTaskExecutionContext,
        SPCExportPackage,
        ExecutionProposal,
        ApprovalVerdict,
        PlanValidationRecord,
        ProjectTrustPolicy,
        PlanCompilationReceipt,
        GateVerdict,
        DomainProfile,
        AgentCapabilityCatalog,
        AgentHandoffPackage,
        ExportManifest,
        RetrievalQuery,
        RetrievalHit,
        LiteratureDocument,
        RawLiteratureArtifact,
        CanonicalTextBlock,
        CanonicalTextArtifact,
        LiteratureIngestionRecord,
        LiteratureIngestionOutcome,
        LiteratureRepresentationSelection,
        LiteratureRepresentationSelectionOutcome,
        HistoricalLiteratureEvidenceAuthorization,
        ExpertProfile,
        ExpertOpinion,
        KnowledgeRelation,
        KnowledgeCurationRecord,
        KnowledgeSnapshot,
        KnowledgeGraphNode,
        KnowledgeGraphEdge,
        KnowledgeGraph,
        RetrievalManifest,
        ScientificContextPacket,
        SourceClaim,
        SourceQuote,
        EvidenceAssessment,
        ReportedResult,
        ResultContext,
        MethodFact,
        ModelFact,
        ConflictSet,
        ComparisonConstraint,
        EvidenceGap,
        InterpretationProposal,
        ScientificEvidencePacket,
        ScientificPlanningInput,
        PlanningLLMResponse,
        ApprovalDimensionScore,
        ApprovalReviewScores,
        ApprovalHardRedFlag,
        ApprovalLLMResponse,
        ApprovalReviewInput,
        ApprovalReviewRecord,
        IndependentApprovalReceipt,
        IntentInterpretation,
        AmbiguityAssessment,
        ObservableDraft,
        ComparisonBaselineDraft,
        CriterionDraft,
        ProposedDeviationDraft,
        CandidateTaskDraft,
        CandidatePlanDraft,
        PlanningProposalSet,
        LiteratureKnowledgeCompilationInput,
        LiteratureKnowledgeChunk,
        LiteratureQuoteProposal,
        LiteratureClaimProposal,
        LiteratureMethodFactProposal,
        LiteratureModelFactProposal,
        LiteratureReportedResultProposal,
        LiteratureRelationProposal,
        LiteratureKnowledgeLLMResponse,
        LiteratureKnowledgeProposalSet,
        LiteratureKnowledgeSupportingQuoteView,
        LiteratureKnowledgeGroundingRecord,
        LiteratureKnowledgeCompilationRecord,
        LiteratureScientificKnowledgeView,
    )
    for path in export_json_schemas(output_dir, models):
        typer.echo(str(path))


if __name__ == "__main__":
    app()
