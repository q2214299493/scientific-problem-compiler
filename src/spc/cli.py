from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

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
    CandidatePlanDraft,
    CandidateTaskDraft,
    ComparisonBaselineDraft,
    ComparisonConstraint,
    ConflictSet,
    CriterionDraft,
    DAGTask,
    DomainProfile,
    EvidenceReference,
    EvidenceAssessment,
    EvidenceGap,
    EvidenceSpan,
    ExportManifest,
    FixResolution,
    GateVerdict,
    HumanDecisionResolution,
    IndependentApprovalReceipt,
    IntentFingerprint,
    IntentInterpretation,
    InterpretationProposal,
    KnowledgeSnapshot,
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
    RequiredFix,
    RetrievalHit,
    RetrievalManifest,
    RetrievalQuery,
    ReportedResult,
    ResultContext,
    ScientificCapability,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificQuestionPlan,
    SourceClaim,
    SourceQuote,
    SourceDocument,
    SystemFingerprint,
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
from .providers import MockProvider
from .retrieval import ScientificContextBuilder
from .repositories import KnowledgeRepositories, SourceEvidenceStore, initialize_state
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
) -> None:
    """Build a validated ScientificEvidencePacket with an offline provider."""
    if provider != "mock":
        raise typer.BadParameter("Phase 2B only supports --provider mock")
    context = load_model(context_file, ScientificContextPacket)
    try:
        packet = ScientificEvidencePacketBuilder(MockInterpretationProvider()).build(
            context,
            SourceEvidenceStore(state_dir),
        )
    except EvidencePacketIntegrityError as error:
        _emit_report(error.report)
        raise typer.Exit(1) from error
    dump_yaml(output, packet)
    typer.echo(str(output))


@app.command()
def plan(
    context_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    evidence_packet_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True)
    ],
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
    evidence_repository = SourceEvidenceStore(state_dir)
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
                raise typer.BadParameter(
                    "--provider llm requires --llm-endpoint and --llm-model"
                )
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
    plan_paths = tuple(
        output_dir / f"{candidate.plan_id}--{candidate.version}.yaml"
        for candidate in result.candidates
    )
    receipt_paths = tuple(
        output_dir / f"{receipt.plan_id}--compilation-receipt.yaml"
        for receipt in result.compilation_receipts
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
        raise typer.BadParameter(
            "refusing to overwrite planning outputs: "
            + ", ".join(str(path) for path in existing)
        )
    dump_yaml(output_paths[0], planning_input)
    dump_yaml(output_paths[1], result.proposal_set)
    dump_yaml(output_paths[3], trust_policy)
    for path, candidate in zip(plan_paths, result.candidates, strict=True):
        require_safe_path_component(candidate.plan_id, field="plan_id")
        dump_yaml(path, candidate)
    for path, receipt in zip(
        receipt_paths, result.compilation_receipts, strict=True
    ):
        dump_yaml(path, receipt)
    validation_payload = {
        "valid": all(report.valid for report in result.reports),
        "reports": [report.model_dump(mode="json") for report in result.reports],
        "candidate_plan_ids": [candidate.plan_id for candidate in result.candidates],
        "compilation_receipt_ids": [
            receipt.receipt_id for receipt in result.compilation_receipts
        ],
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
            SourceEvidenceStore(state_dir),
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
    evidence_packet_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    planning_input_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    candidate_plan_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    validation_record_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output")],
    provider: Annotated[str, typer.Option("--provider")] = "mock",
    verdict_output: Annotated[
        Path | None, typer.Option("--verdict-output")
    ] = None,
    review_input_output: Annotated[
        Path | None, typer.Option("--review-input-output")
    ] = None,
    receipt_output: Annotated[
        Path | None, typer.Option("--receipt-output")
    ] = None,
    approver_id: Annotated[
        str, typer.Option("--approver-id")
    ] = "independent-scientific-approver",
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    knowledge_dir: Annotated[
        Path, typer.Option("--knowledge-dir")
    ] = Path("knowledge"),
    llm_endpoint: Annotated[str | None, typer.Option("--llm-endpoint")] = None,
    llm_model: Annotated[str | None, typer.Option("--llm-model")] = None,
    llm_api_key_env: Annotated[
        str, typer.Option("--llm-api-key-env")
    ] = "SPC_LLM_API_KEY",
    temperature: Annotated[float, typer.Option("--temperature")] = 0.0,
    max_attempts: Annotated[int, typer.Option("--max-attempts")] = 2,
) -> None:
    """Independently review one candidate without modifying or gating it."""
    context = load_model(context_file, ScientificContextPacket)
    evidence_packet = load_model(evidence_packet_file, ScientificEvidencePacket)
    planning_input = load_model(planning_input_file, ScientificPlanningInput)
    candidate_plan = load_model(candidate_plan_file, ScientificQuestionPlan)
    validation_record = load_model(validation_record_file, PlanValidationRecord)
    evidence_repository = SourceEvidenceStore(state_dir)
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
                raise typer.BadParameter(
                    "--provider llm requires --llm-endpoint and --llm-model"
                )
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
    final_review_input_output = review_input_output or output.with_name(
        "approval-review-input.yaml"
    )
    final_receipt_output = receipt_output or output.with_name(
        "independent-approval-receipt.yaml"
    )
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
        raise typer.BadParameter(
            "refusing to overwrite approval outputs: "
            + ", ".join(str(path) for path in existing)
        )
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
    required_fixes_file: Annotated[
        Path | None, typer.Option("--required-fixes", exists=True, dir_okay=False)
    ] = None,
    fix_resolutions_file: Annotated[
        Path | None, typer.Option("--fix-resolutions", exists=True, dir_okay=False)
    ] = None,
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
        RequiredFix.model_validate(item)
        for item in (load_data(required_fixes_file) if required_fixes_file else [])
    )
    fix_resolutions = tuple(
        FixResolution.model_validate(item)
        for item in (load_data(fix_resolutions_file) if fix_resolutions_file else [])
    )
    human_decisions_required = tuple(
        str(item)
        for item in (
            load_data(human_decisions_required_file) if human_decisions_required_file else []
        )
    )
    human_decision_resolutions = tuple(
        HumanDecisionResolution.model_validate(item)
        for item in (
            load_data(human_decision_resolutions_file)
            if human_decision_resolutions_file
            else []
        )
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
    validation_file: Annotated[
        Path, typer.Option("--validation-record", exists=True, dir_okay=False)
    ],
    gate_file: Annotated[Path, typer.Option("--gate", exists=True, dir_okay=False)],
    export_id: Annotated[str, typer.Option("--export-id")],
    trust_policy_file: Annotated[
        Path, typer.Option("--trust-policy", exists=True, dir_okay=False)
    ],
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
        load_model(compilation_receipt_file, PlanCompilationReceipt)
        if compilation_receipt_file is not None
        else None
    )
    review_input = (
        load_model(review_input_file, ApprovalReviewInput)
        if review_input_file is not None
        else None
    )
    review_record = (
        load_model(review_record_file, ApprovalReviewRecord)
        if review_record_file is not None
        else None
    )
    receipt = (
        load_model(receipt_file, IndependentApprovalReceipt)
        if receipt_file is not None
        else None
    )
    try:
        path = GenericExportService(
            exports_dir,
            SourceEvidenceStore(state_dir),
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
        SourceDocument,
        EvidenceSpan,
        EvidenceReference,
        ScientificCapability,
        IntentFingerprint,
        SystemFingerprint,
        MethodFingerprint,
        DAGTask,
        ScientificQuestionPlan,
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
        KnowledgeSnapshot,
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
    )
    for path in export_json_schemas(output_dir, models):
        typer.echo(str(path))


if __name__ == "__main__":
    app()
