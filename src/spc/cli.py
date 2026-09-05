from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .adapters.ft_agent import FTAgentAdapter
from .approval import ScientificPlanApprover
from .compiler import ScientificProblemCompiler
from .domains import DomainPackLoader
from .export import ExportError, GenericExportService
from .models import (
    AgentCapabilityCatalog,
    AgentHandoffPackage,
    ApprovalScores,
    ApprovalVerdict,
    DAGTask,
    DomainProfile,
    EvidenceReference,
    EvidenceSpan,
    ExportManifest,
    FixResolution,
    GateVerdict,
    HumanDecisionResolution,
    IntentFingerprint,
    KnowledgeSnapshot,
    MethodFingerprint,
    PlanValidationRecord,
    RequiredFix,
    RetrievalHit,
    RetrievalManifest,
    RetrievalQuery,
    ScientificCapability,
    ScientificContextPacket,
    ScientificQuestionPlan,
    SourceDocument,
    SystemFingerprint,
)
from .providers import MockProvider
from .retrieval import ScientificContextBuilder
from .repositories import SourceEvidenceStore, initialize_state
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

app = typer.Typer(no_args_is_help=True, help="Compile evidence-grounded scientific question plans offline.")


def _emit_report(report: object) -> None:
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


@app.command()
def ingest(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    source_id: Annotated[str, typer.Option("--source-id")],
    version: Annotated[str, typer.Option("--version")],
    state_dir: Annotated[Path, typer.Option("--state-dir")] = Path(".spc"),
    title: Annotated[str | None, typer.Option("--title")] = None,
) -> None:
    """Copy a source into the versioned, read-only evidence store."""
    record = SourceEvidenceStore(state_dir).ingest(source, source_id, version, title)
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
    """Create an independent, hash-bound approval verdict without modifying the plan."""
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
    typer.echo(str(output))


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
    )
    for path in export_json_schemas(output_dir, models):
        typer.echo(str(path))


if __name__ == "__main__":
    app()
