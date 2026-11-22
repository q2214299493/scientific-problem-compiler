from __future__ import annotations

from contextlib import contextmanager

from ..compiler import CompilationResult
from ..interpretation.validators import validate_evidence_packet_integrity
from ..models import (
    PlanCompilationReceipt,
    PlanningProposalSet,
    PlanValidationRecord,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificProblemRun,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
)
from ..planning.validators import validate_planning_proposal_set
from ..repositories import (
    CompositeEvidenceStore,
)
from ..retrieval.ranker import RETRIEVER_VERSION
from ..serialization import content_hash
from ..validators import (
    validate_plan_compilation_receipt,
    validate_question_plan,
)
from .repository import ScientificProblemRunRepository
from .state import make_run, snapshot_hash


class ProviderOutcomeUncertain(ValueError):
    """The input was claimed but its provider output cannot be verified."""


@contextmanager
def locked_run(runs: ScientificProblemRunRepository, run_id: str, *, initial=None):
    """Load or create the current projection only after acquiring the run lock."""
    with runs.acquire_run_lock(run_id):
        try:
            run = runs.get(run_id)
        except FileNotFoundError:
            if initial is None:
                raise
            run = make_run({"run_id": run_id, **initial, "status": ScientificProblemRunStatus.CREATED})
            runs.save(run)
        yield run


def reuse_context(runs, run, repositories, evidence, profile):
    if run.context_id is None:
        return None
    snapshot = repositories.create_snapshot(evidence, profile)
    if snapshot.snapshot_id != run.knowledge_snapshot_id or snapshot_hash(snapshot) != run.knowledge_snapshot_hash:
        return None
    context = runs.load_artifact(run.run_id, "context.yaml", ScientificContextPacket)
    if (context.context_id, context.content_hash) != (run.context_id, run.context_hash):
        raise ValueError("stored context does not match the run projection")
    if context.retrieval_manifest.retriever_version != RETRIEVER_VERSION:
        return None
    return context


def stored_planning_input(runs, run_id):
    try:
        return runs.load_artifact(run_id, "planning-input.yaml", ScientificPlanningInput)
    except FileNotFoundError:
        return None


def reuse_evidence_packet(
    runs: ScientificProblemRunRepository,
    run: ScientificProblemRun,
    context: ScientificContextPacket,
    evidence: CompositeEvidenceStore,
) -> ScientificEvidencePacket | None:
    if run.context_hash != context.content_hash or run.evidence_packet_id is None:
        return None
    try:
        packet = runs.load_artifact(run.run_id, "evidence-packet.yaml", ScientificEvidencePacket)
    except (FileNotFoundError, OSError, ValueError):
        return None
    report = validate_evidence_packet_integrity(packet, context, evidence)
    return packet if report.valid and packet.content_hash == run.evidence_packet_hash else None


def reuse_compilation(runs: ScientificProblemRunRepository, run, planning_input, evidence):
    if run.planning_input_hash != planning_input.content_hash or not run.candidate_plans:
        return None
    proposal = runs.load_artifact(run.run_id, "planning-proposal.yaml", PlanningProposalSet)
    if (
        proposal.proposal_id != run.planning_proposal_id
        or content_hash(proposal) != run.planning_proposal_hash
        or not validate_planning_proposal_set(proposal, planning_input).valid
    ):
        raise ValueError("stored planning proposal does not bind the current input and run")
    candidates = tuple(
        runs.load_bound_artifact(run.run_id, binding, ScientificQuestionPlan) for binding in run.candidate_plans
    )
    receipts = tuple(
        runs.load_bound_artifact(run.run_id, binding, PlanCompilationReceipt)
        for binding in run.candidate_compilation_receipts
    )
    reports = []
    for plan, receipt, binding in zip(candidates, receipts, run.plan_validation_records, strict=True):
        validation = runs.load_bound_artifact(run.run_id, binding, PlanValidationRecord)
        report = validate_question_plan(plan, planning_input.scientific_capabilities, evidence)
        if (
            not report.valid
            or not validate_plan_compilation_receipt(plan, receipt).valid
            or validation.plan_id != plan.plan_id
            or validation.plan_content_hash != content_hash(plan)
        ):
            raise ValueError("stored planning output failed integrity validation")
        reports.append(report)
    return CompilationResult(candidates, tuple(reports), proposal, receipts)
