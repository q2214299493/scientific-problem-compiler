from __future__ import annotations

from pathlib import Path

from ..adapters.ft_agent import FTAgentAdapter
from ..domains import DomainPackLoader
from ..export import GenericExportService
from ..models import (
    ApprovalReviewRecord,
    ApprovalVerdict,
    GateVerdict,
    IndependentApprovalReceipt,
    PlanCompilationReceipt,
    PlanValidationRecord,
    ProjectTrustPolicy,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
    parse_approval_review_input,
)
from ..serialization import file_sha256, load_data
from .repository import ScientificProblemRunRepository
from .stages.planning import open_knowledge
from .state import update_run


def export_downstream(
    runs: ScientificProblemRunRepository,
    domain_loader: DomainPackLoader,
    run_id: str,
    output_root: Path,
    export_id: str,
    *,
    selected_candidate_id: str | None = None,
) -> Path:
    run = runs.get(run_id)
    if run.status not in {ScientificProblemRunStatus.APPROVED, ScientificProblemRunStatus.EXPORTED}:
        raise ValueError("downstream export requires an independently APPROVED run")
    if run.selected_candidate_id is None:
        raise ValueError("approved run has no selected candidate")
    revision_chain = runs.load_revision_chain(run_id)
    if revision_chain is not None and revision_chain.revisions_used > 0:
        if selected_candidate_id != run.selected_candidate_id:
            raise ValueError("revised-plan export requires explicit selection of the final candidate ID")
    elif selected_candidate_id not in {None, run.selected_candidate_id}:
        raise ValueError("selected export candidate does not match the approved plan")
    index = tuple(item.artifact_id for item in run.candidate_plans).index(run.selected_candidate_id)
    plan = runs.load_artifact(run_id, run.candidate_plans[index].relative_path, ScientificQuestionPlan)
    validation = runs.load_artifact(run_id, run.plan_validation_records[index].relative_path, PlanValidationRecord)
    compilation_receipt = runs.load_artifact(
        run_id, run.candidate_compilation_receipts[index].relative_path, PlanCompilationReceipt
    )
    approval_prefix = runs.approval_prefix(run_id)
    review_input = parse_approval_review_input(
        load_data(runs.resolve_artifact_path(run_id, f"{approval_prefix}/approval-review-input.yaml"))
    )
    review = runs.load_artifact(run_id, f"{approval_prefix}/approval-review.yaml", ApprovalReviewRecord)
    verdict = runs.load_artifact(run_id, f"{approval_prefix}/approval-verdict.yaml", ApprovalVerdict)
    receipt = runs.load_artifact(
        run_id,
        f"{approval_prefix}/independent-approval-receipt.yaml",
        IndependentApprovalReceipt,
    )
    gate = runs.load_artifact(run_id, f"{approval_prefix}/plan-gate.yaml", GateVerdict)
    policy = runs.load_artifact(run_id, f"{approval_prefix}/project-trust-policy.yaml", ProjectTrustPolicy)
    _, evidence = open_knowledge(Path(run.knowledge_dir), Path(run.state_dir), run.domain, domain_loader)
    path = GenericExportService(output_root, evidence).export(
        plan=plan,
        verdict=verdict,
        validation_record=validation,
        gate=gate,
        human_selected=True,
        adapter=FTAgentAdapter(),
        export_id=export_id,
        trust_policy=policy,
        compilation_receipt=compilation_receipt,
        review_input=review_input,
        review=review,
        receipt=receipt,
    )
    checksums = path / "checksums.json"
    updated = update_run(
        run,
        status=ScientificProblemRunStatus.EXPORTED,
        export_path=str(path),
        export_hash=file_sha256(checksums),
    )
    runs.save(updated)
    return path
