from __future__ import annotations

from pathlib import Path
from typing import Any

from yaml import YAMLError

from ..models import (
    CurationStatus,
    KnowledgeCurationRecord,
    ResultEvidenceAssessment,
    ResultEvidenceIntakeStatus,
    ResultEvidenceMaterializationReceipt,
    ResultEvidenceSubmission,
    ResultEvidenceType,
    SuccessorParentBinding,
    SuccessorPlanningCycle,
    SuccessorPlanningInvocation,
)
from ..repositories import CompositeEvidenceStore, KnowledgeEvidenceStore, ProjectEvidenceStore
from ..workflow.repository import ScientificProblemRunRepository
from .artifacts import successor_cycle_index
from .submission import ResultFeedbackError
from .successor import SuccessorStageReader


def _has_stage_artifacts(directory: Path) -> bool:
    # A linked output directory cannot establish that the stage is empty.
    return directory.is_symlink() or any(directory.glob("*.yaml"))


def _successor_stage_status(
    reader: SuccessorStageReader,
    run_id: str,
    directory: Path,
    binding: SuccessorParentBinding | None,
) -> tuple[str, SuccessorPlanningCycle | None]:
    runs = reader.runs
    prefix = directory.relative_to(runs.run_dir(run_id)).as_posix()
    approval_claimed = (directory / "provider-calls/approval.yaml").exists()
    planning_claimed = (directory / "provider-calls/planning.yaml").exists()
    has_cycle_record = (directory / "cycle.yaml").exists()
    if not has_cycle_record:
        if approval_claimed or _has_stage_artifacts(directory / "approval"):
            return "SUCCESSOR_APPROVAL_OUTCOME_UNCERTAIN", None
        if not planning_claimed:
            # Selection is the handoff to approval. With a planning claim, the
            # verified planning stage below can still report that it awaits approval.
            if (directory / "selection.yaml").exists():
                return "SUCCESSOR_APPROVAL_OUTCOME_UNCERTAIN", None
            invocation_path = directory / "invocation.yaml"
            if (
                not invocation_path.is_file()
                or any(path != invocation_path for path in directory.glob("*.yaml"))
                or any(
                    _has_stage_artifacts(directory / stage)
                    for stage in ("candidates", "compilation-receipts", "validation")
                )
            ):
                return "SUCCESSOR_PLANNING_OUTCOME_UNCERTAIN", None
            return "SUCCESSOR_INVOCATION_RESERVED", None

    try:
        if binding is None or binding.parent_run_id != run_id:
            raise ValueError("missing successor binding for this parent run")
        cycle = (
            runs.load_artifact(run_id, f"{prefix}/cycle.yaml", SuccessorPlanningCycle)
            if has_cycle_record else None
        )
        # Historical final records can predate invocation and provider-call artifacts.
        invocation = None
        if (
            cycle is None or cycle.invocation_id is not None
            or (directory / "invocation.yaml").exists()
        ):
            invocation = runs.load_artifact(
                run_id, f"{prefix}/invocation.yaml", SuccessorPlanningInvocation
            )
            if cycle is None or cycle.planning_proposal_id is not None:
                claim = runs.load_artifact(
                    run_id, f"{prefix}/provider-calls/planning.yaml", SuccessorPlanningInvocation
                )
                if claim != invocation:
                    raise ValueError("planning claim does not bind the stored invocation")
        if cycle is not None:
            # Current invocations persist an unselected proposal as a planning outcome.
            # Older invocations included the approval provider and could finalize at selection.
            if (
                invocation is not None and invocation.approval_provider_id is None
                and cycle.planning_proposal_id is not None and cycle.selected_plan_id is None
            ):
                raise ValueError("an unselected planning outcome is not a final cycle")
            reader.validate_reusable_cycle(
                parent_run_id=run_id, prefix=prefix, cycle=cycle, invocation=invocation
            )
            return cycle.status.value, cycle

        outcome = None
        if (directory / "planning-outcome.yaml").exists():
            outcome = runs.load_artifact(
                run_id, f"{prefix}/planning-outcome.yaml", SuccessorPlanningCycle
            )
        compilation, _ = reader.load_complete_planning_stage(
            parent_run_id=run_id, prefix=prefix, invocation=invocation, recorded_outcome=outcome
        )
        if not all(report.valid for report in compilation.reports):
            raise ValueError("successor planning artifacts failed validation")
        if outcome is not None:
            if len(compilation.candidates) < 2:
                raise ValueError("human candidate selection requires multiple candidates")
            return outcome.status.value, None
        return "PLANNING_COMPLETE_AWAITING_SELECTION_OR_APPROVAL", None
    except (OSError, ValueError, YAMLError):
        # A query may observe an interrupted or in-progress write. It never repairs it.
        if approval_claimed:
            return "SUCCESSOR_APPROVAL_OUTCOME_UNCERTAIN", None
        return "SUCCESSOR_PLANNING_OUTCOME_UNCERTAIN", None


def result_feedback_status(
    parent_run_id: str,
    repository: ScientificProblemRunRepository,
) -> dict[str, Any]:
    run_root = repository.run_dir(parent_run_id)
    submissions: list[ResultEvidenceSubmission] = []
    assessments: list[ResultEvidenceAssessment] = []
    receipts: list[ResultEvidenceMaterializationReceipt] = []
    curations: list[KnowledgeCurationRecord] = []
    for path in sorted(run_root.glob("result-evidence/submissions/*/submission.yaml")):
        submissions.append(
            repository.load_artifact(
                parent_run_id,
                path.relative_to(run_root).as_posix(),
                ResultEvidenceSubmission,
            )
        )
    for path in sorted(run_root.glob("result-evidence/submissions/*/assessment.yaml")):
        assessments.append(
            repository.load_artifact(
                parent_run_id,
                path.relative_to(run_root).as_posix(),
                ResultEvidenceAssessment,
            )
        )
    for path in sorted(run_root.glob("result-evidence/accepted/*/receipt.yaml")):
        receipts.append(
            repository.load_artifact(
                parent_run_id,
                path.relative_to(run_root).as_posix(),
                ResultEvidenceMaterializationReceipt,
            )
        )
    for path in sorted(run_root.glob("result-evidence/curations/*/*.yaml")):
        curations.append(
            repository.load_artifact(
                parent_run_id,
                path.relative_to(run_root).as_posix(),
                KnowledgeCurationRecord,
            )
        )
    cycles_by_index: dict[int, SuccessorPlanningCycle] = {}
    attempted_indexes: set[int] = set()
    incomplete_cycles: list[dict[str, Any]] = []
    bindings_by_index: dict[int, SuccessorParentBinding] = {}
    successor_root = repository.resolve_artifact_path(parent_run_id, "successor-planning")
    if successor_root.exists():
        run = repository.get(parent_run_id)
        reader = SuccessorStageReader(
            repository,
            CompositeEvidenceStore(
                KnowledgeEvidenceStore(Path(run.knowledge_dir), initialize=False),
                ProjectEvidenceStore(repository.root.parent, initialize=False),
            ),
        )
        for directory in successor_root.glob("cycle-*"):
            index = successor_cycle_index(directory, attempted_indexes, run_root=run_root)
            if index is None:
                continue
            attempted_indexes.add(index)
            binding_path = directory / "parent-binding.yaml"
            if binding_path.is_symlink():
                raise ResultFeedbackError(
                    "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                    f"symlinked successor parent binding: {binding_path.name}",
                )
            try:
                binding = repository.load_artifact(
                    parent_run_id,
                    binding_path.relative_to(run_root).as_posix(),
                    SuccessorParentBinding,
                )
            except (OSError, ValueError, YAMLError):
                binding = None
            if binding is not None:
                if binding.successor_cycle_index != index:
                    raise ResultFeedbackError(
                        "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                        "successor cycle directory index does not match "
                        "its parent-binding lineage index",
                    )
                if binding.parent_run_id == parent_run_id:
                    bindings_by_index[index] = binding
            status, cycle = _successor_stage_status(
                reader, parent_run_id, directory, bindings_by_index.get(index)
            )
            if cycle is not None:
                cycles_by_index[index] = cycle
            else:
                incomplete_cycles.append({"cycle": index, "status": status})
    cycles = [cycles_by_index[index] for index in sorted(cycles_by_index)]
    assessment_by_submission = {item.submission_id: item for item in assessments}
    materialized_ids = {item.submission_id for item in receipts}
    superseded_curation_ids = {
        item.supersedes_curation_id
        for item in curations
        if item.supersedes_curation_id is not None
    }
    current_curation_by_submission = {
        item.target_id: item
        for item in curations
        if item.curation_id not in superseded_curation_ids
    }
    accepted_ids = {
        submission_id
        for submission_id, curation in current_curation_by_submission.items()
        if curation.status == CurationStatus.ACCEPTED
        and submission_id in materialized_ids
    }
    blocked_statuses = {
        ResultEvidenceIntakeStatus.BLOCKED_CONTENT_BINDING,
        ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING,
        ResultEvidenceIntakeStatus.BLOCKED_CONTEXT_COMPATIBILITY,
        ResultEvidenceIntakeStatus.BLOCKED_SCIENTIFIC_ADMISSIBILITY,
    }
    latest_complete_index = max(cycles_by_index, default=None)
    latest_attempted_index = max(attempted_indexes, default=None)
    incomplete_by_index = {item["cycle"]: item for item in incomplete_cycles}
    if latest_attempted_index in cycles_by_index:
        latest_status = cycles_by_index[latest_attempted_index].status.value
    elif latest_attempted_index in incomplete_by_index:
        latest_status = incomplete_by_index[latest_attempted_index]["status"]
    else:
        latest_status = None
    binding_index = latest_attempted_index
    triggering_result_ids: list[str] = []
    latest_binding = (
        bindings_by_index.get(binding_index)
        if binding_index is not None
        else None
    )
    if latest_binding is not None:
        triggering_result_ids = list(latest_binding.result_submission_ids)
    return {
        "result_evidence": {
            "submissions": len(submissions),
            "accepted": len(accepted_ids),
            "rejected_or_blocked": sum(
                1
                for item in submissions
                if (
                    assessment_by_submission.get(item.submission_id) is not None
                    and assessment_by_submission[item.submission_id].status
                    in blocked_statuses
                )
                or (
                    current_curation_by_submission.get(item.submission_id)
                    is not None
                    and current_curation_by_submission[item.submission_id].status
                    == CurationStatus.REJECTED
                )
            ),
            "pending_curation": sum(
                1
                for item in submissions
                if item.submission_id not in accepted_ids
                and assessment_by_submission.get(item.submission_id) is not None
                and assessment_by_submission[item.submission_id].status
                == ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
                and current_curation_by_submission.get(item.submission_id) is None
            ),
            "failed_execution": sum(
                item.result_type == ResultEvidenceType.FAILED_EXECUTION
                for item in submissions
            ),
            "partial_results": sum(
                item.result_type == ResultEvidenceType.PARTIAL_RESULT
                for item in submissions
            ),
            "parent_plan_binding": [
                {
                    "submission_id": item.submission_id,
                    "parent_plan_id": item.parent_plan_id,
                    "parent_plan_hash": item.parent_plan_hash,
                    "task_id": item.task_id,
                }
                for item in submissions
            ],
        },
        "successor_planning": {
            "cycles": len(cycles),
            "latest_status": latest_status,
            "cycle": latest_complete_index,
            "latest_complete_cycle": latest_complete_index,
            "latest_attempted_cycle": latest_attempted_index,
            "incomplete_cycles": sorted(
                incomplete_cycles, key=lambda item: item["cycle"]
            ),
            "parent_run": parent_run_id,
            "parent_plan": (
                latest_binding.parent_plan_id
                if latest_binding is not None
                else submissions[-1].parent_plan_id if submissions else None
            ),
            "triggering_result_ids": triggering_result_ids,
        },
    }
