from __future__ import annotations

from typing import Any

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
)
from ..workflow.repository import ScientificProblemRunRepository
from .submission import ResultFeedbackError


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
    successor_root = run_root / "successor-planning"
    if successor_root.exists():
        for directory in successor_root.glob("cycle-*"):
            if directory.is_symlink():
                raise ResultFeedbackError(
                    "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                    f"symlinked successor cycle directory: {directory.name}",
                )
            if not directory.is_dir():
                continue
            suffix = directory.name.removeprefix("cycle-")
            try:
                index = int(suffix)
            except ValueError as error:
                raise ResultFeedbackError(
                    "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                    f"invalid successor cycle directory: {directory.name}",
                ) from error
            if index < 1 or suffix != str(index) or index in attempted_indexes:
                raise ResultFeedbackError(
                    "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                    f"non-canonical successor cycle directory: {directory.name}",
                )
            attempted_indexes.add(index)
            binding_path = directory / "parent-binding.yaml"
            if binding_path.is_symlink():
                raise ResultFeedbackError(
                    "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                    f"symlinked successor parent binding: {binding_path.name}",
                )
            if binding_path.exists():
                binding = repository.load_artifact(
                    parent_run_id,
                    binding_path.relative_to(run_root).as_posix(),
                    SuccessorParentBinding,
                )
                if binding.successor_cycle_index != index:
                    raise ResultFeedbackError(
                        "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                        "successor cycle directory index does not match "
                        "its parent-binding lineage index",
                    )
                bindings_by_index[index] = binding
            cycle_path = directory / "cycle.yaml"
            if cycle_path.exists():
                cycle = repository.load_artifact(
                    parent_run_id,
                    cycle_path.relative_to(run_root).as_posix(),
                    SuccessorPlanningCycle,
                )
                binding = bindings_by_index.get(index)
                if binding is None or (
                    cycle.parent_binding_id,
                    cycle.parent_binding_hash,
                ) != (binding.binding_id, binding.content_hash):
                    raise ResultFeedbackError(
                        "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                        "completed successor cycle does not bind the "
                        "parent-binding stored in its directory",
                    )
                cycles_by_index[index] = cycle
                continue
            planning_outcome_path = directory / "planning-outcome.yaml"
            if planning_outcome_path.exists():
                planning_outcome = repository.load_artifact(
                    parent_run_id,
                    planning_outcome_path.relative_to(run_root).as_posix(),
                    SuccessorPlanningCycle,
                )
                incomplete_cycles.append(
                    {"cycle": index, "status": planning_outcome.status.value}
                )
                continue
            approval_claimed = (directory / "provider-calls" / "approval.yaml").exists()
            planning_claimed = (directory / "provider-calls" / "planning.yaml").exists()
            if approval_claimed:
                status = "SUCCESSOR_APPROVAL_OUTCOME_UNCERTAIN"
            elif planning_claimed and (directory / "planning-proposal.yaml").exists():
                status = "PLANNING_COMPLETE_AWAITING_SELECTION_OR_APPROVAL"
            elif planning_claimed:
                status = "SUCCESSOR_PLANNING_OUTCOME_UNCERTAIN"
            else:
                status = "SUCCESSOR_INVOCATION_RESERVED"
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
