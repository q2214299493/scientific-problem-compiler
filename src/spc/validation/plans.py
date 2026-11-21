"""Scientific plan, compilation, and deterministic validation rules."""

from __future__ import annotations

import re
from collections import Counter, deque
from typing import Iterable

from ..models import (
    DAGTask,
    ExecutionPolicy,
    PlanCompilationReceipt,
    PlanValidationRecord,
    ScientificCapability,
    ScientificQuestionPlan,
)
from ..serialization import content_hash
from .common import (
    ValidationIssue,
    ValidationReport,
    content_binding_error,
    validation_report,
    walk_values,
)
from .evidence import EvidenceSpanRepository, validate_evidence_links


def validate_dag(tasks: Iterable[DAGTask]) -> ValidationReport:
    tasks = tuple(tasks)
    issues: list[ValidationIssue] = []
    counts = Counter(task.task_id for task in tasks)
    duplicates = {task_id for task_id, count in counts.items() if count > 1}
    for task_id in sorted(duplicates):
        issues.append(ValidationIssue(code="DUPLICATE_TASK_ID", message=f"duplicate task ID: {task_id}"))
    known = set(counts)
    adjacency: dict[str, list[str]] = {task_id: [] for task_id in known}
    indegree = {task_id: 0 for task_id in known}
    for task in tasks:
        for dependency in task.depends_on:
            if dependency not in known:
                issues.append(ValidationIssue(code="UNKNOWN_DEPENDENCY", message=f"{task.task_id} depends on unknown task {dependency}"))
                continue
            adjacency[dependency].append(task.task_id)
            indegree[task.task_id] += 1
    queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        task_id = queue.popleft()
        visited += 1
        for child in adjacency[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(known):
        issues.append(ValidationIssue(code="DAG_CYCLE", message="task graph contains a cycle"))
    return validation_report(issues)


def compare_method_fingerprints(left: ScientificQuestionPlan, right: ScientificQuestionPlan) -> ValidationReport:
    issues: list[ValidationIssue] = []
    keys = set(left.method_fingerprint.attributes) | set(right.method_fingerprint.attributes)
    disclosed_fields = {
        item.field
        for item in (*left.fingerprint_differences, *right.fingerprint_differences)
        if item.disclosed_deviation
    }
    for key in sorted(keys):
        if left.method_fingerprint.attributes.get(key) != right.method_fingerprint.attributes.get(key) and key not in disclosed_fields:
            issues.append(ValidationIssue(code="UNDISCLOSED_METHOD_DIFFERENCE", message=f"method difference is not disclosed: {key}"))
    return validation_report(issues)


def validate_question_plan(
    plan: ScientificQuestionPlan,
    capabilities: Iterable[ScientificCapability] = (),
    evidence_repository: EvidenceSpanRepository | None = None,
) -> ValidationReport:
    issues = list(validate_evidence_links(plan, evidence_repository).issues) + list(validate_dag(plan.tasks).issues)
    concern_tokens = {
        token for token in re.findall(r"[a-z0-9]+", plan.latent_concern.lower()) if len(token) >= 4
    }
    question_tokens = {
        token
        for question in plan.atomic_questions
        for token in re.findall(r"[a-z0-9]+", question.text.lower())
        if len(token) >= 4
    }
    if concern_tokens and concern_tokens.isdisjoint(question_tokens):
        issues.append(
            ValidationIssue(
                code="INTENT_QUESTION_MISMATCH",
                message="atomic questions do not preserve any explicit latent-concern term",
                path="atomic_questions",
            )
        )
    observable_ids = {item.observable_id for item in plan.observables}
    for criterion in (*plan.acceptance_criteria, *plan.falsification_criteria):
        if criterion.observable_id not in observable_ids:
            issues.append(ValidationIssue(code="UNKNOWN_OBSERVABLE", message=f"criterion references unknown observable: {criterion.observable_id}"))
    baseline_ids = {baseline.baseline_id for baseline in plan.comparison_baselines}
    for index, deviation in enumerate(plan.proposed_deviations):
        if deviation.baseline_ref not in baseline_ids:
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_DEVIATION_BASELINE_REF",
                    message=(
                        "proposed deviation baseline_ref does not identify a plan baseline"
                    ),
                    path=f"proposed_deviations[{index}].baseline_ref",
                )
            )
    task_capabilities = {task.capability_id for task in plan.tasks}
    if not task_capabilities.issubset(set(plan.scientific_capability_ids)):
        issues.append(ValidationIssue(code="UNDECLARED_CAPABILITY", message="a DAG task uses a capability not declared by the plan"))
    available = {item.capability_id for item in capabilities}
    if available:
        for capability_id in sorted(set(plan.scientific_capability_ids) - available):
            issues.append(ValidationIssue(code="UNKNOWN_CAPABILITY", message=f"domain pack does not provide capability: {capability_id}"))
    issues.extend(validate_phase1_boundary(ExecutionPolicy(), plan.tasks).issues)
    return validation_report(issues)


def validate_candidate_set(plans: Iterable[ScientificQuestionPlan]) -> ValidationReport:
    plans = tuple(plans)
    issues: list[ValidationIssue] = []
    if not 1 <= len(plans) <= 4:
        issues.append(ValidationIssue(code="CANDIDATE_COUNT", message="candidate count must be between one and four"))
        return validation_report(issues)
    if len(plans) == 1:
        return validation_report(issues)
    axes = [plan.distinguishing_axis.strip().lower() if plan.distinguishing_axis else "" for plan in plans]
    for index, axis in enumerate(axes):
        if not axis:
            issues.append(ValidationIssue(code="MISSING_DISTINGUISHING_AXIS", message="multi-candidate plans require a distinguishing axis", path=f"candidates[{index}]"))
    if len(set(axes)) != len(axes):
        issues.append(ValidationIssue(code="DUPLICATE_DISTINGUISHING_AXIS", message="candidate axes must be distinct"))
    tuning_terms = ("encut", "k-point", "kpoint", "step count", "steps", "threshold", "cutoff")
    for index, axis in enumerate(axes):
        if axis and any(term in axis for term in tuning_terms):
            issues.append(ValidationIssue(code="PSEUDO_DIVERSITY", message="numerical tuning alone is not a scientific distinguishing axis", path=f"candidates[{index}]"))
    return validation_report(issues)


def validate_plan_compilation_receipt(
    plan: ScientificQuestionPlan,
    receipt: PlanCompilationReceipt,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if content_binding_error(receipt, id_field="receipt_id", prefix="plan-compilation-receipt") is not None:
        issues.append(
            ValidationIssue(
                code="TAMPERED_PLAN_COMPILATION_RECEIPT",
                message="PlanCompilationReceipt identity or content hash is invalid",
            )
        )
    if (receipt.plan_id, receipt.plan_hash) != (plan.plan_id, content_hash(plan)):
        issues.append(
            ValidationIssue(
                code="PLAN_COMPILATION_RECEIPT_MISMATCH",
                message="PlanCompilationReceipt is not bound to the supplied plan",
            )
        )
    if receipt.origin != "phase2c_grounded_compiler":
        issues.append(
            ValidationIssue(
                code="INVALID_PLAN_COMPILATION_ORIGIN",
                message="PlanCompilationReceipt does not identify the grounded compiler",
            )
        )
    return validation_report(issues)


def build_plan_validation_record(
    plan: ScientificQuestionPlan,
    report: ValidationReport,
    *,
    validation_id: str,
) -> PlanValidationRecord:
    return PlanValidationRecord(
        validation_id=validation_id,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        plan_content_hash=content_hash(plan),
        domain=plan.domain,
        domain_pack_version=plan.domain_pack_version,
        valid=report.valid,
        issue_codes=tuple(issue.code for issue in report.issues),
    )


def validation_record_matches_plan(plan: ScientificQuestionPlan, record: PlanValidationRecord) -> bool:
    return (
        record.plan_id,
        record.plan_version,
        record.plan_content_hash,
        record.domain,
        record.domain_pack_version,
    ) == (
        plan.plan_id,
        plan.version,
        content_hash(plan),
        plan.domain,
        plan.domain_pack_version,
    )


def validate_plan_validation_record(
    plan: ScientificQuestionPlan,
    record: PlanValidationRecord,
    current_report: ValidationReport,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if not validation_record_matches_plan(plan, record):
        issues.append(
            ValidationIssue(
                code="STALE_PLAN_VALIDATION",
                message="PlanValidationRecord is not bound to the current plan ID, version, hash, and domain",
            )
        )
    current_codes = tuple(issue.code for issue in current_report.issues)
    if record.valid != current_report.valid or record.issue_codes != current_codes:
        issues.append(
            ValidationIssue(
                code="PLAN_VALIDATION_RECORD_MISMATCH",
                message="PlanValidationRecord does not match the current deterministic validation result",
            )
        )
    if not record.valid or not current_report.valid:
        issues.append(ValidationIssue(code="PLAN_VALIDATION_FAILED", message="invalid plans cannot be exported"))
    return validation_report(issues)


def validate_phase1_boundary(policy: ExecutionPolicy, tasks: Iterable[DAGTask]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if policy.mode != "planning_only":
        issues.append(ValidationIssue(code="INVALID_EXECUTION_MODE", message="phase 1 requires planning_only mode"))
    if policy.runnable or any(task.runnable for task in tasks):
        issues.append(ValidationIssue(code="PHASE1_RUNNABLE_TASK", message="phase 1 requires runnable=false"))
    prohibited_keys = {"command", "commands", "executable", "script", "submit_command", "scheduler_command"}
    for task in tasks:
        for path, value in walk_values(task.inputs, f"tasks.{task.task_id}.inputs"):
            if isinstance(value, dict) and prohibited_keys.intersection(value):
                issues.append(
                    ValidationIssue(
                        code="PHASE1_EXECUTION_PAYLOAD",
                        message="phase 1 task inputs cannot contain executable command or script fields",
                        path=path,
                    )
                )
    return validation_report(issues)
