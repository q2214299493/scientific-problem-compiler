from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import re
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict

from .models import (
    AgentHandoffPackage,
    ApprovalDecision,
    ApprovalVerdict,
    DAGTask,
    EvidenceClassification,
    ExecutionPolicy,
    GateVerdict,
    GroundedStatement,
    ScientificCapability,
    ScientificQuestionPlan,
)
from .serialization import content_hash, file_sha256, load_data


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str
    path: str | None = None


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    valid: bool
    issues: tuple[ValidationIssue, ...] = ()


def _report(issues: Iterable[ValidationIssue]) -> ValidationReport:
    result = tuple(issues)
    return ValidationReport(valid=not result, issues=result)


def _walk(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _walk(getattr(value, name), f"{path}.{name}" if path else name)
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def validate_evidence_links(plan: ScientificQuestionPlan) -> ValidationReport:
    issues: list[ValidationIssue] = []
    declared = {ref.evidence_id for ref in plan.evidence_refs}
    if len(declared) != len(plan.evidence_refs):
        issues.append(ValidationIssue(code="DUPLICATE_EVIDENCE_ID", message="evidence IDs must be unique"))
    for path, value in _walk(plan):
        if isinstance(value, GroundedStatement):
            if value.classification == EvidenceClassification.EVIDENCE and not value.evidence_refs:
                issues.append(ValidationIssue(code="MISSING_EVIDENCE_REF", message="evidence statement has no reference", path=path))
        if isinstance(value, BaseModel) and value is not plan and hasattr(value, "evidence_refs"):
            for evidence_id in getattr(value, "evidence_refs"):
                if evidence_id not in declared:
                    issues.append(ValidationIssue(code="UNKNOWN_EVIDENCE_REF", message=f"unknown evidence ID: {evidence_id}", path=path))
    if plan.system_fingerprint.attributes and not plan.system_fingerprint.evidence_refs:
        issues.append(
            ValidationIssue(
                code="UNGROUNDED_SYSTEM_FINGERPRINT",
                message="system fingerprint facts require evidence_refs",
                path="system_fingerprint",
            )
        )
    if plan.method_fingerprint.attributes and not (
        plan.method_fingerprint.evidence_refs or plan.method_fingerprint.proposed_deviation_refs
    ):
        issues.append(
            ValidationIssue(
                code="UNGROUNDED_METHOD_FINGERPRINT",
                message="method choices require evidence_refs or proposed_deviation_refs",
                path="method_fingerprint",
            )
        )
    declared_deviations = {item.deviation_id for item in plan.proposed_deviations}
    for deviation_id in plan.method_fingerprint.proposed_deviation_refs:
        if deviation_id not in declared_deviations:
            issues.append(ValidationIssue(code="UNKNOWN_DEVIATION_REF", message=f"unknown deviation ID: {deviation_id}", path="method_fingerprint"))
    return _report(issues)


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
    return _report(issues)


def compare_method_fingerprints(left: ScientificQuestionPlan, right: ScientificQuestionPlan) -> ValidationReport:
    issues: list[ValidationIssue] = []
    keys = set(left.method_fingerprint.attributes) | set(right.method_fingerprint.attributes)
    disclosed_fields = {item.field for item in left.fingerprint_differences} | {item.field for item in right.fingerprint_differences}
    for key in sorted(keys):
        if left.method_fingerprint.attributes.get(key) != right.method_fingerprint.attributes.get(key) and key not in disclosed_fields:
            issues.append(ValidationIssue(code="UNDISCLOSED_METHOD_DIFFERENCE", message=f"method difference is not disclosed: {key}"))
    return _report(issues)


def validate_question_plan(
    plan: ScientificQuestionPlan, capabilities: Iterable[ScientificCapability] = ()
) -> ValidationReport:
    issues = list(validate_evidence_links(plan).issues) + list(validate_dag(plan.tasks).issues)
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
    task_capabilities = {task.capability_id for task in plan.tasks}
    if not task_capabilities.issubset(set(plan.scientific_capability_ids)):
        issues.append(ValidationIssue(code="UNDECLARED_CAPABILITY", message="a DAG task uses a capability not declared by the plan"))
    available = {item.capability_id for item in capabilities}
    if available:
        for capability_id in sorted(set(plan.scientific_capability_ids) - available):
            issues.append(ValidationIssue(code="UNKNOWN_CAPABILITY", message=f"domain pack does not provide capability: {capability_id}"))
    issues.extend(validate_phase1_boundary(ExecutionPolicy(), plan.tasks).issues)
    return _report(issues)


def validate_candidate_set(plans: Iterable[ScientificQuestionPlan]) -> ValidationReport:
    plans = tuple(plans)
    issues: list[ValidationIssue] = []
    if not 1 <= len(plans) <= 4:
        issues.append(ValidationIssue(code="CANDIDATE_COUNT", message="candidate count must be between one and four"))
        return _report(issues)
    if len(plans) == 1:
        return _report(issues)
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
    return _report(issues)


def validate_approval_boundary(
    plan: ScientificQuestionPlan, verdict: ApprovalVerdict
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    actual_hash = content_hash(plan)
    if verdict.candidate_id != plan.plan_id:
        issues.append(ValidationIssue(code="APPROVAL_ID_MISMATCH", message="verdict candidate ID does not match plan"))
    if verdict.candidate_version != plan.version:
        issues.append(ValidationIssue(code="APPROVAL_VERSION_MISMATCH", message="verdict version does not match plan"))
    if verdict.candidate_content_hash != actual_hash:
        issues.append(ValidationIssue(code="STALE_APPROVAL", message="verdict content hash does not match current plan"))
    return _report(issues)


def validate_phase1_boundary(policy: ExecutionPolicy, tasks: Iterable[DAGTask]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if policy.mode != "planning_only":
        issues.append(ValidationIssue(code="INVALID_EXECUTION_MODE", message="phase 1 requires planning_only mode"))
    if policy.runnable or any(task.runnable for task in tasks):
        issues.append(ValidationIssue(code="PHASE1_RUNNABLE_TASK", message="phase 1 requires runnable=false"))
    prohibited_keys = {"command", "commands", "executable", "script", "submit_command", "scheduler_command"}
    for task in tasks:
        for path, value in _walk(task.inputs, f"tasks.{task.task_id}.inputs"):
            if isinstance(value, dict) and prohibited_keys.intersection(value):
                issues.append(
                    ValidationIssue(
                        code="PHASE1_EXECUTION_PAYLOAD",
                        message="phase 1 task inputs cannot contain executable command or script fields",
                        path=path,
                    )
                )
    return _report(issues)


def validate_handoff_package(
    plan: ScientificQuestionPlan,
    verdict: ApprovalVerdict,
    gate: GateVerdict,
    handoff: AgentHandoffPackage,
    *,
    human_selected: bool,
) -> ValidationReport:
    issues = list(validate_approval_boundary(plan, verdict).issues)
    plan_hash = content_hash(plan)
    if verdict.decision not in {ApprovalDecision.APPROVE, ApprovalDecision.APPROVE_WITH_CONDITIONS}:
        issues.append(ValidationIssue(code="PLAN_NOT_APPROVED", message="approval decision does not permit export"))
    if not gate.passed:
        issues.append(ValidationIssue(code="PLAN_GATE_FAILED", message="plan gate has not passed"))
    if (gate.candidate_id, gate.candidate_version, gate.candidate_content_hash) != (plan.plan_id, plan.version, plan_hash):
        issues.append(ValidationIssue(code="STALE_PLAN_GATE", message="plan gate is not bound to current plan content"))
    if not human_selected:
        issues.append(ValidationIssue(code="HUMAN_SELECTION_REQUIRED", message="explicit human selection is required"))
    if (handoff.source_plan_id, handoff.source_plan_version, handoff.source_plan_hash) != (plan.plan_id, plan.version, plan_hash):
        issues.append(ValidationIssue(code="HANDOFF_PLAN_MISMATCH", message="handoff is not bound to current plan"))
    for binding in handoff.capability_bindings:
        if binding.status != "available":
            issues.append(
                ValidationIssue(
                    code="TARGET_CAPABILITY_UNAVAILABLE",
                    message=f"target capability unavailable: {binding.scientific_capability_id}",
                )
            )
    issues.extend(validate_phase1_boundary(handoff.execution_policy, plan.tasks).issues)
    return _report(issues)


def validate_export(export_dir: Path) -> ValidationReport:
    issues: list[ValidationIssue] = []
    checksums_path = export_dir / "checksums.json"
    if not checksums_path.is_file():
        return _report([ValidationIssue(code="MISSING_CHECKSUMS", message="checksums.json is missing")])
    checksums = load_data(checksums_path)
    if not isinstance(checksums, dict):
        return _report([ValidationIssue(code="INVALID_CHECKSUMS", message="checksums.json must be an object")])
    for relative_path, expected in checksums.items():
        path = export_dir / relative_path
        if not path.is_file():
            issues.append(ValidationIssue(code="MISSING_EXPORT_FILE", message=f"missing export file: {relative_path}"))
        elif file_sha256(path) != expected:
            issues.append(ValidationIssue(code="CHECKSUM_MISMATCH", message=f"checksum mismatch: {relative_path}"))
    return _report(issues)
