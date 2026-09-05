from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Protocol

from pydantic import BaseModel, ConfigDict

from .models import (
    AgentHandoffPackage,
    ApprovalDecision,
    ApprovalVerdict,
    DAGTask,
    EvidenceClassification,
    EvidenceSpan,
    ExecutionPolicy,
    ExportManifest,
    GateVerdict,
    GroundedStatement,
    PlanValidationRecord,
    ScientificCapability,
    ScientificQuestionPlan,
    SourceDocument,
)
from .serialization import content_hash, file_sha256, load_data


class EvidenceSpanRepository(Protocol):
    def get(self, key: str) -> EvidenceSpan: ...

    def verify_evidence_integrity(self, evidence: EvidenceSpan) -> SourceDocument: ...


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


def validate_evidence_links(
    plan: ScientificQuestionPlan,
    evidence_repository: EvidenceSpanRepository | None,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    declared = {ref.evidence_id for ref in plan.evidence_refs}
    if len(declared) != len(plan.evidence_refs):
        issues.append(ValidationIssue(code="DUPLICATE_EVIDENCE_ID", message="evidence IDs must be unique"))
    if evidence_repository is None:
        issues.append(
            ValidationIssue(
                code="EVIDENCE_REPOSITORY_REQUIRED",
                message="validation requires a real EvidenceSpan repository",
            )
        )
    else:
        for reference in plan.evidence_refs:
            try:
                evidence = evidence_repository.get(reference.evidence_id)
            except (FileNotFoundError, KeyError):
                issues.append(
                    ValidationIssue(
                        code="EVIDENCE_SPAN_NOT_FOUND",
                        message=f"EvidenceSpan repository has no record: {reference.evidence_id}",
                    )
                )
                continue
            except (OSError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="EVIDENCE_REPOSITORY_ERROR",
                        message=f"cannot read EvidenceSpan {reference.evidence_id}: {error}",
                    )
                )
                continue
            if (evidence.source_id, evidence.source_version) != (
                reference.source_id,
                reference.source_version,
            ):
                issues.append(
                    ValidationIssue(
                        code="EVIDENCE_REFERENCE_MISMATCH",
                        message=f"EvidenceReference does not match stored EvidenceSpan: {reference.evidence_id}",
                    )
                )
                continue
            try:
                evidence_repository.verify_evidence_integrity(evidence)
            except AttributeError:
                issues.append(
                    ValidationIssue(
                        code="EVIDENCE_INTEGRITY_REPOSITORY_REQUIRED",
                        message="repository cannot verify EvidenceSpan-to-source-file integrity",
                    )
                )
            except (FileNotFoundError, OSError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="SOURCE_INTEGRITY_FAILURE",
                        message=f"EvidenceSpan source integrity failed for {reference.evidence_id}: {error}",
                    )
                )
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
    disclosed_fields = {
        item.field
        for item in (*left.fingerprint_differences, *right.fingerprint_differences)
        if item.disclosed_deviation
    }
    for key in sorted(keys):
        if left.method_fingerprint.attributes.get(key) != right.method_fingerprint.attributes.get(key) and key not in disclosed_fields:
            issues.append(ValidationIssue(code="UNDISCLOSED_METHOD_DIFFERENCE", message=f"method difference is not disclosed: {key}"))
    return _report(issues)


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


def validate_plan_validation_record(
    plan: ScientificQuestionPlan,
    record: PlanValidationRecord,
    current_report: ValidationReport,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    expected_binding = (
        plan.plan_id,
        plan.version,
        content_hash(plan),
        plan.domain,
        plan.domain_pack_version,
    )
    actual_binding = (
        record.plan_id,
        record.plan_version,
        record.plan_content_hash,
        record.domain,
        record.domain_pack_version,
    )
    if actual_binding != expected_binding:
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
    return _report(issues)


def validate_conditional_approval(verdict: ApprovalVerdict) -> ValidationReport:
    issues: list[ValidationIssue] = []
    fix_counts = Counter(item.fix_id for item in verdict.required_fixes)
    for fix_id, count in fix_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_REQUIRED_FIX",
                    message=f"required fix is declared more than once: {fix_id}",
                )
            )
    resolution_counts = Counter(item.fix_id for item in verdict.fix_resolutions)
    for fix_id, count in resolution_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_FIX_RESOLUTION",
                    message=f"blocking fix has multiple resolutions: {fix_id}",
                )
            )
    known_fixes = {item.fix_id for item in verdict.required_fixes}
    for resolution in verdict.fix_resolutions:
        if resolution.fix_id not in known_fixes:
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_FIX_RESOLUTION",
                    message=f"resolution references unknown required fix: {resolution.fix_id}",
                )
            )
    if verdict.decision in {
        ApprovalDecision.APPROVE,
        ApprovalDecision.APPROVE_WITH_CONDITIONS,
    }:
        resolutions = {item.fix_id: item for item in verdict.fix_resolutions}
        for fix in verdict.required_fixes:
            resolution = resolutions.get(fix.fix_id)
            if fix.blocking and (
                resolution is None or not resolution.resolved or not resolution.resolution.strip()
            ):
                issues.append(
                    ValidationIssue(
                        code="UNRESOLVED_BLOCKING_FIX",
                        message=f"exportable approval has unresolved blocking fix: {fix.fix_id}",
                    )
                )
    return _report(issues)


def validate_approval_state(
    plan: ScientificQuestionPlan,
    verdict: ApprovalVerdict,
) -> ValidationReport:
    issues = list(validate_conditional_approval(verdict).issues)
    exportable = verdict.decision in {
        ApprovalDecision.APPROVE,
        ApprovalDecision.APPROVE_WITH_CONDITIONS,
    }
    if exportable and verdict.hard_red_flags:
        issues.append(
            ValidationIssue(
                code="HARD_RED_FLAGS_BLOCK_EXPORT",
                message="an approval with hard red flags cannot be exported",
            )
        )

    required_decision_counts = Counter(verdict.human_decisions_required)
    for decision_id, count in required_decision_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_REQUIRED_HUMAN_DECISION",
                    message=f"human decision is required more than once: {decision_id}",
                )
            )
    plan_decisions = {item.decision_id: item for item in plan.required_human_decisions}
    verdict_decisions = set(verdict.human_decisions_required)
    plan_decision_ids = set(plan_decisions)
    for decision_id in sorted(verdict_decisions - plan_decision_ids):
        issues.append(
            ValidationIssue(
                code="UNKNOWN_REQUIRED_HUMAN_DECISION",
                message=f"approval requires a human decision not declared by the plan: {decision_id}",
            )
        )
    for decision_id in sorted(plan_decision_ids - verdict_decisions):
        issues.append(
            ValidationIssue(
                code="HUMAN_DECISION_STATE_MISMATCH",
                message=f"approval omits a human decision required by the plan: {decision_id}",
            )
        )

    resolution_counts = Counter(
        item.decision_id for item in verdict.human_decision_resolutions
    )
    for decision_id, count in resolution_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_HUMAN_DECISION_RESOLUTION",
                    message=f"human decision has multiple resolutions: {decision_id}",
                )
            )
    resolutions = {
        item.decision_id: item for item in verdict.human_decision_resolutions
    }
    for decision_id in sorted(set(resolutions) - verdict_decisions):
        issues.append(
            ValidationIssue(
                code="UNKNOWN_HUMAN_DECISION_RESOLUTION",
                message=f"resolution references a human decision not required by the approval: {decision_id}",
            )
        )
    if exportable:
        for decision_id in sorted(plan_decision_ids | verdict_decisions):
            resolution = resolutions.get(decision_id)
            if (
                resolution is None
                or not resolution.resolved
                or not resolution.selected_option.strip()
                or not resolution.rationale.strip()
            ):
                issues.append(
                    ValidationIssue(
                        code="UNRESOLVED_HUMAN_DECISION",
                        message=f"exportable approval has unresolved human decision: {decision_id}",
                    )
                )
                continue
            planned = plan_decisions.get(decision_id)
            if planned is not None and resolution.selected_option not in planned.options:
                issues.append(
                    ValidationIssue(
                        code="INVALID_HUMAN_DECISION_OPTION",
                        message=f"selected option is not declared for human decision: {decision_id}",
                    )
                )
    if (
        verdict.decision == ApprovalDecision.APPROVE_WITH_CONDITIONS
        and not verdict.required_fixes
        and not verdict.human_decisions_required
    ):
        issues.append(
            ValidationIssue(
                code="CONDITIONAL_APPROVAL_WITHOUT_CONDITIONS",
                message="approve_with_conditions requires at least one fix or human decision",
            )
        )
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
    validation_record: PlanValidationRecord,
    gate: GateVerdict,
    handoff: AgentHandoffPackage,
    *,
    human_selected: bool,
) -> ValidationReport:
    issues = list(validate_approval_boundary(plan, verdict).issues)
    plan_hash = content_hash(plan)
    if verdict.decision not in {ApprovalDecision.APPROVE, ApprovalDecision.APPROVE_WITH_CONDITIONS}:
        issues.append(ValidationIssue(code="PLAN_NOT_APPROVED", message="approval decision does not permit export"))
    approval_state = validate_approval_state(plan, verdict)
    issues.extend(approval_state.issues)
    if gate.passed and not approval_state.valid:
        issues.append(
            ValidationIssue(
                code="GATE_STATE_INCONSISTENT",
                message="a passed gate is inconsistent with the approval state",
            )
        )
    declared_evidence = {reference.evidence_id for reference in plan.evidence_refs}
    for resolution in verdict.fix_resolutions:
        for evidence_id in resolution.evidence_refs:
            if evidence_id not in declared_evidence:
                issues.append(
                    ValidationIssue(
                        code="FIX_RESOLUTION_EVIDENCE_MISMATCH",
                        message=f"fix resolution references evidence outside the validated plan: {evidence_id}",
                    )
                )
    if not gate.passed:
        issues.append(ValidationIssue(code="PLAN_GATE_FAILED", message="plan gate has not passed"))
    if (gate.candidate_id, gate.candidate_version, gate.candidate_content_hash) != (plan.plan_id, plan.version, plan_hash):
        issues.append(ValidationIssue(code="STALE_PLAN_GATE", message="plan gate is not bound to current plan content"))
    if (gate.approval_verdict_id, gate.approval_verdict_hash) != (
        verdict.verdict_id,
        content_hash(verdict),
    ):
        issues.append(
            ValidationIssue(
                code="GATE_APPROVAL_BINDING_MISMATCH",
                message="GateVerdict is not bound to the supplied ApprovalVerdict",
            )
        )
    if (gate.plan_validation_id, gate.plan_validation_hash) != (
        validation_record.validation_id,
        content_hash(validation_record),
    ):
        issues.append(
            ValidationIssue(
                code="GATE_VALIDATION_BINDING_MISMATCH",
                message="GateVerdict is not bound to the supplied PlanValidationRecord",
            )
        )
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


def _safe_export_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and path.as_posix() == value and all(part not in {"", ".", ".."} for part in path.parts)


REQUIRED_EXPORT_FILES = frozenset(
    {
        "manifest.yaml",
        "selected-plan.yaml",
        "handoff-package.yaml",
        "task-graph.yaml",
        "approvals/plan-review.yaml",
        "approvals/plan-validation.yaml",
        "approvals/plan-gate.yaml",
        "capability-bindings.yaml",
        "evidence-manifest.jsonl",
        "decisions.jsonl",
        "execution-policy.yaml",
        "checksums.json",
    }
)


def _load_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _semantic_export_issues(export_dir: Path) -> list[ValidationIssue]:
    try:
        plan = ScientificQuestionPlan.model_validate(load_data(export_dir / "selected-plan.yaml"))
        manifest = ExportManifest.model_validate(load_data(export_dir / "manifest.yaml"))
        handoff = AgentHandoffPackage.model_validate(load_data(export_dir / "handoff-package.yaml"))
        verdict = ApprovalVerdict.model_validate(load_data(export_dir / "approvals/plan-review.yaml"))
        validation = PlanValidationRecord.model_validate(
            load_data(export_dir / "approvals/plan-validation.yaml")
        )
        gate = GateVerdict.model_validate(load_data(export_dir / "approvals/plan-gate.yaml"))
        task_graph = load_data(export_dir / "task-graph.yaml")
        bindings = load_data(export_dir / "capability-bindings.yaml")
        policy = ExecutionPolicy.model_validate(load_data(export_dir / "execution-policy.yaml"))
        evidence_manifest = _load_jsonl(export_dir / "evidence-manifest.jsonl")
        decisions = _load_jsonl(export_dir / "decisions.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [
            ValidationIssue(
                code="INVALID_EXPORT_CONTENT",
                message=f"cannot parse required export content: {error}",
            )
        ]

    issues: list[ValidationIssue] = []
    plan_hash = content_hash(plan)
    plan_binding = (plan.plan_id, plan.version, plan_hash)
    if (manifest.source_plan_id, manifest.source_plan_version, manifest.source_plan_hash) != plan_binding:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="manifest plan binding does not match selected plan"))
    if (handoff.source_plan_id, handoff.source_plan_version, handoff.source_plan_hash) != plan_binding:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="handoff plan binding does not match selected plan"))
    if (verdict.candidate_id, verdict.candidate_version, verdict.candidate_content_hash) != plan_binding:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="approval plan binding does not match selected plan"))
    if (validation.plan_id, validation.plan_version, validation.plan_content_hash) != plan_binding:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="validation plan binding does not match selected plan"))
    if (validation.domain, validation.domain_pack_version) != (
        plan.domain,
        plan.domain_pack_version,
    ):
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="validation domain binding does not match selected plan"))
    if (gate.candidate_id, gate.candidate_version, gate.candidate_content_hash) != plan_binding:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="gate plan binding does not match selected plan"))
    if (manifest.export_id, manifest.target_agent) != (handoff.export_id, handoff.target_agent):
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="manifest export target does not match handoff"))
    if (gate.approval_verdict_id, gate.approval_verdict_hash) != (
        verdict.verdict_id,
        content_hash(verdict),
    ):
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="gate does not bind the exported approval"))
    if (gate.plan_validation_id, gate.plan_validation_hash) != (
        validation.validation_id,
        content_hash(validation),
    ):
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="gate does not bind the exported validation record"))
    if not validation.valid or not gate.passed:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="export contains a failed validation record or gate"))
    issues.extend(validate_approval_state(plan, verdict).issues)
    declared_evidence = {reference.evidence_id for reference in plan.evidence_refs}
    for resolution in verdict.fix_resolutions:
        if not set(resolution.evidence_refs).issubset(declared_evidence):
            issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message=f"fix resolution evidence does not belong to selected plan: {resolution.fix_id}"))
    issues.extend(validate_phase1_boundary(policy, plan.tasks).issues)

    expected_task_graph = {
        "plan_id": plan.plan_id,
        "plan_hash": plan_hash,
        "tasks": [task.task_id for task in plan.tasks],
    }
    if task_graph != expected_task_graph:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="task graph does not match selected plan"))
    expected_bindings = [
        item.model_dump(mode="json", exclude_none=True)
        for item in handoff.capability_bindings
    ]
    if bindings != expected_bindings:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="capability bindings do not match handoff"))
    if policy != handoff.execution_policy:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="execution policy does not match handoff"))
    expected_evidence = [item.model_dump(mode="json") for item in plan.evidence_refs]
    if evidence_manifest != expected_evidence:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="evidence manifest does not match selected plan"))
    expected_decisions = [
        {
            "decision": "human_selected",
            "plan_id": plan.plan_id,
            "plan_version": plan.version,
            "plan_hash": plan_hash,
        }
    ]
    if decisions != expected_decisions:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="decision record does not confirm selected plan"))

    expected_task_files = {f"tasks/{task.task_id}.yaml" for task in plan.tasks}
    actual_task_files = {
        path.relative_to(export_dir).as_posix()
        for path in (export_dir / "tasks").glob("*.yaml")
        if path.is_file()
    }
    if actual_task_files != expected_task_files:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="exported task files do not match selected plan"))
    binding_by_capability = {
        item.scientific_capability_id: item for item in handoff.capability_bindings
    }
    for task in plan.tasks:
        binding = binding_by_capability.get(task.capability_id)
        if binding is None:
            issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message=f"task has no exported capability binding: {task.task_id}"))
            continue
        expected_task = {
            **task.model_dump(mode="json"),
            "source_plan_id": plan.plan_id,
            "source_plan_hash": plan_hash,
            "target_capability_binding": binding.model_dump(mode="json", exclude_none=True),
            "intent_fingerprint_ref": plan.intent_fingerprint.fingerprint_id,
            "system_fingerprint_ref": plan.system_fingerprint.fingerprint_id,
            "method_fingerprint_ref": plan.method_fingerprint.fingerprint_id,
            "execution_policy": handoff.execution_policy.model_dump(mode="json"),
            "runnable": False,
        }
        try:
            actual_task = load_data(export_dir / f"tasks/{task.task_id}.yaml")
        except (OSError, ValueError) as error:
            issues.append(ValidationIssue(code="INVALID_EXPORT_CONTENT", message=f"cannot parse exported task {task.task_id}: {error}"))
        else:
            if actual_task != expected_task:
                issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message=f"exported task does not match selected plan: {task.task_id}"))

    fingerprints = (
        plan.intent_fingerprint,
        plan.system_fingerprint,
        plan.method_fingerprint,
    )
    expected_fingerprint_files = {
        f"fingerprints/{item.fingerprint_id}.yaml" for item in fingerprints
    }
    actual_fingerprint_files = {
        path.relative_to(export_dir).as_posix()
        for path in (export_dir / "fingerprints").glob("*.yaml")
        if path.is_file()
    }
    if actual_fingerprint_files != expected_fingerprint_files:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="exported fingerprint files do not match selected plan"))
    for fingerprint in fingerprints:
        try:
            actual_fingerprint = load_data(
                export_dir / f"fingerprints/{fingerprint.fingerprint_id}.yaml"
            )
        except (OSError, ValueError) as error:
            issues.append(ValidationIssue(code="INVALID_EXPORT_CONTENT", message=f"cannot parse exported fingerprint {fingerprint.fingerprint_id}: {error}"))
        else:
            if actual_fingerprint != fingerprint.model_dump(mode="json", exclude_none=True):
                issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message=f"exported fingerprint does not match selected plan: {fingerprint.fingerprint_id}"))
    expected_contract_files = (
        set(REQUIRED_EXPORT_FILES)
        | expected_task_files
        | expected_fingerprint_files
    )
    actual_contract_files = {
        path.relative_to(export_dir).as_posix()
        for path in export_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    for relative_path in sorted(actual_contract_files - expected_contract_files):
        issues.append(
            ValidationIssue(
                code="EXTRA_EXPORT_FILE",
                message=f"file is outside the export contract: {relative_path}",
            )
        )
    for relative_path in sorted(expected_contract_files - actual_contract_files):
        issues.append(
            ValidationIssue(
                code="MISSING_REQUIRED_EXPORT_FILE",
                message=f"required export-contract file is missing: {relative_path}",
            )
        )
    expected_manifest_files = expected_contract_files - {"manifest.yaml", "checksums.json"}
    if set(manifest.files) != expected_manifest_files:
        issues.append(
            ValidationIssue(
                code="EXPORT_MANIFEST_FILE_MISMATCH",
                message="manifest does not contain the exact export-contract file set",
            )
        )
    return issues


def validate_export(export_dir: Path) -> ValidationReport:
    issues: list[ValidationIssue] = []
    export_root = export_dir.resolve()
    checksums_path = export_dir / "checksums.json"
    if checksums_path.is_symlink():
        return _report(
            [
                ValidationIssue(
                    code="UNSAFE_EXPORT_SYMLINK",
                    message="checksums.json cannot be a symlink",
                )
            ]
        )
    if not checksums_path.is_file():
        return _report([ValidationIssue(code="MISSING_CHECKSUMS", message="checksums.json is missing")])
    try:
        checksums = load_data(checksums_path)
    except (OSError, ValueError) as error:
        return _report([ValidationIssue(code="INVALID_CHECKSUMS", message=f"cannot read checksums.json: {error}")])
    if not isinstance(checksums, dict):
        return _report([ValidationIssue(code="INVALID_CHECKSUMS", message="checksums.json must be an object")])
    safe_checksum_paths: set[str] = set()
    for relative_path, expected in checksums.items():
        if not _safe_export_relative_path(relative_path):
            issues.append(
                ValidationIssue(
                    code="UNSAFE_CHECKSUM_PATH",
                    message=f"checksum path is not a safe package-relative path: {relative_path!r}",
                )
            )
            continue
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            issues.append(
                ValidationIssue(
                    code="INVALID_CHECKSUM_VALUE",
                    message=f"invalid SHA-256 value for: {relative_path}",
                )
            )
            continue
        safe_checksum_paths.add(relative_path)
        path = export_dir / relative_path
        resolved = path.resolve()
        if not resolved.is_relative_to(export_root):
            issues.append(
                ValidationIssue(
                    code="UNSAFE_CHECKSUM_PATH",
                    message=f"checksum path escapes export root: {relative_path}",
                )
            )
            continue
        current = path
        symlink_found = False
        while current != export_dir:
            if current.is_symlink():
                symlink_found = True
                break
            current = current.parent
        if symlink_found:
            issues.append(
                ValidationIssue(
                    code="UNSAFE_EXPORT_SYMLINK",
                    message=f"export package cannot contain symlinks: {relative_path}",
                )
            )
            continue
        if not path.is_file():
            issues.append(ValidationIssue(code="MISSING_EXPORT_FILE", message=f"missing export file: {relative_path}"))
        elif file_sha256(path) != expected:
            issues.append(ValidationIssue(code="CHECKSUM_MISMATCH", message=f"checksum mismatch: {relative_path}"))
    actual_files = {
        path.relative_to(export_dir).as_posix()
        for path in export_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected_files = safe_checksum_paths | {"checksums.json"}
    for relative_path in sorted(REQUIRED_EXPORT_FILES - actual_files):
        issues.append(
            ValidationIssue(
                code="MISSING_REQUIRED_EXPORT_FILE",
                message=f"required export file is missing: {relative_path}",
            )
        )
    for relative_path in sorted(actual_files - expected_files):
        issues.append(
            ValidationIssue(
                code="EXTRA_EXPORT_FILE",
                message=f"export contains an unchecked extra file: {relative_path}",
            )
        )
    manifest_path = export_dir / "manifest.yaml"
    if manifest_path.is_file():
        try:
            manifest = ExportManifest.model_validate(load_data(manifest_path))
        except (OSError, ValueError) as error:
            issues.append(ValidationIssue(code="INVALID_EXPORT_MANIFEST", message=str(error)))
        else:
            expected_manifest_files = safe_checksum_paths - {"manifest.yaml"}
            if set(manifest.files) != expected_manifest_files:
                issues.append(
                    ValidationIssue(
                        code="EXPORT_MANIFEST_FILE_MISMATCH",
                        message="manifest file inventory does not match checksummed package files",
                    )
                )
    if REQUIRED_EXPORT_FILES.issubset(actual_files):
        issues.extend(_semantic_export_issues(export_dir))
    return _report(issues)
