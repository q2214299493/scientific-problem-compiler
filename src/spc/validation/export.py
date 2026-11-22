"""In-memory handoff and on-disk export validation."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from ..models import (
    AgentHandoffPackage,
    ApprovalDecision,
    ApprovalMode,
    ApprovalReviewInput,
    ApprovalReviewRecord,
    ApprovalVerdict,
    ExecutionPolicy,
    ExportManifest,
    GateVerdict,
    IndependentApprovalReceipt,
    PlanCompilationReceipt,
    PlanValidationRecord,
    ProjectTrustPolicy,
    ScientificQuestionPlan,
    parse_approval_review_input,
)
from ..serialization import content_hash, file_sha256, load_data
from .approval import (
    gate_binds_approval_receipt,
    gate_binds_compilation_receipt,
    gate_binds_plan_version,
    gate_binds_policy,
    gate_binds_validation,
    gate_binds_verdict,
    validate_approval_boundary,
    validate_approval_state,
    validate_independent_approval_chain,
)
from .common import ValidationIssue, ValidationReport, validation_report
from .plans import validate_phase1_boundary, validate_plan_compilation_receipt


def validate_handoff_package(
    plan: ScientificQuestionPlan,
    verdict: ApprovalVerdict,
    validation_record: PlanValidationRecord,
    gate: GateVerdict,
    handoff: AgentHandoffPackage,
    *,
    human_selected: bool,
    trust_policy: ProjectTrustPolicy,
    compilation_receipt: PlanCompilationReceipt | None = None,
    review_input: ApprovalReviewInput | None = None,
    review: ApprovalReviewRecord | None = None,
    receipt: IndependentApprovalReceipt | None = None,
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
    if not gate_binds_plan_version(gate, (plan.plan_id, plan.version, plan_hash)):
        issues.append(ValidationIssue(code="STALE_PLAN_GATE", message="plan gate is not bound to current plan content"))
    if not gate_binds_verdict(gate, verdict):
        issues.append(
            ValidationIssue(
                code="GATE_APPROVAL_BINDING_MISMATCH",
                message="GateVerdict is not bound to the supplied ApprovalVerdict",
            )
        )
    if not gate_binds_validation(gate, validation_record):
        issues.append(
            ValidationIssue(
                code="GATE_VALIDATION_BINDING_MISMATCH",
                message="GateVerdict is not bound to the supplied PlanValidationRecord",
            )
        )
    if not gate_binds_policy(gate, trust_policy):
        issues.append(
            ValidationIssue(
                code="GATE_TRUST_POLICY_BINDING_MISMATCH",
                message="GateVerdict is not bound to the supplied ProjectTrustPolicy",
            )
        )
    if trust_policy.approval_mode == ApprovalMode.INDEPENDENT_REQUIRED:
        if compilation_receipt is None:
            issues.append(
                ValidationIssue(
                    code="MISSING_PLAN_COMPILATION_RECEIPT",
                    message="independent approval policy requires grounded compiler lineage",
                )
            )
        else:
            issues.extend(
                validate_plan_compilation_receipt(plan, compilation_receipt).issues
            )
            if not gate_binds_compilation_receipt(gate, compilation_receipt):
                issues.append(
                    ValidationIssue(
                        code="GATE_COMPILATION_RECEIPT_BINDING_MISMATCH",
                        message="GateVerdict is not bound to the PlanCompilationReceipt",
                    )
                )
        if review_input is None or review is None or receipt is None:
            issues.append(
                ValidationIssue(
                    code="MISSING_INDEPENDENT_APPROVAL",
                    message="Phase 2D plan requires an independent review input, review record, and receipt",
                )
            )
        else:
            chain = validate_independent_approval_chain(
                plan, verdict, review_input, review, receipt
            )
            issues.extend(chain.issues)
            if not gate_binds_approval_receipt(gate, receipt):
                issues.append(
                    ValidationIssue(
                        code="GATE_INDEPENDENT_APPROVAL_BINDING_MISMATCH",
                        message="GateVerdict is not bound to the independent approval receipt",
                    )
                )
    elif compilation_receipt is not None:
        issues.extend(validate_plan_compilation_receipt(plan, compilation_receipt).issues)
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
    return validation_report(issues)


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
        "approvals/project-trust-policy.yaml",
        "capability-bindings.yaml",
        "evidence-manifest.jsonl",
        "decisions.jsonl",
        "execution-policy.yaml",
        "checksums.json",
    }
)


INDEPENDENT_APPROVAL_EXPORT_FILES = frozenset(
    {
        "approvals/approval-review-input.yaml",
        "approvals/approval-review-record.yaml",
        "approvals/independent-approval-receipt.yaml",
        "provenance/plan-compilation-receipt.yaml",
    }
)


def _load_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _semantic_export_issues(export_dir: Path) -> list[ValidationIssue]:
    try:
        plan = ScientificQuestionPlan.model_validate(load_data(export_dir / "selected-plan.yaml"))
        trust_policy = ProjectTrustPolicy.model_validate(
            load_data(export_dir / "approvals/project-trust-policy.yaml")
        )
        review_input = None
        review = None
        receipt = None
        compilation_receipt = None
        if trust_policy.approval_mode == ApprovalMode.INDEPENDENT_REQUIRED:
            review_input = parse_approval_review_input(
                load_data(export_dir / "approvals/approval-review-input.yaml")
            )
            review = ApprovalReviewRecord.model_validate(
                load_data(export_dir / "approvals/approval-review-record.yaml")
            )
            receipt = IndependentApprovalReceipt.model_validate(
                load_data(export_dir / "approvals/independent-approval-receipt.yaml")
            )
            compilation_receipt = PlanCompilationReceipt.model_validate(
                load_data(export_dir / "provenance/plan-compilation-receipt.yaml")
            )
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
    if not gate_binds_plan_version(gate, plan_binding):
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="gate plan binding does not match selected plan"))
    if (manifest.export_id, manifest.target_agent) != (handoff.export_id, handoff.target_agent):
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="manifest export target does not match handoff"))
    if not gate_binds_verdict(gate, verdict):
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="gate does not bind the exported approval"))
    if not gate_binds_validation(gate, validation):
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="gate does not bind the exported validation record"))
    if not validation.valid or not gate.passed:
        issues.append(ValidationIssue(code="EXPORT_SEMANTIC_MISMATCH", message="export contains a failed validation record or gate"))
    issues.extend(validate_approval_state(plan, verdict).issues)
    if not gate_binds_policy(gate, trust_policy):
        issues.append(
            ValidationIssue(
                code="EXPORT_SEMANTIC_MISMATCH",
                message="gate does not bind the exported ProjectTrustPolicy",
            )
        )
    if trust_policy.approval_mode == ApprovalMode.INDEPENDENT_REQUIRED:
        if (
            review_input is None
            or review is None
            or receipt is None
            or compilation_receipt is None
        ):
            issues.append(
                ValidationIssue(
                    code="MISSING_INDEPENDENT_APPROVAL",
                    message="export lacks the required independent approval artifacts",
                )
            )
        else:
            issues.extend(
                validate_plan_compilation_receipt(
                    plan, compilation_receipt
                ).issues
            )
            issues.extend(
                validate_independent_approval_chain(
                    plan, verdict, review_input, review, receipt
                ).issues
            )
            if not gate_binds_approval_receipt(gate, receipt):
                issues.append(
                    ValidationIssue(
                        code="EXPORT_SEMANTIC_MISMATCH",
                        message="gate does not bind the exported independent approval receipt",
                    )
                )
            if not gate_binds_compilation_receipt(gate, compilation_receipt):
                issues.append(
                    ValidationIssue(
                        code="EXPORT_SEMANTIC_MISMATCH",
                        message="gate does not bind the exported PlanCompilationReceipt",
                    )
                )
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
        | (
            set(INDEPENDENT_APPROVAL_EXPORT_FILES)
            if trust_policy.approval_mode == ApprovalMode.INDEPENDENT_REQUIRED
            else set()
        )
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
        return validation_report(
            [
                ValidationIssue(
                    code="UNSAFE_EXPORT_SYMLINK",
                    message="checksums.json cannot be a symlink",
                )
            ]
        )
    if not checksums_path.is_file():
        return validation_report([ValidationIssue(code="MISSING_CHECKSUMS", message="checksums.json is missing")])
    try:
        checksums = load_data(checksums_path)
    except (OSError, ValueError) as error:
        return validation_report([ValidationIssue(code="INVALID_CHECKSUMS", message=f"cannot read checksums.json: {error}")])
    if not isinstance(checksums, dict):
        return validation_report([ValidationIssue(code="INVALID_CHECKSUMS", message="checksums.json must be an object")])
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
    required_export_files = set(REQUIRED_EXPORT_FILES)
    trust_policy_path = export_dir / "approvals/project-trust-policy.yaml"
    if trust_policy_path.is_file():
        try:
            trust_policy = ProjectTrustPolicy.model_validate(
                load_data(trust_policy_path)
            )
        except (OSError, ValueError):
            pass
        else:
            if trust_policy.approval_mode == ApprovalMode.INDEPENDENT_REQUIRED:
                required_export_files.update(INDEPENDENT_APPROVAL_EXPORT_FILES)
    for relative_path in sorted(required_export_files - actual_files):
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
    if required_export_files.issubset(actual_files):
        issues.extend(_semantic_export_issues(export_dir))
    return validation_report(issues)
