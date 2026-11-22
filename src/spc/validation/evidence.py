"""Evidence reference, source integrity, and provenance validation."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from ..models import (
    EvidenceClassification,
    EvidenceSpan,
    GroundedStatement,
    ScientificQuestionPlan,
    SourceDocument,
)
from .common import ValidationIssue, ValidationReport, validation_report, walk_values


class EvidenceSpanRepository(Protocol):
    def get(self, key: str) -> EvidenceSpan: ...

    def verify_evidence_integrity(self, evidence: EvidenceSpan) -> SourceDocument: ...


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
    for path, value in walk_values(plan):
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
    return validation_report(issues)
