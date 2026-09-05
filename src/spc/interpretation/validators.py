from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
import re
from typing import Iterable

from ..models import (
    EpistemicStatus,
    ReportedResult,
    ScientificContextPacket,
    ScientificEvidencePacket,
)
from ..serialization import content_hash
from ..validators import EvidenceSpanRepository, ValidationIssue, ValidationReport


def _report(issues: Iterable[ValidationIssue]) -> ValidationReport:
    result = tuple(issues)
    return ValidationReport(valid=not result, issues=result)


def _context_evidence_ids(context: ScientificContextPacket) -> set[str]:
    return {hit.record_id for hit in context.evidence_hits}


def _check_refs(
    refs: tuple[str, ...],
    *,
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
    path: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    allowed = _context_evidence_ids(context)
    for evidence_id in refs:
        if evidence_id not in allowed:
            issues.append(
                ValidationIssue(
                    code="INTERPRETATION_EVIDENCE_NOT_RETRIEVED",
                    message=f"evidence reference was not retrieved into the context: {evidence_id}",
                    path=path,
                )
            )
            continue
        try:
            evidence = evidence_repository.get(evidence_id)
            evidence_repository.verify_evidence_integrity(evidence)
        except (FileNotFoundError, KeyError, OSError, ValueError) as error:
            issues.append(
                ValidationIssue(
                    code="INTERPRETATION_EVIDENCE_INTEGRITY_FAILURE",
                    message=f"cannot verify EvidenceSpan {evidence_id}: {error}",
                    path=path,
                )
            )
            continue
        snapshot_hash = context.knowledge_snapshot.evidence_span_hashes.get(evidence_id)
        if snapshot_hash != content_hash(evidence):
            issues.append(
                ValidationIssue(
                    code="INTERPRETATION_EVIDENCE_SNAPSHOT_MISMATCH",
                    message=f"EvidenceSpan differs from the context knowledge snapshot: {evidence_id}",
                    path=path,
                )
            )
    return issues


def validate_claim_evidence_refs(
    packet: ScientificEvidencePacket,
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for index, claim in enumerate(packet.source_claims):
        issues.extend(
            _check_refs(
                claim.evidence_refs,
                context=context,
                evidence_repository=evidence_repository,
                path=f"source_claims[{index}]",
            )
        )
    return _report(issues)


def validate_claim_source_binding(
    packet: ScientificEvidencePacket,
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
) -> ValidationReport:
    del context
    issues: list[ValidationIssue] = []
    for index, claim in enumerate(packet.source_claims):
        if claim.source_role.casefold() in {"spc", "agent", "mock", "interpreter"} and (
            claim.epistemic_status != EpistemicStatus.SOURCE_INTERPRETATION
        ):
            issues.append(
                ValidationIssue(
                    code="AGENT_INTERPRETATION_MISLABELED",
                    message="agent-created interpretation must be tagged source_interpretation",
                    path=f"source_claims[{index}]",
                )
            )
        if claim.claim_type == "hypothesis" and claim.epistemic_status != EpistemicStatus.SOURCE_HYPOTHESIS:
            issues.append(
                ValidationIssue(
                    code="SOURCE_HYPOTHESIS_PROMOTED",
                    message="an author hypothesis must remain source_hypothesis",
                    path=f"source_claims[{index}]",
                )
            )
        if claim.claim_type == "reviewer_question" and claim.epistemic_status != EpistemicStatus.UNRESOLVED:
            issues.append(
                ValidationIssue(
                    code="REVIEWER_QUESTION_PROMOTED",
                    message="a reviewer question must remain unresolved",
                    path=f"source_claims[{index}]",
                )
            )
        for evidence_id in claim.evidence_refs:
            try:
                evidence = evidence_repository.get(evidence_id)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue
            if claim.epistemic_status != EpistemicStatus.SOURCE_INTERPRETATION and claim.text not in evidence.text:
                issues.append(
                    ValidationIssue(
                        code="CLAIM_SOURCE_TEXT_MISMATCH",
                        message=f"claim text is not present in EvidenceSpan {evidence_id}",
                        path=f"source_claims[{index}]",
                    )
                )
    return _report(issues)


def validate_result_units(packet: ScientificEvidencePacket) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for index, result in enumerate(packet.reported_results):
        if not math.isfinite(result.value) or not result.unit.strip():
            issues.append(
                ValidationIssue(
                    code="INVALID_REPORTED_RESULT_UNIT",
                    message="reported numerical results require a finite value and explicit unit",
                    path=f"reported_results[{index}]",
                )
            )
    return _report(issues)


def validate_result_provenance(
    packet: ScientificEvidencePacket,
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for index, result in enumerate(packet.reported_results):
        path = f"reported_results[{index}]"
        if (
            result.method_context.get("method_family") == "model_prediction"
            and result.result_status.value != "predicted_reported"
        ):
            issues.append(
                ValidationIssue(
                    code="PREDICTION_MISLABELED",
                    message="a model-derived numerical result must remain predicted_reported",
                    path=path,
                )
            )
        issues.extend(
            _check_refs(
                result.evidence_refs,
                context=context,
                evidence_repository=evidence_repository,
                path=path,
            )
        )
        found = False
        for evidence_id in result.evidence_refs:
            try:
                text = evidence_repository.get(evidence_id).text
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue
            for match in re.finditer(
                r"([-+]?\d+(?:\.\d+)?)\s*(meV|eV|kJ\s*/\s*mol|kcal\s*/\s*mol|K|bar|Pa|atm|%)\b",
                text,
                re.I,
            ):
                unit = re.sub(r"\s+", "", match.group(2)).casefold()
                if float(match.group(1)) == result.value and unit == result.unit.casefold():
                    found = True
                    break
            if found:
                break
        if not found:
            issues.append(
                ValidationIssue(
                    code="RESULT_NOT_PRESENT_IN_SOURCE",
                    message=f"reported value and unit are not present in referenced evidence: {result.result_id}",
                    path=path,
                )
            )
    return _report(issues)


def _mismatched_context_fields(left: ReportedResult, right: ReportedResult) -> set[str]:
    left_fields = {
        **{f"system_context.{key}": value for key, value in left.system_context.items()},
        **{f"method_context.{key}": value for key, value in left.method_context.items()},
    }
    right_fields = {
        **{f"system_context.{key}": value for key, value in right.system_context.items()},
        **{f"method_context.{key}": value for key, value in right.method_context.items()},
    }
    mismatches = {
        field
        for field in set(left_fields) | set(right_fields)
        if left_fields.get(field) != right_fields.get(field)
        or "not specified" in str(left_fields.get(field, "")).casefold()
        or "not specified" in str(right_fields.get(field, "")).casefold()
    }
    if left.unit != right.unit:
        mismatches.add("unit")
    return mismatches


def validate_result_comparability(packet: ScientificEvidencePacket) -> ValidationReport:
    issues: list[ValidationIssue] = []
    constraints = {constraint.comparison_target: constraint for constraint in packet.comparison_constraints}
    for left, right in combinations(packet.reported_results, 2):
        if left.quantity != right.quantity:
            continue
        mismatches = _mismatched_context_fields(left, right)
        if not mismatches:
            continue
        target = f"{left.result_id} vs {right.result_id}"
        constraint = constraints.get(target)
        if constraint is None or not mismatches.issubset(set(constraint.must_match_fields)):
            issues.append(
                ValidationIssue(
                    code="UNGUARDED_INCOMPARABLE_RESULTS",
                    message=f"system or method mismatch is not guarded for {target}",
                    path="comparison_constraints",
                )
            )
    return _report(issues)


def validate_conflict_sets(packet: ScientificEvidencePacket) -> ValidationReport:
    issues: list[ValidationIssue] = []
    claim_ids = {claim.claim_id for claim in packet.source_claims}
    for index, conflict in enumerate(packet.conflict_sets):
        if not set(conflict.claim_refs).issubset(claim_ids):
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_CONFLICT_CLAIM",
                    message="ConflictSet references an unknown SourceClaim",
                    path=f"conflict_sets[{index}]",
                )
            )
        if conflict.resolution_status != "unresolved":
            issues.append(
                ValidationIssue(
                    code="UNSUPPORTED_CONFLICT_RESOLUTION",
                    message="Phase 2B cannot silently resolve conflicting source claims",
                    path=f"conflict_sets[{index}]",
                )
            )
    return _report(issues)


def validate_comparison_constraints(
    packet: ScientificEvidencePacket,
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    result_ids = {result.result_id for result in packet.reported_results}
    for index, constraint in enumerate(packet.comparison_constraints):
        path = f"comparison_constraints[{index}]"
        target_ids = tuple(part.strip() for part in constraint.comparison_target.split(" vs "))
        if len(target_ids) != 2 or not set(target_ids).issubset(result_ids):
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_COMPARISON_TARGET",
                    message="ComparisonConstraint target must name two reported results",
                    path=path,
                )
            )
        if not set(constraint.disclosure_required_fields).issubset(set(constraint.must_match_fields)):
            issues.append(
                ValidationIssue(
                    code="INVALID_COMPARISON_DISCLOSURE",
                    message="disclosure-required fields must be comparison fields",
                    path=path,
                )
            )
        issues.extend(
            _check_refs(
                constraint.evidence_refs,
                context=context,
                evidence_repository=evidence_repository,
                path=path,
            )
        )
    return _report(issues)


def validate_gap_capabilities(
    packet: ScientificEvidencePacket,
    context: ScientificContextPacket,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    capability_ids = {hit.record_id for hit in context.capability_hits}
    allowed_evidence = _context_evidence_ids(context)
    for index, gap in enumerate(packet.evidence_gaps):
        path = f"evidence_gaps[{index}]"
        for capability_id in gap.candidate_capabilities:
            if capability_id not in capability_ids:
                issues.append(
                    ValidationIssue(
                        code="UNRETRIEVED_GAP_CAPABILITY",
                        message=f"gap names a capability not retrieved into context: {capability_id}",
                        path=path,
                    )
                )
        if not set(gap.evidence_refs).issubset(allowed_evidence):
            issues.append(
                ValidationIssue(
                    code="UNRETRIEVED_GAP_EVIDENCE",
                    message="gap references evidence not retrieved into context",
                    path=path,
                )
            )
    return _report(issues)


def validate_evidence_packet_integrity(
    packet: ScientificEvidencePacket,
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if packet.context_id != context.context_id or packet.context_hash != context.content_hash:
        issues.append(
            ValidationIssue(
                code="EVIDENCE_PACKET_CONTEXT_MISMATCH",
                message="ScientificEvidencePacket is not bound to the supplied context ID and hash",
            )
        )
    manifest = packet.provenance_manifest
    expected_manifest = {
        "context_id": context.context_id,
        "context_hash": context.content_hash,
        "retrieval_id": context.retrieval_manifest.retrieval_id,
        "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        issues.append(
            ValidationIssue(
                code="EVIDENCE_PACKET_PROVENANCE_MISMATCH",
                message="provenance manifest does not bind the source context and retrieval snapshot",
            )
        )
    id_values = [
        *(item.claim_id for item in packet.source_claims),
        *(item.result_id for item in packet.reported_results),
        *(item.fact_id for item in packet.method_facts),
        *(item.fact_id for item in packet.model_facts),
        *(item.assessment_id for item in packet.evidence_assessments),
        *(item.conflict_id for item in packet.conflict_sets),
        *(item.constraint_id for item in packet.comparison_constraints),
        *(item.gap_id for item in packet.evidence_gaps),
    ]
    for identifier, count in Counter(id_values).items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_INTERPRETATION_ID",
                    message=f"interpretation entity ID is duplicated: {identifier}",
                )
            )
    claim_ids = {claim.claim_id for claim in packet.source_claims}
    for assessment in packet.evidence_assessments:
        if assessment.claim_ref not in claim_ids:
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_ASSESSMENT_CLAIM",
                    message=f"assessment references unknown claim: {assessment.claim_ref}",
                )
            )
        issues.extend(
            _check_refs(
                tuple(
                    dict.fromkeys(
                        (*assessment.supporting_evidence_refs, *assessment.contradicting_evidence_refs)
                    )
                ),
                context=context,
                evidence_repository=evidence_repository,
                path=f"evidence_assessments[{assessment.assessment_id}]",
            )
        )
    for collection_name, facts in (("method_facts", packet.method_facts), ("model_facts", packet.model_facts)):
        for index, fact in enumerate(facts):
            issues.extend(
                _check_refs(
                    fact.evidence_refs,
                    context=context,
                    evidence_repository=evidence_repository,
                    path=f"{collection_name}[{index}]",
                )
            )
    issues.extend(validate_claim_evidence_refs(packet, context, evidence_repository).issues)
    issues.extend(validate_claim_source_binding(packet, context, evidence_repository).issues)
    issues.extend(validate_result_units(packet).issues)
    issues.extend(validate_result_provenance(packet, context, evidence_repository).issues)
    issues.extend(validate_result_comparability(packet).issues)
    issues.extend(validate_conflict_sets(packet).issues)
    issues.extend(validate_comparison_constraints(packet, context, evidence_repository).issues)
    issues.extend(validate_gap_capabilities(packet, context).issues)
    known_capabilities = {hit.record_id for hit in context.capability_hits}
    if not set(packet.capability_candidates).issubset(known_capabilities):
        issues.append(
            ValidationIssue(
                code="UNRETRIEVED_PACKET_CAPABILITY",
                message="packet capability candidates must come from context retrieval hits",
                path="capability_candidates",
            )
        )
    return _report(issues)
