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
    SourceRole,
    SourceQuote,
    SourceType,
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


def _source_quote_issues(
    packet: ScientificEvidencePacket,
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
) -> tuple[list[ValidationIssue], dict[str, SourceQuote]]:
    issues: list[ValidationIssue] = []
    quotes_by_id = {quote.quote_id: quote for quote in packet.source_quotes}
    source_document_hashes: dict[str, str] = {}
    if len(quotes_by_id) != len(packet.source_quotes):
        issues.append(
            ValidationIssue(
                code="DUPLICATE_SOURCE_QUOTE_ID",
                message="SourceQuote IDs must be unique",
                path="source_quotes",
            )
        )
    for index, quote in enumerate(packet.source_quotes):
        path = f"source_quotes[{index}]"
        if quote.evidence_ref not in _context_evidence_ids(context):
            issues.append(
                ValidationIssue(
                    code="SOURCE_QUOTE_NOT_RETRIEVED",
                    message=f"SourceQuote evidence was not retrieved: {quote.evidence_ref}",
                    path=path,
                )
            )
            continue
        try:
            evidence = evidence_repository.get(quote.evidence_ref)
            source = evidence_repository.verify_evidence_integrity(evidence)
            source_document_hashes[f"{source.source_id}@{source.version}"] = content_hash(source)
        except (FileNotFoundError, KeyError, OSError, ValueError) as error:
            issues.append(
                ValidationIssue(
                    code="SOURCE_QUOTE_INTEGRITY_FAILURE",
                    message=f"cannot verify SourceQuote: {error}",
                    path=path,
                )
            )
            continue
        if quote.relative_end_offset > len(evidence.text):
            issues.append(
                ValidationIssue(
                    code="SOURCE_QUOTE_OUT_OF_BOUNDS",
                    message="SourceQuote offsets exceed its EvidenceSpan",
                    path=path,
                )
            )
        elif evidence.text[
            quote.relative_start_offset : quote.relative_end_offset
        ] != quote.text:
            issues.append(
                ValidationIssue(
                    code="SOURCE_QUOTE_TEXT_MISMATCH",
                    message="SourceQuote offsets must recover its exact text from EvidenceSpan",
                    path=path,
                )
            )
        if (
            quote.source_id,
            quote.source_version,
            quote.source_role,
            quote.source_type,
        ) != (
            source.source_id,
            source.version,
            source.source_role,
            source.source_type,
        ):
            issues.append(
                ValidationIssue(
                    code="SOURCE_QUOTE_PROVENANCE_MISMATCH",
                    message="SourceQuote provenance does not match SourceDocument",
                    path=path,
                )
            )
    if packet.provenance_manifest.get("source_document_hashes") != source_document_hashes:
        issues.append(
            ValidationIssue(
                code="SOURCE_DOCUMENT_PROVENANCE_MISMATCH",
                message="packet provenance does not bind the referenced SourceDocument records",
                path="provenance_manifest.source_document_hashes",
            )
        )
    return issues, quotes_by_id


def validate_source_quotes(
    packet: ScientificEvidencePacket,
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
) -> ValidationReport:
    issues, _ = _source_quote_issues(packet, context, evidence_repository)
    return _report(issues)


def validate_claim_source_binding(
    packet: ScientificEvidencePacket,
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
) -> ValidationReport:
    issues, quotes_by_id = _source_quote_issues(packet, context, evidence_repository)
    for index, claim in enumerate(packet.source_claims):
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
        referenced_quotes = [
            quotes_by_id[quote_id]
            for quote_id in claim.source_quote_refs
            if quote_id in quotes_by_id
        ]
        if len(referenced_quotes) != len(claim.source_quote_refs) or not referenced_quotes:
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_SOURCE_QUOTE_REF",
                    message="SourceClaim must reference existing SourceQuote records",
                    path=f"source_claims[{index}]",
                )
            )
        elif set(claim.evidence_refs) != {quote.evidence_ref for quote in referenced_quotes}:
            issues.append(
                ValidationIssue(
                    code="CLAIM_QUOTE_EVIDENCE_MISMATCH",
                    message="SourceClaim evidence_refs must match its SourceQuote evidence",
                    path=f"source_claims[{index}]",
                )
            )
        explicit_roles = {
            quote.source_role
            for quote in referenced_quotes
            if quote.source_role != SourceRole.UNSPECIFIED
        }
        if len(explicit_roles) == 1 and claim.source_role not in explicit_roles:
            issues.append(
                ValidationIssue(
                    code="CLAIM_SOURCE_ROLE_MISMATCH",
                    message="SourceClaim role does not match SourceDocument provenance",
                    path=f"source_claims[{index}]",
                )
            )
        source_types = {quote.source_type for quote in referenced_quotes}
        if (
            SourceRole.REVIEWER in explicit_roles
            or SourceType.REVIEWER_COMMENT in source_types
        ) and claim.epistemic_status != EpistemicStatus.UNRESOLVED:
            issues.append(
                ValidationIssue(
                    code="REVIEWER_CLAIM_PROMOTED",
                    message="reviewer provenance must remain epistemically unresolved",
                    path=f"source_claims[{index}]",
                )
            )
        if SourceType.AUTHOR_RESPONSE in source_types and claim.source_role != SourceRole.AUTHOR:
            issues.append(
                ValidationIssue(
                    code="AUTHOR_RESPONSE_ROLE_MISMATCH",
                    message="author_response provenance requires author source role",
                    path=f"source_claims[{index}]",
                )
            )
        if SourceType.LITERATURE_ARTICLE in source_types and claim.source_role == SourceRole.REVIEWER:
            issues.append(
                ValidationIssue(
                    code="LITERATURE_ROLE_MISCLASSIFIED",
                    message="literature_article punctuation cannot imply reviewer provenance",
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


def validate_result_context(packet: ScientificEvidencePacket) -> ValidationReport:
    issues: list[ValidationIssue] = []
    method_facts = {fact.fact_id: fact for fact in packet.method_facts}
    model_facts = {fact.fact_id: fact for fact in packet.model_facts}
    for index, result in enumerate(packet.reported_results):
        path = f"reported_results[{index}].result_context"
        result_context = result.result_context
        if result_context is None:
            issues.append(
                ValidationIssue(
                    code="MISSING_RESULT_CONTEXT",
                    message="reported results require a ResultContext",
                    path=path,
                )
            )
            continue
        if (
            result_context.system_context != result.system_context
            or result_context.method_context != result.method_context
        ):
            issues.append(
                ValidationIssue(
                    code="RESULT_CONTEXT_MISMATCH",
                    message="ResultContext must preserve the result system and method context",
                    path=path,
                )
            )
        if not set(result_context.method_fact_refs).issubset(method_facts):
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_RESULT_METHOD_FACT",
                    message="ResultContext references an unknown MethodFact",
                    path=path,
                )
            )
        if not set(result_context.model_fact_refs).issubset(model_facts):
            issues.append(
                ValidationIssue(
                    code="UNKNOWN_RESULT_MODEL_FACT",
                    message="ResultContext references an unknown ModelFact",
                    path=path,
                )
            )
        applicable_methods = {
            fact.fact_id
            for fact in method_facts.values()
            if set(fact.evidence_refs) & set(result.evidence_refs)
        }
        applicable_models = {
            fact.fact_id
            for fact in model_facts.values()
            if set(fact.evidence_refs) & set(result.evidence_refs)
        }
        if not applicable_methods.issubset(set(result_context.method_fact_refs)):
            issues.append(
                ValidationIssue(
                    code="MISSING_RESULT_METHOD_FACT_REF",
                    message="ResultContext omits an applicable MethodFact",
                    path=path,
                )
            )
        if not applicable_models.issubset(set(result_context.model_fact_refs)):
            issues.append(
                ValidationIssue(
                    code="MISSING_RESULT_MODEL_FACT_REF",
                    message="ResultContext omits an applicable ModelFact",
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
    issues.extend(validate_result_context(packet).issues)
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
