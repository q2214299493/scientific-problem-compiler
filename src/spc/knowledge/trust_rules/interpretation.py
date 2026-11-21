"""Trust checks for relations, quotes, claims and reported results."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
)

from pydantic import BaseModel

from ...interpretation.validators import (
    _source_quote_binding_issues,
    result_context_record_issues,
    source_claim_binding_issues,
)
from ...models import (
    CurationStatus,
    KnowledgeRelation,
    MethodFact,
    ModelFact,
    ReportedResult,
    SourceClaim,
    SourceQuote,
)
from . import (
    CURATION_REQUIRED_TYPES,
    REPRESENTATION_COMPONENT_TYPES,
    Curations,
    RecordKey,
    RememberEvidence,
    RequireRecord,
    TrustedKnowledgeError,
    VerifyEvidence,
    VerifyEvidenceRefs,
    VisitRecord,
)

if TYPE_CHECKING:
    from ...repositories import EvidenceStore


def raise_issues(issues: list[Any]) -> None:
    if issues:
        issue = issues[0]
        raise TrustedKnowledgeError(issue.code, issue.message)


def validate_relation(
    relation: KnowledgeRelation,
    *,
    visit: VisitRecord,
    require_record: RequireRecord,
    verify_evidence_refs: VerifyEvidenceRefs,
    current: Curations,
) -> None:
    verify_evidence_refs(relation.evidence_refs)
    for key in (
        (relation.subject_type, relation.subject_id),
        (relation.object_type, relation.object_id),
    ):
        if key[0] in REPRESENTATION_COMPONENT_TYPES:
            raise TrustedKnowledgeError(
                "UNTRUSTED_RELATION_ENDPOINT",
                "relations must reference the common literature representation, "
                f"not an internal component: {key[0]}:{key[1]}",
            )
        require_record(*key)
        curation = current.get(key)
        if (key[0] in CURATION_REQUIRED_TYPES or curation is not None) and (
            curation is None or curation.status != CurationStatus.ACCEPTED
        ):
            raise TrustedKnowledgeError(
                "UNTRUSTED_RELATION_ENDPOINT",
                f"accepted relation points to untrusted target {key[0]}:{key[1]}",
            )
        visit(key)


def validate_source_quote(
    quote: SourceQuote,
    *,
    evidence_store: EvidenceStore | None,
    verify_evidence: VerifyEvidence,
    remember_evidence: RememberEvidence,
) -> None:
    if evidence_store is None:
        raise TrustedKnowledgeError(
            "EVIDENCE_STORE_REQUIRED",
            "trusted knowledge validation requires an EvidenceStore",
        )
    evidence, source = verify_evidence(quote.evidence_ref)
    issues = _source_quote_binding_issues(
        quote, evidence, source, path=f"source_quote:{quote.quote_id}",
    )
    raise_issues(issues)
    remember_evidence(evidence, source)


def validate_source_claim(
    claim: SourceClaim,
    *,
    records: Mapping[RecordKey, BaseModel],
    visit: VisitRecord,
    verify_evidence_refs: VerifyEvidenceRefs,
) -> None:
    quotes_by_id = {
        quote_id: quote
        for quote_id in claim.source_quote_refs
        if isinstance(
            (quote := records.get(("source_quote", quote_id))), SourceQuote
        )
    }
    raise_issues(
        source_claim_binding_issues(
            claim,
            quotes_by_id,
            path=f"source_claim:{claim.claim_id}",
        )
    )
    verify_evidence_refs(claim.evidence_refs)
    for quote_id in claim.source_quote_refs:
        visit(("source_quote", quote_id))


def validate_reported_result(
    result: ReportedResult,
    *,
    records: Mapping[RecordKey, BaseModel],
    visit: VisitRecord,
    verify_evidence_refs: VerifyEvidenceRefs,
) -> None:
    verify_evidence_refs(result.evidence_refs)
    result_context = result.result_context
    if result_context is None:
        return
    method_facts = {
        fact_id: fact
        for fact_id in result_context.method_fact_refs
        if isinstance((fact := records.get(("method_fact", fact_id))), MethodFact)
    }
    model_facts = {
        fact_id: fact
        for fact_id in result_context.model_fact_refs
        if isinstance((fact := records.get(("model_fact", fact_id))), ModelFact)
    }
    raise_issues(
        result_context_record_issues(
            result,
            method_facts,
            model_facts,
            path=f"reported_result:{result.result_id}.result_context",
            require_context=False,
        )
    )
    for fact_id in result_context.method_fact_refs:
        visit(("method_fact", fact_id))
    for fact_id in result_context.model_fact_refs:
        visit(("model_fact", fact_id))
