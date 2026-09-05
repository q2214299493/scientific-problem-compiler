from __future__ import annotations

from ..models import (
    EvidenceAssessment,
    EvidenceAssessmentStatus,
    InterpretationProposal,
    ScientificContextPacket,
)
from ..serialization import content_hash
from ..validators import EvidenceSpanRepository
from .claim_extractor import extract_source_claims, extract_source_quotes
from .comparison_analyzer import analyze_comparisons
from .conflict_detector import detect_conflicts
from .gap_analyzer import analyze_gaps
from .method_fact_extractor import extract_method_facts
from .model_fact_extractor import extract_model_facts
from .result_extractor import extract_reported_results

MOCK_PROVIDER_VERSION = "mock-interpretation-1.0.0"


class MockInterpretationProvider:
    provider_id = "mock"
    provider_version = MOCK_PROVIDER_VERSION

    def __init__(self, evidence_repository: EvidenceSpanRepository | None = None) -> None:
        self._evidence_repository = evidence_repository

    def bind_evidence_repository(self, repository: EvidenceSpanRepository) -> None:
        self._evidence_repository = repository

    def interpret(self, context: ScientificContextPacket) -> InterpretationProposal:
        if self._evidence_repository is None:
            raise ValueError("MockInterpretationProvider requires an EvidenceSpan repository")
        source_quotes = extract_source_quotes(context, self._evidence_repository)
        claims = extract_source_claims(context, source_quotes)
        method_facts = extract_method_facts(claims)
        model_facts = extract_model_facts(claims)
        results = extract_reported_results(claims, method_facts, model_facts)
        conflicts = detect_conflicts(claims, context)
        conflict_claims = {claim_id for conflict in conflicts for claim_id in conflict.claim_refs}
        assessments = tuple(
            EvidenceAssessment(
                assessment_id=f"assessment-{content_hash({'claim_ref': claim.claim_id})[:24]}",
                claim_ref=claim.claim_id,
                supporting_evidence_refs=claim.evidence_refs,
                contradicting_evidence_refs=tuple(
                    dict.fromkeys(
                        evidence_ref
                        for conflict in conflicts
                        if claim.claim_id in conflict.claim_refs
                        for other_id in conflict.claim_refs
                        if other_id != claim.claim_id
                        for other in claims
                        if other.claim_id == other_id
                        for evidence_ref in other.evidence_refs
                    )
                ),
                assessment=EvidenceAssessmentStatus.UNRESOLVED,
                limitations=(
                    "Retrieval establishes source provenance, not the scientific truth of the claim.",
                ),
                confidence_basis=(
                    "Traceable source wording with explicit disagreement retained."
                    if claim.claim_id in conflict_claims
                    else "Traceable source wording only; no independent truth assessment was performed."
                ),
            )
            for claim in claims
        )
        payload = {
            "context_id": context.context_id,
            "context_hash": context.content_hash,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "source_quotes": source_quotes,
            "source_claims": claims,
            "reported_results": results,
            "method_facts": method_facts,
            "model_facts": model_facts,
            "evidence_assessments": assessments,
            "conflict_sets": conflicts,
            "comparison_constraints": analyze_comparisons(results),
            "evidence_gaps": analyze_gaps(context, claims, results),
            "unknowns": tuple(item.statement for item in context.unknowns),
            "assumption_candidates": tuple(item.statement for item in context.assumptions),
            "capability_candidates": tuple(hit.record_id for hit in context.capability_hits),
        }
        return InterpretationProposal(
            proposal_id=f"interpretation-{content_hash(payload)[:24]}",
            **payload,
        )
