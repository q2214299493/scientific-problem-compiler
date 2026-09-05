from __future__ import annotations

import re

from ..models import EvidenceGap, ReportedResult, ScientificContextPacket, SourceClaim
from ..serialization import content_hash


def analyze_gaps(
    context: ScientificContextPacket,
    claims: tuple[SourceClaim, ...],
    results: tuple[ReportedResult, ...],
) -> tuple[EvidenceGap, ...]:
    searchable = " ".join((context.original_request, *(claim.text for claim in claims))).casefold()
    asks_for_barrier = bool(re.search(r"\b(?:barrier|transition state|ts barrier|activation energy)\b", searchable))
    has_barrier = any(result.quantity == "activation_barrier" for result in results)
    gaps: list[EvidenceGap] = []
    if asks_for_barrier and not has_barrier:
        capability_ids = tuple(hit.record_id for hit in context.capability_hits)
        identity = {
            "context_id": context.context_id,
            "missing": "traceable transition-state activation barrier",
        }
        gaps.append(
            EvidenceGap(
                gap_id=f"gap-{content_hash(identity)[:24]}",
                scientific_question=context.original_request,
                missing_evidence="A traceable transition-state activation barrier is not present in the retrieved evidence.",
                why_it_matters="The requested barrier cannot be assessed from source claims alone.",
                blocking=True,
                candidate_capabilities=capability_ids,
                evidence_refs=tuple(hit.record_id for hit in context.evidence_hits),
            )
        )
    return tuple(gaps)
