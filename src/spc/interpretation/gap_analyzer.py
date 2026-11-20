from __future__ import annotations

import re
import unicodedata

from ..models import EvidenceGap, ReportedResult, ScientificContextPacket, SourceClaim
from ..serialization import content_hash


def analyze_gaps(
    context: ScientificContextPacket,
    claims: tuple[SourceClaim, ...],
    results: tuple[ReportedResult, ...],
) -> tuple[EvidenceGap, ...]:
    searchable = " ".join((context.original_request, *(claim.text for claim in claims))).casefold()
    asks_for_barrier = (
        "activation barrier" in context.retrieval_query.desired_observables
        or bool(re.search(r"\b(?:barrier|transition state|ts barrier|activation energy)\b", searchable))
    )
    requested_systems = context.retrieval_query.system_terms
    requested_topics = tuple(
        concept
        for concept in context.retrieval_query.concepts
        if " " in concept
        and concept not in (*requested_systems, *context.retrieval_query.desired_observables)
    )

    def system_key(value: str) -> str:
        return "".join(
            character for character in unicodedata.normalize("NFKC", value).casefold()
            if character.isalnum()
        )

    def topic_text(value: str) -> str:
        return " ".join(
            re.findall(
                r"[a-z0-9]+|[\u3400-\u9fff]+",
                unicodedata.normalize("NFKC", value).casefold().replace("_", " "),
            )
        )

    has_barrier = any(
        result.quantity == "activation_barrier"
        and (
            not requested_systems
            or all(
                system_key(system) in system_key(str(dict(result.system_context)))
                for system in requested_systems
            )
        )
        and (
            not requested_topics
            or any(
                set(claim.evidence_refs) & set(result.evidence_refs)
                and any(
                    topic_text(topic) in topic_text(claim.text)
                    for topic in requested_topics
                )
                for claim in claims
            )
        )
        for result in results
    )
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
