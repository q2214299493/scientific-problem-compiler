from __future__ import annotations

import re

from ..models import ScientificContextPacket, SourceClaim
from ..serialization import content_hash
from .claim_classifier import classify_claim


def normalize_claim_text(text: str) -> str:
    normalized = " ".join(text.split())
    normalized = re.sub(
        r"^(?:the authors?\s+)?(?:hypothes(?:ize|izes|ized)|propose(?:s|d)?)\s+that\s+",
        "",
        normalized,
        flags=re.I,
    )
    normalized = re.sub(r"^we\s+(?:hypothesize|propose)\s+that\s+", "", normalized, flags=re.I)
    normalized = re.sub(r"^reviewer\s+(?:asks?|questions?)\s+(?:whether\s+)?", "", normalized, flags=re.I)
    return normalized


def _quote_id(evidence_id: str) -> str:
    return f"quote-{content_hash({'evidence_ref': evidence_id})[:24]}"


def _atomic_statements(text: str) -> tuple[str, ...]:
    statements = tuple(item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip())
    return statements or (text,)


def extract_source_claims(context: ScientificContextPacket) -> tuple[SourceClaim, ...]:
    hit_ids = {hit.record_id for hit in context.evidence_hits}
    claims: list[SourceClaim] = []
    for statement in context.retrieved_statements:
        evidence_refs = tuple(ref for ref in statement.evidence_refs if ref in hit_ids)
        if not evidence_refs:
            continue
        quote_refs = tuple(_quote_id(evidence_id) for evidence_id in evidence_refs)
        for atomic_index, atomic_text in enumerate(_atomic_statements(statement.text), start=1):
            claim_type, source_role, strength, status = classify_claim(atomic_text)
            normalized_text = normalize_claim_text(atomic_text)
            identity = {
                "text": normalized_text,
                "evidence_refs": evidence_refs,
                "atomic_index": atomic_index,
            }
            claims.append(
                SourceClaim(
                    claim_id=f"claim-{content_hash(identity)[:24]}",
                    text=normalized_text,
                    claim_type=claim_type,
                    source_role=source_role,
                    evidence_refs=evidence_refs,
                    source_quote_refs=quote_refs,
                    claim_strength=strength,
                    epistemic_status=status,
                )
            )
    return tuple(sorted(claims, key=lambda claim: claim.claim_id))
