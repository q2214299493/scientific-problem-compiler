from __future__ import annotations

from ..models import ScientificContextPacket, SourceClaim
from ..serialization import content_hash
from .claim_classifier import classify_claim


def extract_source_claims(context: ScientificContextPacket) -> tuple[SourceClaim, ...]:
    hit_ids = {hit.record_id for hit in context.evidence_hits}
    claims: list[SourceClaim] = []
    for statement in context.known_facts:
        evidence_refs = tuple(ref for ref in statement.evidence_refs if ref in hit_ids)
        if not evidence_refs:
            continue
        claim_type, source_role, strength, status = classify_claim(statement.text)
        identity = {"text": statement.text, "evidence_refs": evidence_refs}
        claims.append(
            SourceClaim(
                claim_id=f"claim-{content_hash(identity)[:24]}",
                text=statement.text,
                claim_type=claim_type,
                source_role=source_role,
                evidence_refs=evidence_refs,
                claim_strength=strength,
                epistemic_status=status,
            )
        )
    return tuple(sorted(claims, key=lambda claim: claim.claim_id))
