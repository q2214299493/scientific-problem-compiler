from __future__ import annotations

import re

from ..models import ScientificContextPacket, SourceClaim, SourceQuote
from ..serialization import content_hash
from ..validators import EvidenceSpanRepository
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


def source_quote_id(evidence_id: str, start: int, end: int, text: str) -> str:
    identity = {
        "evidence_ref": evidence_id,
        "relative_start_offset": start,
        "relative_end_offset": end,
        "text_hash": content_hash({"text": text}),
    }
    return f"quote-{content_hash(identity)[:24]}"


def _atomic_segments(text: str) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (match.start(), match.end(), match.group())
        for match in re.finditer(r"\S.*?(?:[.!?](?=\s|\Z)|\Z)", text, flags=re.S)
        if match.group()
    )


def extract_source_quotes(
    context: ScientificContextPacket,
    evidence_repository: EvidenceSpanRepository,
) -> tuple[SourceQuote, ...]:
    quotes: list[SourceQuote] = []
    for hit in context.evidence_hits:
        evidence = evidence_repository.get(hit.record_id)
        source = evidence_repository.verify_evidence_integrity(evidence)
        for start, end, text in _atomic_segments(evidence.text):
            quotes.append(
                SourceQuote(
                    quote_id=source_quote_id(evidence.evidence_id, start, end, text),
                    evidence_ref=evidence.evidence_id,
                    relative_start_offset=start,
                    relative_end_offset=end,
                    text=text,
                    source_id=source.source_id,
                    source_version=source.version,
                    source_role=source.source_role,
                    source_type=source.source_type,
                )
            )
    return tuple(quotes)


def extract_source_claims(
    context: ScientificContextPacket,
    source_quotes: tuple[SourceQuote, ...],
) -> tuple[SourceClaim, ...]:
    hit_ids = {hit.record_id for hit in context.evidence_hits}
    claims: list[SourceClaim] = []
    for quote in source_quotes:
        if quote.evidence_ref not in hit_ids:
            continue
        claim_type, source_role, strength, status = classify_claim(
            quote.text,
            quote.source_role,
            quote.source_type,
        )
        normalized_text = normalize_claim_text(quote.text)
        identity = {"text": normalized_text, "source_quote_ref": quote.quote_id}
        claims.append(
            SourceClaim(
                claim_id=f"claim-{content_hash(identity)[:24]}",
                text=normalized_text,
                claim_type=claim_type,
                source_role=source_role,
                evidence_refs=(quote.evidence_ref,),
                source_quote_refs=(quote.quote_id,),
                claim_strength=strength,
                epistemic_status=status,
            )
        )
    return tuple(sorted(claims, key=lambda claim: claim.claim_id))
