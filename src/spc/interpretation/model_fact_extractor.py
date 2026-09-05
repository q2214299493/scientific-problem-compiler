from __future__ import annotations

import re

from ..immutable import FrozenDict
from ..models import EpistemicStatus, ModelFact, SourceClaim
from ..serialization import content_hash

_MODEL_TERMS = ("GAME", "BEP", "model", "descriptor", "scaling relation")


def extract_model_facts(claims: tuple[SourceClaim, ...]) -> tuple[ModelFact, ...]:
    facts: list[ModelFact] = []
    for claim in claims:
        matched = tuple(term for term in _MODEL_TERMS if re.search(rf"\b{re.escape(term)}\b", claim.text, re.I))
        if not matched:
            continue
        identity = {"claim_id": claim.claim_id, "terms": matched}
        facts.append(
            ModelFact(
                fact_id=f"model-fact-{content_hash(identity)[:24]}",
                text=claim.text,
                attributes=FrozenDict({"reported_model_terms": matched}),
                evidence_refs=claim.evidence_refs,
                epistemic_status=EpistemicStatus.MODEL_STATEMENT,
            )
        )
    return tuple(facts)
