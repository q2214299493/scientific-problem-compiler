from __future__ import annotations

import re

from ..immutable import FrozenDict
from ..models import EpistemicStatus, MethodFact, SourceClaim
from ..serialization import content_hash

_METHOD_TERMS = ("DFT", "PBE", "VASP", "NEB", "experiment", "calculation")


def extract_method_facts(claims: tuple[SourceClaim, ...]) -> tuple[MethodFact, ...]:
    facts: list[MethodFact] = []
    for claim in claims:
        matched = tuple(term for term in _METHOD_TERMS if re.search(rf"\b{re.escape(term)}\b", claim.text, re.I))
        if not matched:
            continue
        identity = {"claim_id": claim.claim_id, "terms": matched}
        facts.append(
            MethodFact(
                fact_id=f"method-fact-{content_hash(identity)[:24]}",
                text=claim.text,
                attributes=FrozenDict({"reported_method_terms": matched}),
                evidence_refs=claim.evidence_refs,
                epistemic_status=EpistemicStatus.METHOD_STATEMENT,
            )
        )
    return tuple(facts)
