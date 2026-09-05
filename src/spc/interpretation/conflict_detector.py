from __future__ import annotations

import re
from itertools import combinations

from ..models import ConflictSet, ScientificContextPacket, SourceClaim
from ..serialization import content_hash

_NEGATION = re.compile(r"\b(?:not|no|never|unlikely|cannot|contradict(?:s|ed)?)\b", re.I)
_TOKENS = re.compile(r"[a-z0-9]+")
_STOP = {"a", "an", "and", "are", "by", "for", "in", "is", "of", "on", "the", "to", "was", "with"}
_MECHANISM_MARKERS = (
    "associative",
    "carbide",
    "direct",
    "dissociative",
    "hydrogen assisted",
    "indirect",
    "insertion",
)


def _scientific_tokens(text: str) -> set[str]:
    return {token for token in _TOKENS.findall(text.casefold()) if len(token) > 2 and token not in _STOP}


def detect_conflicts(
    claims: tuple[SourceClaim, ...], context: ScientificContextPacket
) -> tuple[ConflictSet, ...]:
    conflicts: list[ConflictSet] = []
    seen_pairs: set[tuple[str, str]] = set()
    for left, right in combinations(claims, 2):
        shared = _scientific_tokens(left.text) & _scientific_tokens(right.text)
        negation_differs = bool(_NEGATION.search(left.text)) != bool(_NEGATION.search(right.text))
        explicit_conflict = "contradict" in left.text.casefold() or "contradict" in right.text.casefold()
        left_markers = {marker for marker in _MECHANISM_MARKERS if marker in left.text.casefold()}
        right_markers = {marker for marker in _MECHANISM_MARKERS if marker in right.text.casefold()}
        mechanism_disagreement = (
            "mechanism" in shared
            and bool(left_markers)
            and bool(right_markers)
            and left_markers.isdisjoint(right_markers)
        )
        if not (
            (len(shared) >= 2 and negation_differs)
            or explicit_conflict
            or mechanism_disagreement
        ):
            continue
        pair = tuple(sorted((left.claim_id, right.claim_id)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        identity = {"claims": pair, "topic": sorted(shared)}
        conflicts.append(
            ConflictSet(
                conflict_id=f"conflict-{content_hash(identity)[:24]}",
                topic=", ".join(sorted(shared)[:6]),
                claim_refs=pair,
                conflict_type="source_claim_disagreement",
                possible_causes=("different source assumptions, systems, methods, or interpretations",),
                required_discrimination=("compare the underlying evidence under compatible conditions",),
                resolution_status="unresolved",
            )
        )
    for index, topic in enumerate(context.conflicting_evidence, start=1):
        matching = tuple(claim.claim_id for claim in claims if _scientific_tokens(topic) & _scientific_tokens(claim.text))
        if len(matching) < 2:
            continue
        identity = {"context_conflict": topic, "claims": matching, "index": index}
        conflicts.append(
            ConflictSet(
                conflict_id=f"conflict-{content_hash(identity)[:24]}",
                topic=topic,
                claim_refs=matching,
                conflict_type="retrieval_reported_conflict",
                possible_causes=("source disagreement retained by retrieval",),
                required_discrimination=("review the conflicting source evidence",),
                resolution_status="unresolved",
            )
        )
    return tuple(sorted(conflicts, key=lambda item: item.conflict_id))
