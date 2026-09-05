from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from ..models import DomainProfile, RetrievalHit, RetrievalQuery, RetrievalSourceType

RETRIEVER_VERSION = "lexical-1.0.0"

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "what",
    "which",
    "with",
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", normalized.replace("_", " ")))


def tokenize(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in normalize_text(value).split()
        if token not in _STOP_WORDS
    )


def ontology_groups(profile: DomainProfile) -> tuple[tuple[str, tuple[str, ...]], ...]:
    combined: dict[str, set[str]] = {}
    for canonical, definition in profile.terminology.items():
        canonical_normalized = normalize_text(canonical)
        combined.setdefault(canonical_normalized, {canonical_normalized}).add(
            normalize_text(definition)
        )
    for mapping in (profile.aliases, profile.synonyms, profile.ontology_relationships):
        for canonical, related in mapping.items():
            canonical_normalized = normalize_text(canonical)
            variants = combined.setdefault(canonical_normalized, {canonical_normalized})
            variants.update(normalize_text(item) for item in related)
    return tuple(
        (canonical, tuple(sorted(variants)))
        for canonical, variants in sorted(combined.items())
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def source_type_allowed(query: RetrievalQuery, source_type: RetrievalSourceType) -> bool:
    return not query.evidence_types or source_type in query.evidence_types


def score_text(
    query: RetrievalQuery,
    record_text: str,
    profile: DomainProfile,
) -> tuple[float, tuple[str, ...], str] | None:
    query_text = normalize_text(query.raw_request)
    candidate_text = normalize_text(record_text)
    if any(_contains_phrase(candidate_text, normalize_text(item)) for item in query.exclusions):
        return None

    exact_phrases: set[str] = set()
    ontology_matches: set[str] = set()
    if _contains_phrase(candidate_text, query_text):
        exact_phrases.add(query_text)
    for canonical, variants in ontology_groups(profile):
        query_variants = {item for item in variants if _contains_phrase(query_text, item)}
        candidate_variants = {item for item in variants if _contains_phrase(candidate_text, item)}
        if not query_variants or not candidate_variants:
            continue
        shared = query_variants & candidate_variants
        if shared:
            exact_phrases.update(shared)
        else:
            ontology_matches.add(canonical)

    overlap = set(tokenize(query.raw_request)) & set(tokenize(record_text))
    score = 50.0 * len(exact_phrases) + 20.0 * len(ontology_matches) + float(len(overlap))
    if score <= 0:
        return None
    matched_terms = tuple(sorted(exact_phrases | ontology_matches | overlap))
    reasons: list[str] = []
    if exact_phrases:
        reasons.append(f"exact phrase: {', '.join(sorted(exact_phrases))}")
    if ontology_matches:
        reasons.append(f"ontology synonym: {', '.join(sorted(ontology_matches))}")
    if overlap:
        reasons.append(f"token overlap: {', '.join(sorted(overlap))}")
    return score, matched_terms, "; ".join(reasons)


def make_hit(
    *,
    query: RetrievalQuery,
    profile: DomainProfile,
    source_type: RetrievalSourceType,
    record_id: str,
    record_text: str,
    evidence_refs: tuple[str, ...] = (),
) -> RetrievalHit | None:
    if not source_type_allowed(query, source_type):
        return None
    scored = score_text(query, record_text, profile)
    if scored is None:
        return None
    score, matched_terms, rationale = scored
    return RetrievalHit(
        hit_id=f"{source_type.value}:{record_id}",
        source_type=source_type,
        record_id=record_id,
        score=score,
        matched_terms=matched_terms,
        rationale=rationale,
        evidence_refs=evidence_refs,
        retriever_version=RETRIEVER_VERSION,
    )


def rank_hits(hits: Iterable[RetrievalHit | None]) -> tuple[RetrievalHit, ...]:
    present = (hit for hit in hits if hit is not None)
    return tuple(sorted(present, key=lambda hit: (-hit.score, hit.record_id, hit.hit_id)))
