from __future__ import annotations

from ..models import DomainProfile, RetrievalQuery, RetrievalSourceType
from ..serialization import content_hash
from .ranker import normalize_text, ontology_groups, tokenize


def _matching_ontology_terms(
    raw_request: str,
    profile: DomainProfile,
    category_names: set[str],
) -> tuple[str, ...]:
    request_text = normalize_text(raw_request)
    groups = dict(ontology_groups(profile))
    matches: set[str] = set()
    for category, values in profile.ontology.items():
        if normalize_text(category) not in category_names:
            continue
        for value in values:
            canonical = normalize_text(value)
            variants = groups.get(canonical, (canonical,))
            if any(f" {variant} " in f" {request_text} " for variant in variants):
                matches.add(canonical)
    return tuple(sorted(matches))


def build_retrieval_query(
    raw_request: str,
    domain: str,
    profile: DomainProfile,
    *,
    evidence_types: tuple[RetrievalSourceType, ...] = (),
    exclusions: tuple[str, ...] = (),
) -> RetrievalQuery:
    if profile.domain_id != domain:
        raise ValueError("retrieval domain and Domain Pack differ")
    request_text = normalize_text(raw_request)
    recognized = {
        canonical
        for canonical, variants in ontology_groups(profile)
        if any(f" {variant} " in f" {request_text} " for variant in variants)
    }
    concepts = tuple(sorted(recognized | set(tokenize(raw_request))))
    fields = {
        "raw_request": raw_request,
        "domain": domain,
        "concepts": concepts,
        "system_terms": _matching_ontology_terms(
            raw_request, profile, {"entities", "systems", "system terms"}
        ),
        "method_terms": _matching_ontology_terms(
            raw_request, profile, {"methods", "method terms"}
        ),
        "desired_observables": _matching_ontology_terms(
            raw_request, profile, {"observables", "desired observables"}
        ),
        "evidence_types": evidence_types,
        "exclusions": exclusions,
    }
    return RetrievalQuery(
        query_id=f"query-{content_hash(fields)[:24]}",
        **fields,
    )
