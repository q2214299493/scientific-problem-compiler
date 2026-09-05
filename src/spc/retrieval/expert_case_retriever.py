from __future__ import annotations

from ..models import DomainProfile, RetrievalHit, RetrievalQuery, RetrievalSourceType
from ..repositories import ExpertCaseRepository
from .ranker import make_hit, rank_hits


def retrieve_expert_cases(
    query: RetrievalQuery,
    repository: ExpertCaseRepository,
    profile: DomainProfile,
) -> tuple[RetrievalHit, ...]:
    hits: list[RetrievalHit | None] = []
    for record in repository.list():
        if record.domain != query.domain:
            continue
        hits.append(
            make_hit(
                query=query,
                profile=profile,
                source_type=RetrievalSourceType.EXPERT_CASE,
                record_id=record.case_id,
                record_text=" ".join(
                    (record.vague_request, *record.translated_questions, record.rationale)
                ),
                evidence_refs=record.evidence_refs,
            )
        )
    return rank_hits(hits)
