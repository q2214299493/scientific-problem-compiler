from __future__ import annotations

from ..models import DomainProfile, RetrievalHit, RetrievalQuery, RetrievalSourceType
from ..repositories import LiteratureWorkflowRepository
from .ranker import make_hit, rank_hits


def retrieve_workflow_patterns(
    query: RetrievalQuery,
    repository: LiteratureWorkflowRepository,
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
                source_type=RetrievalSourceType.WORKFLOW_PATTERN,
                record_id=record.pattern_id,
                record_text=" ".join(
                    (record.trigger, *record.workflow_capabilities, *record.limitations)
                ),
                evidence_refs=record.evidence_refs,
            )
        )
    return rank_hits(hits)
