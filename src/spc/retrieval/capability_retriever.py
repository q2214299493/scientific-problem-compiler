from __future__ import annotations

from ..models import DomainProfile, RetrievalHit, RetrievalQuery, RetrievalSourceType
from ..repositories import ScientificCapabilityRepository
from .ranker import make_hit, rank_hits


def retrieve_capabilities(
    query: RetrievalQuery,
    repository: ScientificCapabilityRepository,
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
                source_type=RetrievalSourceType.SCIENTIFIC_CAPABILITY,
                record_id=record.capability_id,
                record_text=" ".join(
                    (
                        record.scientific_goal,
                        *record.required_inputs,
                        *record.outputs,
                        *record.dag_expansion,
                        *record.limitations,
                        *record.failure_branches,
                    )
                ),
            )
        )
    return rank_hits(hits)
