from __future__ import annotations

from ..models import DomainProfile, RetrievalHit, RetrievalQuery, RetrievalSourceType
from ..repositories import SourceEvidenceStore
from ..serialization import load_data
from .ranker import make_hit, rank_hits


class RetrievalIntegrityError(RuntimeError):
    pass


def retrieve_evidence_spans(
    query: RetrievalQuery,
    repository: SourceEvidenceStore,
    profile: DomainProfile,
) -> tuple[RetrievalHit, ...]:
    project = load_data(repository.state_root / "project.yaml")
    project_domain = project.get("domain") if isinstance(project, dict) else None
    if project_domain not in {"unselected", query.domain}:
        return ()

    hits: list[RetrievalHit | None] = []
    for evidence in repository.evidence_records.list():
        try:
            repository.verify_evidence_integrity(evidence)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise RetrievalIntegrityError(
                f"EvidenceSpan integrity failed for {evidence.evidence_id}: {error}"
            ) from error
        hits.append(
            make_hit(
                query=query,
                profile=profile,
                source_type=RetrievalSourceType.EVIDENCE_SPAN,
                record_id=evidence.evidence_id,
                record_text=evidence.text,
                evidence_refs=(evidence.evidence_id,),
            )
        )
    return rank_hits(hits)
