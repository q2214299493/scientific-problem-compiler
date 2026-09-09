from __future__ import annotations

from ..models import DomainProfile, RetrievalHit, RetrievalQuery, RetrievalSourceType
from ..repositories import EvidenceStore
from ..serialization import load_data
from .ranker import make_hit, rank_hits


class RetrievalIntegrityError(RuntimeError):
    pass


def retrieve_evidence_spans(
    query: RetrievalQuery,
    repository: EvidenceStore,
    profile: DomainProfile,
) -> tuple[RetrievalHit, ...]:
    if repository.project_state_root is not None:
        project = load_data(repository.project_state_root / "project.yaml")
        project_domain = project.get("domain") if isinstance(project, dict) else None
        if project_domain not in {"unselected", query.domain}:
            return ()

    hits: list[RetrievalHit | None] = []
    try:
        evidence_records = repository.list_evidence()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RetrievalIntegrityError(f"Evidence store integrity failed: {error}") from error
    for evidence in evidence_records:
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
