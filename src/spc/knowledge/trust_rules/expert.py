"""Trust checks for expert provenance, attribution, opinions and cases."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...models import (
    EvidenceSpan,
    ExpertAttributionRecord,
    ExpertCase,
    ExpertOpinion,
)
from . import (
    RecordKey,
    RememberEvidence,
    TrustedKnowledgeError,
    VerifyEvidence,
    VerifyEvidenceRefs,
    VerifySource,
    VisitRecord,
)

if TYPE_CHECKING:
    from ...repositories import KnowledgeRepositories


def validate_expert_source(
    record: BaseModel,
    *,
    repositories: KnowledgeRepositories,
    visit: VisitRecord,
) -> None:
    from ..expert_knowledge.contracts import ExpertSourceRecord
    from ..expert_knowledge.source import validate_expert_source as validate_provenance

    if not isinstance(record, ExpertSourceRecord):
        raise TrustedKnowledgeError(
            "INVALID_EXPERT_SOURCE",
            "expert source repository contains an incompatible record",
        )
    try:
        validate_provenance(record, repositories)
    except (FileNotFoundError, ValueError) as error:
        raise TrustedKnowledgeError(
            "INVALID_EXPERT_SOURCE",
            f"expert source provenance is invalid: {record.expert_source_id}",
        ) from error
    visit(("expert_profile", record.expert_id))
    attribution = visit(("expert_attribution", record.attribution_id))
    if (
        not isinstance(attribution, ExpertAttributionRecord)
        or attribution.content_hash != record.attribution_hash
    ):
        raise TrustedKnowledgeError(
            "INVALID_EXPERT_SOURCE_ATTRIBUTION",
            f"expert source attribution binding is invalid: {record.expert_source_id}",
        )


def validate_expert_case(
    record: ExpertCase,
    *,
    visit: VisitRecord,
    verify_evidence_refs: VerifyEvidenceRefs,
) -> None:
    verify_evidence_refs(record.evidence_refs)
    if not record.opinion_refs:
        return
    opinion_evidence: set[str] = set()
    for opinion_id in record.opinion_refs:
        opinion = visit(("expert_opinion", opinion_id))
        if not isinstance(opinion, ExpertOpinion):
            raise TrustedKnowledgeError(
                "INVALID_EXPERT_CASE_OPINION",
                f"expert case dependency is not an opinion: {opinion_id}",
            )
        opinion_evidence.update(opinion.evidence_refs)
    if not set(record.evidence_refs).issubset(opinion_evidence):
        raise TrustedKnowledgeError(
            "EXPERT_CASE_EVIDENCE_MISMATCH",
            f"expert case evidence is not derived from its opinions: {record.case_id}",
        )


def validate_attribution(
    attribution: ExpertAttributionRecord,
    *,
    reachable: dict[RecordKey, BaseModel],
    verify_evidence: VerifyEvidence,
    verify_source: VerifySource,
    remember_evidence: RememberEvidence,
) -> None:
    if attribution.attribution_status not in {None, "confirmed"}:
        raise TrustedKnowledgeError(
            "UNRESOLVED_EXPERT_ATTRIBUTION",
            f"expert attribution is not confirmed: {attribution.attribution_id}",
        )
    source = verify_source(attribution.source_id, attribution.source_version)
    for evidence_id in attribution.evidence_refs:
        evidence, evidence_source = verify_evidence(evidence_id)
        if (evidence.source_id, evidence.source_version) != (
            attribution.source_id,
            attribution.source_version,
        ):
            raise TrustedKnowledgeError(
                "EXPERT_ATTRIBUTION_SOURCE_MISMATCH",
                f"attribution evidence comes from another source: {evidence_id}",
            )
        remember_evidence(evidence, evidence_source)
    reachable[("source_document", f"{source.source_id}@{source.version}")] = source


def validate_opinion(
    opinion: ExpertOpinion,
    *,
    visit: VisitRecord,
    verify_evidence_refs: VerifyEvidenceRefs,
    verify_evidence: VerifyEvidence,
) -> None:
    visit(("expert_profile", opinion.expert_id))
    if not opinion.attribution_refs:
        raise TrustedKnowledgeError(
            "MISSING_EXPERT_ATTRIBUTION",
            f"accepted ExpertOpinion has no attribution: {opinion.opinion_id}",
        )
    attributed_evidence: set[str] = set()
    attributed_spans: list[EvidenceSpan] = []
    for attribution_id in opinion.attribution_refs:
        attribution = visit(("expert_attribution", attribution_id))
        if not isinstance(attribution, ExpertAttributionRecord):
            raise TrustedKnowledgeError(
                "INVALID_EXPERT_ATTRIBUTION",
                f"record is not an ExpertAttributionRecord: {attribution_id}",
            )
        if attribution.expert_id != opinion.expert_id:
            raise TrustedKnowledgeError(
                "EXPERT_ATTRIBUTION_OWNER_MISMATCH",
                f"attribution belongs to another expert: {attribution_id}",
            )
        attributed_evidence.update(attribution.evidence_refs)
        attributed_spans.extend(
            verify_evidence(evidence_id)[0]
            for evidence_id in attribution.evidence_refs
        )
    for evidence_id in opinion.evidence_refs:
        evidence = verify_evidence(evidence_id)[0]
        covered = evidence_id in attributed_evidence or any(
            span.source_id == evidence.source_id
            and span.source_version == evidence.source_version
            and span.start_offset <= evidence.start_offset
            and span.end_offset >= evidence.end_offset
            for span in attributed_spans
        )
        if not covered:
            raise TrustedKnowledgeError(
                "EXPERT_OPINION_EVIDENCE_NOT_ATTRIBUTED",
                f"opinion evidence is not covered by its attributions: {opinion.opinion_id}",
            )
    verify_evidence_refs(opinion.evidence_refs)
    for claim_id in opinion.related_claim_refs:
        visit(("source_claim", claim_id))
    for workflow_id in opinion.related_workflow_refs:
        visit(("workflow_pattern", workflow_id))
    if opinion.supersedes is not None:
        visit(("expert_opinion", opinion.supersedes))
