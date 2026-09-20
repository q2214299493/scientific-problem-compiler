from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from ..models import (
    ApprovalDecision,
    ApprovalReviewRecord,
    EvidenceAcquisitionProposal,
    EvidenceClassification,
    EvidenceGap,
    EvidenceGapResolutionKind,
    EvidenceResolutionRecord,
    EvidenceResolutionSet,
    EvidenceResolutionStatus,
    KnowledgeSnapshot,
    KnowledgeViewMode,
    PlanningEvidenceGapClassification,
    PlanningEvidenceRequest,
    PlanningEvidenceRequestLLMResponse,
    PlanningEvidenceRequestProposal,
    PlanningEvidenceRequestSet,
    PlanningEvidenceType,
    PlanningStrategy,
    GroundedStatement,
    ResearchDirectionSet,
    RetrievalHit,
    RetrievalManifest,
    RetrievalSourceType,
    ScientificPlanningInput,
    ScientificContextPacket,
    scientific_context_semantic_hash,
)
from ..repositories import EvidenceStore, KnowledgeRepositories
from ..retrieval.knowledge_retriever import PersistentKnowledgeRetriever
from ..retrieval.query_builder import build_retrieval_query
from ..serialization import content_hash

EVIDENCE_REQUEST_POLICY_VERSION = "planning-evidence-request-1.0.0"

_ACCEPTABLE_SOURCE_CLASSES = frozenset(
    {
        "literature_reported",
        "expert_opinion",
        "expert_case",
        "supporting_context",
    }
)
_CALCULATION_TERMS = frozenset(
    {
        "calculate",
        "new calculation",
        "run dft",
        "simulation",
        "true barrier",
        "actual barrier",
        "this system",
    }
)
_EXPERIMENT_TERMS = frozenset(
    {"new experiment", "measure experimentally", "experimental campaign"}
)
_HUMAN_TERMS = frozenset(
    {
        "user choice",
        "user must choose",
        "human decision",
        "choose whether",
        "scope decision",
        "expert decision",
    }
)
_RETRIEVAL_TERMS = frozenset(
    {
        "reported",
        "source",
        "literature",
        "paper",
        "method",
        "model",
        "condition",
        "context",
        "slab thickness",
        "prior evidence",
        "citation",
    }
)


class PlanningEvidenceError(ValueError):
    pass


def _provider_config(provider: Any) -> dict[str, object]:
    config = dict(getattr(provider, "provider_config", {}))
    model_id = getattr(getattr(provider, "transport", None), "model_id", None)
    if model_id is not None:
        config["model_id"] = model_id
    for field_name in ("temperature", "max_attempts"):
        value = getattr(provider, field_name, None)
        if value is not None:
            config[field_name] = value
    return config or {"mode": "deterministic", "network": False}


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def classify_evidence_gap(gap: EvidenceGap) -> PlanningEvidenceGapClassification:
    text = " ".join(
        (gap.scientific_question, gap.missing_evidence, gap.why_it_matters)
    ).casefold()
    if any(term in text for term in _HUMAN_TERMS):
        kind = EvidenceGapResolutionKind.HUMAN_DECISION_REQUIRED
        rationale = "The gap changes user or expert scope and cannot be closed by retrieval."
    elif any(term in text for term in _EXPERIMENT_TERMS):
        kind = EvidenceGapResolutionKind.EXPERIMENT_REQUIRED
        rationale = "The missing information requires a new experiment, not source retrieval."
    elif any(term in text for term in _CALCULATION_TERMS):
        kind = EvidenceGapResolutionKind.CALCULATION_REQUIRED
        rationale = "The missing information requires a new scientific calculation."
    elif any(term in text for term in _RETRIEVAL_TERMS):
        kind = EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE
        rationale = "The gap asks for reported source, method, model, or comparison context."
    else:
        kind = EvidenceGapResolutionKind.UNDETERMINED
        rationale = "The current trusted context does not establish a safe resolution route."
    return PlanningEvidenceGapClassification(
        gap_id=gap.gap_id,
        resolution_kind=kind,
        rationale=rationale,
    )


def proposal_for_gap(
    gap: EvidenceGap,
    planning_input: ScientificPlanningInput,
) -> PlanningEvidenceRequestProposal:
    evidence_type = PlanningEvidenceType.COMPARISON_CONTEXT
    text = f"{gap.scientific_question} {gap.missing_evidence}".casefold()
    for marker, candidate in (
        ("method", PlanningEvidenceType.METHOD_FACT),
        ("model", PlanningEvidenceType.MODEL_FACT),
        ("result", PlanningEvidenceType.REPORTED_RESULT),
        ("condition", PlanningEvidenceType.EXPERIMENTAL_CONDITION),
        ("mechan", PlanningEvidenceType.PRIOR_MECHANISTIC_EVIDENCE),
        ("claim", PlanningEvidenceType.SOURCE_CLAIM),
    ):
        if marker in text:
            evidence_type = candidate
            break
    related_claims = tuple(
        claim.claim_id
        for claim in planning_input.source_claims
        if set(claim.evidence_refs).intersection(gap.evidence_refs)
    )
    return PlanningEvidenceRequestProposal(
        request_key=f"gap-{gap.gap_id}",
        scientific_question=gap.scientific_question,
        gap_id=gap.gap_id,
        triggering_question=None,
        why_needed=(
            f"Resolving this evidence gap would determine whether the planning comparison "
            f"can proceed: {gap.why_it_matters}"
        ),
        evidence_type_needed=evidence_type,
        target_entities=(),
        concepts=tuple(
            dict.fromkeys(
                token
                for token in re.findall(
                    r"[A-Za-z0-9][A-Za-z0-9_-]+", gap.scientific_question
                )
                if len(token) >= 3
            )
        )[:12],
        comparison_conditions=(gap.missing_evidence,),
        acceptable_source_classes=("literature_reported",),
        evidence_that_would_resolve_gap=(
            "A trusted, exactly grounded record reporting the missing condition or context."
        ),
        evidence_that_would_not_resolve_gap=(
            "An uncurated snippet, unsupported interpretation, or relevance score alone."
        ),
        related_claim_refs=related_claims,
        related_evidence_refs=gap.evidence_refs,
        related_direction_refs=(),
        related_candidate_refs=(),
        priority_reason="The originating EvidenceGap is marked blocking.",
        blocking=gap.blocking,
    )


def proposal_for_review(
    review: ApprovalReviewRecord,
    planning_input: ScientificPlanningInput,
) -> PlanningEvidenceRequestProposal:
    del planning_input
    response = getattr(review.response, "review", review.response)
    if response.decision_recommendation != ApprovalDecision.INSUFFICIENT_EVIDENCE:
        raise PlanningEvidenceError(
            "only an INSUFFICIENT_EVIDENCE review can trigger review-directed retrieval"
        )
    descriptions = tuple(item.description for item in response.required_fixes)
    red_flags = tuple(item.description for item in response.hard_red_flags)
    trigger = " ".join((response.summary, *descriptions, *red_flags)).strip()
    synthetic_gap = EvidenceGap(
        gap_id=f"review-gap-{review.review_id}",
        scientific_question=trigger,
        missing_evidence=trigger,
        why_it_matters="Independent review identified a source-grounding deficiency.",
        blocking=True,
    )
    classification = classify_evidence_gap(synthetic_gap)
    if classification.resolution_kind != EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE:
        raise PlanningEvidenceError(
            f"review evidence deficiency is {classification.resolution_kind.value}"
        )
    evidence_type = PlanningEvidenceType.COMPARISON_CONTEXT
    lowered = trigger.casefold()
    for marker, candidate in (
        ("method", PlanningEvidenceType.METHOD_FACT),
        ("model", PlanningEvidenceType.MODEL_FACT),
        ("result", PlanningEvidenceType.REPORTED_RESULT),
        ("condition", PlanningEvidenceType.EXPERIMENTAL_CONDITION),
        ("mechan", PlanningEvidenceType.PRIOR_MECHANISTIC_EVIDENCE),
        ("claim", PlanningEvidenceType.SOURCE_CLAIM),
    ):
        if marker in lowered:
            evidence_type = candidate
            break
    return PlanningEvidenceRequestProposal(
        request_key=f"review-{review.review_id}",
        scientific_question=trigger,
        gap_id=None,
        triggering_question=trigger,
        why_needed=(
            "The missing trusted source grounding prevents independent approval of the "
            "current planning judgment."
        ),
        evidence_type_needed=evidence_type,
        target_entities=(),
        concepts=tuple(
            dict.fromkeys(
                token
                for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", trigger)
                if len(token) >= 3
            )
        )[:12],
        comparison_conditions=descriptions or red_flags or (response.summary,),
        acceptable_source_classes=("literature_reported",),
        evidence_that_would_resolve_gap=(
            "A trusted exact source record addressing the review's stated grounding gap."
        ),
        evidence_that_would_not_resolve_gap=(
            "A relevance score, uncurated source, new calculation, or planner assertion."
        ),
        related_claim_refs=(),
        related_evidence_refs=(),
        related_direction_refs=(),
        related_candidate_refs=(),
        priority_reason="The independent review decision is INSUFFICIENT_EVIDENCE.",
        blocking=True,
    )
def _request_signature(proposal: PlanningEvidenceRequestProposal) -> tuple[object, ...]:
    return (
        proposal.gap_id,
        proposal.evidence_type_needed.value,
        tuple(sorted(_normalized(item) for item in proposal.target_entities)),
        tuple(sorted(_normalized(item) for item in proposal.concepts)),
        tuple(sorted(_normalized(item) for item in proposal.comparison_conditions)),
        tuple(sorted(proposal.acceptable_source_classes)),
    )


def _materialize_request(
    proposal: PlanningEvidenceRequestProposal,
) -> PlanningEvidenceRequest:
    identity = proposal.model_dump(mode="json")
    request_id = f"planning-evidence-request-{content_hash(identity)[:24]}"
    return PlanningEvidenceRequest(
        request_id=request_id,
        **identity,
        content_hash=content_hash({"request_id": request_id, **identity}),
    )


def build_planning_evidence_request_set(
    planning_input: ScientificPlanningInput,
    response: PlanningEvidenceRequestLLMResponse | None,
    provider: Any,
    *,
    planning_strategy: PlanningStrategy | str,
    knowledge_snapshot_hash: str,
    source_acquisition_allowed: bool = False,
    direction_set: ResearchDirectionSet | None = None,
    triage_id: str | None = None,
    triage_hash: str | None = None,
    triggering_review_id: str | None = None,
    triggering_review_hash: str | None = None,
) -> PlanningEvidenceRequestSet:
    classifications = tuple(
        classify_evidence_gap(gap)
        for gap in planning_input.evidence_gaps
        if gap.blocking
    )
    proposals = response.requests if response is not None else ()
    gaps = {item.gap_id: item for item in planning_input.evidence_gaps}
    allowed_claims = set(planning_input.allowed_claim_ids)
    allowed_evidence = set(planning_input.allowed_evidence_ids)
    allowed_directions = (
        {item.direction_id for item in direction_set.directions}
        | {item.direction_key for item in direction_set.directions}
        if direction_set is not None
        else set()
    )
    seen: set[tuple[object, ...]] = set()
    requests: list[PlanningEvidenceRequest] = []
    for proposal in proposals:
        if proposal.gap_id is not None:
            gap = gaps.get(proposal.gap_id)
            if gap is None:
                raise PlanningEvidenceError("FABRICATED_EVIDENCE_GAP_ID")
            classification = next(
                item for item in classifications if item.gap_id == proposal.gap_id
            )
            if classification.resolution_kind != EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE:
                raise PlanningEvidenceError("EVIDENCE_GAP_NOT_RETRIEVAL_RESOLVABLE")
        if not set(proposal.related_claim_refs).issubset(allowed_claims):
            raise PlanningEvidenceError("FABRICATED_EVIDENCE_REQUEST_CLAIM_ID")
        if not set(proposal.related_evidence_refs).issubset(allowed_evidence):
            raise PlanningEvidenceError("FABRICATED_EVIDENCE_REQUEST_EVIDENCE_ID")
        if proposal.related_direction_refs and not set(
            proposal.related_direction_refs
        ).issubset(allowed_directions):
            raise PlanningEvidenceError("FABRICATED_EVIDENCE_REQUEST_DIRECTION_ID")
        if not set(proposal.acceptable_source_classes).issubset(
            _ACCEPTABLE_SOURCE_CLASSES
        ):
            raise PlanningEvidenceError("UNSUPPORTED_EVIDENCE_SOURCE_CLASS")
        signature = _request_signature(proposal)
        if signature in seen:
            continue
        seen.add(signature)
        requests.append(_materialize_request(proposal))
    if len(requests) > 5:
        raise PlanningEvidenceError("at most five evidence requests are allowed per cycle")
    provenance = dict(planning_input.provenance_manifest)
    identity = {
        "planning_input_id": planning_input.planning_input_id,
        "planning_input_hash": planning_input.content_hash,
        "context_id": planning_input.context_id,
        "context_hash": planning_input.context_hash,
        "domain": planning_input.domain,
        "knowledge_snapshot_id": provenance["knowledge_snapshot_id"],
        "knowledge_snapshot_hash": knowledge_snapshot_hash,
        "domain_pack_version": planning_input.domain_pack_version,
        "planning_strategy": PlanningStrategy(planning_strategy),
        "provider_id": provider.provider_id,
        "provider_version": provider.provider_version,
        "provider_config": {
            **_provider_config(provider),
            "evidence_request_policy_version": EVIDENCE_REQUEST_POLICY_VERSION,
        },
        "source_acquisition_allowed": source_acquisition_allowed,
        "direction_set_id": direction_set.direction_set_id if direction_set else None,
        "direction_set_hash": direction_set.content_hash if direction_set else None,
        "triage_id": triage_id,
        "triage_hash": triage_hash,
        "triggering_review_id": triggering_review_id,
        "triggering_review_hash": triggering_review_hash,
        "gap_classifications": classifications,
        "requests": tuple(requests),
    }
    identity = PlanningEvidenceRequestSet.model_construct(
        request_set_id="pending",
        **identity,
        content_hash="0" * 64,
    ).model_dump(mode="json", exclude={"request_set_id", "content_hash"})
    request_set_id = f"planning-evidence-request-set-{content_hash(identity)[:24]}"
    return PlanningEvidenceRequestSet(
        request_set_id=request_set_id,
        **identity,
        content_hash=content_hash({"request_set_id": request_set_id, **identity}),
    )


def retrieval_resolvable_gaps(
    planning_input: ScientificPlanningInput,
) -> tuple[EvidenceGap, ...]:
    kinds = {
        item.gap_id: item.resolution_kind
        for item in (
            classify_evidence_gap(gap)
            for gap in planning_input.evidence_gaps
            if gap.blocking
        )
    }
    return tuple(
        gap
        for gap in planning_input.evidence_gaps
        if gap.blocking
        and kinds.get(gap.gap_id) == EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE
    )


def _retrieval_source_types(
    evidence_type: PlanningEvidenceType,
) -> tuple[RetrievalSourceType, ...]:
    mapping = {
        PlanningEvidenceType.REPORTED_RESULT: (RetrievalSourceType.REPORTED_RESULT,),
        PlanningEvidenceType.METHOD_FACT: (RetrievalSourceType.METHOD_FACT,),
        PlanningEvidenceType.MODEL_FACT: (RetrievalSourceType.MODEL_FACT,),
        PlanningEvidenceType.SOURCE_CLAIM: (RetrievalSourceType.SOURCE_CLAIM,),
        PlanningEvidenceType.EXPERIMENTAL_CONDITION: (
            RetrievalSourceType.REPORTED_RESULT,
            RetrievalSourceType.METHOD_FACT,
            RetrievalSourceType.SOURCE_CLAIM,
        ),
        PlanningEvidenceType.COMPARISON_CONTEXT: (
            RetrievalSourceType.REPORTED_RESULT,
            RetrievalSourceType.METHOD_FACT,
            RetrievalSourceType.MODEL_FACT,
            RetrievalSourceType.SOURCE_CLAIM,
        ),
        PlanningEvidenceType.PRIOR_MECHANISTIC_EVIDENCE: (
            RetrievalSourceType.SOURCE_CLAIM,
            RetrievalSourceType.REPORTED_RESULT,
            RetrievalSourceType.EXPERT_OPINION,
            RetrievalSourceType.EXPERT_CASE,
        ),
    }
    return mapping[evidence_type]


def _all_hits(context: Any) -> tuple[RetrievalHit, ...]:
    return (
        *context.initial_literature_hits,
        *context.initial_expert_hits,
        *context.graph_expanded_hits,
    )


def _source_class_allows(
    hit: RetrievalHit,
    acceptable_source_classes: tuple[str, ...],
) -> bool:
    source_types = {
        "literature_reported": {
            RetrievalSourceType.LITERATURE_DOCUMENT,
            RetrievalSourceType.SOURCE_CLAIM,
            RetrievalSourceType.METHOD_FACT,
            RetrievalSourceType.MODEL_FACT,
            RetrievalSourceType.REPORTED_RESULT,
        },
        "expert_opinion": {RetrievalSourceType.EXPERT_OPINION},
        "expert_case": {RetrievalSourceType.EXPERT_CASE},
        "supporting_context": set(RetrievalSourceType),
    }
    allowed = {
        source_type
        for source_class in acceptable_source_classes
        for source_type in source_types[source_class]
    }
    return hit.source_type in allowed


def _make_acquisition_proposal(
    request: PlanningEvidenceRequest,
    query: Any,
) -> EvidenceAcquisitionProposal:
    identity = {
        "request_id": request.request_id,
        "request_hash": request.content_hash,
        "query": query,
        "acceptable_source_classes": request.acceptable_source_classes,
        "rationale": (
            "No trusted or pending local match was found; acquisition remains a separate "
            "proposal and has not been executed."
        ),
        "status": "proposed_not_executed",
    }
    identity = EvidenceAcquisitionProposal.model_construct(
        proposal_id="pending",
        **identity,
        content_hash="0" * 64,
    ).model_dump(mode="json", exclude={"proposal_id", "content_hash"})
    proposal_id = f"evidence-acquisition-proposal-{content_hash(identity)[:24]}"
    return EvidenceAcquisitionProposal(
        proposal_id=proposal_id,
        **identity,
        content_hash=content_hash({"proposal_id": proposal_id, **identity}),
    )


def _make_resolution(
    request: PlanningEvidenceRequest,
    *,
    status: EvidenceResolutionStatus,
    retrieval_context: Any | None,
    hits: Iterable[RetrievalHit] = (),
    rationale: str,
    uncertainty: tuple[str, ...],
    acquisition: EvidenceAcquisitionProposal | None = None,
) -> EvidenceResolutionRecord:
    matched = tuple(hits)
    identity = {
        "request_id": request.request_id,
        "request_hash": request.content_hash,
        "status": status,
        "retrieval_context_id": (
            retrieval_context.retrieval_context_id if retrieval_context else None
        ),
        "retrieval_context_hash": (
            retrieval_context.content_hash if retrieval_context else None
        ),
        "retrieval_context": retrieval_context,
        "matched_hits": matched,
        "evidence_refs": tuple(
            sorted({ref for hit in matched for ref in hit.evidence_refs})
        ),
        "source_refs": tuple(
            sorted({ref for hit in matched for ref in hit.source_refs})
        ),
        "record_refs": tuple(
            sorted(f"{hit.source_type.value}:{hit.record_id}" for hit in matched)
        ),
        "resolution_rationale": rationale,
        "remaining_uncertainty": uncertainty,
        "acquisition_proposal_id": acquisition.proposal_id if acquisition else None,
        "acquisition_proposal_hash": acquisition.content_hash if acquisition else None,
    }
    identity = EvidenceResolutionRecord.model_construct(
        resolution_id="pending",
        **identity,
        content_hash="0" * 64,
    ).model_dump(mode="json", exclude={"resolution_id", "content_hash"})
    resolution_id = f"evidence-resolution-{content_hash(identity)[:24]}"
    return EvidenceResolutionRecord(
        resolution_id=resolution_id,
        **identity,
        content_hash=content_hash({"resolution_id": resolution_id, **identity}),
    )


def resolve_planning_evidence_requests(
    request_set: PlanningEvidenceRequestSet,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
    profile: Any,
    snapshot: KnowledgeSnapshot,
) -> EvidenceResolutionSet:
    snapshot_hash = content_hash(
        snapshot.model_dump(mode="json", exclude={"created_at"})
    )
    if (
        request_set.knowledge_snapshot_id != snapshot.snapshot_id
        or request_set.knowledge_snapshot_hash != snapshot_hash
    ):
        raise PlanningEvidenceError("EVIDENCE_REQUEST_SNAPSHOT_MISMATCH")
    if (
        request_set.domain != profile.domain_id
        or request_set.domain_pack_version != profile.version
    ):
        raise PlanningEvidenceError("EVIDENCE_REQUEST_DOMAIN_MISMATCH")
    records: list[EvidenceResolutionRecord] = []
    acquisitions: list[EvidenceAcquisitionProposal] = []
    retriever = PersistentKnowledgeRetriever()
    for request in request_set.requests:
        raw_query = " ".join(
            (
                request.scientific_question,
                *request.target_entities,
                *request.concepts,
                *request.comparison_conditions,
            )
        )
        query = build_retrieval_query(
            raw_query,
            profile.domain_id,
            profile,
            evidence_types=_retrieval_source_types(request.evidence_type_needed),
        )
        trusted = retriever.retrieve(
            query,
            repositories,
            evidence_store,
            profile,
            snapshot,
            mode=KnowledgeViewMode.TRUSTED,
        )
        trusted_hits = tuple(
            hit
            for hit in _all_hits(trusted)
            if _source_class_allows(hit, request.acceptable_source_classes)
        )
        if trusted_hits:
            conflicting = bool(trusted.conflict_relation_refs)
            records.append(
                _make_resolution(
                    request,
                    status=(
                        EvidenceResolutionStatus.CONFLICTING_EVIDENCE
                        if conflicting
                        else EvidenceResolutionStatus.RESOLVED_TRUSTED
                    ),
                    retrieval_context=trusted,
                    hits=trusted_hits,
                    rationale=(
                        "Trusted matches were found, including an explicit contradiction relation."
                        if conflicting
                        else "Trusted, snapshot-bound knowledge matches the bounded request."
                    ),
                    uncertainty=(
                        ("Conflicting trusted records require explicit planning treatment.",)
                        if conflicting
                        else ()
                    ),
                )
            )
            continue
        audit = retriever.retrieve(
            query,
            repositories,
            evidence_store,
            profile,
            snapshot,
            mode=KnowledgeViewMode.AUDIT,
        )
        audit_hits = tuple(
            hit
            for hit in _all_hits(audit)
            if hit.authority_status != "trusted_current"
            and _source_class_allows(hit, request.acceptable_source_classes)
        )
        if audit_hits:
            records.append(
                _make_resolution(
                    request,
                    status=EvidenceResolutionStatus.REQUIRES_SOURCE_CURATION,
                    retrieval_context=audit,
                    hits=audit_hits,
                    rationale=(
                        "Matching records exist only outside the trusted-current curation boundary."
                    ),
                    uncertainty=("Human curation is required before planning may use these records.",),
                )
            )
            continue
        acquisition = (
            _make_acquisition_proposal(request, query)
            if request_set.source_acquisition_allowed
            else None
        )
        if acquisition is not None:
            acquisitions.append(acquisition)
        records.append(
            _make_resolution(
                request,
                status=EvidenceResolutionStatus.NO_MATCH,
                retrieval_context=trusted,
                rationale=(
                    "No qualifying trusted match was found within the declared source scope, "
                    "knowledge snapshot, and retrieval policy."
                ),
                uncertainty=(
                    "NO_MATCH does not establish novelty or absence from the literature.",
                ),
                acquisition=acquisition,
            )
        )
    return build_evidence_resolution_set(
        request_set,
        records=tuple(records),
        acquisition_proposals=tuple(acquisitions),
    )


def build_evidence_resolution_set(
    request_set: PlanningEvidenceRequestSet,
    *,
    records: tuple[EvidenceResolutionRecord, ...] = (),
    acquisition_proposals: tuple[EvidenceAcquisitionProposal, ...] = (),
) -> EvidenceResolutionSet:
    if {item.request_id for item in records} != {
        item.request_id for item in request_set.requests
    }:
        raise PlanningEvidenceError(
            "evidence resolution must dispose every request exactly once"
        )
    identity = {
        "request_set_id": request_set.request_set_id,
        "request_set_hash": request_set.content_hash,
        "records": records,
        "acquisition_proposals": acquisition_proposals,
    }
    identity = EvidenceResolutionSet.model_construct(
        resolution_set_id="pending",
        **identity,
        content_hash="0" * 64,
    ).model_dump(mode="json", exclude={"resolution_set_id", "content_hash"})
    resolution_set_id = f"evidence-resolution-set-{content_hash(identity)[:24]}"
    return EvidenceResolutionSet(
        resolution_set_id=resolution_set_id,
        **identity,
        content_hash=content_hash({"resolution_set_id": resolution_set_id, **identity}),
    )


def augment_scientific_context(
    context: ScientificContextPacket,
    resolutions: EvidenceResolutionSet,
    evidence_store: EvidenceStore,
) -> ScientificContextPacket:
    trusted_hits = tuple(
        hit
        for record in resolutions.records
        if record.status
        in {
            EvidenceResolutionStatus.RESOLVED_TRUSTED,
            EvidenceResolutionStatus.PARTIALLY_RESOLVED,
            EvidenceResolutionStatus.CONFLICTING_EVIDENCE,
        }
        for hit in record.matched_hits
        if hit.authority_status == "trusted_current"
    )
    if not trusted_hits:
        raise PlanningEvidenceError(
            "evidence resolution contains no trusted matches for context rebuild"
        )

    def merged_hits(
        existing: tuple[RetrievalHit, ...],
        allowed: frozenset[RetrievalSourceType],
        *,
        graph_expanded: bool,
    ) -> tuple[RetrievalHit, ...]:
        by_id = {item.hit_id: item for item in existing}
        for hit in trusted_hits:
            if hit.source_type in allowed and hit.graph_expanded == graph_expanded:
                current = by_id.get(hit.hit_id)
                if current is not None:
                    if (
                        current.source_type != hit.source_type
                        or current.record_id != hit.record_id
                        or current.record_hash != hit.record_hash
                    ):
                        raise PlanningEvidenceError(
                            f"conflicting retrieval hit identity: {hit.hit_id}"
                        )
                    continue
                by_id[hit.hit_id] = hit
        return tuple(sorted(by_id.values(), key=lambda item: item.hit_id))

    literature_types = frozenset(
        {
            RetrievalSourceType.LITERATURE_DOCUMENT,
            RetrievalSourceType.SOURCE_CLAIM,
            RetrievalSourceType.METHOD_FACT,
            RetrievalSourceType.MODEL_FACT,
            RetrievalSourceType.REPORTED_RESULT,
        }
    )
    literature_hits = merged_hits(
        context.literature_knowledge_hits,
        literature_types,
        graph_expanded=False,
    )
    expert_opinion_hits = merged_hits(
        context.expert_opinion_hits,
        frozenset({RetrievalSourceType.EXPERT_OPINION}),
        graph_expanded=False,
    )
    expert_case_hits = merged_hits(
        context.expert_case_hits,
        frozenset({RetrievalSourceType.EXPERT_CASE}),
        graph_expanded=False,
    )
    graph_hits = merged_hits(
        context.graph_expanded_hits,
        frozenset(
            {
                RetrievalSourceType.LITERATURE_DOCUMENT,
                RetrievalSourceType.SOURCE_CLAIM,
                RetrievalSourceType.METHOD_FACT,
                RetrievalSourceType.MODEL_FACT,
                RetrievalSourceType.REPORTED_RESULT,
                RetrievalSourceType.EXPERT_PROFILE,
                RetrievalSourceType.EXPERT_OPINION,
                RetrievalSourceType.EXPERT_CASE,
                RetrievalSourceType.WORKFLOW_PATTERN,
                RetrievalSourceType.SCIENTIFIC_CAPABILITY,
            }
        ),
        graph_expanded=True,
    )
    statements = list(context.retrieved_statements)
    included_evidence = {
        evidence_id
        for statement in statements
        for evidence_id in statement.evidence_refs
    }
    for evidence_id in tuple(
        dict.fromkeys(
            evidence_id
            for hit in trusted_hits
            for evidence_id in hit.evidence_refs
            if evidence_id not in included_evidence
        )
    ):
        evidence = evidence_store.get_evidence(evidence_id)
        evidence_store.verify_evidence_integrity(evidence)
        statements.append(
            GroundedStatement(
                statement_id=f"retrieved-statement-{len(statements) + 1}",
                text=evidence.text,
                classification=EvidenceClassification.EVIDENCE,
                evidence_refs=(evidence_id,),
            )
        )
        included_evidence.add(evidence_id)
    categorized = (
        context.evidence_hits,
        literature_hits,
        expert_opinion_hits,
        expert_case_hits,
        context.workflow_pattern_hits,
        context.capability_hits,
        graph_hits,
    )
    result_ids = tuple(hit.hit_id for hits in categorized for hit in hits)
    if len(set(result_ids)) != len(result_ids):
        raise PlanningEvidenceError(
            "planning-directed retrieval produced duplicate categorized hit IDs"
        )
    result_hashes = tuple(content_hash(hit) for hits in categorized for hit in hits)
    binding = {
        "query_hash": content_hash(context.retrieval_query),
        "domain_pack_id": context.domain,
        "domain_pack_version": context.retrieval_manifest.domain_pack_version,
        "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
        "retriever_version": context.retrieval_manifest.retriever_version,
        "result_ids": result_ids,
        "result_hashes": result_hashes,
        "retrieval_policy_hash": resolutions.content_hash,
    }
    manifest = RetrievalManifest(
        retrieval_id=f"retrieval-{content_hash(binding)[:24]}",
        timestamp=datetime.now(timezone.utc),
        **binding,
    )
    conflicts = tuple(
        sorted(
            set(context.conflicting_evidence)
            | {
                relation_ref
                for record in resolutions.records
                if record.status == EvidenceResolutionStatus.CONFLICTING_EVIDENCE
                for hit in record.matched_hits
                for relation_ref in hit.relation_refs
            }
        )
    )
    payload = {
        "context_id": f"context-{manifest.retrieval_id.removeprefix('retrieval-')}",
        "original_request": context.original_request,
        "domain": context.domain,
        "retrieval_query": context.retrieval_query,
        "evidence_hits": context.evidence_hits,
        "literature_knowledge_hits": literature_hits,
        "expert_opinion_hits": expert_opinion_hits,
        "expert_case_hits": expert_case_hits,
        "workflow_pattern_hits": context.workflow_pattern_hits,
        "capability_hits": context.capability_hits,
        "graph_expanded_hits": graph_hits,
        "knowledge_retrieval_context": None,
        "retrieved_statements": tuple(statements),
        "assumptions": context.assumptions,
        "conflicting_evidence": conflicts,
        "unknowns": context.unknowns,
        "retrieval_manifest": manifest,
        "knowledge_snapshot": context.knowledge_snapshot,
    }
    return ScientificContextPacket(
        **payload,
        content_hash=scientific_context_semantic_hash(payload),
    )
