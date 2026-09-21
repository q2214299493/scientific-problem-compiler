from __future__ import annotations

import re
from typing import Any

from ..models import (
    AmbiguityAssessment,
    DirectionCandidateBinding,
    DirectionDisposition,
    DirectionExpansionLLMResponse,
    DirectionTriageItem,
    DirectionTriageLLMResponse,
    DirectionTriageRecord,
    EvidenceGapResolutionKind,
    HierarchicalPlanningExpansion,
    HierarchicalPlanningStageAttempt,
    ResearchDirection,
    ResearchDirectionLLMResponse,
    ResearchDirectionSet,
    ScientificPlanningInput,
)
from ..serialization import content_hash
from ..validators import ValidationIssue, ValidationReport
from .mock_provider import build_proposal_set
from .validators import validate_planning_proposal_set

HIERARCHICAL_PLANNING_STRATEGY_VERSION = "hierarchical-planning-1.0.0"

_GENERIC_RELEVANCE_TOKENS = frozenset(
    {
        "analysis",
        "and",
        "compare",
        "comparison",
        "distinguish",
        "distinguishes",
        "does",
        "evidence",
        "explanation",
        "explanations",
        "further",
        "from",
        "has",
        "have",
        "how",
        "hypothesis",
        "mechanism",
        "mechanisms",
        "observation",
        "observable",
        "one",
        "plan",
        "proposed",
        "research",
        "scientific",
        "study",
        "the",
        "test",
        "testing",
        "that",
        "this",
        "two",
        "under",
        "which",
        "with",
        "without",
    }
)


class HierarchicalPlanningError(ValueError):
    pass


class HierarchicalPlanningBlocked(HierarchicalPlanningError):
    def __init__(self, blocking_items: tuple[str, ...]) -> None:
        self.blocking_items = blocking_items
        super().__init__("hierarchical planning blocked: " + "; ".join(blocking_items))


def planning_provider_config(provider: Any) -> dict[str, object]:
    config = dict(getattr(provider, "provider_config", {}))
    model_id = getattr(getattr(provider, "transport", None), "model_id", None)
    if model_id is not None:
        config["model_id"] = model_id
    for field_name in ("temperature", "max_attempts"):
        value = getattr(provider, field_name, None)
        if value is not None:
            config[field_name] = value
    if not config:
        config = {"mode": "deterministic", "network": False}
    return config


def build_hierarchical_stage_attempt(
    stage: str,
    planning_input: ScientificPlanningInput,
    provider: Any,
    *,
    previous_stage_id: str | None = None,
    previous_stage_hash: str | None = None,
) -> HierarchicalPlanningStageAttempt:
    provenance = dict(planning_input.provenance_manifest)
    identity = {
        "stage": stage,
        "planning_input_id": planning_input.planning_input_id,
        "planning_input_hash": planning_input.content_hash,
        "knowledge_snapshot_id": provenance["knowledge_snapshot_id"],
        "retrieval_id": provenance["retrieval_id"],
        "previous_stage_id": previous_stage_id,
        "previous_stage_hash": previous_stage_hash,
        "provider_id": provider.provider_id,
        "provider_version": provider.provider_version,
        "provider_config": planning_provider_config(provider),
    }
    attempt_id = f"hierarchical-attempt-{content_hash(identity)[:24]}"
    payload = {"attempt_id": attempt_id, **identity}
    return HierarchicalPlanningStageAttempt(
        **payload,
        content_hash=content_hash(payload),
    )


def _materialize_direction(
    proposal: object,
) -> ResearchDirection:
    payload = proposal.model_dump(mode="json")
    direction_id = f"research-direction-{content_hash(payload)[:24]}"
    return ResearchDirection(
        direction_id=direction_id,
        **payload,
        content_hash=content_hash({"direction_id": direction_id, **payload}),
    )


def build_research_direction_set(
    planning_input: ScientificPlanningInput,
    response: ResearchDirectionLLMResponse,
    provider: Any,
) -> ResearchDirectionSet:
    provenance = dict(planning_input.provenance_manifest)
    identity = {
        "planning_input_id": planning_input.planning_input_id,
        "planning_input_hash": planning_input.content_hash,
        "domain_pack_version": planning_input.domain_pack_version,
        "context_id": planning_input.context_id,
        "context_hash": planning_input.context_hash,
        "knowledge_snapshot_id": provenance["knowledge_snapshot_id"],
        "retrieval_id": provenance["retrieval_id"],
        "strategy_version": HIERARCHICAL_PLANNING_STRATEGY_VERSION,
        "provider_id": provider.provider_id,
        "provider_version": provider.provider_version,
        "provider_config": planning_provider_config(provider),
        "directions": tuple(_materialize_direction(item) for item in response.directions),
    }
    direction_set_id = f"research-direction-set-{content_hash(identity)[:24]}"
    payload = {"direction_set_id": direction_set_id, **identity}
    return ResearchDirectionSet(**payload, content_hash=content_hash(payload))


def _normalized_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 3
    )


def _meaningful_tokens(value: str) -> frozenset[str]:
    return _normalized_tokens(value) - _GENERIC_RELEVANCE_TOKENS


def _direction_anchor_tokens(
    planning_input: ScientificPlanningInput,
    direction: ResearchDirection,
) -> frozenset[str]:
    claims = {item.claim_id: item for item in planning_input.source_claims}
    capabilities = {
        item.capability_id: item for item in planning_input.scientific_capabilities
    }
    conflicts = {item.conflict_id: item for item in planning_input.conflict_sets}
    gaps = {item.gap_id: item for item in planning_input.evidence_gaps}
    values = [
        claims[item].text for item in direction.claim_refs if item in claims
    ]
    values.extend(
        capabilities[item].scientific_goal
        for item in direction.capability_refs
        if item in capabilities
    )
    for conflict_ref in direction.conflict_refs:
        conflict = conflicts.get(conflict_ref)
        if conflict is not None:
            values.extend((conflict.topic, *conflict.required_discrimination))
    for gap_ref in direction.blocking_gaps:
        gap = gaps.get(gap_ref)
        if gap is not None:
            values.extend(
                (gap.scientific_question, gap.missing_evidence, gap.why_it_matters)
            )
    for expert_case in planning_input.expert_cases:
        values.extend(expert_case.translated_questions)
        values.extend(expert_case.atomic_questions)
        if expert_case.latent_concern is not None:
            values.append(expert_case.latent_concern)
    return _meaningful_tokens(" ".join(values))


def validate_research_direction_set(
    directions: ResearchDirectionSet,
    planning_input: ScientificPlanningInput,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    provenance = dict(planning_input.provenance_manifest)
    if (
        directions.planning_input_id != planning_input.planning_input_id
        or directions.planning_input_hash != planning_input.content_hash
        or directions.domain_pack_version != planning_input.domain_pack_version
        or directions.context_id != planning_input.context_id
        or directions.context_hash != planning_input.context_hash
        or directions.knowledge_snapshot_id != provenance["knowledge_snapshot_id"]
        or directions.retrieval_id != provenance["retrieval_id"]
        or directions.strategy_version != HIERARCHICAL_PLANNING_STRATEGY_VERSION
    ):
        issues.append(
            ValidationIssue(
                code="HIERARCHICAL_INPUT_MISMATCH",
                message="research directions do not bind the supplied planning input",
            )
        )
    allowed_evidence = set(planning_input.allowed_evidence_ids)
    allowed_claims = set(planning_input.allowed_claim_ids)
    allowed_capabilities = set(planning_input.allowed_capability_ids)
    conflicts = {
        item.conflict_id: item
        for item in planning_input.conflict_sets
        if item.resolution_status == "unresolved"
    }
    blocking_gaps = {item.gap_id for item in planning_input.evidence_gaps if item.blocking}
    request_tokens = _meaningful_tokens(planning_input.original_request)
    signatures: dict[tuple[object, ...], int] = {}
    generic_observations = {
        "do more calculations",
        "perform more experiments",
        "further study the mechanism",
        "further research",
    }
    for index, direction in enumerate(directions.directions):
        path = f"directions[{index}]"
        for refs, allowed, code, field in (
            (direction.evidence_refs, allowed_evidence, "FABRICATED_EVIDENCE_ID", "evidence_refs"),
            (direction.claim_refs, allowed_claims, "FABRICATED_CLAIM_ID", "claim_refs"),
            (
                direction.capability_refs,
                allowed_capabilities,
                "FABRICATED_CAPABILITY_ID",
                "capability_refs",
            ),
            (direction.conflict_refs, set(conflicts), "FABRICATED_CONFLICT_ID", "conflict_refs"),
            (direction.blocking_gaps, blocking_gaps, "FABRICATED_GAP_ID", "blocking_gaps"),
        ):
            if not set(refs).issubset(allowed):
                issues.append(
                    ValidationIssue(
                        code=code,
                        message=f"research direction contains a non-allowlisted {field}",
                        path=f"{path}.{field}",
                    )
                )
        for conflict_id in direction.conflict_refs:
            conflict = conflicts.get(conflict_id)
            if conflict is not None and not set(conflict.claim_refs).issubset(
                set(direction.claim_refs)
            ):
                issues.append(
                    ValidationIssue(
                        code="CONFLICT_CLAIMS_NOT_PRESERVED",
                        message=f"direction does not retain every claim in {conflict_id}",
                        path=f"{path}.claim_refs",
                    )
                )
        if direction.distinguishing_observation.casefold().strip() in generic_observations:
            issues.append(
                ValidationIssue(
                    code="DIRECTION_NOT_DECISION_RELEVANT",
                    message="direction does not state a distinguishing observation",
                    path=f"{path}.distinguishing_observation",
                )
            )
        direction_tokens = _meaningful_tokens(
            " ".join(
                (
                    direction.scientific_question,
                    direction.hypothesis_or_claim_to_test,
                    *direction.competing_explanations,
                    direction.distinguishing_observation,
                    direction.rationale,
                )
            )
        )
        anchor_tokens = _direction_anchor_tokens(planning_input, direction)
        request_aligned = bool(request_tokens.intersection(direction_tokens))
        structured_context_aligned = bool(anchor_tokens.intersection(direction_tokens))
        if anchor_tokens and not (request_aligned or structured_context_aligned):
            issues.append(
                ValidationIssue(
                    code="DIRECTION_RELEVANCE_REQUIRES_HUMAN_REVIEW",
                    message=(
                        "direction has no verifiable alignment with the original request or "
                        "its referenced structured planning context"
                    ),
                    path=f"{path}.scientific_question",
                )
            )
        signature = (
            direction.scientific_question.casefold().strip(),
            direction.hypothesis_or_claim_to_test.casefold().strip(),
            tuple(item.casefold().strip() for item in direction.competing_explanations),
            direction.distinguishing_observation.casefold().strip(),
        )
        if signature in signatures:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_RESEARCH_DIRECTION",
                    message="research directions must be scientifically distinct",
                    path=path,
                )
            )
        else:
            signatures[signature] = index
    covered_conflicts = {
        item for direction in directions.directions for item in direction.conflict_refs
    }
    for conflict_id in sorted(set(conflicts) - covered_conflicts):
        issues.append(
            ValidationIssue(
                code="UNRESOLVED_CONFLICT_DROPPED",
                message=f"unresolved conflict {conflict_id} is absent from all directions",
            )
        )
    covered_gaps = {
        item for direction in directions.directions for item in direction.blocking_gaps
    }
    for gap_id in sorted(blocking_gaps - covered_gaps):
        issues.append(
            ValidationIssue(
                code="BLOCKING_EVIDENCE_GAP_DROPPED",
                message=f"blocking evidence gap {gap_id} is absent from all directions",
            )
        )
    return ValidationReport(valid=not issues, issues=tuple(issues))


def build_direction_triage_record(
    planning_input: ScientificPlanningInput,
    directions: ResearchDirectionSet,
    response: DirectionTriageLLMResponse,
    provider: Any,
) -> DirectionTriageRecord:
    by_key = {item.direction_key: item for item in directions.directions}
    supplied_keys = tuple(item.direction_key for item in response.dispositions)
    unknown_keys = tuple(sorted(set(supplied_keys) - set(by_key)))
    if unknown_keys:
        raise HierarchicalPlanningError(
            "UNKNOWN_TRIAGE_DIRECTION_KEY: " + ", ".join(unknown_keys)
        )
    duplicate_keys = tuple(
        sorted(key for key in set(supplied_keys) if supplied_keys.count(key) > 1)
    )
    if duplicate_keys:
        raise HierarchicalPlanningError(
            "DUPLICATE_TRIAGE_DIRECTION_KEY: " + ", ".join(duplicate_keys)
        )
    missing_keys = tuple(sorted(set(by_key) - set(supplied_keys)))
    if missing_keys:
        raise HierarchicalPlanningError(
            "DIRECTION_TRIAGE_INCOMPLETE: " + ", ".join(missing_keys)
        )
    items = tuple(
        DirectionTriageItem(
            **item.model_dump(mode="python"),
            direction_id=by_key[item.direction_key].direction_id,
            direction_hash=by_key[item.direction_key].content_hash,
        )
        for item in response.dispositions
    )
    blocking_items = _direction_triage_blocking_items(
        planning_input, directions, items
    )
    provenance = dict(planning_input.provenance_manifest)
    identity = {
        "planning_input_id": planning_input.planning_input_id,
        "planning_input_hash": planning_input.content_hash,
        "domain_pack_version": planning_input.domain_pack_version,
        "context_id": planning_input.context_id,
        "context_hash": planning_input.context_hash,
        "knowledge_snapshot_id": provenance["knowledge_snapshot_id"],
        "retrieval_id": provenance["retrieval_id"],
        "strategy_version": HIERARCHICAL_PLANNING_STRATEGY_VERSION,
        "provider_id": provider.provider_id,
        "provider_version": provider.provider_version,
        "provider_config": planning_provider_config(provider),
        "direction_set_id": directions.direction_set_id,
        "direction_set_hash": directions.content_hash,
        "dispositions": items,
        "blocking_items": blocking_items,
    }
    triage_id = f"direction-triage-{content_hash(identity)[:24]}"
    payload = {"triage_id": triage_id, **identity}
    return DirectionTriageRecord(**payload, content_hash=content_hash(payload))


def _direction_triage_blocking_items(
    planning_input: ScientificPlanningInput,
    directions: ResearchDirectionSet,
    dispositions: tuple[DirectionTriageItem, ...],
) -> tuple[str, ...]:
    blocking = [
        (
            "requires_human_choice:"
            f"{item.direction_id}:{item.direction_key}:{item.reason}"
        )
        for item in dispositions
        if item.disposition == DirectionDisposition.REQUIRES_HUMAN_CHOICE
    ]
    by_key = {item.direction_key: item for item in directions.directions}
    active = tuple(
        by_key[item.direction_key]
        for item in dispositions
        if item.disposition
        in {DirectionDisposition.RETAIN, DirectionDisposition.REQUIRES_HUMAN_CHOICE}
    )
    for conflict in planning_input.conflict_sets:
        if conflict.resolution_status == "unresolved" and not any(
            conflict.conflict_id in direction.conflict_refs for direction in active
        ):
            blocking.append(f"unresolved_conflict:{conflict.conflict_id}")
    for gap in planning_input.evidence_gaps:
        if gap.blocking and not any(
            gap.gap_id in direction.blocking_gaps for direction in active
        ):
            blocking.append(f"blocking_evidence_gap:{gap.gap_id}")
    return tuple(blocking)


def unresolved_human_choice_items(
    triage: DirectionTriageRecord,
) -> tuple[str, ...]:
    return tuple(
        (
            "requires_human_choice:"
            f"{item.direction_id}:{item.direction_key}:{item.reason}"
        )
        for item in triage.dispositions
        if item.disposition == DirectionDisposition.REQUIRES_HUMAN_CHOICE
    )


def validate_direction_triage(
    triage: DirectionTriageRecord,
    directions: ResearchDirectionSet,
    planning_input: ScientificPlanningInput,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    provenance = dict(planning_input.provenance_manifest)
    if (
        triage.planning_input_id != planning_input.planning_input_id
        or triage.planning_input_hash != planning_input.content_hash
        or triage.direction_set_id != directions.direction_set_id
        or triage.direction_set_hash != directions.content_hash
        or triage.domain_pack_version != planning_input.domain_pack_version
        or triage.context_id != planning_input.context_id
        or triage.context_hash != planning_input.context_hash
        or triage.knowledge_snapshot_id != provenance["knowledge_snapshot_id"]
        or triage.retrieval_id != provenance["retrieval_id"]
        or triage.strategy_version != HIERARCHICAL_PLANNING_STRATEGY_VERSION
    ):
        issues.append(
            ValidationIssue(
                code="DIRECTION_TRIAGE_BINDING_MISMATCH",
                message="triage does not bind its planning input and direction set",
            )
        )
    by_key = {item.direction_key: item for item in directions.directions}
    if {item.direction_key for item in triage.dispositions} != set(by_key):
        issues.append(
            ValidationIssue(
                code="DIRECTION_TRIAGE_INCOMPLETE",
                message="triage must dispose every research direction exactly once",
            )
        )
    for item in triage.dispositions:
        direction = by_key.get(item.direction_key)
        if direction is None:
            continue
        if (
            item.direction_id != direction.direction_id
            or item.direction_hash != direction.content_hash
        ):
            issues.append(
                ValidationIssue(
                    code="DIRECTION_TRIAGE_DIRECTION_MISMATCH",
                    message=f"triage item {item.direction_key} has a stale direction binding",
                )
            )
        for need in item.evidence_needs:
            if need.related_direction_ref not in {
                direction.direction_key,
                direction.direction_id,
            }:
                issues.append(
                    ValidationIssue(
                        code="DIRECTION_EVIDENCE_NEED_BINDING_MISMATCH",
                        message=(
                            f"evidence need {need.need_key} does not bind triage "
                            f"direction {direction.direction_key}"
                        ),
                    )
                )
            if (
                need.resolution_kind
                == EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE
                and not need.comparison_conditions
            ):
                issues.append(
                    ValidationIssue(
                        code="DIRECTION_EVIDENCE_NEED_HAS_NO_CRITERION",
                        message=(
                            f"retrieval evidence need {need.need_key} has no explicit "
                            "comparison condition"
                        ),
                    )
                )
    retained = {
        item.direction_key
        for item in triage.dispositions
        if item.disposition == DirectionDisposition.RETAIN
    }
    if len(retained) > 4:
        issues.append(
            ValidationIssue(
                code="TOO_MANY_RETAINED_DIRECTIONS",
                message="at most four directions may be expanded",
            )
        )
    active = {
        item.direction_key
        for item in triage.dispositions
        if item.disposition
        in {DirectionDisposition.RETAIN, DirectionDisposition.REQUIRES_HUMAN_CHOICE}
    }
    active_directions = tuple(by_key[key] for key in active if key in by_key)
    for conflict in planning_input.conflict_sets:
        if conflict.resolution_status == "unresolved" and not any(
            conflict.conflict_id in item.conflict_refs for item in active_directions
        ):
            issues.append(
                ValidationIssue(
                    code="UNRESOLVED_CONFLICT_DROPPED",
                    message=f"triage removed every direction for conflict {conflict.conflict_id}",
                )
            )
    for gap in planning_input.evidence_gaps:
        if gap.blocking and not any(
            gap.gap_id in item.blocking_gaps for item in active_directions
        ):
            issues.append(
                ValidationIssue(
                    code="BLOCKING_EVIDENCE_GAP_DROPPED",
                    message=f"triage removed every direction for blocking gap {gap.gap_id}",
                )
            )
    return ValidationReport(valid=not issues, issues=tuple(issues))


def build_hierarchical_expansion(
    planning_input: ScientificPlanningInput,
    directions: ResearchDirectionSet,
    triage: DirectionTriageRecord,
    response: DirectionExpansionLLMResponse,
    provider: Any,
) -> HierarchicalPlanningExpansion:
    retained = {
        item.direction_key: item
        for item in triage.dispositions
        if item.disposition == DirectionDisposition.RETAIN
    }
    by_key = {item.direction_key: item for item in directions.directions}
    expanded_keys = tuple(item.direction_key for item in response.candidates)
    if set(expanded_keys) != set(retained) or len(expanded_keys) != len(retained):
        raise HierarchicalPlanningError(
            "plan expansion must produce exactly one candidate per retained direction"
        )
    candidates = tuple(item.candidate for item in response.candidates)
    axes = tuple(dict.fromkeys(item.distinguishing_axis for item in candidates))
    provider_config = planning_provider_config(provider)
    provider_config.update(
        {
            "planning_strategy": "hierarchical",
            "strategy_version": HIERARCHICAL_PLANNING_STRATEGY_VERSION,
            "direction_set_id": directions.direction_set_id,
            "direction_set_hash": directions.content_hash,
            "triage_id": triage.triage_id,
            "triage_hash": triage.content_hash,
            "candidate_direction_keys": expanded_keys,
        }
    )
    proposal = build_proposal_set(
        planning_input,
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        provider_config=provider_config,
        intent=response.intent,
        ambiguity_assessment=AmbiguityAssessment(
            multiple_candidates_required=len(candidates) > 1,
            rationale=(
                "Retained research directions require distinct formal plans."
                if len(candidates) > 1
                else "Triage retained one decision-relevant research direction."
            ),
            scientifically_distinct_axes=axes if len(candidates) > 1 else (),
        ),
        candidates=candidates,
    )
    bindings = tuple(
        DirectionCandidateBinding(
            candidate_key=item.candidate.candidate_key,
            candidate_hash=content_hash(item.candidate),
            direction_id=by_key[item.direction_key].direction_id,
            direction_hash=by_key[item.direction_key].content_hash,
            triage_item_hash=content_hash(retained[item.direction_key]),
        )
        for item in response.candidates
    )
    provenance = dict(planning_input.provenance_manifest)
    identity = {
        "planning_input_id": planning_input.planning_input_id,
        "planning_input_hash": planning_input.content_hash,
        "domain_pack_version": planning_input.domain_pack_version,
        "context_id": planning_input.context_id,
        "context_hash": planning_input.context_hash,
        "knowledge_snapshot_id": provenance["knowledge_snapshot_id"],
        "retrieval_id": provenance["retrieval_id"],
        "strategy_version": HIERARCHICAL_PLANNING_STRATEGY_VERSION,
        "provider_id": provider.provider_id,
        "provider_version": provider.provider_version,
        "provider_config": planning_provider_config(provider),
        "direction_set_id": directions.direction_set_id,
        "direction_set_hash": directions.content_hash,
        "triage_id": triage.triage_id,
        "triage_hash": triage.content_hash,
        "candidate_bindings": bindings,
        "planning_proposal": proposal,
        "planning_proposal_hash": content_hash(proposal),
    }
    normalized_identity = HierarchicalPlanningExpansion.model_construct(
        expansion_id="pending",
        content_hash="0" * 64,
        **identity,
    ).model_dump(mode="json", exclude={"expansion_id", "content_hash"})
    expansion_id = f"hierarchical-expansion-{content_hash(normalized_identity)[:24]}"
    payload = {"expansion_id": expansion_id, **normalized_identity}
    expansion = HierarchicalPlanningExpansion(
        **payload,
        content_hash=content_hash(payload),
    )
    report = validate_hierarchical_expansion(
        expansion, directions, triage, planning_input
    )
    if not report.valid:
        raise HierarchicalPlanningError(
            ", ".join(item.code for item in report.issues)
        )
    return expansion


def validate_hierarchical_expansion(
    expansion: HierarchicalPlanningExpansion,
    directions: ResearchDirectionSet,
    triage: DirectionTriageRecord,
    planning_input: ScientificPlanningInput,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    provenance = dict(planning_input.provenance_manifest)
    if (
        expansion.planning_input_id != planning_input.planning_input_id
        or expansion.planning_input_hash != planning_input.content_hash
        or expansion.direction_set_id != directions.direction_set_id
        or expansion.direction_set_hash != directions.content_hash
        or expansion.triage_id != triage.triage_id
        or expansion.triage_hash != triage.content_hash
        or expansion.domain_pack_version != planning_input.domain_pack_version
        or expansion.context_id != planning_input.context_id
        or expansion.context_hash != planning_input.context_hash
        or expansion.knowledge_snapshot_id != provenance["knowledge_snapshot_id"]
        or expansion.retrieval_id != provenance["retrieval_id"]
        or expansion.strategy_version != HIERARCHICAL_PLANNING_STRATEGY_VERSION
    ):
        issues.append(
            ValidationIssue(
                code="HIERARCHICAL_EXPANSION_BINDING_MISMATCH",
                message="expansion does not bind its input, directions, and triage",
            )
        )
    by_direction_id = {item.direction_id: item for item in directions.directions}
    triage_by_direction_id = {
        item.direction_id: item
        for item in triage.dispositions
        if item.disposition == DirectionDisposition.RETAIN
    }
    candidates = {
        item.candidate_key: item for item in expansion.planning_proposal.candidates
    }
    atomic_questions = {
        item.casefold().strip() for item in expansion.planning_proposal.intent.atomic_questions
    }
    for binding in expansion.candidate_bindings:
        direction = by_direction_id.get(binding.direction_id)
        triage_item = triage_by_direction_id.get(binding.direction_id)
        candidate = candidates.get(binding.candidate_key)
        if direction is None or triage_item is None or candidate is None:
            issues.append(
                ValidationIssue(
                    code="CANDIDATE_DIRECTION_BINDING_MISMATCH",
                    message="candidate does not bind a retained direction",
                )
            )
            continue
        if (
            binding.direction_hash != direction.content_hash
            or binding.triage_item_hash != content_hash(triage_item)
            or binding.candidate_hash != content_hash(candidate)
        ):
            issues.append(
                ValidationIssue(
                    code="CANDIDATE_DIRECTION_BINDING_MISMATCH",
                    message="candidate direction or triage binding is stale",
                )
            )
        if candidate.primary_hypothesis.casefold().strip() != (
            direction.hypothesis_or_claim_to_test.casefold().strip()
        ):
            issues.append(
                ValidationIssue(
                    code="EXPANSION_DIRECTION_CHANGED",
                    message="candidate changed the retained direction hypothesis",
                    path=f"candidates.{candidate.candidate_key}.primary_hypothesis",
                )
            )
        if direction.scientific_question.casefold().strip() not in atomic_questions:
            issues.append(
                ValidationIssue(
                    code="EXPANSION_DIRECTION_CHANGED",
                    message="candidate proposal omitted the retained scientific question",
                    path="intent.atomic_questions",
                )
            )
        for required, actual, code in (
            (set(direction.evidence_refs), set(candidate.evidence_refs), "DIRECTION_EVIDENCE_DROPPED"),
            (set(direction.claim_refs), set(candidate.claim_refs), "DIRECTION_CLAIMS_DROPPED"),
            (
                set(direction.capability_refs),
                set(candidate.capability_ids),
                "DIRECTION_CAPABILITIES_DROPPED",
            ),
        ):
            if not required.issubset(actual):
                issues.append(
                    ValidationIssue(
                        code=code,
                        message="formal candidate dropped a retained direction constraint",
                        path=f"candidates.{candidate.candidate_key}",
                    )
                )
    proposal_report = validate_planning_proposal_set(
        expansion.planning_proposal, planning_input
    )
    issues.extend(proposal_report.issues)
    return ValidationReport(valid=not issues, issues=tuple(issues))
