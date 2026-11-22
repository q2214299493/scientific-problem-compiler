from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ..immutable import FrozenDict

from .base import NonBlankStr, Sha256Str, StrictModel
from .evidence import (
    ComparisonConstraint,
    ConflictSet,
    EvidenceAssessment,
    EvidenceGap,
    MethodFact,
    ModelFact,
    ReportedObservation,
    ReportedResult,
    SourceClaim,
    SourceQuote,
)
from .knowledge import ExpertCase
from .literature import LiteratureWorkflowPattern
from .retrieval import KnowledgeRetrievalContext, RetrievalHit, RetrievalQuery
from .scientific import RequiredHumanDecision, ScientificCapability


class PlanningEvidenceType(StrEnum):
    REPORTED_RESULT = "reported_result"
    METHOD_FACT = "method_fact"
    MODEL_FACT = "model_fact"
    SOURCE_CLAIM = "source_claim"
    EXPERIMENTAL_CONDITION = "experimental_condition"
    COMPARISON_CONTEXT = "comparison_context"
    PRIOR_MECHANISTIC_EVIDENCE = "prior_mechanistic_evidence"


class EvidenceGapResolutionKind(StrEnum):
    RETRIEVAL_RESOLVABLE = "retrieval_resolvable"
    CALCULATION_REQUIRED = "calculation_required"
    EXPERIMENT_REQUIRED = "experiment_required"
    HUMAN_DECISION_REQUIRED = "human_decision_required"
    UNDETERMINED = "undetermined"


class EvidenceResolutionStatus(StrEnum):
    RESOLVED_TRUSTED = "resolved_trusted"
    PARTIALLY_RESOLVED = "partially_resolved"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    NO_MATCH = "no_match"
    REQUIRES_SOURCE_CURATION = "requires_source_curation"
    NOT_RETRIEVAL_RESOLVABLE = "not_retrieval_resolvable"
    HUMAN_DECISION_REQUIRED = "human_decision_required"


class RetrievalMatchStatus(StrEnum):
    TRUSTED_MATCH = "trusted_match"
    CONFLICTING_MATCH = "conflicting_match"
    AMBIGUOUS_MATCH = "ambiguous_match"
    NO_TRUSTED_MATCH = "no_trusted_match"
    REQUIRES_CURATION = "requires_curation"


class PlanningEvidenceRequestProposal(StrictModel):
    request_key: NonBlankStr
    scientific_question: NonBlankStr
    gap_id: NonBlankStr | None = None
    triggering_question: NonBlankStr | None = None
    why_needed: NonBlankStr
    evidence_type_needed: PlanningEvidenceType
    target_entities: tuple[NonBlankStr, ...]
    concepts: tuple[NonBlankStr, ...]
    comparison_conditions: tuple[NonBlankStr, ...]
    acceptable_source_classes: tuple[NonBlankStr, ...]
    evidence_that_would_resolve_gap: NonBlankStr
    evidence_that_would_not_resolve_gap: NonBlankStr
    related_claim_refs: tuple[NonBlankStr, ...]
    related_evidence_refs: tuple[NonBlankStr, ...]
    related_direction_refs: tuple[NonBlankStr, ...]
    related_candidate_refs: tuple[NonBlankStr, ...]
    priority_reason: NonBlankStr
    blocking: bool

    @model_validator(mode="after")
    def validate_trigger(self) -> PlanningEvidenceRequestProposal:
        if (self.gap_id is None) == (self.triggering_question is None):
            raise ValueError(
                "evidence request requires exactly one gap_id or triggering_question"
            )
        return self


class PlanningEvidenceRequestLLMResponse(StrictModel):
    requests: tuple[PlanningEvidenceRequestProposal, ...] = Field(
        min_length=1, max_length=5
    )


class PlanningEvidenceGapClassification(StrictModel):
    gap_id: NonBlankStr
    resolution_kind: EvidenceGapResolutionKind
    rationale: NonBlankStr


class PlanningEvidenceRequest(PlanningEvidenceRequestProposal):
    request_id: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> PlanningEvidenceRequest:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"request_id", "content_hash"}
        )
        expected = f"planning-evidence-request-{content_hash(identity)[:24]}"
        if self.request_id != expected:
            raise ValueError("PlanningEvidenceRequest request_id is not content-bound")
        if self.content_hash != content_hash({"request_id": expected, **identity}):
            raise ValueError("PlanningEvidenceRequest content_hash is invalid")
        return self


class PlanningStrategy(StrEnum):
    DIRECT = "direct"
    HIERARCHICAL = "hierarchical"


class PlanningEvidenceRequestSet(StrictModel):
    request_set_id: NonBlankStr
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    context_id: NonBlankStr
    context_hash: Sha256Str
    domain: NonBlankStr
    knowledge_snapshot_id: NonBlankStr
    knowledge_snapshot_hash: Sha256Str
    domain_pack_version: NonBlankStr
    planning_strategy: PlanningStrategy
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config: FrozenDict
    source_acquisition_allowed: bool
    direction_set_id: NonBlankStr | None = None
    direction_set_hash: Sha256Str | None = None
    triage_id: NonBlankStr | None = None
    triage_hash: Sha256Str | None = None
    triggering_review_id: NonBlankStr | None = None
    triggering_review_hash: Sha256Str | None = None
    triggering_review_input_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    triggering_review_input_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    triggering_candidate_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    triggering_candidate_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    gap_classifications: tuple[PlanningEvidenceGapClassification, ...]
    requests: tuple[PlanningEvidenceRequest, ...] = Field(max_length=5)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> PlanningEvidenceRequestSet:
        from ..serialization import content_hash

        for left, right, name in (
            (self.direction_set_id, self.direction_set_hash, "direction set"),
            (self.triage_id, self.triage_hash, "triage"),
            (self.triggering_review_id, self.triggering_review_hash, "review"),
            (
                self.triggering_review_input_id,
                self.triggering_review_input_hash,
                "review input",
            ),
            (
                self.triggering_candidate_id,
                self.triggering_candidate_hash,
                "triggering candidate",
            ),
        ):
            if (left is None) != (right is None):
                raise ValueError(f"{name} ID and hash must be supplied together")
        new_trigger_bindings = (
            self.triggering_review_input_id,
            self.triggering_candidate_id,
        )
        if any(item is not None for item in new_trigger_bindings) and (
            self.triggering_review_id is None
            or any(item is None for item in new_trigger_bindings)
        ):
            raise ValueError(
                "review input and candidate bindings require a triggering review"
            )
        if len({item.request_key for item in self.requests}) != len(self.requests):
            raise ValueError("planning evidence request keys must be unique")
        if len({item.request_id for item in self.requests}) != len(self.requests):
            raise ValueError("planning evidence request IDs must be unique")
        identity = self.model_dump(
            mode="json", exclude={"request_set_id", "content_hash"}
        )
        expected = f"planning-evidence-request-set-{content_hash(identity)[:24]}"
        if self.request_set_id != expected:
            raise ValueError("PlanningEvidenceRequestSet ID is not content-bound")
        if self.content_hash != content_hash({"request_set_id": expected, **identity}):
            raise ValueError("PlanningEvidenceRequestSet content_hash is invalid")
        return self


class EvidenceAcquisitionProposal(StrictModel):
    proposal_id: NonBlankStr
    request_id: NonBlankStr
    request_hash: Sha256Str
    query: RetrievalQuery
    acceptable_source_classes: tuple[NonBlankStr, ...]
    rationale: NonBlankStr
    status: Literal["proposed_not_executed"] = "proposed_not_executed"
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> EvidenceAcquisitionProposal:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"proposal_id", "content_hash"}
        )
        expected = f"evidence-acquisition-proposal-{content_hash(identity)[:24]}"
        if self.proposal_id != expected:
            raise ValueError("EvidenceAcquisitionProposal ID is not content-bound")
        if self.content_hash != content_hash({"proposal_id": expected, **identity}):
            raise ValueError("EvidenceAcquisitionProposal content_hash is invalid")
        return self


class EvidenceResolutionRecord(StrictModel):
    resolution_id: NonBlankStr
    request_id: NonBlankStr
    request_hash: Sha256Str
    status: EvidenceResolutionStatus
    retrieval_status: RetrievalMatchStatus | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    retrieval_context_id: NonBlankStr | None = None
    retrieval_context_hash: Sha256Str | None = None
    retrieval_context: KnowledgeRetrievalContext | None = None
    matched_hits: tuple[RetrievalHit, ...] = ()
    evidence_refs: tuple[NonBlankStr, ...] = ()
    source_refs: tuple[NonBlankStr, ...] = ()
    record_refs: tuple[NonBlankStr, ...] = ()
    supporting_hit_ids: tuple[NonBlankStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    satisfied_conditions: tuple[NonBlankStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    unsatisfied_conditions: tuple[NonBlankStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    applied_resolution_criterion: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    applied_nonresolution_criterion: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    resolution_rationale: NonBlankStr
    remaining_uncertainty: tuple[NonBlankStr, ...]
    acquisition_proposal_id: NonBlankStr | None = None
    acquisition_proposal_hash: Sha256Str | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> EvidenceResolutionRecord:
        from ..serialization import content_hash

        if (self.retrieval_context_id is None) != (
            self.retrieval_context_hash is None
        ):
            raise ValueError("retrieval context ID and hash must be supplied together")
        if self.retrieval_context is not None and (
            self.retrieval_context_id != self.retrieval_context.retrieval_context_id
            or self.retrieval_context_hash != self.retrieval_context.content_hash
        ):
            raise ValueError("embedded retrieval context does not match its binding")
        if self.retrieval_context is None and self.retrieval_context_id is not None:
            raise ValueError("bound retrieval context must be embedded for audit")
        if (self.acquisition_proposal_id is None) != (
            self.acquisition_proposal_hash is None
        ):
            raise ValueError("acquisition proposal ID and hash must be supplied together")
        identity = self.model_dump(
            mode="json", exclude={"resolution_id", "content_hash"}
        )
        expected = f"evidence-resolution-{content_hash(identity)[:24]}"
        if self.resolution_id != expected:
            raise ValueError("EvidenceResolutionRecord ID is not content-bound")
        if self.content_hash != content_hash({"resolution_id": expected, **identity}):
            raise ValueError("EvidenceResolutionRecord content_hash is invalid")
        return self


class EvidenceResolutionSet(StrictModel):
    resolution_set_id: NonBlankStr
    request_set_id: NonBlankStr
    request_set_hash: Sha256Str
    records: tuple[EvidenceResolutionRecord, ...]
    acquisition_proposals: tuple[EvidenceAcquisitionProposal, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> EvidenceResolutionSet:
        from ..serialization import content_hash

        if len({item.request_id for item in self.records}) != len(self.records):
            raise ValueError("evidence resolution request IDs must be unique")
        identity = self.model_dump(
            mode="json", exclude={"resolution_set_id", "content_hash"}
        )
        expected = f"evidence-resolution-set-{content_hash(identity)[:24]}"
        if self.resolution_set_id != expected:
            raise ValueError("EvidenceResolutionSet ID is not content-bound")
        if self.content_hash != content_hash({"resolution_set_id": expected, **identity}):
            raise ValueError("EvidenceResolutionSet content_hash is invalid")
        return self


class PlanningEvidenceCyclePolicy(StrictModel):
    policy_id: NonBlankStr
    run_id: NonBlankStr
    max_cycles: int = Field(ge=1, le=3)
    source_acquisition_allowed: bool
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> PlanningEvidenceCyclePolicy:
        from ..serialization import content_hash

        identity = self.model_dump(mode="json", exclude={"policy_id", "content_hash"})
        expected = f"planning-evidence-policy-{content_hash(identity)[:24]}"
        if self.policy_id != expected:
            raise ValueError("PlanningEvidenceCyclePolicy ID is not content-bound")
        if self.content_hash != content_hash({"policy_id": expected, **identity}):
            raise ValueError("PlanningEvidenceCyclePolicy content_hash is invalid")
        return self


class PlanningEvidenceRequestAttempt(StrictModel):
    attempt_id: NonBlankStr
    run_id: NonBlankStr
    cycle_index: int = Field(ge=1)
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    context_id: NonBlankStr
    context_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config_hash: Sha256Str
    triggering_review_id: NonBlankStr | None = None
    triggering_review_hash: Sha256Str | None = None
    triggering_review_input_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    triggering_review_input_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    triggering_candidate_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    triggering_candidate_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    direction_set_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    direction_set_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    triage_id: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    triage_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> PlanningEvidenceRequestAttempt:
        from ..serialization import content_hash

        if (self.triggering_review_id is None) != (
            self.triggering_review_hash is None
        ):
            raise ValueError("triggering review ID and hash must be supplied together")
        for left, right, name in (
            (
                self.triggering_review_input_id,
                self.triggering_review_input_hash,
                "triggering review input",
            ),
            (
                self.triggering_candidate_id,
                self.triggering_candidate_hash,
                "triggering candidate",
            ),
            (self.direction_set_id, self.direction_set_hash, "direction set"),
            (self.triage_id, self.triage_hash, "triage"),
        ):
            if (left is None) != (right is None):
                raise ValueError(f"{name} ID and hash must be supplied together")
        identity = self.model_dump(
            mode="json", exclude={"attempt_id", "content_hash"}
        )
        expected = f"planning-evidence-attempt-{content_hash(identity)[:24]}"
        if self.attempt_id != expected:
            raise ValueError("PlanningEvidenceRequestAttempt ID is not content-bound")
        if self.content_hash != content_hash({"attempt_id": expected, **identity}):
            raise ValueError("PlanningEvidenceRequestAttempt content_hash is invalid")
        return self


class PlanningEvidenceCycleRecord(StrictModel):
    cycle_id: NonBlankStr
    cycle_index: int = Field(ge=1)
    parent_context_id: NonBlankStr
    parent_context_hash: Sha256Str
    parent_planning_input_id: NonBlankStr
    parent_planning_input_hash: Sha256Str
    request_set_id: NonBlankStr
    request_set_hash: Sha256Str
    resolution_set_id: NonBlankStr
    resolution_set_hash: Sha256Str
    knowledge_snapshot_id: NonBlankStr
    knowledge_snapshot_hash: Sha256Str
    child_context_id: NonBlankStr | None = None
    child_context_hash: Sha256Str | None = None
    child_evidence_packet_id: NonBlankStr | None = None
    child_evidence_packet_hash: Sha256Str | None = None
    child_planning_input_id: NonBlankStr | None = None
    child_planning_input_hash: Sha256Str | None = None
    terminal_reason: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> PlanningEvidenceCycleRecord:
        from ..serialization import content_hash

        child_pairs = (
            (self.child_context_id, self.child_context_hash),
            (self.child_evidence_packet_id, self.child_evidence_packet_hash),
            (self.child_planning_input_id, self.child_planning_input_hash),
        )
        if any((left is None) != (right is None) for left, right in child_pairs):
            raise ValueError("evidence-cycle child IDs and hashes must be paired")
        has_child = tuple(left is not None for left, _ in child_pairs)
        if len(set(has_child)) != 1:
            raise ValueError("evidence-cycle child artifacts must be supplied together")
        if has_child[0] == (self.terminal_reason is not None):
            raise ValueError("evidence cycle requires either child artifacts or terminal_reason")
        identity = self.model_dump(mode="json", exclude={"cycle_id", "content_hash"})
        expected = f"planning-evidence-cycle-{content_hash(identity)[:24]}"
        if self.cycle_id != expected:
            raise ValueError("PlanningEvidenceCycleRecord ID is not content-bound")
        if self.content_hash != content_hash({"cycle_id": expected, **identity}):
            raise ValueError("PlanningEvidenceCycleRecord content_hash is invalid")
        return self


class PlanningStrategyClass(StrEnum):
    MINIMAL_DECISIVE_TEST = "minimal_decisive_test"
    MECHANISM_DISCRIMINATION = "mechanism_discrimination"
    ROBUSTNESS_SENSITIVITY = "robustness_sensitivity"
    MODEL_DISCRIMINATION = "model_discrimination"
    EVIDENCE_GAP_RESOLUTION = "evidence_gap_resolution"


class DirectionDisposition(StrEnum):
    RETAIN = "retain"
    DEFER = "defer"
    EXCLUDE = "exclude"
    REQUIRES_HUMAN_CHOICE = "requires_human_choice"


class ScientificPlanningInput(StrictModel):
    planning_input_id: NonBlankStr
    original_request: NonBlankStr
    domain: NonBlankStr
    domain_pack_version: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    evidence_packet_id: NonBlankStr
    evidence_packet_hash: Sha256Str
    source_quotes: tuple[SourceQuote, ...] = ()
    source_claims: tuple[SourceClaim, ...] = ()
    reported_results: tuple[ReportedResult, ...] = ()
    reported_observations: tuple[ReportedObservation, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    method_facts: tuple[MethodFact, ...] = ()
    model_facts: tuple[ModelFact, ...] = ()
    evidence_assessments: tuple[EvidenceAssessment, ...] = ()
    conflict_sets: tuple[ConflictSet, ...] = ()
    comparison_constraints: tuple[ComparisonConstraint, ...] = ()
    evidence_gaps: tuple[EvidenceGap, ...] = ()
    unknowns: tuple[NonBlankStr, ...] = ()
    assumption_candidates: tuple[NonBlankStr, ...] = ()
    expert_cases: tuple[ExpertCase, ...] = ()
    workflow_patterns: tuple[LiteratureWorkflowPattern, ...] = ()
    scientific_capabilities: tuple[ScientificCapability, ...] = ()
    allowed_evidence_ids: tuple[NonBlankStr, ...] = ()
    allowed_claim_ids: tuple[NonBlankStr, ...] = ()
    allowed_capability_ids: tuple[NonBlankStr, ...] = ()
    required_human_decisions: tuple[RequiredHumanDecision, ...] = ()
    provenance_manifest: FrozenDict
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_allowlists(self) -> ScientificPlanningInput:
        from ..serialization import content_hash

        for field_name in (
            "allowed_evidence_ids",
            "allowed_claim_ids",
            "allowed_capability_ids",
        ):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must contain unique IDs")
        if set(self.allowed_claim_ids) != {claim.claim_id for claim in self.source_claims}:
            raise ValueError("allowed_claim_ids must match the included SourceClaim records")
        if set(self.allowed_capability_ids) != {
            capability.capability_id for capability in self.scientific_capabilities
        }:
            raise ValueError(
                "allowed_capability_ids must match the included ScientificCapability records"
            )
        allowed_evidence = set(self.allowed_evidence_ids)
        referenced_evidence = {
            evidence_id
            for records in (
                self.source_quotes,
                self.source_claims,
                self.reported_results,
                self.reported_observations,
                self.method_facts,
                self.model_facts,
                self.comparison_constraints,
                self.evidence_gaps,
            )
            for record in records
            for evidence_id in (
                (record.evidence_ref,)
                if isinstance(record, SourceQuote)
                else record.evidence_refs
            )
        }
        if not referenced_evidence.issubset(allowed_evidence):
            raise ValueError("planning input contains evidence outside allowed_evidence_ids")
        identity = self.model_dump(
            mode="json", exclude={"planning_input_id", "content_hash"}
        )
        expected_id = f"planning-input-{content_hash(identity)[:24]}"
        if self.planning_input_id != expected_id:
            raise ValueError("ScientificPlanningInput planning_input_id is not content-bound")
        payload = {"planning_input_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ScientificPlanningInput content_hash is invalid")
        return self


class IntentInterpretation(StrictModel):
    target_claim: NonBlankStr
    latent_concern: NonBlankStr
    atomic_questions: tuple[NonBlankStr, ...] = Field(min_length=1)
    excluded_substitutions: tuple[NonBlankStr, ...] = ()
    decision_relevant_observables: tuple[NonBlankStr, ...] = Field(min_length=1)
    evidence_basis: tuple[NonBlankStr, ...] = Field(min_length=1)
    unresolved_points: tuple[NonBlankStr, ...] = ()


class AmbiguityAssessment(StrictModel):
    multiple_candidates_required: bool
    rationale: NonBlankStr
    scientifically_distinct_axes: tuple[NonBlankStr, ...] = ()

    @model_validator(mode="after")
    def validate_axes(self) -> AmbiguityAssessment:
        if self.multiple_candidates_required and not self.scientifically_distinct_axes:
            raise ValueError("multiple candidates require at least one scientific axis")
        if len(set(self.scientifically_distinct_axes)) != len(
            self.scientifically_distinct_axes
        ):
            raise ValueError("scientifically distinct axes must be unique")
        return self


class ObservableDraft(StrictModel):
    observable_key: NonBlankStr
    description: NonBlankStr
    unit: NonBlankStr | None = None
    evidence_refs: tuple[NonBlankStr, ...] = ()


class ComparisonBaselineDraft(StrictModel):
    baseline_key: NonBlankStr
    description: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()


class CriterionDraft(StrictModel):
    statement: NonBlankStr
    observable_key: NonBlankStr


class ProposedDeviationDraft(StrictModel):
    field: NonBlankStr
    statement: NonBlankStr
    baseline_ref: NonBlankStr
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()


class CandidateTaskDraft(StrictModel):
    task_key: NonBlankStr
    scientific_objective: NonBlankStr
    capability_id: NonBlankStr
    inputs: FrozenDict = Field(default_factory=FrozenDict)
    outputs: tuple[NonBlankStr, ...] = Field(min_length=1)
    depends_on: tuple[NonBlankStr, ...] = ()
    success_criteria: tuple[NonBlankStr, ...] = Field(min_length=1)
    falsification_relevance: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    release_gates: tuple[NonBlankStr, ...] = (
        "deterministic-plan-validation",
        "human-selection",
    )
    failure_policy: NonBlankStr = "stop and request review"
    provenance_requirements: tuple[NonBlankStr, ...] = (
        "planning input hash",
        "evidence references",
    )
    cost_estimate: NonBlankStr = "unknown"


class CandidatePlanDraft(StrictModel):
    candidate_key: NonBlankStr
    strategy_class: PlanningStrategyClass
    distinguishing_axis: NonBlankStr
    distinguishing_value: NonBlankStr
    primary_hypothesis: NonBlankStr
    null_hypothesis: NonBlankStr
    model_definition: NonBlankStr
    observables: tuple[ObservableDraft, ...] = Field(min_length=1)
    comparison_baselines: tuple[ComparisonBaselineDraft, ...] = Field(min_length=1)
    acceptance_criteria: tuple[CriterionDraft, ...] = Field(min_length=1)
    falsification_criteria: tuple[CriterionDraft, ...] = Field(min_length=1)
    assumptions: tuple[NonBlankStr, ...] = ()
    unknowns: tuple[NonBlankStr, ...] = ()
    proposed_deviations: tuple[ProposedDeviationDraft, ...] = ()
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    claim_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    capability_ids: tuple[NonBlankStr, ...] = Field(min_length=1)
    task_drafts: tuple[CandidateTaskDraft, ...] = Field(min_length=1)
    cost_tier: NonBlankStr
    risks: tuple[NonBlankStr, ...] = ()
    limitations: tuple[NonBlankStr, ...] = ()
    human_decisions_required: tuple[NonBlankStr, ...] = ()


class PlanningLLMResponse(StrictModel):
    intent: IntentInterpretation
    ambiguity_assessment: AmbiguityAssessment
    candidates: tuple[CandidatePlanDraft, ...] = Field(min_length=1, max_length=4)


class ResearchDirectionProposal(StrictModel):
    direction_key: NonBlankStr
    scientific_question: NonBlankStr
    hypothesis_or_claim_to_test: NonBlankStr
    competing_explanations: tuple[NonBlankStr, ...] = Field(min_length=1)
    distinguishing_observation: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    claim_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    capability_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    conflict_refs: tuple[NonBlankStr, ...] = ()
    assumptions: tuple[NonBlankStr, ...] = ()
    blocking_gaps: tuple[NonBlankStr, ...] = ()
    limitations: tuple[NonBlankStr, ...] = ()
    rationale: NonBlankStr


class ResearchDirectionLLMResponse(StrictModel):
    directions: tuple[ResearchDirectionProposal, ...] = Field(min_length=1, max_length=6)


class ResearchDirection(ResearchDirectionProposal):
    direction_id: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ResearchDirection:
        from ..serialization import content_hash

        payload = self.model_dump(
            mode="json", exclude={"direction_id", "content_hash"}
        )
        expected = f"research-direction-{content_hash(payload)[:24]}"
        if self.direction_id != expected:
            raise ValueError("ResearchDirection direction_id is not content-bound")
        if self.content_hash != content_hash({"direction_id": expected, **payload}):
            raise ValueError("ResearchDirection content_hash is invalid")
        return self


class ResearchDirectionSet(StrictModel):
    direction_set_id: NonBlankStr
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    domain_pack_version: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    knowledge_snapshot_id: NonBlankStr
    retrieval_id: NonBlankStr
    strategy_version: NonBlankStr
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config: FrozenDict
    directions: tuple[ResearchDirection, ...] = Field(min_length=1, max_length=6)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ResearchDirectionSet:
        from ..serialization import content_hash

        keys = tuple(item.direction_key for item in self.directions)
        if len(set(keys)) != len(keys):
            raise ValueError("research direction keys must be unique")
        identity = self.model_dump(
            mode="json", exclude={"direction_set_id", "content_hash"}
        )
        expected = f"research-direction-set-{content_hash(identity)[:24]}"
        if self.direction_set_id != expected:
            raise ValueError("ResearchDirectionSet direction_set_id is not content-bound")
        payload = {"direction_set_id": expected, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ResearchDirectionSet content_hash is invalid")
        return self


class DirectionEvidenceNeed(StrictModel):
    need_key: NonBlankStr
    related_direction_ref: NonBlankStr
    scientific_question: NonBlankStr
    missing_evidence: NonBlankStr
    why_direction_comparison_changes: NonBlankStr
    required_evidence_type: PlanningEvidenceType
    comparison_conditions: tuple[NonBlankStr, ...] = Field(min_length=1)
    resolution_kind: EvidenceGapResolutionKind


class DirectionTriageProposal(StrictModel):
    direction_key: NonBlankStr
    disposition: DirectionDisposition
    reason: NonBlankStr
    evidence_needs: tuple[DirectionEvidenceNeed, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


class DirectionTriageLLMResponse(StrictModel):
    dispositions: tuple[DirectionTriageProposal, ...] = Field(min_length=1, max_length=6)


class DirectionTriageItem(DirectionTriageProposal):
    direction_id: NonBlankStr
    direction_hash: Sha256Str


class DirectionTriageRecord(StrictModel):
    triage_id: NonBlankStr
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    domain_pack_version: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    knowledge_snapshot_id: NonBlankStr
    retrieval_id: NonBlankStr
    strategy_version: NonBlankStr
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config: FrozenDict
    direction_set_id: NonBlankStr
    direction_set_hash: Sha256Str
    dispositions: tuple[DirectionTriageItem, ...] = Field(min_length=1, max_length=6)
    blocking_items: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> DirectionTriageRecord:
        from ..serialization import content_hash

        keys = tuple(item.direction_key for item in self.dispositions)
        if len(set(keys)) != len(keys):
            raise ValueError("direction triage keys must be unique")
        identity = self.model_dump(mode="json", exclude={"triage_id", "content_hash"})
        expected = f"direction-triage-{content_hash(identity)[:24]}"
        if self.triage_id != expected:
            raise ValueError("DirectionTriageRecord triage_id is not content-bound")
        payload = {"triage_id": expected, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("DirectionTriageRecord content_hash is invalid")
        return self


class DirectionExpansionCandidate(StrictModel):
    direction_key: NonBlankStr
    candidate: CandidatePlanDraft


class DirectionExpansionLLMResponse(StrictModel):
    intent: IntentInterpretation
    candidates: tuple[DirectionExpansionCandidate, ...] = Field(min_length=1, max_length=4)


class DirectionCandidateBinding(StrictModel):
    candidate_key: NonBlankStr
    candidate_hash: Sha256Str
    direction_id: NonBlankStr
    direction_hash: Sha256Str
    triage_item_hash: Sha256Str


class PlanningProposalSet(StrictModel):
    proposal_id: NonBlankStr
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config: FrozenDict = Field(default_factory=FrozenDict)
    intent: IntentInterpretation
    ambiguity_assessment: AmbiguityAssessment
    candidates: tuple[CandidatePlanDraft, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_identity_and_candidates(self) -> PlanningProposalSet:
        from ..serialization import content_hash

        if len({candidate.candidate_key for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("candidate keys must be unique")
        multiple = len(self.candidates) > 1
        if self.ambiguity_assessment.multiple_candidates_required != multiple:
            raise ValueError("ambiguity assessment does not match candidate count")
        candidate_distinctions = tuple(
            (candidate.distinguishing_axis, candidate.distinguishing_value)
            for candidate in self.candidates
        )
        if len(set(candidate_distinctions)) != len(candidate_distinctions):
            raise ValueError("candidate distinguishing axis/value pairs must be unique")
        declared_axes = set(self.ambiguity_assessment.scientifically_distinct_axes)
        candidate_axes = {candidate.distinguishing_axis for candidate in self.candidates}
        if multiple and declared_axes != candidate_axes:
            raise ValueError("ambiguity assessment axes do not match candidate axes")
        identity = self.model_dump(mode="json", exclude={"proposal_id"})
        if self.proposal_id != f"planning-proposal-{content_hash(identity)[:24]}":
            raise ValueError("PlanningProposalSet proposal_id is not content-bound")
        return self


class HierarchicalPlanningExpansion(StrictModel):
    expansion_id: NonBlankStr
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    domain_pack_version: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    knowledge_snapshot_id: NonBlankStr
    retrieval_id: NonBlankStr
    strategy_version: NonBlankStr
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config: FrozenDict
    direction_set_id: NonBlankStr
    direction_set_hash: Sha256Str
    triage_id: NonBlankStr
    triage_hash: Sha256Str
    candidate_bindings: tuple[DirectionCandidateBinding, ...] = Field(min_length=1)
    planning_proposal: PlanningProposalSet
    planning_proposal_hash: Sha256Str
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> HierarchicalPlanningExpansion:
        from ..serialization import content_hash

        if self.planning_proposal.planning_input_id != self.planning_input_id:
            raise ValueError("hierarchical expansion proposal has wrong planning input")
        if self.planning_proposal.planning_input_hash != self.planning_input_hash:
            raise ValueError("hierarchical expansion proposal has wrong planning input hash")
        if self.planning_proposal_hash != content_hash(self.planning_proposal):
            raise ValueError("hierarchical expansion proposal hash is invalid")
        candidate_keys = tuple(item.candidate_key for item in self.candidate_bindings)
        if candidate_keys != tuple(
            item.candidate_key for item in self.planning_proposal.candidates
        ):
            raise ValueError("hierarchical candidate bindings do not match proposal")
        identity = self.model_dump(mode="json", exclude={"expansion_id", "content_hash"})
        expected = f"hierarchical-expansion-{content_hash(identity)[:24]}"
        if self.expansion_id != expected:
            raise ValueError("HierarchicalPlanningExpansion expansion_id is not content-bound")
        payload = {"expansion_id": expected, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("HierarchicalPlanningExpansion content_hash is invalid")
        return self


class HierarchicalPlanningStageAttempt(StrictModel):
    attempt_id: NonBlankStr
    stage: Literal["directions", "triage", "expansion"]
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    knowledge_snapshot_id: NonBlankStr
    retrieval_id: NonBlankStr
    previous_stage_id: NonBlankStr | None = None
    previous_stage_hash: Sha256Str | None = None
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    provider_config: FrozenDict
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> HierarchicalPlanningStageAttempt:
        from ..serialization import content_hash

        if (self.previous_stage_id is None) != (self.previous_stage_hash is None):
            raise ValueError("hierarchical stage previous binding must be complete")
        identity = self.model_dump(mode="json", exclude={"attempt_id", "content_hash"})
        expected = f"hierarchical-attempt-{content_hash(identity)[:24]}"
        if self.attempt_id != expected:
            raise ValueError("HierarchicalPlanningStageAttempt attempt_id is not content-bound")
        payload = {"attempt_id": expected, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("HierarchicalPlanningStageAttempt content_hash is invalid")
        return self


class PlanCompilationReceipt(StrictModel):
    receipt_id: NonBlankStr
    plan_id: NonBlankStr
    plan_hash: Sha256Str
    planning_input_id: NonBlankStr
    planning_input_hash: Sha256Str
    planning_proposal_id: NonBlankStr
    planning_proposal_hash: Sha256Str
    materializer_version: NonBlankStr
    origin: NonBlankStr = "phase2c_grounded_compiler"
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> PlanCompilationReceipt:
        from ..serialization import content_hash

        if self.origin != "phase2c_grounded_compiler":
            raise ValueError("PlanCompilationReceipt origin is invalid")
        identity = self.model_dump(
            mode="json", exclude={"receipt_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"plan-compilation-receipt-{content_hash(identity)[:24]}"
        if self.receipt_id != expected_id:
            raise ValueError("PlanCompilationReceipt receipt_id is not content-bound")
        payload = {"receipt_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("PlanCompilationReceipt content_hash is invalid")
        return self
