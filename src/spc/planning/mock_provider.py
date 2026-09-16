from __future__ import annotations

from ..models import (
    AmbiguityAssessment,
    CandidatePlanDraft,
    CandidateTaskDraft,
    ComparisonBaselineDraft,
    CriterionDraft,
    IntentInterpretation,
    ObservableDraft,
    PlanRevisionDisposition,
    PlanRevisionFeedbackResponse,
    PlanRevisionInput,
    PlanRevisionLLMResponse,
    PlanningProposalSet,
    PlanningStrategyClass,
    ScientificPlanningInput,
    SourceRole,
)
from ..serialization import content_hash

MOCK_PLANNING_PROVIDER_VERSION = "mock-planning-1.1.0"


def build_proposal_set(
    planning_input: ScientificPlanningInput,
    *,
    provider_id: str,
    provider_version: str,
    provider_config: dict[str, object],
    intent: IntentInterpretation,
    ambiguity_assessment: AmbiguityAssessment,
    candidates: tuple[CandidatePlanDraft, ...],
) -> PlanningProposalSet:
    identity = {
        "planning_input_id": planning_input.planning_input_id,
        "planning_input_hash": planning_input.content_hash,
        "provider_id": provider_id,
        "provider_version": provider_version,
        "provider_config": provider_config,
        "intent": intent,
        "ambiguity_assessment": ambiguity_assessment,
        "candidates": candidates,
    }
    serialized_identity = PlanningProposalSet.model_construct(
        proposal_id="pending", **identity
    ).model_dump(mode="json", exclude={"proposal_id"})
    return PlanningProposalSet(
        proposal_id=f"planning-proposal-{content_hash(serialized_identity)[:24]}",
        **identity,
    )


class MockPlanningProvider:
    provider_id = "mock-planning"
    provider_version = MOCK_PLANNING_PROVIDER_VERSION

    def propose(self, planning_input: ScientificPlanningInput) -> PlanningProposalSet:
        claims_by_id = {claim.claim_id: claim for claim in planning_input.source_claims}
        non_reviewer_claims = tuple(
            claim
            for claim in planning_input.source_claims
            if claim.source_role != SourceRole.REVIEWER
        )
        reviewer_claims = tuple(
            claim
            for claim in planning_input.source_claims
            if claim.source_role == SourceRole.REVIEWER
        )
        target_claim = (
            non_reviewer_claims[0].text
            if non_reviewer_claims
            else planning_input.original_request
        )
        latent_concern = (
            reviewer_claims[0].text
            if reviewer_claims
            else planning_input.original_request
        )
        questions = tuple(
            dict.fromkeys(
                question
                for case in planning_input.expert_cases
                if case.positive
                for question in case.translated_questions
            )
        ) or (
            planning_input.original_request
            if planning_input.original_request.rstrip().endswith("?")
            else f"{planning_input.original_request.rstrip('.')}?",
        )
        observable_names = tuple(
            dict.fromkeys(result.quantity for result in planning_input.reported_results)
        ) or ("decision-relevant observable",)
        evidence_basis = (
            planning_input.allowed_claim_ids or planning_input.allowed_evidence_ids
        )
        if not evidence_basis:
            raise ValueError("planning requires at least one grounded claim or evidence record")
        intent = IntentInterpretation(
            target_claim=target_claim,
            latent_concern=latent_concern,
            atomic_questions=questions,
            excluded_substitutions=tuple(
                case.rationale for case in planning_input.expert_cases if not case.positive
            ),
            decision_relevant_observables=observable_names,
            evidence_basis=evidence_basis,
            unresolved_points=tuple(
                dict.fromkeys(
                    (*planning_input.unknowns, *(gap.missing_evidence for gap in planning_input.evidence_gaps))
                )
            ),
        )

        unresolved_conflicts = tuple(
            conflict
            for conflict in planning_input.conflict_sets
            if conflict.resolution_status == "unresolved"
        )
        candidate_claim_groups: tuple[tuple[str, ...], ...]
        if unresolved_conflicts:
            conflict = unresolved_conflicts[0]
            candidate_claim_groups = tuple(
                (claim_id, *tuple(item for item in conflict.claim_refs if item != claim_id))
                for claim_id in conflict.claim_refs[:4]
            )
        else:
            candidate_claim_groups = (planning_input.allowed_claim_ids,)

        workflow_capabilities = tuple(
            capability_id
            for pattern in planning_input.workflow_patterns
            for capability_id in pattern.workflow_capabilities
            if capability_id in planning_input.allowed_capability_ids
        )
        selected_capability = next(
            iter(dict.fromkeys((*workflow_capabilities, *planning_input.allowed_capability_ids))),
            None,
        )
        if selected_capability is None:
            raise ValueError("planning requires at least one retrieved scientific capability")

        candidates: list[CandidatePlanDraft] = []
        for index, claim_group in enumerate(candidate_claim_groups, start=1):
            primary_claim = claims_by_id.get(claim_group[0]) if claim_group else None
            evidence_refs = tuple(
                dict.fromkeys(
                    evidence_id
                    for claim_id in claim_group
                    for evidence_id in claims_by_id[claim_id].evidence_refs
                    if evidence_id in planning_input.allowed_evidence_ids
                )
            ) or planning_input.allowed_evidence_ids
            gap_capabilities = {
                capability_id
                for gap in planning_input.evidence_gaps
                if gap.blocking
                for capability_id in gap.candidate_capabilities
            }
            addresses_gaps = selected_capability in gap_capabilities
            unresolved_gap_ids = tuple(
                gap.gap_id for gap in planning_input.evidence_gaps if gap.blocking
            )
            decision_ids = () if addresses_gaps else tuple(
                decision.decision_id for decision in planning_input.required_human_decisions
            )
            if unresolved_conflicts:
                strategy = PlanningStrategyClass.MECHANISM_DISCRIMINATION
                axis = "competing scientific claim"
                value = claim_group[0]
            elif unresolved_gap_ids:
                strategy = PlanningStrategyClass.EVIDENCE_GAP_RESOLUTION
                axis = "evidence gap resolution strategy"
                value = (
                    f"address with {selected_capability}"
                    if addresses_gaps
                    else "retain for human decision"
                )
            else:
                strategy = PlanningStrategyClass.MINIMAL_DECISIVE_TEST
                axis = "scientific strategy"
                value = "single evidence-grounded decisive test"
            observable = ObservableDraft(
                observable_key="observable-1",
                description=f"Measure or assess {observable_names[0]} for hypothesis discrimination.",
                unit=(planning_input.reported_results[0].unit if planning_input.reported_results else None),
                evidence_refs=evidence_refs,
            )
            task = CandidateTaskDraft(
                task_key="task-1",
                scientific_objective=(
                    f"Discriminate the proposed hypothesis using {observable.description}"
                ),
                capability_id=selected_capability,
                inputs={"evidence_gap_ids": unresolved_gap_ids},
                outputs=("evidence-grounded discrimination record",),
                success_criteria=(
                    "Apply all declared comparison constraints or disclose deviations.",
                ),
                falsification_relevance="The task must be capable of retaining the null hypothesis.",
                evidence_refs=evidence_refs,
                cost_estimate="bounded planning estimate",
            )
            limitations = tuple(
                dict.fromkeys(
                    (
                        "Planning only; no scientific software is executed.",
                        *(
                            f"Blocking evidence gap retained: {gap_id}"
                            for gap_id in unresolved_gap_ids
                            if not addresses_gaps
                        ),
                    )
                )
            )
            candidates.append(
                CandidatePlanDraft(
                    candidate_key=f"candidate-{index}",
                    strategy_class=strategy,
                    distinguishing_axis=axis,
                    distinguishing_value=value,
                    primary_hypothesis=(
                        primary_claim.text
                        if primary_claim is not None
                        and primary_claim.source_role != SourceRole.REVIEWER
                        else target_claim
                    ),
                    null_hypothesis="The proposed primary hypothesis is not supported by the decisive test.",
                    model_definition="Use the evidence-bounded comparison defined by the planning input.",
                    observables=(observable,),
                    comparison_baselines=(
                        ComparisonBaselineDraft(
                            baseline_key="baseline-1",
                            description="The evidence-grounded source comparison retained in the planning input.",
                            evidence_refs=evidence_refs,
                        ),
                    ),
                    acceptance_criteria=(
                        CriterionDraft(
                            statement="The declared observable discriminates the primary and null hypotheses.",
                            observable_key=observable.observable_key,
                        ),
                    ),
                    falsification_criteria=(
                        CriterionDraft(
                            statement="Retain the null hypothesis when the declared observable is non-discriminating.",
                            observable_key=observable.observable_key,
                        ),
                    ),
                    assumptions=planning_input.assumption_candidates,
                    unknowns=planning_input.unknowns,
                    proposed_deviations=(),
                    evidence_refs=evidence_refs,
                    claim_refs=claim_group,
                    capability_ids=(selected_capability,),
                    task_drafts=(task,),
                    cost_tier="medium",
                    risks=("Available evidence may remain insufficient for discrimination.",),
                    limitations=limitations,
                    human_decisions_required=decision_ids,
                )
            )

        axes = tuple(
            dict.fromkeys(candidate.distinguishing_axis for candidate in candidates)
        )
        ambiguity = AmbiguityAssessment(
            multiple_candidates_required=len(candidates) > 1,
            rationale=(
                "Unresolved source claims require scientifically distinct alternatives."
                if len(candidates) > 1
                else "The current evidence supports one honest candidate strategy."
            ),
            scientifically_distinct_axes=axes if len(candidates) > 1 else (),
        )
        return build_proposal_set(
            planning_input,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            provider_config={"mode": "deterministic", "network": False},
            intent=intent,
            ambiguity_assessment=ambiguity,
            candidates=tuple(candidates),
        )

    def revise(self, revision_input: PlanRevisionInput) -> PlanRevisionLLMResponse:
        """Deterministically repair a small allowlist of plan-local review defects."""
        parent = next(
            candidate
            for candidate in revision_input.parent_proposal.candidates
            if candidate.candidate_key == revision_input.parent_candidate_key
        )
        candidate = parent
        intent = revision_input.parent_proposal.intent
        responses: list[PlanRevisionFeedbackResponse] = []
        for feedback in revision_input.feedback:
            changed_paths: tuple[str, ...] = ()
            disposition = PlanRevisionDisposition.UNRESOLVED
            rationale = (
                "The offline mock reviser cannot safely resolve this issue inside the fixed "
                "trusted context."
            )
            if feedback.code == "MISSING_BASELINE_OR_CONTROL":
                baseline = candidate.comparison_baselines[0].model_copy(
                    update={
                        "description": (
                            "Use an explicit evidence-grounded control that preserves the same "
                            "system and method fields while testing the competing explanation."
                        )
                    }
                )
                candidate = candidate.model_copy(
                    update={
                        "comparison_baselines": (
                            baseline,
                            *candidate.comparison_baselines[1:],
                        )
                    }
                )
                changed_paths = ("comparison_baselines",)
                disposition = PlanRevisionDisposition.ADDRESSED
                rationale = "Added an explicit matched control for hypothesis discrimination."
            elif feedback.code == "MISSING_OR_WEAK_OBSERVABLE":
                observable = candidate.observables[0].model_copy(
                    update={
                        "description": (
                            "Measure a declared decision-relevant observable that differs between "
                            "the primary and competing explanations."
                        )
                    }
                )
                candidate = candidate.model_copy(
                    update={"observables": (observable, *candidate.observables[1:])}
                )
                changed_paths = ("observables",)
                disposition = PlanRevisionDisposition.ADDRESSED
                rationale = "Replaced the weak observable with an explicit discriminating one."
            elif feedback.code == "NON_FALSIFIABLE_PLAN":
                criterion = candidate.falsification_criteria[0].model_copy(
                    update={
                        "statement": (
                            "Reject the primary hypothesis and retain the null hypothesis when the "
                            "declared observable matches the competing explanation."
                        )
                    }
                )
                candidate = candidate.model_copy(
                    update={
                        "falsification_criteria": (
                            criterion,
                            *candidate.falsification_criteria[1:],
                        )
                    }
                )
                changed_paths = ("falsification_criteria",)
                disposition = PlanRevisionDisposition.ADDRESSED
                rationale = "Added an explicit rejecting outcome for the primary hypothesis."
            elif feedback.code == "ADJACENT_OR_WRONG_SCIENTIFIC_QUESTION":
                question = revision_input.planning_input.original_request
                if not question.rstrip().endswith("?"):
                    question = f"{question.rstrip('.')}?"
                intent = intent.model_copy(update={"atomic_questions": (question,)})
                changed_paths = ("atomic_questions",)
                disposition = PlanRevisionDisposition.ADDRESSED
                rationale = "Restored the original user's scientific subject in the atomic question."
            elif feedback.code == "PSEUDO_DIVERSE_CANDIDATE":
                candidate = candidate.model_copy(
                    update={
                        "distinguishing_axis": "scientific strategy",
                        "distinguishing_value": "matched decisive control",
                        "strategy_class": PlanningStrategyClass.MINIMAL_DECISIVE_TEST,
                    }
                )
                changed_paths = ("method_fingerprint",)
                disposition = PlanRevisionDisposition.ADDRESSED
                rationale = "Replaced parameter-only diversity with a scientific strategy axis."
            responses.append(
                PlanRevisionFeedbackResponse(
                    feedback_id=feedback.feedback_id,
                    disposition=disposition,
                    changed_plan_paths=changed_paths,
                    rationale=rationale,
                )
            )
        return PlanRevisionLLMResponse(
            intent=intent,
            candidate=candidate,
            feedback_responses=tuple(responses),
        )
