from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..models import (
    AcceptanceCriterion,
    AssumptionRecord,
    CandidatePlanDraft,
    ComparisonBaseline,
    DAGTask,
    EvidenceClassification,
    EvidenceReference,
    FalsificationCriterion,
    FingerprintDifference,
    GroundedStatement,
    Hypothesis,
    IntentFingerprint,
    MethodFingerprint,
    ModelDefinition,
    ObservableDefinition,
    PlanningProposalSet,
    ProposedDeviation,
    RequiredHumanDecision,
    ScientificPlanningInput,
    ScientificQuestion,
    ScientificQuestionPlan,
    SystemFingerprint,
    UnknownRecord,
)
from ..serialization import content_hash

MATERIALIZER_VERSION = "plan-materializer-1.1.0"


def _entity_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{content_hash(value)[:24]}"


def _statement(prefix: str, text: str) -> GroundedStatement:
    payload = {
        "text": text,
        "classification": EvidenceClassification.ASSUMPTION,
        "evidence_refs": (),
    }
    return GroundedStatement(statement_id=_entity_id(prefix, payload), **payload)


class PlanMaterializer:
    def materialize(
        self,
        proposal: PlanningProposalSet,
        planning_input: ScientificPlanningInput,
    ) -> tuple[ScientificQuestionPlan, ...]:
        if (
            proposal.planning_input_id != planning_input.planning_input_id
            or proposal.planning_input_hash != planning_input.content_hash
        ):
            raise ValueError("PlanningProposalSet is not bound to ScientificPlanningInput")
        return tuple(
            self.materialize_candidate(candidate, proposal, planning_input)
            for candidate in proposal.candidates
        )

    def materialize_candidate(
        self,
        candidate: CandidatePlanDraft,
        proposal: PlanningProposalSet,
        planning_input: ScientificPlanningInput,
    ) -> ScientificQuestionPlan:
        allowed_evidence = set(planning_input.allowed_evidence_ids)
        if not set(candidate.evidence_refs).issubset(allowed_evidence):
            raise ValueError("candidate evidence references are not allowlisted")
        bindings = dict(planning_input.provenance_manifest).get("evidence_bindings", {})
        evidence_references: list[EvidenceReference] = []
        for evidence_id in candidate.evidence_refs:
            binding = dict(bindings).get(evidence_id)
            if not isinstance(binding, dict):
                try:
                    binding = dict(binding)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"planning input has no source binding for evidence {evidence_id}"
                    ) from error
            evidence_references.append(
                EvidenceReference(
                    evidence_id=evidence_id,
                    source_id=binding["source_id"],
                    source_version=binding["source_version"],
                )
            )

        questions = tuple(
            ScientificQuestion(
                question_id=_entity_id(
                    "question",
                    {
                        "candidate_key": candidate.candidate_key,
                        "text": text,
                        "evidence_refs": candidate.evidence_refs,
                    },
                ),
                text=text,
                evidence_refs=candidate.evidence_refs,
            )
            for text in proposal.intent.atomic_questions
        )
        hypothesis = Hypothesis(
            primary=_statement("hypothesis", candidate.primary_hypothesis),
            null=_statement("hypothesis", candidate.null_hypothesis),
        )
        model_statement = _statement("model-statement", candidate.model_definition)
        model = ModelDefinition(
            model_id=_entity_id(
                "model", {"candidate_key": candidate.candidate_key, "description": model_statement}
            ),
            description=model_statement,
        )

        observable_ids = {
            item.observable_key: _entity_id(
                "observable",
                {
                    "candidate_key": candidate.candidate_key,
                    "observable_key": item.observable_key,
                    "description": item.description,
                    "unit": item.unit,
                },
            )
            for item in candidate.observables
        }
        if len(observable_ids) != len(candidate.observables):
            raise ValueError("observable keys must be unique")
        observables = tuple(
            ObservableDefinition(
                observable_id=observable_ids[item.observable_key],
                description=_statement("observable-statement", item.description),
                unit=item.unit,
            )
            for item in candidate.observables
        )
        baseline_ids = {
            item.baseline_key: _entity_id(
                "baseline",
                {
                    "candidate_key": candidate.candidate_key,
                    "baseline_key": item.baseline_key,
                    "description": item.description,
                },
            )
            for item in candidate.comparison_baselines
        }
        if len(baseline_ids) != len(candidate.comparison_baselines):
            raise ValueError("comparison baseline keys must be unique")
        unknown_baselines = tuple(
            item.baseline_ref
            for item in candidate.proposed_deviations
            if item.baseline_ref not in baseline_ids
        )
        if unknown_baselines:
            raise ValueError(
                "proposed deviation references unknown comparison baseline: "
                + ", ".join(unknown_baselines)
            )
        baselines = tuple(
            ComparisonBaseline(
                baseline_id=baseline_ids[item.baseline_key],
                description=_statement("baseline-statement", item.description),
            )
            for item in candidate.comparison_baselines
        )

        def criteria(
            values: tuple[Any, ...],
            model_type: type[AcceptanceCriterion] | type[FalsificationCriterion],
            prefix: str,
        ) -> tuple[AcceptanceCriterion | FalsificationCriterion, ...]:
            built: list[AcceptanceCriterion | FalsificationCriterion] = []
            for item in values:
                observable_id = observable_ids.get(item.observable_key)
                if observable_id is None:
                    raise ValueError(
                        f"criterion references unknown observable key {item.observable_key}"
                    )
                payload = {
                    "candidate_key": candidate.candidate_key,
                    "statement": item.statement,
                    "observable_id": observable_id,
                }
                built.append(
                    model_type(
                        criterion_id=_entity_id(prefix, payload),
                        statement=item.statement,
                        observable_id=observable_id,
                    )
                )
            return tuple(built)

        acceptance = criteria(
            candidate.acceptance_criteria, AcceptanceCriterion, "acceptance"
        )
        falsification = criteria(
            candidate.falsification_criteria,
            FalsificationCriterion,
            "falsification",
        )

        deviations = tuple(
            ProposedDeviation(
                deviation_id=_entity_id(
                    "deviation",
                    {
                        "candidate_key": candidate.candidate_key,
                        "field": item.field,
                        "statement": item.statement,
                        "baseline_ref": item.baseline_ref,
                        "rationale": item.rationale,
                        "evidence_refs": item.evidence_refs,
                    },
                ),
                statement=item.statement,
                baseline_ref=baseline_ids[item.baseline_ref],
                rationale=item.rationale,
                evidence_refs=item.evidence_refs,
            )
            for item in candidate.proposed_deviations
        )
        deviation_fields = {item.field for item in candidate.proposed_deviations}
        constraint_fields = tuple(
            dict.fromkeys(
                field
                for constraint in planning_input.comparison_constraints
                for field in constraint.must_match_fields
            )
        )
        fingerprint_differences = tuple(
            FingerprintDifference(
                field=field,
                left="comparison baseline",
                right="candidate plan",
                disclosed_deviation=field in deviation_fields,
            )
            for field in constraint_fields
        )

        system_values: dict[str, list[Any]] = defaultdict(list)
        method_values: dict[str, list[Any]] = defaultdict(list)
        for result in planning_input.reported_results:
            for key, value in result.system_context.items():
                if value not in system_values[key]:
                    system_values[key].append(value)
            for key, value in result.method_context.items():
                if value not in method_values[key]:
                    method_values[key].append(value)
        system_attributes: dict[str, Any] = {
            "domain": planning_input.domain,
            **{
                key: values[0] if len(values) == 1 else tuple(values)
                for key, values in sorted(system_values.items())
            },
        }
        method_attributes: dict[str, Any] = {
            "strategy_class": candidate.strategy_class.value,
            **{
                key: values[0] if len(values) == 1 else tuple(values)
                for key, values in sorted(method_values.items())
            },
            "result_context_refs": {
                result.result_id: {
                    "method_fact_refs": result.result_context.method_fact_refs,
                    "model_fact_refs": result.result_context.model_fact_refs,
                }
                for result in planning_input.reported_results
                if result.result_context is not None
            },
        }
        for field in constraint_fields:
            target = system_attributes if field.startswith("system_context.") else method_attributes
            target[f"constraint:{field}"] = "must match or be explicitly disclosed"

        intent_payload = {
            "objective": proposal.intent.latent_concern,
            "constraints": (*proposal.intent.excluded_substitutions, *constraint_fields),
            "requested_outputs": proposal.intent.decision_relevant_observables,
        }
        intent_fingerprint = IntentFingerprint(
            fingerprint_id=_entity_id("intent", intent_payload), **intent_payload
        )
        system_payload = {
            "attributes": system_attributes,
            "evidence_refs": candidate.evidence_refs,
        }
        system_fingerprint = SystemFingerprint(
            fingerprint_id=_entity_id("system", system_payload), **system_payload
        )
        method_payload = {
            "attributes": method_attributes,
            "evidence_refs": candidate.evidence_refs,
            "proposed_deviation_refs": tuple(item.deviation_id for item in deviations),
        }
        method_fingerprint = MethodFingerprint(
            fingerprint_id=_entity_id("method", method_payload), **method_payload
        )

        task_ids = {
            task.task_key: _entity_id(
                "task",
                {
                    "candidate_key": candidate.candidate_key,
                    **task.model_dump(mode="json"),
                },
            )
            for task in candidate.task_drafts
        }
        if len(task_ids) != len(candidate.task_drafts):
            raise ValueError("task draft keys must be unique")
        constraint_success = tuple(
            f"Match or explicitly disclose comparison field: {field}"
            for field in constraint_fields
        )
        tasks = tuple(
            DAGTask(
                task_id=task_ids[item.task_key],
                scientific_objective=item.scientific_objective,
                capability_id=item.capability_id,
                inputs=dict(item.inputs),
                outputs=item.outputs,
                depends_on=tuple(task_ids[key] for key in item.depends_on),
                success_criteria=tuple(
                    dict.fromkeys((*item.success_criteria, *constraint_success))
                ),
                falsification_relevance=item.falsification_relevance,
                evidence_refs=item.evidence_refs,
                release_gates=item.release_gates,
                failure_policy=item.failure_policy,
                provenance_requirements=item.provenance_requirements,
                cost_estimate=item.cost_estimate,
                runnable=False,
            )
            for item in candidate.task_drafts
        )
        assumptions = tuple(
            AssumptionRecord(
                assumption_id=_entity_id(
                    "assumption", {"candidate_key": candidate.candidate_key, "text": text}
                ),
                statement=text,
                impact="Reassess the candidate if this assumption is not valid.",
            )
            for text in candidate.assumptions
        )
        unknowns = tuple(
            UnknownRecord(
                unknown_id=_entity_id(
                    "unknown", {"candidate_key": candidate.candidate_key, "text": text}
                ),
                statement=text,
                resolution="Resolve through the declared candidate tasks or human review.",
            )
            for text in candidate.unknowns
        )
        decisions_by_id = {
            item.decision_id: item for item in planning_input.required_human_decisions
        }
        decisions: tuple[RequiredHumanDecision, ...] = tuple(
            decisions_by_id[decision_id]
            for decision_id in candidate.human_decisions_required
        )

        identity = {
            "version": "1.0.0",
            "domain": planning_input.domain,
            "domain_pack_version": planning_input.domain_pack_version,
            "original_question": planning_input.original_request,
            "original_comment_id": None,
            "latent_concern": proposal.intent.latent_concern,
            "atomic_questions": questions,
            "hypothesis": hypothesis,
            "model": model,
            "observables": observables,
            "comparison_baselines": baselines,
            "acceptance_criteria": acceptance,
            "falsification_criteria": falsification,
            "intent_fingerprint": intent_fingerprint,
            "system_fingerprint": system_fingerprint,
            "method_fingerprint": method_fingerprint,
            "fingerprint_differences": fingerprint_differences,
            "evidence_refs": tuple(evidence_references),
            "assumptions": assumptions,
            "defaults": (),
            "unknowns": unknowns,
            "proposed_deviations": deviations,
            "scientific_capability_ids": candidate.capability_ids,
            "tasks": tasks,
            "distinguishing_axis": (
                f"{candidate.distinguishing_axis}: {candidate.distinguishing_value}"
            ),
            "cost_tier": candidate.cost_tier,
            "risks": candidate.risks,
            "limitations": candidate.limitations,
            "required_human_decisions": decisions,
            "source_query_manifest": (
                planning_input.context_id,
                planning_input.evidence_packet_id,
                planning_input.planning_input_id,
                proposal.proposal_id,
                MATERIALIZER_VERSION,
                f"distinguishing-axis:{candidate.distinguishing_axis}",
                f"distinguishing-value:{candidate.distinguishing_value}",
                *(f"claim:{claim_id}" for claim_id in candidate.claim_refs),
            ),
            "target_agent_capability_requirements": candidate.capability_ids,
            "wave_id": "wave-1",
            "follow_up_of": None,
            "source_proposal": proposal.proposal_id,
        }
        return ScientificQuestionPlan(
            plan_id=_entity_id("plan", identity),
            **identity,
        )
