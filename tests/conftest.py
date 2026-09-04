from __future__ import annotations

from typing import Any

import pytest

from spc.models import (
    AcceptanceCriterion,
    AssumptionRecord,
    ComparisonBaseline,
    DAGTask,
    EvidenceClassification,
    EvidenceReference,
    EvidenceSpan,
    FalsificationCriterion,
    GroundedStatement,
    Hypothesis,
    IntentFingerprint,
    MethodFingerprint,
    ModelDefinition,
    ObservableDefinition,
    ScientificQuestion,
    ScientificQuestionPlan,
    SystemFingerprint,
)
from spc.repositories import SourceEvidenceStore


def statement(statement_id: str, text: str, *, evidence: bool = True) -> GroundedStatement:
    return GroundedStatement(
        statement_id=statement_id,
        text=text,
        classification=EvidenceClassification.EVIDENCE if evidence else EvidenceClassification.ASSUMPTION,
        evidence_refs=("ev-1",) if evidence else (),
    )


@pytest.fixture
def evidence_repository(tmp_path):
    source = tmp_path / "source-1.txt"
    content = "evidence text for the scientific plan"
    source.write_text(content, encoding="utf-8")
    store = SourceEvidenceStore(tmp_path / ".spc")
    record = store.ingest(source, "source-1", "v1")
    store.add_evidence(
        EvidenceSpan(
            evidence_id="ev-1",
            source_id="source-1",
            source_version="v1",
            content_sha256=record.content_sha256,
            start_offset=0,
            end_offset=len(content),
            text=content,
        )
    )
    return store


@pytest.fixture
def make_plan():
    def factory(
        *,
        plan_id: str = "plan-1",
        domain: str = "fischer_tropsch",
        latent_concern: str = "CO activation and chain growth mechanism",
        question: str = "Does CO activation rather than chain growth explain the declared mechanistic comparison?",
        capability_id: str = "ft_pathway_comparison_plan",
        task_overrides: dict[str, Any] | None = None,
        distinguishing_axis: str | None = None,
        method_evidence: bool = True,
    ) -> ScientificQuestionPlan:
        task_data: dict[str, Any] = {
            "task_id": "task-1",
            "scientific_objective": "Define a falsifiable comparison",
            "capability_id": capability_id,
            "outputs": ("comparison-spec",),
            "falsification_relevance": "Tests the null hypothesis",
            "evidence_refs": ("ev-1",),
            "release_gates": ("plan-approval", "human-selection"),
            "failure_policy": "stop and request evidence",
            "provenance_requirements": ("source hashes",),
        }
        task_data.update(task_overrides or {})
        return ScientificQuestionPlan(
            plan_id=plan_id,
            version="1.0.0",
            domain=domain,
            domain_pack_version="1.0.0",
            original_question="Determine which mechanistic concern is supported.",
            original_comment_id="reviewer-2-comment-4",
            latent_concern=latent_concern,
            atomic_questions=(ScientificQuestion(question_id="q-1", text=question, evidence_refs=("ev-1",)),),
            hypothesis=Hypothesis(
                primary=statement("h-primary", "CO activation controls the comparison."),
                null=statement("h-null", "CO activation does not control the comparison."),
            ),
            model=ModelDefinition(
                model_id="model-1",
                description=statement("model-description", "Use one declared comparative model."),
                parameters=(statement("model-parameter", "Keep the comparison method consistent."),),
            ),
            observables=(
                ObservableDefinition(
                    observable_id="obs-1",
                    description=statement("observable-description", "A declared discriminating observable."),
                    unit="dimensionless",
                ),
            ),
            comparison_baselines=(
                ComparisonBaseline(
                    baseline_id="baseline-1",
                    description=statement("baseline-description", "The evidence-defined reference pathway."),
                ),
            ),
            acceptance_criteria=(
                AcceptanceCriterion(
                    criterion_id="accept-1",
                    statement="The observable satisfies the predeclared comparison rule.",
                    observable_id="obs-1",
                ),
            ),
            falsification_criteria=(
                FalsificationCriterion(
                    criterion_id="falsify-1",
                    statement="The null is retained when the predeclared rule is not satisfied.",
                    observable_id="obs-1",
                ),
            ),
            intent_fingerprint=IntentFingerprint(
                fingerprint_id="intent-1", objective="Discriminate mechanism", requested_outputs=("comparison",)
            ),
            system_fingerprint=SystemFingerprint(
                fingerprint_id="system-1",
                attributes={"system_class": "surface_reaction"},
                evidence_refs=("ev-1",),
            ),
            method_fingerprint=MethodFingerprint(
                fingerprint_id="method-1",
                attributes={"comparison_rule": "fixed"},
                evidence_refs=("ev-1",) if method_evidence else (),
            ),
            evidence_refs=(
                EvidenceReference(evidence_id="ev-1", source_id="source-1", source_version="v1"),
            ),
            assumptions=(
                AssumptionRecord(
                    assumption_id="assumption-1",
                    statement="The declared observable is measurable.",
                    impact="If false, revise the plan before export.",
                ),
            ),
            scientific_capability_ids=(capability_id,),
            tasks=(DAGTask(**task_data),),
            distinguishing_axis=distinguishing_axis,
            cost_tier="medium",
            risks=("insufficient evidence",),
            limitations=("planning only",),
            target_agent_capability_requirements=(capability_id,),
        )

    return factory
