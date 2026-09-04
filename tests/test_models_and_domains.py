from __future__ import annotations

import pytest
from pydantic import ValidationError

from spc.domains import DomainPackLoader
from spc.models import EvidenceClassification, GroundedStatement
from spc.serialization import content_hash
from spc.validators import compare_method_fingerprints, validate_candidate_set, validate_question_plan


def test_ft_co_activation_chain_growth_golden_fixture(make_plan) -> None:
    plan = make_plan()
    pack = DomainPackLoader().load("fischer_tropsch")
    report = validate_question_plan(plan, pack.capabilities)
    assert report.valid
    assert "ft_pathway_comparison_plan" in {item.capability_id for item in pack.capabilities}


def test_cross_domain_core_has_no_ft_requirement(make_plan) -> None:
    plan = make_plan(
        domain="base",
        latent_concern="OER mechanism",
        question="Does the OER mechanism differ from the declared baseline?",
        capability_id="comparative_analysis",
    )
    pack = DomainPackLoader().load("base")
    assert validate_question_plan(plan, pack.capabilities).valid
    schema_text = str(type(plan).model_json_schema())
    assert "Fe" not in schema_text
    assert "Fischer" not in schema_text


def test_adjacent_question_adversarial_fixture(make_plan) -> None:
    plan = make_plan(
        latent_concern="CO activation",
        question="Does product selectivity change under the comparison?",
    )
    report = validate_question_plan(plan)
    assert "INTENT_QUESTION_MISMATCH" in {item.code for item in report.issues}


def test_evidence_statement_without_reference_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GroundedStatement(
            statement_id="fact-1",
            text="Claimed fact",
            classification=EvidenceClassification.EVIDENCE,
        )


def test_method_fingerprint_without_evidence_or_deviation_is_rejected(make_plan) -> None:
    report = validate_question_plan(make_plan(method_evidence=False))
    assert "UNGROUNDED_METHOD_FINGERPRINT" in {item.code for item in report.issues}


def test_undisclosed_method_fingerprint_difference_is_rejected(make_plan) -> None:
    left = make_plan(plan_id="left")
    right_method = left.method_fingerprint.model_copy(
        update={"attributes": {"comparison_rule": "changed"}}
    )
    right = make_plan(plan_id="right").model_copy(update={"method_fingerprint": right_method})
    report = compare_method_fingerprints(left, right)
    assert "UNDISCLOSED_METHOD_DIFFERENCE" in {item.code for item in report.issues}


def test_dag_cycle_is_rejected(make_plan) -> None:
    plan = make_plan(task_overrides={"depends_on": ("task-1",)})
    report = validate_question_plan(plan)
    assert "DAG_CYCLE" in {item.code for item in report.issues}


def test_pseudo_diversity_is_rejected(make_plan) -> None:
    plans = (
        make_plan(plan_id="a", distinguishing_axis="ENCUT 400 versus 450"),
        make_plan(plan_id="b", distinguishing_axis="k-point density"),
    )
    report = validate_candidate_set(plans)
    assert "PSEUDO_DIVERSITY" in {item.code for item in report.issues}


def test_single_reasonable_candidate_is_valid(make_plan) -> None:
    assert validate_candidate_set((make_plan(),)).valid


def test_content_hash_is_deterministic(make_plan) -> None:
    assert content_hash(make_plan()) == content_hash(make_plan())
