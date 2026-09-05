from __future__ import annotations

import pytest
from pydantic import ValidationError

from spc.domains import DomainPackLoader
from spc.models import (
    EvidenceSpan,
    EvidenceClassification,
    FingerprintDifference,
    GroundedStatement,
    SystemFingerprint,
    ScientificQuestionPlan,
)
from spc.serialization import content_hash
from spc.validators import compare_method_fingerprints, validate_candidate_set, validate_question_plan


def test_ft_co_activation_chain_growth_golden_fixture(make_plan, evidence_repository) -> None:
    plan = make_plan()
    pack = DomainPackLoader().load("fischer_tropsch")
    report = validate_question_plan(plan, pack.capabilities, evidence_repository)
    assert report.valid
    assert "ft_pathway_comparison_plan" in {item.capability_id for item in pack.capabilities}


def test_cross_domain_core_has_no_ft_requirement(make_plan, evidence_repository) -> None:
    plan = make_plan(
        domain="base",
        latent_concern="OER mechanism",
        question="Does the OER mechanism differ from the declared baseline?",
        capability_id="comparative_analysis",
    )
    pack = DomainPackLoader().load("base")
    assert validate_question_plan(plan, pack.capabilities, evidence_repository).valid
    schema_text = str(type(plan).model_json_schema())
    assert "Fe" not in schema_text
    assert "Fischer" not in schema_text


def test_adjacent_question_adversarial_fixture(make_plan, evidence_repository) -> None:
    plan = make_plan(
        latent_concern="CO activation",
        question="Does product selectivity change under the comparison?",
    )
    report = validate_question_plan(plan, evidence_repository=evidence_repository)
    assert "INTENT_QUESTION_MISMATCH" in {item.code for item in report.issues}


def test_evidence_statement_without_reference_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GroundedStatement(
            statement_id="fact-1",
            text="Claimed fact",
            classification=EvidenceClassification.EVIDENCE,
        )


def test_method_fingerprint_without_evidence_or_deviation_is_rejected(make_plan, evidence_repository) -> None:
    report = validate_question_plan(
        make_plan(method_evidence=False), evidence_repository=evidence_repository
    )
    assert "UNGROUNDED_METHOD_FINGERPRINT" in {item.code for item in report.issues}


def test_undisclosed_method_fingerprint_difference_is_rejected(make_plan) -> None:
    left = make_plan(plan_id="left")
    right_method = left.method_fingerprint.model_copy(
        update={"attributes": {"comparison_rule": "changed"}}
    )
    right = make_plan(plan_id="right").model_copy(update={"method_fingerprint": right_method})
    report = compare_method_fingerprints(left, right)
    assert "UNDISCLOSED_METHOD_DIFFERENCE" in {item.code for item in report.issues}


def test_false_disclosure_cannot_hide_method_difference(make_plan) -> None:
    left = make_plan(plan_id="left")
    right = make_plan(plan_id="right").model_copy(
        update={
            "method_fingerprint": left.method_fingerprint.model_copy(
                update={"attributes": {"comparison_rule": "changed"}}
            ),
            "fingerprint_differences": (
                FingerprintDifference(
                    field="comparison_rule",
                    left="fixed",
                    right="changed",
                    disclosed_deviation=False,
                ),
            ),
        }
    )
    report = compare_method_fingerprints(left, right)
    assert "UNDISCLOSED_METHOD_DIFFERENCE" in {item.code for item in report.issues}


def test_dag_cycle_is_rejected(make_plan, evidence_repository) -> None:
    plan = make_plan(task_overrides={"depends_on": ("task-1",)})
    report = validate_question_plan(plan, evidence_repository=evidence_repository)
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


def test_nested_mutable_fingerprint_fields_are_prevented() -> None:
    source = {"surface": {"layers": [1, 2, 3]}}
    fingerprint = SystemFingerprint(
        fingerprint_id="system-nested",
        attributes=source,
        evidence_refs=("ev-1",),
    )
    source["surface"]["layers"].append(4)
    assert fingerprint.attributes["surface"]["layers"] == (1, 2, 3)
    with pytest.raises(TypeError):
        fingerprint.attributes["surface"]["layers"][0] = 9
    with pytest.raises(TypeError):
        fingerprint.attributes["surface"]["new"] = "mutable"
    with pytest.raises(TypeError):
        fingerprint.attributes._data["surface"] = "mutable"


@pytest.mark.parametrize(
    "field",
    (
        "atomic_questions",
        "observables",
        "comparison_baselines",
        "acceptance_criteria",
        "falsification_criteria",
        "evidence_refs",
        "scientific_capability_ids",
        "tasks",
    ),
)
def test_plan_rejects_empty_required_collections(make_plan, field) -> None:
    data = make_plan().model_dump(mode="python")
    data[field] = ()
    with pytest.raises(ValidationError, match=f"{field} must not be empty"):
        ScientificQuestionPlan.model_validate(data)


def test_plan_rejects_globally_duplicate_entity_ids(make_plan) -> None:
    data = make_plan().model_dump(mode="python")
    data["tasks"][0]["task_id"] = data["atomic_questions"][0]["question_id"]
    with pytest.raises(ValidationError, match="entity IDs must be globally unique"):
        ScientificQuestionPlan.model_validate(data)


@pytest.mark.parametrize(
    "path",
    (
        ("original_question",),
        ("latent_concern",),
        ("atomic_questions", 0, "text"),
        ("hypothesis", "primary", "text"),
        ("model", "description", "text"),
        ("observables", 0, "description", "text"),
        ("comparison_baselines", 0, "description", "text"),
        ("acceptance_criteria", 0, "statement"),
        ("falsification_criteria", 0, "statement"),
        ("assumptions", 0, "statement"),
        ("assumptions", 0, "impact"),
        ("intent_fingerprint", "objective"),
        ("tasks", 0, "scientific_objective"),
        ("tasks", 0, "falsification_relevance"),
        ("tasks", 0, "failure_policy"),
    ),
)
def test_core_scientific_text_rejects_blank_values(make_plan, path) -> None:
    data = make_plan().model_dump(mode="python")
    target = data
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = " \t\n"
    with pytest.raises(ValidationError, match="text must not be blank"):
        ScientificQuestionPlan.model_validate(data)


def test_core_scientific_text_collections_reject_blank_items(make_plan) -> None:
    data = make_plan().model_dump(mode="python")
    data["risks"] = (" ",)
    with pytest.raises(ValidationError, match="text must not be blank"):
        ScientificQuestionPlan.model_validate(data)


def test_evidence_span_text_rejects_blank_value(evidence_repository) -> None:
    data = evidence_repository.get("ev-1").model_dump(mode="python")
    data["text"] = " \n"
    with pytest.raises(ValidationError, match="text must not be blank"):
        EvidenceSpan.model_validate(data)
