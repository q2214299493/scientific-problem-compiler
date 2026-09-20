from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import spc.workflow.service as workflow_service
from spc.cli import app
from spc.models import (
    ApprovalDecision,
    ApprovalHardRedFlag,
    ApprovalRedFlagSeverity,
    DirectionDisposition,
    DirectionExpansionLLMResponse,
    DirectionTriageRecord,
    DirectionTriageLLMResponse,
    DirectionTriageProposal,
    ResearchDirectionLLMResponse,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
)
from spc.planning import (
    FakeLLMTransport,
    HierarchicalPlanningError,
    MockPlanningProvider,
    PlanMaterializer,
    StructuredLLMPlanningProvider,
    build_direction_triage_record,
    build_hierarchical_expansion,
    build_research_direction_set,
    derive_candidate_task_id,
    validate_hierarchical_expansion,
    validate_direction_triage,
    validate_research_direction_set,
)
from spc.planning.mock_provider import build_proposal_set
from spc.serialization import content_hash, dump_yaml
from spc.workflow import ScientificProblemWorkflow, scientific_run_status
from spc.approval import MockApprovalProvider
from test_knowledge_retrieval import _prepare
from test_planning import build_grounded_inputs


REQUEST = "The mechanism is not sufficiently convincing without a distinguishing observation."


def _hierarchy(
    tmp_path: Path, *, request: str = "CO activation pathway comparison plan"
):
    *_, planning_input, _, _ = build_grounded_inputs(tmp_path, request)
    provider = MockPlanningProvider()
    direction_response = provider.propose_directions(planning_input)
    directions = build_research_direction_set(
        planning_input, direction_response, provider
    )
    triage_response = provider.triage_directions(planning_input, directions)
    triage = build_direction_triage_record(
        planning_input, directions, triage_response, provider
    )
    expansion_response = provider.expand_directions(
        planning_input, directions, triage
    )
    expansion = build_hierarchical_expansion(
        planning_input, directions, triage, expansion_response, provider
    )
    return planning_input, provider, direction_response, directions, triage, expansion


def _two_direction_response(
    response: ResearchDirectionLLMResponse,
) -> ResearchDirectionLLMResponse:
    first = response.directions[0]
    second = first.model_copy(
        update={
            "direction_key": "direction-2",
            "scientific_question": (
                "Which independent pathway control distinguishes the competing explanation?"
            ),
            "hypothesis_or_claim_to_test": (
                "An independent pathway control separates the competing explanation."
            ),
            "competing_explanations": (
                "The control does not distinguish the competing explanation.",
            ),
            "distinguishing_observation": (
                "A matched control gives a different response for the two explanations."
            ),
            "rationale": (
                "The independent control supplies a second decision-relevant contrast."
            ),
        }
    )
    return ResearchDirectionLLMResponse(directions=(first, second))


def test_shared_task_identity_preserves_existing_materialized_ids(tmp_path: Path) -> None:
    *_, planning_input, proposal, _ = build_grounded_inputs(tmp_path)
    candidate = proposal.candidates[0]
    plan = PlanMaterializer().materialize(proposal, planning_input)[0]

    legacy_ids = {
        task.task_key: "task-"
        + content_hash(
            {
                "candidate_key": candidate.candidate_key,
                **task.model_dump(mode="json"),
            }
        )[:24]
        for task in candidate.task_drafts
    }
    shared_ids = {
        task.task_key: derive_candidate_task_id(
            candidate.candidate_key, task.model_dump(mode="json")
        )
        for task in candidate.task_drafts
    }

    assert shared_ids == legacy_ids
    assert set(shared_ids.values()) == {task.task_id for task in plan.tasks}


def test_direct_is_default_and_does_not_create_hierarchy(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    run = workflow.start(REQUEST, "base").run

    assert run.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    assert not workflow.runs.resolve_artifact_path(
        run.run_id, "hierarchical-planning"
    ).exists()


class CountingHierarchicalProvider(MockPlanningProvider):
    directions_calls = 0
    triage_calls = 0
    expansion_calls = 0

    def propose_directions(self, planning_input):
        type(self).directions_calls += 1
        return super().propose_directions(planning_input)

    def triage_directions(self, planning_input, directions):
        type(self).triage_calls += 1
        return super().triage_directions(planning_input, directions)

    def expand_directions(self, planning_input, directions, triage):
        type(self).expansion_calls += 1
        return super().expand_directions(planning_input, directions, triage)


def test_hierarchical_workflow_runs_three_stages_and_resume_reuses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories, *_ = _prepare(tmp_path)
    CountingHierarchicalProvider.directions_calls = 0
    CountingHierarchicalProvider.triage_calls = 0
    CountingHierarchicalProvider.expansion_calls = 0
    monkeypatch.setattr(
        workflow_service, "MockPlanningProvider", CountingHierarchicalProvider
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    first = workflow.start(
        REQUEST, "base", planning_strategy="hierarchical"
    ).run
    assert first.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    assert (
        CountingHierarchicalProvider.directions_calls,
        CountingHierarchicalProvider.triage_calls,
        CountingHierarchicalProvider.expansion_calls,
    ) == (1, 1, 1)
    for name in ("directions.yaml", "triage.yaml", "expansion.yaml"):
        assert workflow.runs.resolve_artifact_path(
            first.run_id, f"hierarchical-planning/{name}"
        ).is_file()

    repeated = workflow.resume(first.run_id).run
    assert repeated.candidate_plans == first.candidate_plans
    assert (
        CountingHierarchicalProvider.directions_calls,
        CountingHierarchicalProvider.triage_calls,
        CountingHierarchicalProvider.expansion_calls,
    ) == (1, 1, 1)


def test_structured_provider_uses_distinct_stage_contracts(tmp_path: Path) -> None:
    planning_input, mock, direction_response, directions, triage, _ = _hierarchy(
        tmp_path
    )
    triage_response = mock.triage_directions(planning_input, directions)
    expansion_response = mock.expand_directions(planning_input, directions, triage)
    transport = FakeLLMTransport(
        (
            direction_response.model_dump(mode="json"),
            triage_response.model_dump(mode="json"),
            expansion_response.model_dump(mode="json"),
        )
    )
    provider = StructuredLLMPlanningProvider(transport, max_attempts=1)

    provider.propose_directions(planning_input)
    provider.triage_directions(planning_input, directions)
    provider.expand_directions(planning_input, directions, triage)

    assert transport.call_count == 3
    assert len({item["system_prompt"] for item in transport.requests}) == 3
    assert [item["response_schema"]["title"] for item in transport.requests] == [
        "ResearchDirectionLLMResponse",
        "DirectionTriageLLMResponse",
        "DirectionExpansionLLMResponse",
    ]


def test_simple_problem_may_generate_one_direction(tmp_path: Path) -> None:
    _, _, _, directions, triage, expansion = _hierarchy(tmp_path)

    assert len(directions.directions) == 1
    assert sum(
        item.disposition == DirectionDisposition.RETAIN
        for item in triage.dispositions
    ) == 1
    assert len(expansion.planning_proposal.candidates) == 1


def test_duplicate_directions_are_rejected(tmp_path: Path) -> None:
    planning_input, provider, response, *_ = _hierarchy(tmp_path)
    first = response.directions[0]
    duplicate = first.model_copy(update={"direction_key": "duplicate-key"})
    directions = build_research_direction_set(
        planning_input,
        ResearchDirectionLLMResponse(directions=(first, duplicate)),
        provider,
    )

    report = validate_research_direction_set(directions, planning_input)
    assert "DUPLICATE_RESEARCH_DIRECTION" in {item.code for item in report.issues}


@pytest.mark.parametrize(
    ("field", "fake_id", "code"),
    (
        ("evidence_refs", "ev-fabricated", "FABRICATED_EVIDENCE_ID"),
        ("claim_refs", "claim-fabricated", "FABRICATED_CLAIM_ID"),
        ("capability_refs", "capability-fabricated", "FABRICATED_CAPABILITY_ID"),
    ),
)
def test_direction_fabricated_references_are_rejected(
    tmp_path: Path, field: str, fake_id: str, code: str
) -> None:
    planning_input, provider, response, *_ = _hierarchy(tmp_path)
    first = response.directions[0]
    changed = first.model_copy(update={field: (*getattr(first, field), fake_id)})
    directions = build_research_direction_set(
        planning_input,
        ResearchDirectionLLMResponse(directions=(changed,)),
        provider,
    )

    report = validate_research_direction_set(directions, planning_input)
    assert code in {item.code for item in report.issues}


def test_synonymous_direction_without_request_token_overlap_is_allowed(
    tmp_path: Path,
) -> None:
    planning_input, provider, response, *_ = _hierarchy(tmp_path)
    first = response.directions[0].model_copy(
        update={
            "scientific_question": (
                "Which measured signal separates dissociative carbon monoxide chemistry "
                "from associative chain propagation?"
            ),
            "hypothesis_or_claim_to_test": (
                "Dissociative carbon monoxide chemistry yields a distinct signal."
            ),
            "competing_explanations": (
                "Associative chain propagation yields the alternative signal.",
            ),
            "distinguishing_observation": (
                "A measured signal separates the dissociative and associative routes."
            ),
            "rationale": (
                "The contrast discriminates the two evidence-grounded explanations."
            ),
        }
    )
    directions = build_research_direction_set(
        planning_input,
        ResearchDirectionLLMResponse(directions=(first,)),
        provider,
    )

    report = validate_research_direction_set(directions, planning_input)

    assert report.valid


@pytest.mark.parametrize(
    ("original_request", "question"),
    (
        (
            "The catalytic mechanism needs a decisive comparison.",
            "Which mechanism controls lunar crater erosion?",
        ),
        (
            "CO activation pathway comparison plan",
            "How should coastal salinity be mapped from satellite images?",
        ),
    ),
)
def test_unrelated_direction_is_not_accepted_by_generic_or_absent_overlap(
    tmp_path: Path,
    original_request: str,
    question: str,
) -> None:
    planning_input, provider, response, *_ = _hierarchy(
        tmp_path, request=original_request
    )
    first = response.directions[0].model_copy(
        update={
            "scientific_question": question,
            "hypothesis_or_claim_to_test": "The unrelated phenomenon has a remote cause.",
            "competing_explanations": ("A local cause explains the phenomenon.",),
            "distinguishing_observation": (
                "A remote measurement separates the two unrelated causes."
            ),
            "rationale": "The unrelated observation distinguishes the two causes.",
        }
    )
    directions = build_research_direction_set(
        planning_input,
        ResearchDirectionLLMResponse(directions=(first,)),
        provider,
    )

    codes = {
        issue.code
        for issue in validate_research_direction_set(
            directions, planning_input
        ).issues
    }

    assert "DIRECTION_RELEVANCE_REQUIRES_HUMAN_REVIEW" in codes


def test_triage_rejects_unknown_duplicate_and_missing_direction_keys(
    tmp_path: Path,
) -> None:
    planning_input, provider, response, *_ = _hierarchy(tmp_path)
    directions = build_research_direction_set(
        planning_input,
        _two_direction_response(response),
        provider,
    )
    valid = tuple(
        DirectionTriageProposal(
            direction_key=item.direction_key,
            disposition=DirectionDisposition.RETAIN,
            reason="Direction is grounded and suitable for expansion.",
        )
        for item in directions.directions
    )

    with pytest.raises(HierarchicalPlanningError, match="UNKNOWN_TRIAGE_DIRECTION_KEY"):
        build_direction_triage_record(
            planning_input,
            directions,
            DirectionTriageLLMResponse(
                dispositions=(
                    *valid,
                    DirectionTriageProposal(
                        direction_key="direction-fake",
                        disposition=DirectionDisposition.EXCLUDE,
                        reason="Fabricated direction must not be silently discarded.",
                    ),
                )
            ),
            provider,
        )
    with pytest.raises(HierarchicalPlanningError, match="DUPLICATE_TRIAGE_DIRECTION_KEY"):
        build_direction_triage_record(
            planning_input,
            directions,
            DirectionTriageLLMResponse(dispositions=(valid[0], valid[0])),
            provider,
        )
    with pytest.raises(HierarchicalPlanningError, match="DIRECTION_TRIAGE_INCOMPLETE"):
        build_direction_triage_record(
            planning_input,
            directions,
            DirectionTriageLLMResponse(dispositions=(valid[0],)),
            provider,
        )

    triage = build_direction_triage_record(
        planning_input,
        directions,
        DirectionTriageLLMResponse(dispositions=valid),
        provider,
    )
    assert validate_direction_triage(triage, directions, planning_input).valid


@pytest.mark.parametrize(
    "non_retained_disposition",
    (DirectionDisposition.DEFER, DirectionDisposition.EXCLUDE),
)
def test_defer_and_exclude_are_audit_dispositions_not_blocking_items(
    tmp_path: Path,
    non_retained_disposition: DirectionDisposition,
) -> None:
    planning_input, provider, response, *_ = _hierarchy(tmp_path)
    directions = build_research_direction_set(
        planning_input,
        _two_direction_response(response),
        provider,
    )
    triage = build_direction_triage_record(
        planning_input,
        directions,
        DirectionTriageLLMResponse(
            dispositions=(
                DirectionTriageProposal(
                    direction_key="direction-1",
                    disposition=DirectionDisposition.RETAIN,
                    reason="Retain the grounded primary direction.",
                ),
                DirectionTriageProposal(
                    direction_key="direction-2",
                    disposition=non_retained_disposition,
                    reason="Keep this non-selected direction for audit.",
                ),
            )
        ),
        provider,
    )

    assert triage.blocking_items == ()
    expansion = build_hierarchical_expansion(
        planning_input,
        directions,
        triage,
        provider.expand_directions(planning_input, directions, triage),
        provider,
    )
    assert len(expansion.planning_proposal.candidates) == 1


def test_conflict_and_blocking_gap_cannot_disappear_in_hierarchy(tmp_path: Path) -> None:
    *_, planning_input, _, _ = build_grounded_inputs(
        tmp_path,
        "transition state barrier for CO activation pathway comparison",
        evidence_records=(
            {
                "evidence_id": "ev-positive",
                "text": "CO activation is the dominant mechanism in the pathway comparison.",
            },
            {
                "evidence_id": "ev-negative",
                "text": "CO activation is not the dominant mechanism in the pathway comparison.",
            },
        ),
    )
    assert planning_input.conflict_sets
    assert any(item.blocking for item in planning_input.evidence_gaps)
    provider = MockPlanningProvider()
    response = provider.propose_directions(planning_input)
    grounded_directions = build_research_direction_set(
        planning_input, response, provider
    )
    excluded_triage = build_direction_triage_record(
        planning_input,
        grounded_directions,
        DirectionTriageLLMResponse(
            dispositions=tuple(
                DirectionTriageProposal(
                    direction_key=item.direction_key,
                    disposition=DirectionDisposition.EXCLUDE,
                    reason="Excluded only to exercise blocking-item semantics.",
                )
                for item in grounded_directions.directions
            )
        ),
        provider,
    )
    assert any(
        item.startswith("unresolved_conflict:")
        for item in excluded_triage.blocking_items
    )
    assert any(
        item.startswith("blocking_evidence_gap:")
        for item in excluded_triage.blocking_items
    )
    assert all(":exclude" not in item for item in excluded_triage.blocking_items)
    stripped = ResearchDirectionLLMResponse(
        directions=tuple(
            item.model_copy(update={"conflict_refs": (), "blocking_gaps": ()})
            for item in response.directions
        )
    )
    directions = build_research_direction_set(planning_input, stripped, provider)

    codes = {
        item.code
        for item in validate_research_direction_set(
            directions, planning_input
        ).issues
    }
    assert "UNRESOLVED_CONFLICT_DROPPED" in codes
    assert "BLOCKING_EVIDENCE_GAP_DROPPED" in codes


class ExcludingProvider(CountingHierarchicalProvider):
    def triage_directions(self, planning_input, directions):
        type(self).triage_calls += 1
        del planning_input
        return DirectionTriageLLMResponse(
            dispositions=tuple(
                DirectionTriageProposal(
                    direction_key=item.direction_key,
                    disposition=DirectionDisposition.EXCLUDE,
                    reason="Offline test excludes this otherwise valid direction.",
                )
                for item in directions.directions
            )
        )


class RetainAndHumanChoiceProvider(CountingHierarchicalProvider):
    def propose_directions(self, planning_input):
        response = super().propose_directions(planning_input)
        return _two_direction_response(response)

    def triage_directions(self, planning_input, directions):
        type(self).triage_calls += 1
        del planning_input
        return DirectionTriageLLMResponse(
            dispositions=(
                DirectionTriageProposal(
                    direction_key=directions.directions[0].direction_key,
                    disposition=DirectionDisposition.RETAIN,
                    reason="The primary direction is grounded.",
                ),
                DirectionTriageProposal(
                    direction_key=directions.directions[1].direction_key,
                    disposition=DirectionDisposition.REQUIRES_HUMAN_CHOICE,
                    reason="The user must choose whether the independent control is in scope.",
                ),
            )
        )


class HumanChoiceOnlyProvider(CountingHierarchicalProvider):
    def triage_directions(self, planning_input, directions):
        type(self).triage_calls += 1
        del planning_input
        return DirectionTriageLLMResponse(
            dispositions=tuple(
                DirectionTriageProposal(
                    direction_key=item.direction_key,
                    disposition=DirectionDisposition.REQUIRES_HUMAN_CHOICE,
                    reason="The user must select the scientific scope.",
                )
                for item in directions.directions
            )
        )


@pytest.mark.parametrize(
    "provider_class",
    (RetainAndHumanChoiceProvider, HumanChoiceOnlyProvider),
)
def test_unresolved_human_choice_blocks_expansion_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_class: type[CountingHierarchicalProvider],
) -> None:
    repositories, *_ = _prepare(tmp_path)
    provider_class.directions_calls = 0
    provider_class.triage_calls = 0
    provider_class.expansion_calls = 0
    monkeypatch.setattr(workflow_service, "MockPlanningProvider", provider_class)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    first = workflow.start(
        REQUEST, "base", planning_strategy="hierarchical"
    ).run

    assert first.status == ScientificProblemRunStatus.HIERARCHICAL_PLANNING_BLOCKED
    assert provider_class.expansion_calls == 0
    assert first.failure is not None
    assert "requires_human_choice" in first.failure.message
    triage = workflow.runs.load_artifact(
        first.run_id,
        "hierarchical-planning/triage.yaml",
        DirectionTriageRecord,
    )
    human_items = tuple(
        item
        for item in triage.dispositions
        if item.disposition == DirectionDisposition.REQUIRES_HUMAN_CHOICE
    )
    assert human_items
    serialized_blockers = " ".join(triage.blocking_items)
    assert all(
        item.direction_id in serialized_blockers
        and item.direction_key in serialized_blockers
        and item.reason in serialized_blockers
        for item in human_items
    )
    calls = (
        provider_class.directions_calls,
        provider_class.triage_calls,
        provider_class.expansion_calls,
    )

    repeated = workflow.resume(first.run_id).run
    assert repeated.status == ScientificProblemRunStatus.HIERARCHICAL_PLANNING_BLOCKED
    assert (
        provider_class.directions_calls,
        provider_class.triage_calls,
        provider_class.expansion_calls,
    ) == calls


def test_all_directions_unavailable_is_blocked_and_resume_does_not_reinvoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories, *_ = _prepare(tmp_path)
    ExcludingProvider.directions_calls = 0
    ExcludingProvider.triage_calls = 0
    ExcludingProvider.expansion_calls = 0
    monkeypatch.setattr(workflow_service, "MockPlanningProvider", ExcludingProvider)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    first = workflow.start(
        REQUEST, "base", planning_strategy="hierarchical"
    ).run
    assert first.status == ScientificProblemRunStatus.HIERARCHICAL_PLANNING_BLOCKED
    assert ExcludingProvider.expansion_calls == 0
    calls = (ExcludingProvider.directions_calls, ExcludingProvider.triage_calls)

    repeated = workflow.resume(first.run_id).run
    assert repeated.status == ScientificProblemRunStatus.HIERARCHICAL_PLANNING_BLOCKED
    assert (ExcludingProvider.directions_calls, ExcludingProvider.triage_calls) == calls


class InterruptedDirectionProvider(MockPlanningProvider):
    directions_calls = 0

    def propose_directions(self, planning_input):
        del planning_input
        type(self).directions_calls += 1
        raise ValueError("simulated uncertain direction provider outcome")


def test_uncertain_direction_attempt_is_not_blindly_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories, *_ = _prepare(tmp_path)
    InterruptedDirectionProvider.directions_calls = 0
    monkeypatch.setattr(
        workflow_service, "MockPlanningProvider", InterruptedDirectionProvider
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    first = workflow.start(
        REQUEST, "base", planning_strategy="hierarchical"
    ).run
    assert first.status == ScientificProblemRunStatus.FAILED
    assert InterruptedDirectionProvider.directions_calls == 1

    repeated = workflow.resume(first.run_id).run
    assert repeated.status == ScientificProblemRunStatus.HIERARCHICAL_PLANNING_BLOCKED
    assert InterruptedDirectionProvider.directions_calls == 1


def test_retained_direction_is_bound_and_question_substitution_fails(
    tmp_path: Path,
) -> None:
    planning_input, provider, _, directions, triage, expansion = _hierarchy(
        tmp_path
    )
    assert validate_hierarchical_expansion(
        expansion, directions, triage, planning_input
    ).valid
    binding = expansion.candidate_bindings[0]
    assert binding.direction_id == directions.directions[0].direction_id

    response = provider.expand_directions(planning_input, directions, triage)
    changed_intent = response.intent.model_copy(
        update={"atomic_questions": ("What unrelated question should be studied?",)}
    )
    with pytest.raises(ValueError, match="EXPANSION_DIRECTION_CHANGED"):
        build_hierarchical_expansion(
            planning_input,
            directions,
            triage,
            DirectionExpansionLLMResponse(
                intent=changed_intent, candidates=response.candidates
            ),
            provider,
        )


def test_triage_does_not_replace_independent_approval(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    run = workflow.start(
        REQUEST, "base", planning_strategy="hierarchical"
    ).run

    assert run.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    assert run.approval_verdict_id is None


class BaselineRepairHierarchicalProvider(MockPlanningProvider):
    def propose(self, planning_input):
        proposal = super().propose(planning_input)
        candidate = proposal.candidates[0]
        baseline = candidate.comparison_baselines[0].model_copy(
            update={"description": "missing baseline"}
        )
        return build_proposal_set(
            planning_input,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            provider_config={"mode": "offline-test", "network": False},
            intent=proposal.intent,
            ambiguity_assessment=proposal.ambiguity_assessment,
            candidates=(
                candidate.model_copy(update={"comparison_baselines": (baseline,)}),
            ),
        )


def test_hierarchical_candidate_uses_existing_bounded_revision_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories, *_ = _prepare(tmp_path)
    monkeypatch.setattr(
        workflow_service,
        "MockPlanningProvider",
        BaselineRepairHierarchicalProvider,
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    run = workflow.start(
        REQUEST,
        "base",
        planning_strategy="hierarchical",
        approval_provider="mock",
        max_plan_revisions=1,
    ).run

    chain = workflow._load_revision_chain(run.run_id)
    assert chain is not None
    assert run.status == ScientificProblemRunStatus.APPROVED
    assert chain.revisions_used == 1
    assert chain.rounds[0].outcome == "request_revision"
    assert chain.rounds[1].round_index == 1
    revised_plan = workflow.runs.load_artifact(
        run.run_id,
        chain.rounds[1].candidate_plan.relative_path,
        ScientificQuestionPlan,
    )
    assert all(not task.runnable for task in revised_plan.tasks)
    assert workflow.runs.resolve_artifact_path(
        run.run_id, "hierarchical-planning/directions.yaml"
    ).is_file()


class ReplanningApprovalProvider(MockApprovalProvider):
    def review(self, review_input):
        response = super().review(review_input)
        flag = ApprovalHardRedFlag(
            code="REQUIRES_REPLANNING",
            severity=ApprovalRedFlagSeverity.BLOCKING,
            description="The selected research direction cannot answer the original question.",
            plan_path="hypothesis.primary",
        )
        return response.model_copy(
            update={
                "decision_recommendation": ApprovalDecision.REQUEST_REVISION,
                "hard_red_flags": (*response.hard_red_flags, flag),
            }
        )


def test_wrong_direction_records_replanning_without_restarting_hierarchy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories, *_ = _prepare(tmp_path)
    CountingHierarchicalProvider.directions_calls = 0
    CountingHierarchicalProvider.triage_calls = 0
    CountingHierarchicalProvider.expansion_calls = 0
    monkeypatch.setattr(
        workflow_service, "MockPlanningProvider", CountingHierarchicalProvider
    )
    monkeypatch.setattr(
        workflow_service, "MockApprovalProvider", ReplanningApprovalProvider
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    run = workflow.start(
        REQUEST,
        "base",
        planning_strategy="hierarchical",
        approval_provider="mock",
        max_plan_revisions=2,
    ).run
    assert run.status == ScientificProblemRunStatus.REQUIRES_REPLANNING
    chain = workflow._load_revision_chain(run.run_id)
    assert chain is not None
    assert chain.revisions_used == 0

    repeated = workflow.resume(
        run.run_id,
        approval_provider="mock",
        max_plan_revisions=2,
    ).run
    assert repeated.status == ScientificProblemRunStatus.REQUIRES_REPLANNING
    assert chain.revisions_used == 0
    assert (
        CountingHierarchicalProvider.directions_calls,
        CountingHierarchicalProvider.triage_calls,
        CountingHierarchicalProvider.expansion_calls,
    ) == (1, 1, 1)


def test_non_ft_hierarchical_fixture_and_audit_comparison(tmp_path: Path) -> None:
    pack_root = tmp_path / "packs" / "oer"
    dump_yaml(
        pack_root / "profile.yaml",
        {
            "domain_id": "oer",
            "version": "1.0.0",
            "name": "Oxygen evolution research",
            "terminology": {},
            "ontology": {},
        },
    )
    dump_yaml(
        pack_root / "capabilities.yaml",
        [
            {
                "capability_id": "oer_evidence_comparison",
                "domain": "oer",
                "scientific_goal": "Compare oxygen evolution mechanism evidence",
                "required_inputs": ["claims"],
                "outputs": ["comparison"],
                "dag_expansion": ["compare"],
            }
        ],
    )
    dump_yaml(pack_root / "expert_cases.yaml", [])
    dump_yaml(pack_root / "workflow_patterns.yaml", [])
    from spc.domains import DomainPackLoader

    loader = DomainPackLoader(search_paths=(tmp_path / "packs",))
    *_, planning_input, direct, _ = build_grounded_inputs(
        tmp_path / "oer-work",
        "Compare oxygen evolution mechanism evidence",
        evidence_records=(
            {
                "evidence_id": "ev-oer",
                "text": "Oxygen evolution mechanism evidence requires comparison.",
            },
        ),
        domain="oer",
        domain_loader=loader,
    )
    provider = MockPlanningProvider()
    directions = build_research_direction_set(
        planning_input, provider.propose_directions(planning_input), provider
    )
    triage = build_direction_triage_record(
        planning_input,
        directions,
        provider.triage_directions(planning_input, directions),
        provider,
    )
    expansion = build_hierarchical_expansion(
        planning_input,
        directions,
        triage,
        provider.expand_directions(planning_input, directions, triage),
        provider,
    )

    audit = {
        "direct": {
            "provider_call_count": 1,
            "generated_direction_count": 0,
            "retained_direction_count": 0,
            "candidate_count": len(direct.candidates),
            "blocking_items": [],
            "provenance_complete": True,
        },
        "hierarchical": {
            "provider_call_count": 3,
            "generated_direction_count": len(directions.directions),
            "retained_direction_count": sum(
                item.disposition == DirectionDisposition.RETAIN
                for item in triage.dispositions
            ),
            "candidate_count": len(expansion.planning_proposal.candidates),
            "blocking_items": list(triage.blocking_items),
            "provenance_complete": bool(
                expansion.direction_set_hash and expansion.triage_hash
            ),
        },
    }
    assert planning_input.domain == "oer"
    assert audit["direct"]["provider_call_count"] == 1
    assert audit["hierarchical"]["provider_call_count"] == 3
    assert audit["hierarchical"]["provenance_complete"] is True


def test_scientific_run_status_reports_hierarchical_audit_counts(
    tmp_path: Path,
) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )
    run = workflow.start(
        REQUEST, "base", planning_strategy="hierarchical"
    ).run

    status = scientific_run_status(run, workflow.runs)
    hierarchy = status["hierarchical_planning"]
    assert hierarchy["generated_direction_count"] >= 1
    assert hierarchy["retained_direction_count"] >= 1
    assert hierarchy["provenance_complete"] is True


def test_plan_cli_accepts_explicit_hierarchical_strategy(tmp_path: Path) -> None:
    _, context, packet, *_ = build_grounded_inputs(tmp_path)
    context_path = tmp_path / "context.yaml"
    packet_path = tmp_path / "packet.yaml"
    output_dir = tmp_path / "hierarchical-output"
    dump_yaml(context_path, context)
    dump_yaml(packet_path, packet)

    result = CliRunner().invoke(
        app,
        [
            "plan",
            str(context_path),
            str(packet_path),
            "--domain",
            "fischer_tropsch",
            "--state-dir",
            str(tmp_path / ".spc"),
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--output-dir",
            str(output_dir),
            "--planning-strategy",
            "hierarchical",
        ],
    )

    assert result.exit_code == 0, result.output
    for name in ("directions.yaml", "triage.yaml", "expansion.yaml"):
        assert (output_dir / "hierarchical-planning" / name).is_file()
