from __future__ import annotations

from pathlib import Path

import pytest

from spc.domains import DomainPackLoader
from spc.interpretation import MockInterpretationProvider, ScientificEvidencePacketBuilder
from spc.models import (
    ApprovalDecision,
    CurationStatus,
    EvidenceGap,
    EvidenceGapResolutionKind,
    EvidenceResolutionStatus,
    PlanningEvidenceRequestLLMResponse,
    PlanningEvidenceType,
    PlanningStrategy,
    InterpretationProposal,
    RequiredFix,
    ScientificPlanningInput,
    ScientificProblemRunStatus,
)
from spc.planning import (
    MockPlanningProvider,
    PlanningContextResolver,
    PlanningEvidenceError,
    augment_scientific_context,
    build_planning_evidence_request_set,
    classify_evidence_gap,
    proposal_for_gap,
    resolve_planning_evidence_requests,
)
from spc.retrieval import PersistentKnowledgeRetriever, ScientificContextBuilder
from spc.serialization import content_hash
from spc.workflow import ScientificProblemWorkflow, scientific_run_status
import spc.workflow.service as workflow_service
from test_knowledge_retrieval import _prepare


def _snapshot_hash(snapshot: object) -> str:
    return content_hash(
        snapshot.model_dump(mode="json", exclude={"created_at"})  # type: ignore[attr-defined]
    )


def _planning_input_with_gap(
    planning_input: ScientificPlanningInput,
    gap: EvidenceGap,
) -> ScientificPlanningInput:
    identity = planning_input.model_dump(
        mode="python", exclude={"planning_input_id", "content_hash"}
    )
    identity["evidence_gaps"] = (gap,)
    planning_input_id = f"planning-input-{content_hash(identity)[:24]}"
    payload = {"planning_input_id": planning_input_id, **identity}
    return ScientificPlanningInput(**payload, content_hash=content_hash(payload))


def _grounded_gap_inputs(
    tmp_path: Path,
    *,
    claim_status: CurationStatus = CurationStatus.ACCEPTED,
    question: str = "Which reported method and comparison condition applies?",
    missing: str = "The literature method and comparison condition are not established.",
):
    repositories, store, *_ = _prepare(tmp_path, claim_status=claim_status)
    loader = DomainPackLoader()
    context = ScientificContextBuilder(loader).build(
        "Which surface ensemble should define the comparison?",
        "base",
        state_dir=tmp_path / ".spc",
        knowledge_dir=repositories.root,
    )
    packet = ScientificEvidencePacketBuilder(MockInterpretationProvider()).build(
        context, store
    )
    planning_input = PlanningContextResolver(loader).resolve(
        context, packet, repositories, store
    )
    gap = EvidenceGap(
        gap_id="gap-method-comparison",
        scientific_question=question,
        missing_evidence=missing,
        why_it_matters="The results cannot be compared without the reported context.",
        blocking=True,
        evidence_refs=(),
    )
    return (
        repositories,
        store,
        loader,
        context,
        _planning_input_with_gap(planning_input, gap),
        gap,
    )


def _request_set(
    planning_input: ScientificPlanningInput,
    context,
    *,
    source_acquisition_allowed: bool = False,
    response: PlanningEvidenceRequestLLMResponse | None = None,
):
    provider = MockPlanningProvider()
    return build_planning_evidence_request_set(
        planning_input,
        response or provider.propose_evidence_requests(planning_input),
        provider,
        planning_strategy=PlanningStrategy.DIRECT,
        knowledge_snapshot_hash=_snapshot_hash(context.knowledge_snapshot),
        source_acquisition_allowed=source_acquisition_allowed,
    )


class GapUntilMethodRetrievedProvider(MockInterpretationProvider):
    def interpret(self, context):
        proposal = super().interpret(context)
        retrieved_ids = {
            hit.record_id
            for hits in (
                context.literature_knowledge_hits,
                context.graph_expanded_hits,
            )
            for hit in hits
        }
        gaps = proposal.evidence_gaps
        if "method-k1h-dft" not in retrieved_ids:
            gaps = (
                EvidenceGap(
                    gap_id="gap-method-comparison",
                    scientific_question=(
                        "Which reported method and model comparison condition applies?"
                    ),
                    missing_evidence=(
                        "The literature method and comparison condition are not established."
                    ),
                    why_it_matters=(
                        "The competing explanations cannot be compared on unmatched methods."
                    ),
                    blocking=True,
                ),
            )
        payload = proposal.model_dump(mode="python", exclude={"proposal_id"})
        payload["evidence_gaps"] = gaps
        return InterpretationProposal(
            proposal_id=f"interpretation-{content_hash(payload)[:24]}",
            **payload,
        )


class CountingHierarchicalEvidenceProvider(MockPlanningProvider):
    directions_calls = 0
    triage_calls = 0
    expansion_calls = 0
    evidence_request_calls = 0

    def propose_evidence_requests(self, planning_input):
        type(self).evidence_request_calls += 1
        return super().propose_evidence_requests(planning_input)

    def propose_directions(self, planning_input):
        type(self).directions_calls += 1
        return super().propose_directions(planning_input)

    def triage_directions(self, planning_input, directions):
        type(self).triage_calls += 1
        return super().triage_directions(planning_input, directions)

    def expand_directions(self, planning_input, directions, triage):
        type(self).expansion_calls += 1
        return super().expand_directions(planning_input, directions, triage)


class InsufficientThenApproveProvider(workflow_service.MockApprovalProvider):
    review_calls = 0

    def review(self, review_input):
        type(self).review_calls += 1
        response = super().review(review_input)
        if type(self).review_calls != 1:
            return response
        return response.model_copy(
            update={
                "decision_recommendation": ApprovalDecision.INSUFFICIENT_EVIDENCE,
                "summary": (
                    "Trusted literature source grounding for the reported method "
                    "comparison condition is insufficient."
                ),
                "required_fixes": (
                    RequiredFix(
                        fix_id="fix-reported-method-source",
                        description=(
                            "Retrieve the literature-reported method and comparison "
                            "condition from a trusted source."
                        ),
                        blocking=True,
                    ),
                ),
            }
        )


class RequestRevisionApprovalProvider(workflow_service.MockApprovalProvider):
    def review(self, review_input):
        return super().review(review_input).model_copy(
            update={"decision_recommendation": ApprovalDecision.REQUEST_REVISION}
        )


@pytest.mark.parametrize(
    ("missing", "expected"),
    (
        (
            "The paper does not report the slab thickness method condition.",
            EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE,
        ),
        (
            "A new DFT calculation is required for the true barrier in this system.",
            EvidenceGapResolutionKind.CALCULATION_REQUIRED,
        ),
        (
            "The user must choose whether the carbide phase is in scope.",
            EvidenceGapResolutionKind.HUMAN_DECISION_REQUIRED,
        ),
    ),
)
def test_evidence_gap_classification_is_bounded(
    missing: str,
    expected: EvidenceGapResolutionKind,
) -> None:
    gap = EvidenceGap(
        gap_id="gap-classification",
        scientific_question="What is needed?",
        missing_evidence=missing,
        why_it_matters="It changes the planning decision.",
        blocking=True,
    )

    assert classify_evidence_gap(gap).resolution_kind == expected


def test_request_validation_rejects_fabricated_refs_and_deduplicates(
    tmp_path: Path,
) -> None:
    _, _, _, context, planning_input, gap = _grounded_gap_inputs(tmp_path)
    proposal = proposal_for_gap(gap, planning_input)
    duplicate = proposal.model_copy(update={"request_key": "same-gap-rephrased"})
    request_set = _request_set(
        planning_input,
        context,
        response=PlanningEvidenceRequestLLMResponse(
            requests=(proposal, duplicate)
        ),
    )
    assert len(request_set.requests) == 1

    for field_name, bad_value, code in (
        ("gap_id", "gap-fabricated", "FABRICATED_EVIDENCE_GAP_ID"),
        (
            "related_claim_refs",
            ("claim-fabricated",),
            "FABRICATED_EVIDENCE_REQUEST_CLAIM_ID",
        ),
        (
            "related_evidence_refs",
            ("ev-fabricated",),
            "FABRICATED_EVIDENCE_REQUEST_EVIDENCE_ID",
        ),
    ):
        invalid = proposal.model_copy(update={field_name: bad_value})
        with pytest.raises(PlanningEvidenceError, match=code):
            _request_set(
                planning_input,
                context,
                response=PlanningEvidenceRequestLLMResponse(requests=(invalid,)),
            )


def test_trusted_match_uses_k1h_without_acquisition_and_rebuilds_context(
    tmp_path: Path,
) -> None:
    repositories, store, loader, context, planning_input, _ = _grounded_gap_inputs(
        tmp_path
    )
    request_set = _request_set(
        planning_input, context, source_acquisition_allowed=True
    )

    resolutions = resolve_planning_evidence_requests(
        request_set,
        repositories,
        store,
        loader.load("base").profile,
        context.knowledge_snapshot,
    )

    assert resolutions.acquisition_proposals == ()
    assert all(
        item.status
        in {
            EvidenceResolutionStatus.RESOLVED_TRUSTED,
            EvidenceResolutionStatus.CONFLICTING_EVIDENCE,
        }
        for item in resolutions.records
    )
    child_context = augment_scientific_context(context, resolutions, store)
    child_packet = ScientificEvidencePacketBuilder(MockInterpretationProvider()).build(
        child_context, store
    )
    child_input = PlanningContextResolver(loader).resolve(
        child_context, child_packet, repositories, store
    )
    assert child_context.context_id != context.context_id
    assert child_context.content_hash != context.content_hash
    assert child_input.planning_input_id != planning_input.planning_input_id
    assert child_input.content_hash != planning_input.content_hash


def test_no_match_acquisition_requires_explicit_policy_and_does_not_claim_novelty(
    tmp_path: Path,
) -> None:
    repositories, store, loader, context, planning_input, gap = _grounded_gap_inputs(
        tmp_path,
        question="Which reported unobtainium spectroscopy condition applies?",
        missing="No source reports the unobtainium spectroscopy condition.",
    )
    proposal = proposal_for_gap(gap, planning_input).model_copy(
        update={
            "evidence_type_needed": PlanningEvidenceType.METHOD_FACT,
            "concepts": ("unobtainium", "spectroscopy"),
            "comparison_conditions": ("unobtainium phase omega",),
            "acceptable_source_classes": ("expert_opinion",),
        }
    )
    response = PlanningEvidenceRequestLLMResponse(requests=(proposal,))
    without_policy = _request_set(
        planning_input,
        context,
        response=response,
        source_acquisition_allowed=False,
    )
    without = resolve_planning_evidence_requests(
        without_policy,
        repositories,
        store,
        loader.load("base").profile,
        context.knowledge_snapshot,
    )
    assert without.records[0].status == EvidenceResolutionStatus.NO_MATCH
    assert without.acquisition_proposals == ()
    assert "novelty" in " ".join(without.records[0].remaining_uncertainty).lower()

    with_policy = _request_set(
        planning_input,
        context,
        response=response,
        source_acquisition_allowed=True,
    )
    with_acquisition = resolve_planning_evidence_requests(
        with_policy,
        repositories,
        store,
        loader.load("base").profile,
        context.knowledge_snapshot,
    )
    assert len(with_acquisition.acquisition_proposals) == 1
    assert with_acquisition.acquisition_proposals[0].status == "proposed_not_executed"


def test_machine_extracted_match_requires_curation_and_never_enters_context(
    tmp_path: Path,
) -> None:
    repositories, store, loader, context, planning_input, _ = _grounded_gap_inputs(
        tmp_path,
        claim_status=CurationStatus.MACHINE_EXTRACTED,
    )
    request_set = _request_set(planning_input, context)

    resolutions = resolve_planning_evidence_requests(
        request_set,
        repositories,
        store,
        loader.load("base").profile,
        context.knowledge_snapshot,
    )

    assert resolutions.records[0].status == EvidenceResolutionStatus.REQUIRES_SOURCE_CURATION
    with pytest.raises(PlanningEvidenceError, match="no trusted matches"):
        augment_scientific_context(context, resolutions, store)


def test_conflicting_trusted_matches_remain_explicit(tmp_path: Path) -> None:
    repositories, store, loader, context, planning_input, gap = _grounded_gap_inputs(
        tmp_path,
        question="Which prior source claims distinguish the competing mechanism?",
        missing="Prior literature source claims about the mechanism are required.",
    )
    proposal = proposal_for_gap(gap, planning_input).model_copy(
        update={
            "evidence_type_needed": PlanningEvidenceType.PRIOR_MECHANISTIC_EVIDENCE,
            "concepts": ("competing", "mechanism", "activation", "barrier"),
            "comparison_conditions": ("competing mechanism",),
            "acceptable_source_classes": ("supporting_context",),
        }
    )
    request_set = _request_set(
        planning_input,
        context,
        response=PlanningEvidenceRequestLLMResponse(requests=(proposal,)),
    )

    resolutions = resolve_planning_evidence_requests(
        request_set,
        repositories,
        store,
        loader.load("base").profile,
        context.knowledge_snapshot,
    )

    assert resolutions.records[0].status == EvidenceResolutionStatus.CONFLICTING_EVIDENCE
    assert resolutions.records[0].retrieval_context is not None
    child = augment_scientific_context(context, resolutions, store)
    assert child.conflicting_evidence


def test_request_resolution_rejects_snapshot_or_domain_mismatch(tmp_path: Path) -> None:
    repositories, store, loader, context, planning_input, _ = _grounded_gap_inputs(
        tmp_path
    )
    request_set = _request_set(planning_input, context)
    def rebound(**updates):
        identity = request_set.model_dump(
            mode="python", exclude={"request_set_id", "content_hash"}
        )
        identity.update(updates)
        request_set_id = (
            f"planning-evidence-request-set-{content_hash(identity)[:24]}"
        )
        payload = {"request_set_id": request_set_id, **identity}
        return type(request_set)(**payload, content_hash=content_hash(payload))

    bad_snapshot = rebound(knowledge_snapshot_hash="0" * 64)

    with pytest.raises(PlanningEvidenceError, match="SNAPSHOT_MISMATCH"):
        resolve_planning_evidence_requests(
            bad_snapshot,
            repositories,
            store,
            loader.load("base").profile,
            context.knowledge_snapshot,
        )
    with pytest.raises(PlanningEvidenceError, match="DOMAIN_MISMATCH"):
        resolve_planning_evidence_requests(
            rebound(domain="other-domain"),
            repositories,
            store,
            loader.load("base").profile,
            context.knowledge_snapshot,
        )


def test_default_workflow_keeps_evidence_resolution_disabled(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    run = workflow.start(
        "The mechanism is not sufficiently convincing without a distinguishing observation.",
        "base",
    ).run

    assert not workflow.runs.resolve_artifact_path(
        run.run_id, "planning-evidence"
    ).exists()
    assert run.planning_input_id is not None


def test_offline_hierarchical_evidence_cycle_replans_and_awaits_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repositories, *_ = _prepare(tmp_path)
    CountingHierarchicalEvidenceProvider.directions_calls = 0
    CountingHierarchicalEvidenceProvider.triage_calls = 0
    CountingHierarchicalEvidenceProvider.expansion_calls = 0
    CountingHierarchicalEvidenceProvider.evidence_request_calls = 0
    evidence_query_calls = 0
    original_retrieve = PersistentKnowledgeRetriever.retrieve

    def counted_retrieve(self, query, *args, **kwargs):
        nonlocal evidence_query_calls
        if "reported method" in query.raw_request.casefold():
            evidence_query_calls += 1
        return original_retrieve(self, query, *args, **kwargs)

    monkeypatch.setattr(PersistentKnowledgeRetriever, "retrieve", counted_retrieve)
    monkeypatch.setattr(
        workflow_service,
        "MockInterpretationProvider",
        GapUntilMethodRetrievedProvider,
    )
    monkeypatch.setattr(
        workflow_service,
        "MockPlanningProvider",
        CountingHierarchicalEvidenceProvider,
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    result = workflow.start(
        "The mechanism is not sufficiently convincing without a distinguishing observation.",
        "base",
        planning_strategy="hierarchical",
        max_evidence_resolution_cycles=1,
    )

    assert result.run.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    assert CountingHierarchicalEvidenceProvider.evidence_request_calls == 1
    assert CountingHierarchicalEvidenceProvider.directions_calls == 2
    assert CountingHierarchicalEvidenceProvider.triage_calls == 1
    assert CountingHierarchicalEvidenceProvider.expansion_calls == 1
    assert evidence_query_calls == 1
    cycles = workflow._load_evidence_cycles(result.run.run_id)
    assert len(cycles) == 1
    cycle = cycles[0]
    assert cycle.child_context_id == result.run.context_id
    assert cycle.child_planning_input_id == result.run.planning_input_id
    assert cycle.parent_context_id != cycle.child_context_id
    assert workflow.runs.resolve_artifact_path(
        result.run.run_id, "planning-evidence/cycle-1/trigger-directions.yaml"
    ).is_file()
    assert workflow.runs.resolve_artifact_path(
        result.run.run_id, "evidence-hierarchy-1/directions.yaml"
    ).is_file()
    assert result.run.approval_receipt_id is None
    status = scientific_run_status(result.run, workflow.runs)
    assert status["evidence_resolution"] == {
        "max_cycles": 1,
        "cycles_used": 1,
        "requests_generated": 1,
        "trusted_matches": 1,
        "conflicting_matches": 0,
        "no_matches": 0,
        "curation_required": 0,
        "unresolved_requests": [],
    }
    plan = workflow.runs.load_artifact(
        result.run.run_id,
        result.run.candidate_plans[0].relative_path,
        workflow_service.ScientificQuestionPlan,
    )
    assert all(not task.runnable for task in plan.tasks)

    before = (
        CountingHierarchicalEvidenceProvider.directions_calls,
        CountingHierarchicalEvidenceProvider.triage_calls,
        CountingHierarchicalEvidenceProvider.expansion_calls,
        CountingHierarchicalEvidenceProvider.evidence_request_calls,
    )
    resumed = workflow.resume(result.run.run_id).run
    assert resumed.status == ScientificProblemRunStatus.AWAITING_APPROVAL
    assert (
        CountingHierarchicalEvidenceProvider.directions_calls,
        CountingHierarchicalEvidenceProvider.triage_calls,
        CountingHierarchicalEvidenceProvider.expansion_calls,
        CountingHierarchicalEvidenceProvider.evidence_request_calls,
    ) == before
    assert evidence_query_calls == 1


def test_nonretrieval_gap_stops_without_retrieval_and_resume_stays_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CalculationGapProvider(MockInterpretationProvider):
        def interpret(self, context):
            proposal = super().interpret(context)
            gap = EvidenceGap(
                gap_id="gap-new-calculation",
                scientific_question="What is the true barrier in this system?",
                missing_evidence="A new DFT calculation is required.",
                why_it_matters="The system-specific barrier is unknown.",
                blocking=True,
            )
            payload = proposal.model_dump(mode="python", exclude={"proposal_id"})
            payload["evidence_gaps"] = (gap,)
            return InterpretationProposal(
                proposal_id=f"interpretation-{content_hash(payload)[:24]}",
                **payload,
            )

    repositories, *_ = _prepare(tmp_path)
    CountingHierarchicalEvidenceProvider.evidence_request_calls = 0
    monkeypatch.setattr(
        workflow_service, "MockInterpretationProvider", CalculationGapProvider
    )
    monkeypatch.setattr(
        workflow_service,
        "MockPlanningProvider",
        CountingHierarchicalEvidenceProvider,
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    first = workflow.start(
        "The mechanism is not sufficiently convincing without a distinguishing observation.",
        "base",
        max_evidence_resolution_cycles=1,
    ).run
    assert first.status == ScientificProblemRunStatus.EVIDENCE_RESOLUTION_BLOCKED
    assert first.failure is not None
    assert first.failure.category == "NOT_RETRIEVAL_RESOLVABLE"
    assert CountingHierarchicalEvidenceProvider.evidence_request_calls == 0

    for _ in range(2):
        resumed = workflow.resume(first.run_id).run
        assert resumed.status == ScientificProblemRunStatus.EVIDENCE_RESOLUTION_BLOCKED
    assert CountingHierarchicalEvidenceProvider.evidence_request_calls == 0
    assert workflow._evidence_attempt_count(first.run_id) == 1


def test_insufficient_evidence_review_starts_new_bound_planning_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repositories, *_ = _prepare(tmp_path)
    InsufficientThenApproveProvider.review_calls = 0
    monkeypatch.setattr(
        workflow_service,
        "MockApprovalProvider",
        InsufficientThenApproveProvider,
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    run = workflow.start(
        "The mechanism is not sufficiently convincing without a distinguishing observation.",
        "base",
        approval_provider="mock",
        max_plan_revisions=1,
        max_evidence_resolution_cycles=1,
    ).run

    assert run.status == ScientificProblemRunStatus.APPROVED
    assert InsufficientThenApproveProvider.review_calls == 2
    cycles = workflow._load_evidence_cycles(run.run_id)
    assert len(cycles) == 1
    request_set = workflow.runs.load_artifact(
        run.run_id,
        "planning-evidence/cycle-1/request-set.yaml",
        workflow_service.PlanningEvidenceRequestSet,
    )
    assert request_set.triggering_review_id is not None
    assert request_set.triggering_review_hash is not None
    assert request_set.requests[0].triggering_question is not None
    assert workflow.runs.resolve_artifact_path(
        run.run_id,
        "planning-evidence/cycle-1/trigger/approval-review.yaml",
    ).is_file()
    triggering_review = workflow.runs.load_artifact(
        run.run_id,
        "planning-evidence/cycle-1/trigger/approval-review.yaml",
        workflow_service.ApprovalReviewRecord,
    )
    assert triggering_review.review_id == request_set.triggering_review_id
    assert triggering_review.content_hash == request_set.triggering_review_hash
    assert run.approval_review_id != triggering_review.review_id
    chain = workflow._load_revision_chain(run.run_id)
    assert chain is not None
    assert chain.context_id == cycles[0].child_context_id


def test_uncertain_evidence_request_attempt_is_not_repeated_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingEvidenceProvider(MockPlanningProvider):
        request_calls = 0

        def propose_evidence_requests(self, planning_input, triggering_review=None):
            del planning_input, triggering_review
            type(self).request_calls += 1
            raise RuntimeError("simulated provider interruption")

    repositories, *_ = _prepare(tmp_path)
    monkeypatch.setattr(
        workflow_service,
        "MockInterpretationProvider",
        GapUntilMethodRetrievedProvider,
    )
    monkeypatch.setattr(
        workflow_service, "MockPlanningProvider", ExplodingEvidenceProvider
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    with pytest.raises(RuntimeError, match="simulated provider interruption"):
        workflow.start(
            "The mechanism is not sufficiently convincing without a distinguishing observation.",
            "base",
            max_evidence_resolution_cycles=1,
        )
    run = workflow.runs.list()[0]
    assert ExplodingEvidenceProvider.request_calls == 1
    assert workflow._evidence_attempt_count(run.run_id) == 1

    for _ in range(3):
        resumed = workflow.resume(run.run_id).run
        assert resumed.status == ScientificProblemRunStatus.EVIDENCE_RESOLUTION_BLOCKED
        assert resumed.failure is not None
        assert resumed.failure.category == "EVIDENCE_REQUEST_OUTCOME_UNCERTAIN"
    assert ExplodingEvidenceProvider.request_calls == 1


def test_evidence_cycle_budget_stops_persistent_gap(tmp_path: Path, monkeypatch) -> None:
    class PersistentGapProvider(GapUntilMethodRetrievedProvider):
        def interpret(self, context):
            proposal = super().interpret(context)
            gap = EvidenceGap(
                gap_id="gap-method-comparison",
                scientific_question=(
                    "Which reported method and model comparison condition applies?"
                ),
                missing_evidence=(
                    "The literature method and comparison condition are not established."
                ),
                why_it_matters="The comparison remains blocked.",
                blocking=True,
            )
            payload = proposal.model_dump(mode="python", exclude={"proposal_id"})
            payload["evidence_gaps"] = (gap,)
            return InterpretationProposal(
                proposal_id=f"interpretation-{content_hash(payload)[:24]}",
                **payload,
            )

    repositories, *_ = _prepare(tmp_path)
    CountingHierarchicalEvidenceProvider.evidence_request_calls = 0
    monkeypatch.setattr(
        workflow_service, "MockInterpretationProvider", PersistentGapProvider
    )
    monkeypatch.setattr(
        workflow_service,
        "MockPlanningProvider",
        CountingHierarchicalEvidenceProvider,
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    run = workflow.start(
        "The mechanism is not sufficiently convincing without a distinguishing observation.",
        "base",
        max_evidence_resolution_cycles=1,
    ).run

    assert run.status == ScientificProblemRunStatus.EVIDENCE_RESOLUTION_BLOCKED
    assert run.failure is not None
    assert run.failure.category == "EVIDENCE_RESOLUTION_BUDGET_EXHAUSTED"
    assert workflow._evidence_attempt_count(run.run_id) == 1
    assert CountingHierarchicalEvidenceProvider.evidence_request_calls == 1


def test_request_revision_does_not_enter_evidence_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repositories, *_ = _prepare(tmp_path)
    monkeypatch.setattr(
        workflow_service,
        "MockApprovalProvider",
        RequestRevisionApprovalProvider,
    )
    workflow = ScientificProblemWorkflow(
        state_dir=tmp_path / ".spc", knowledge_dir=repositories.root
    )

    run = workflow.start(
        "The mechanism is not sufficiently convincing without a distinguishing observation.",
        "base",
        approval_provider="mock",
        max_evidence_resolution_cycles=1,
    ).run

    assert run.status == ScientificProblemRunStatus.REJECTED
    assert workflow._evidence_attempt_count(run.run_id) == 0
