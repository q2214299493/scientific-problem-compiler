from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from spc.adapters.ft_agent import FTAgentAdapter
from spc.approval import (
    ApprovalContextResolver,
    IndependentApprovalService,
    MockApprovalProvider,
    bind_gate_verdict,
)
from spc.compiler import ScientificProblemCompiler
from spc.downstream import (
    DownstreamImportError,
    DownstreamImportValidator,
    ExecutionProposalBuilder,
)
from spc.export import GenericExportService
from spc.interpretation import MockInterpretationProvider, ScientificEvidencePacketBuilder
from spc.models import (
    AgentCapability,
    AgentCapabilityCatalog,
    AgentHandoffPackage,
    ApprovalMode,
    CapabilityBinding,
    EvidenceSpan,
    ExecutionPolicy,
    ProjectTrustPolicy,
    SPCExportPackage,
)
from spc.planning import MockPlanningProvider, PlanningContextResolver
from spc.repositories import KnowledgeRepositories, SourceEvidenceStore
from spc.retrieval import ScientificContextBuilder
from spc.serialization import content_hash
from spc.validators import build_plan_validation_record, validate_question_plan

from test_independent_approval import (
    INDEPENDENT_TRUST_POLICY,
    build_review_case,
    independently_approved,
)
from test_export import approved_inputs, run_export


def export_review_case(tmp_path: Path):
    case = build_review_case(tmp_path / "case")
    independent = independently_approved(case)
    gate = bind_gate_verdict(
        case.plan,
        independent.verdict,
        case.validation_record,
        trust_policy=INDEPENDENT_TRUST_POLICY,
        gate_id="phase3-gate",
        passed=True,
        review_input=case.review_input,
        review=independent.review,
        receipt=independent.receipt,
        compilation_receipt=case.compilation_receipt,
    )
    output = GenericExportService(tmp_path / "exports", case.store).export(
        plan=case.plan,
        verdict=independent.verdict,
        validation_record=case.validation_record,
        gate=gate,
        human_selected=True,
        adapter=FTAgentAdapter(),
        export_id="phase3-export",
        trust_policy=INDEPENDENT_TRUST_POLICY,
        compilation_receipt=case.compilation_receipt,
        review_input=case.review_input,
        review=independent.review,
        receipt=independent.receipt,
    )
    return case, output


def test_tampered_export_is_rejected(tmp_path) -> None:
    _, output = export_review_case(tmp_path)
    plan_path = output / "selected-plan.yaml"
    plan_path.write_text(
        plan_path.read_text(encoding="utf-8") + "\n# modified downstream\n",
        encoding="utf-8",
    )
    with pytest.raises(DownstreamImportError) as caught:
        DownstreamImportValidator().import_package(output)
    assert "CHECKSUM_MISMATCH" in {issue.code for issue in caught.value.report.issues}


def test_wrong_target_agent_is_rejected(tmp_path) -> None:
    _, output = export_review_case(tmp_path)
    with pytest.raises(DownstreamImportError) as caught:
        DownstreamImportValidator().import_package(
            output,
            expected_target_agent="different-agent",
        )
    assert "WRONG_TARGET_AGENT" in {issue.code for issue in caught.value.report.issues}


def test_legacy_manual_export_is_rejected(
    tmp_path, make_plan, evidence_repository
) -> None:
    plan = make_plan()
    verdict, validation_record, gate = approved_inputs(plan, evidence_repository)
    output = run_export(
        tmp_path,
        plan,
        evidence_repository,
        verdict,
        validation_record,
        gate,
    )
    with pytest.raises(DownstreamImportError) as caught:
        DownstreamImportValidator().import_package(output)
    assert "INDEPENDENT_APPROVAL_REQUIRED" in {
        issue.code for issue in caught.value.report.issues
    }


def test_missing_capability_mapping_is_rejected(tmp_path) -> None:
    _, output = export_review_case(tmp_path)
    package = DownstreamImportValidator().import_package(output)
    catalog = AgentCapabilityCatalog(agent_id="ft-agent", version="empty", capabilities=())
    with pytest.raises(DownstreamImportError) as caught:
        ExecutionProposalBuilder().build(
            package,
            task_id=package.source_plan.tasks[0].task_id,
            catalog=catalog,
            adapter=FTAgentAdapter(catalog),
            target_environment="ft-agent-staging",
        )
    assert "MISSING_CAPABILITY_MAPPING" in {
        issue.code for issue in caught.value.report.issues
    }


def test_plan_a_export_cannot_propose_task_for_plan_b(tmp_path) -> None:
    _, output = export_review_case(tmp_path)
    package = DownstreamImportValidator().import_package(output)
    plan_b = package.source_plan.model_copy(update={"plan_id": "plan-b"})
    data = {
        field_name: getattr(package, field_name)
        for field_name in type(package).model_fields
    }
    data.update({"source_plan": plan_b, "source_plan_hash": content_hash(plan_b)})
    forged = SPCExportPackage.model_construct(**data)
    with pytest.raises(DownstreamImportError) as caught:
        ExecutionProposalBuilder().build(
            forged,
            task_id=plan_b.tasks[0].task_id,
            catalog=FTAgentAdapter().catalog,
            adapter=FTAgentAdapter(),
            target_environment="ft-agent-staging",
        )
    assert "DOWNSTREAM_PACKAGE_INVALID" in {
        issue.code for issue in caught.value.report.issues
    }


def test_execution_proposal_is_unauthorized_and_non_runnable(tmp_path) -> None:
    _, output = export_review_case(tmp_path)
    package = DownstreamImportValidator().import_package(output)
    adapter = FTAgentAdapter()
    proposal = ExecutionProposalBuilder().build(
        package,
        task_id=package.source_plan.tasks[0].task_id,
        catalog=adapter.catalog,
        adapter=adapter,
        target_environment="ft-agent-staging",
    )
    assert proposal.authorized is False
    assert proposal.runnable is False
    assert proposal.source_plan_hash == package.source_plan_hash
    assert proposal.gate_hash == package.gate_hash
    assert proposal.approval_receipt_hash == package.approval_receipt.content_hash
    assert proposal.agent_capability_catalog_hash == content_hash(adapter.catalog)
    with pytest.raises(ValidationError, match="unauthorized and non-runnable"):
        proposal.model_copy(update={"authorized": True})


class InspectingAdapter:
    adapter_id = "generic.inspecting-adapter"
    adapter_version = "1.0.0"
    target_agent = "ft-agent"
    supported_environments = ("inspection-only",)

    def __init__(self) -> None:
        self.received: tuple[object, ...] = ()

    def map_capability(self, scientific_capability_id, catalog):
        self.received = (scientific_capability_id, catalog)
        return next(
            item.capability_id
            for item in catalog.capabilities
            if scientific_capability_id in item.supports_scientific_capability_ids
        )

    def resource_requirements(self, executable_capability_id, target_environment):
        self.received += (executable_capability_id, target_environment)
        return {"allocation": "none"}


def test_execution_adapter_never_receives_scientific_plan(tmp_path) -> None:
    _, output = export_review_case(tmp_path)
    package = DownstreamImportValidator().import_package(output)
    plan_hash_before = content_hash(package.source_plan)
    adapter = InspectingAdapter()
    ExecutionProposalBuilder().build(
        package,
        task_id=package.source_plan.tasks[0].task_id,
        catalog=FTAgentAdapter().catalog,
        adapter=adapter,
        target_environment="inspection-only",
    )
    assert all(not isinstance(value, type(package.source_plan)) for value in adapter.received)
    assert content_hash(package.source_plan) == plan_hash_before


@dataclass(frozen=True)
class GenericReviewCase:
    store: SourceEvidenceStore
    plan: object
    validation_record: object
    review_input: object
    compilation_receipt: object


class GenericAgentAdapter:
    target_agent = "generic-science-agent"
    supported_domains = ("base",)
    adapter_id = "generic.catalog-adapter"
    adapter_version = "1.0.0"
    supported_environments = ("generic-staging",)

    def __init__(self, catalog: AgentCapabilityCatalog) -> None:
        self.catalog = catalog

    def bind_capabilities(self, plan):
        return tuple(
            CapabilityBinding(
                scientific_capability_id=capability_id,
                target_capability_id=f"generic.{capability_id}",
                status="available",
            )
            for capability_id in plan.scientific_capability_ids
        )

    def build_handoff(self, plan, export_id):
        return AgentHandoffPackage(
            export_id=export_id,
            target_agent=self.target_agent,
            source_plan_id=plan.plan_id,
            source_plan_version=plan.version,
            source_plan_hash=content_hash(plan),
            capability_bindings=self.bind_capabilities(plan),
            execution_policy=ExecutionPolicy(),
        )

    def map_capability(self, scientific_capability_id, catalog):
        match = [
            item.capability_id
            for item in catalog.capabilities
            if scientific_capability_id in item.supports_scientific_capability_ids
        ]
        return match[0] if len(match) == 1 else None

    def resource_requirements(self, executable_capability_id, target_environment):
        return {
            "capability": executable_capability_id,
            "environment": target_environment,
            "allocation": "not_authorized",
        }


def build_generic_review_case(tmp_path: Path):
    text = "Compare two declared models using one evidence-grounded observable."
    source_path = tmp_path / "source.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(text, encoding="utf-8")
    store = SourceEvidenceStore(tmp_path / ".spc")
    source = store.ingest(source_path, "source-generic", "v1")
    store.add_evidence(
        EvidenceSpan(
            evidence_id="ev-generic",
            source_id=source.source_id,
            source_version=source.version,
            content_sha256=source.content_sha256,
            start_offset=0,
            end_offset=len(text),
            text=text,
        )
    )
    knowledge_dir = tmp_path / "knowledge"
    context = ScientificContextBuilder().build(
        "Compare two declared scientific models",
        "base",
        state_dir=tmp_path / ".spc",
        knowledge_dir=knowledge_dir,
    )
    evidence_packet = ScientificEvidencePacketBuilder(
        MockInterpretationProvider()
    ).build(context, store)
    knowledge = KnowledgeRepositories(knowledge_dir)
    planning_input = PlanningContextResolver().resolve(
        context, evidence_packet, knowledge, store
    )
    compilation = ScientificProblemCompiler(
        MockPlanningProvider(), evidence_repository=store
    ).compile(planning_input)
    plan = compilation.candidates[0]
    report = validate_question_plan(plan, planning_input.scientific_capabilities, store)
    validation_record = build_plan_validation_record(
        plan, report, validation_id="validation-generic"
    )
    review_input = ApprovalContextResolver().resolve(
        context,
        evidence_packet,
        planning_input,
        plan,
        validation_record,
        knowledge,
        store,
    )
    return GenericReviewCase(
        store=store,
        plan=plan,
        validation_record=validation_record,
        review_input=review_input,
        compilation_receipt=compilation.compilation_receipts[0],
    )


def test_generic_non_ft_fixture_creates_proposal(tmp_path) -> None:
    case = build_generic_review_case(tmp_path / "generic-case")
    independent = IndependentApprovalService(
        MockApprovalProvider(), approver_id="generic-independent-reviewer"
    ).review(case.review_input)
    trust_policy = ProjectTrustPolicy(
        approval_mode=ApprovalMode.INDEPENDENT_REQUIRED,
        policy_version="1.0.0",
    )
    gate = bind_gate_verdict(
        case.plan,
        independent.verdict,
        case.validation_record,
        trust_policy=trust_policy,
        gate_id="generic-gate",
        passed=True,
        review_input=case.review_input,
        review=independent.review,
        receipt=independent.receipt,
        compilation_receipt=case.compilation_receipt,
    )
    capabilities = tuple(
        AgentCapability(
            capability_id=f"generic.{capability_id}",
            version="1.0.0",
            supports_scientific_capability_ids=(capability_id,),
        )
        for capability_id in case.plan.scientific_capability_ids
    )
    catalog = AgentCapabilityCatalog(
        agent_id="generic-science-agent",
        version="1.0.0",
        capabilities=capabilities,
    )
    adapter = GenericAgentAdapter(catalog)
    output = GenericExportService(tmp_path / "exports", case.store).export(
        plan=case.plan,
        verdict=independent.verdict,
        validation_record=case.validation_record,
        gate=gate,
        human_selected=True,
        adapter=adapter,
        export_id="generic-export",
        trust_policy=trust_policy,
        compilation_receipt=case.compilation_receipt,
        review_input=case.review_input,
        review=independent.review,
        receipt=independent.receipt,
    )
    package = DownstreamImportValidator().import_package(
        output, expected_target_agent="generic-science-agent"
    )
    proposal = ExecutionProposalBuilder().build(
        package,
        task_id=case.plan.tasks[0].task_id,
        catalog=catalog,
        adapter=adapter,
        target_environment="generic-staging",
    )
    assert proposal.target_agent == "generic-science-agent"
    assert proposal.scientific_capability_id in case.plan.scientific_capability_ids
    assert proposal.runnable is False
