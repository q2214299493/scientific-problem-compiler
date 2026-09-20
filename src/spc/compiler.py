from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domains import DomainPackLoader
from .models import (
    PlanCompilationReceipt,
    PlanRevisionInput,
    PlanRevisionLLMResponse,
    PlanningProposalSet,
    ScientificPlanningInput,
    ScientificQuestionPlan,
)
from .planning.materializer import PlanMaterializer
from .planning.revision import (
    build_revision_proposal_set,
    validate_plan_revision_response,
)
from .planning.validators import PlanningProposalError, validate_planning_proposal_set
from .validators import EvidenceSpanRepository, ValidationReport, validate_candidate_set, validate_question_plan


@dataclass(frozen=True)
class CompilationResult:
    candidates: tuple[ScientificQuestionPlan, ...]
    reports: tuple[ValidationReport, ...]
    proposal_set: PlanningProposalSet | None = None
    compilation_receipts: tuple[PlanCompilationReceipt, ...] = ()


@dataclass(frozen=True)
class PlanRevisionResult:
    plan: ScientificQuestionPlan
    reports: tuple[ValidationReport, ...]
    proposal_set: PlanningProposalSet
    compilation_receipt: PlanCompilationReceipt
    response: PlanRevisionLLMResponse


class ScientificProblemCompiler:
    def __init__(
        self,
        provider: Any,
        domain_loader: DomainPackLoader | None = None,
        evidence_repository: EvidenceSpanRepository | None = None,
        materializer: PlanMaterializer | None = None,
    ) -> None:
        self.provider = provider
        self.domain_loader = domain_loader or DomainPackLoader()
        self.evidence_repository = evidence_repository
        self.materializer = materializer or PlanMaterializer()

    def compile(
        self,
        planning_input: ScientificPlanningInput | str,
        domain: str | None = None,
    ) -> CompilationResult:
        """Compile the grounded Phase 2C path, with a marked Phase 1 compatibility path."""
        if isinstance(planning_input, ScientificPlanningInput):
            if domain is not None:
                raise TypeError("domain is already bound by ScientificPlanningInput")
            return self._compile_grounded(planning_input)
        if domain is None:
            raise TypeError("legacy compilation requires an explicit domain")
        return self.compile_legacy(planning_input, domain)

    def _compile_grounded(self, planning_input: ScientificPlanningInput) -> CompilationResult:
        propose = getattr(self.provider, "propose", None)
        if not callable(propose):
            raise TypeError("grounded compilation requires a PlanningProvider")
        proposal = propose(planning_input)
        return self.compile_proposal(planning_input, proposal)

    def compile_proposal(
        self,
        planning_input: ScientificPlanningInput,
        proposal: PlanningProposalSet,
    ) -> CompilationResult:
        """Validate and materialize an already-produced grounded proposal."""
        proposal_report = validate_planning_proposal_set(proposal, planning_input)
        if not proposal_report.valid:
            raise PlanningProposalError(proposal_report)
        plans = self.materializer.materialize(proposal, planning_input)
        reports = tuple(
            validate_question_plan(
                plan,
                planning_input.scientific_capabilities,
                self.evidence_repository,
            )
            for plan in plans
        )
        set_report = validate_candidate_set(plans)
        receipts = tuple(
            self.materializer.build_compilation_receipt(plan, proposal, planning_input)
            for plan in plans
        )
        return CompilationResult(
            plans,
            (proposal_report, *reports, set_report),
            proposal,
            receipts,
        )

    def compile_legacy(self, request: str, domain: str) -> CompilationResult:
        """Phase 1 compatibility path for test-supplied immutable plans."""
        pack = self.domain_loader.load(domain)
        plans = tuple(self.provider.compile(request, domain_context=pack.profile.model_dump_json()))
        if not 1 <= len(plans) <= 4:
            raise ValueError("compiler must produce between one and four candidates")
        reports = tuple(
            validate_question_plan(plan, pack.capabilities, self.evidence_repository) for plan in plans
        )
        set_report = validate_candidate_set(plans)
        return CompilationResult(plans, reports + (set_report,))

    def revise(self, revision_input: PlanRevisionInput) -> PlanRevisionResult:
        revise = getattr(self.provider, "revise", None)
        if not callable(revise):
            raise TypeError("bounded plan revision requires a revision-capable PlanningProvider")
        response = revise(revision_input)
        response_report = validate_plan_revision_response(response, revision_input)
        if not response_report.valid:
            raise PlanningProposalError(response_report)
        provider_config = dict(getattr(self.provider, "provider_config", {}))
        model_id = getattr(getattr(self.provider, "transport", None), "model_id", None)
        if model_id is not None:
            provider_config["model_id"] = model_id
        for field_name in ("temperature", "max_attempts"):
            value = getattr(self.provider, field_name, None)
            if value is not None:
                provider_config[field_name] = value
        proposal = build_revision_proposal_set(
            revision_input,
            response,
            provider_id=self.provider.provider_id,
            provider_version=self.provider.provider_version,
            provider_config=provider_config,
        )
        proposal_report = validate_planning_proposal_set(
            proposal, revision_input.planning_input
        )
        if not proposal_report.valid:
            raise PlanningProposalError(proposal_report)
        plans = self.materializer.materialize(proposal, revision_input.planning_input)
        if len(plans) != 1:
            raise ValueError("bounded revision must materialize exactly one candidate")
        plan = plans[0]
        plan_report = validate_question_plan(
            plan,
            revision_input.planning_input.scientific_capabilities,
            self.evidence_repository,
        )
        set_report = validate_candidate_set(plans)
        receipt = self.materializer.build_compilation_receipt(
            plan, proposal, revision_input.planning_input
        )
        return PlanRevisionResult(
            plan=plan,
            reports=(response_report, proposal_report, plan_report, set_report),
            proposal_set=proposal,
            compilation_receipt=receipt,
            response=response,
        )
