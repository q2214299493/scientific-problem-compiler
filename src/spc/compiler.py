from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domains import DomainPackLoader
from .models import PlanningProposalSet, ScientificPlanningInput, ScientificQuestionPlan
from .planning.materializer import PlanMaterializer
from .planning.validators import PlanningProposalError, validate_planning_proposal_set
from .validators import EvidenceSpanRepository, ValidationReport, validate_candidate_set, validate_question_plan


@dataclass(frozen=True)
class CompilationResult:
    candidates: tuple[ScientificQuestionPlan, ...]
    reports: tuple[ValidationReport, ...]
    proposal_set: PlanningProposalSet | None = None


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
        return CompilationResult(plans, (proposal_report, *reports, set_report), proposal)

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
