from __future__ import annotations

from dataclasses import dataclass

from .domains import DomainPackLoader
from .models import ScientificQuestionPlan
from .providers import Provider
from .validators import EvidenceSpanRepository, ValidationReport, validate_candidate_set, validate_question_plan


@dataclass(frozen=True)
class CompilationResult:
    candidates: tuple[ScientificQuestionPlan, ...]
    reports: tuple[ValidationReport, ...]


class ScientificProblemCompiler:
    def __init__(
        self,
        provider: Provider,
        domain_loader: DomainPackLoader | None = None,
        evidence_repository: EvidenceSpanRepository | None = None,
    ) -> None:
        self.provider = provider
        self.domain_loader = domain_loader or DomainPackLoader()
        self.evidence_repository = evidence_repository

    def compile(self, request: str, domain: str) -> CompilationResult:
        pack = self.domain_loader.load(domain)
        plans = tuple(self.provider.compile(request, domain_context=pack.profile.model_dump_json()))
        if not 1 <= len(plans) <= 4:
            raise ValueError("compiler must produce between one and four candidates")
        reports = tuple(
            validate_question_plan(plan, pack.capabilities, self.evidence_repository) for plan in plans
        )
        set_report = validate_candidate_set(plans)
        return CompilationResult(plans, reports + (set_report,))
