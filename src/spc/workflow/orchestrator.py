from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..domains import DomainPackLoader
from ..models import (
    PlanningStrategy,
    ScientificProblemRun,
    ScientificProblemRunStatus,
    ScientificRunFailure,
)
from ..planning.hierarchical import (
    HierarchicalPlanningBlocked,
)
from ..serialization import content_hash
from .downstream import export_downstream
from .providers import ProviderSettings
from .repository import ScientificProblemRunRepository
from .resume import ProviderOutcomeUncertain, locked_run
from .stages.approval import ApprovalStage
from .stages.evidence_resolution import EvidenceResolutionBlocked, EvidenceResolutionStage
from .stages.hierarchy import HierarchicalPlanningStage
from .stages.planning import PlanningStage, blocking_curations, open_knowledge
from .stages.revision import RevisionStage
from .state import WorkflowResult, fail_run, sanitize_failure, snapshot_hash, update_run
from .status import dry_run_report


@dataclass(frozen=True, kw_only=True)
class _RunOptions:
    dry_run: bool = False
    interpretation_provider: str = "mock"
    planning_provider: str = "mock"
    approval_provider: str = "none"
    selected_candidate_id: str | None = None
    llm_endpoint: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = field(default=None, repr=False)
    temperature: float = 0.0
    max_attempts: int = 1
    max_plan_revisions: int = 0
    max_evidence_resolution_cycles: int = 0
    allow_evidence_source_acquisition: bool = False
    planning_strategy: str = "direct"


class ScientificProblemWorkflow:
    """Coordinate run transitions using the bound outputs of each stage."""

    def __init__(self, *, state_dir: Path, knowledge_dir: Path) -> None:
        self.state_dir = state_dir.resolve()
        self.knowledge_dir = knowledge_dir.resolve()
        self.runs = ScientificProblemRunRepository(self.state_dir)
        self.domain_loader = DomainPackLoader()
        self.planning = PlanningStage(self.runs, self.domain_loader)
        self.evidence = EvidenceResolutionStage(self.runs, self.domain_loader)
        self.hierarchy = HierarchicalPlanningStage(self.runs)
        self.approval = ApprovalStage(self.runs, self.domain_loader)
        self.revision = RevisionStage(self.runs, self.approval)

    def start(
        self,
        request: str,
        domain: str,
        *,
        dry_run: bool = False,
        interpretation_provider: str = "mock",
        planning_provider: str = "mock",
        approval_provider: str = "none",
        selected_candidate_id: str | None = None,
        llm_endpoint: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        temperature: float = 0.0,
        max_attempts: int = 1,
        max_plan_revisions: int = 0,
        max_evidence_resolution_cycles: int = 0,
        allow_evidence_source_acquisition: bool = False,
        planning_strategy: str = "direct",
    ) -> WorkflowResult:
        if not request.strip():
            raise ValueError("scientific request must not be blank")
        identity = {
            "original_request": request,
            "domain": domain,
            "state_dir": str(self.state_dir),
            "knowledge_dir": str(self.knowledge_dir),
        }
        run_id = f"scientific-run-{content_hash(identity)[:24]}"
        with locked_run(self.runs, run_id, initial=identity) as run:
            options = _RunOptions(
                dry_run=dry_run,
                interpretation_provider=interpretation_provider,
                planning_provider=planning_provider,
                approval_provider=approval_provider,
                selected_candidate_id=selected_candidate_id,
                llm_endpoint=llm_endpoint,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                temperature=temperature,
                max_attempts=max_attempts,
                max_plan_revisions=max_plan_revisions,
                max_evidence_resolution_cycles=max_evidence_resolution_cycles,
                allow_evidence_source_acquisition=allow_evidence_source_acquisition,
                planning_strategy=planning_strategy,
            )
            return self._advance(run, options)

    def resume(self, run_id: str, **options: Any) -> WorkflowResult:
        with locked_run(self.runs, run_id) as run:
            return self._advance(run, _RunOptions(**options))

    def _advance(self, run: ScientificProblemRun, options: _RunOptions) -> WorkflowResult:
        max_plan_revisions = options.max_plan_revisions
        try:
            strategy = PlanningStrategy(options.planning_strategy)
        except ValueError as error:
            raise ValueError("planning_strategy must be 'direct' or 'hierarchical'") from error
        if not 0 <= max_plan_revisions <= 10:
            raise ValueError("max_plan_revisions must be between zero and ten")
        if not 0 <= options.max_evidence_resolution_cycles <= 3:
            raise ValueError("max_evidence_resolution_cycles must be between zero and three")
        evidence_policy = self.evidence.ensure_policy(
            run.run_id,
            options.max_evidence_resolution_cycles,
            options.allow_evidence_source_acquisition,
        )
        if evidence_policy is not None:
            try:
                run = self.evidence.recover_committed_child(run)
            except (OSError, ValueError) as error:
                return fail_run(self.runs, run, "planning_evidence_resolution", error, retryable=False)
        revision_chain = self.runs.load_revision_chain(run.run_id)
        if revision_chain is not None:
            if max_plan_revisions not in {0, revision_chain.max_revisions}:
                raise ValueError("persisted revision budget cannot be changed during resume")
            max_plan_revisions = revision_chain.max_revisions
        if revision_chain is not None and run.context_id is not None and run.context_id != revision_chain.context_id:
            revision_chain = None
        if revision_chain is not None and revision_chain.termination_reason is not None:
            try:
                self.revision.validate_chain(revision_chain)
            except (FileNotFoundError, OSError, ValueError) as error:
                return fail_run(
                    self.runs,
                    run,
                    "plan_revision_integrity",
                    error,
                    retryable=False,
                )
            if evidence_policy is None or not self.revision.needs_evidence(revision_chain):
                return WorkflowResult(run)
        repositories, evidence = open_knowledge(self.knowledge_dir, self.state_dir, run.domain, self.domain_loader)
        try:
            blocking = blocking_curations(repositories, evidence, run.domain)
        except (FileNotFoundError, OSError, ValueError) as error:
            return fail_run(self.runs, run, "knowledge_trust", error, retryable=False)
        if blocking:
            blocked = update_run(
                run,
                status=ScientificProblemRunStatus.BLOCKED_SOURCE_CURATION,
                blocking_items=blocking,
                failure=None,
            )
            self.runs.save(blocked)
            return WorkflowResult(blocked, dry_run_report(blocked, None))

        ready = update_run(
            run,
            status=ScientificProblemRunStatus.READY,
            last_successful_status=None,
            blocking_items=(),
            failure=None,
        )
        self.runs.save(ready)
        try:
            base_context = self.planning.retrieve(ready, repositories, evidence)
            context = self.evidence.reuse_context(ready.run_id, base_context)
        except (FileNotFoundError, OSError, ValueError) as error:
            return fail_run(self.runs, ready, "trusted_retrieval", error, retryable=False)
        if revision_chain is not None and (
            revision_chain.context_id != context.context_id
            or revision_chain.context_hash != context.content_hash
            or revision_chain.knowledge_snapshot_id != context.knowledge_snapshot.snapshot_id
            or revision_chain.knowledge_snapshot_hash != snapshot_hash(context.knowledge_snapshot)
        ):
            return self.revision.block(
                ready,
                revision_chain,
                "TRUSTED_CONTEXT_CHANGED",
                "trusted context or knowledge authority changed; start a new compiled context",
            )
        context_run = self.planning.record_context(ready, context)
        if options.dry_run:
            return WorkflowResult(
                context_run,
                dry_run_report(context_run, context),
            )

        if options.interpretation_provider != "mock":
            return fail_run(
                self.runs,
                context_run,
                "interpretation",
                ValueError("available interpretation provider: mock"),
                retryable=False,
            )
        try:
            interpreted, packet = self.planning.interpret(context_run, context, evidence)
        except (OSError, ValueError) as error:
            return fail_run(self.runs, context_run, "interpretation", error, retryable=True)

        return self._plan_and_review(
            run,
            interpreted,
            context,
            packet,
            repositories,
            evidence,
            strategy,
            revision_chain,
            max_plan_revisions,
            evidence_policy,
            options,
        )

    def _plan_and_review(
        self,
        run,
        interpreted,
        context,
        packet,
        repositories,
        evidence,
        strategy,
        revision_chain,
        max_plan_revisions,
        evidence_policy,
        options,
    ) -> WorkflowResult:
        selected_candidate_id = options.selected_candidate_id
        settings = ProviderSettings(
            options.llm_endpoint, options.llm_model, options.llm_api_key, options.temperature, options.max_attempts
        )
        try:
            planning_input = self.planning.resolve_input(interpreted, context, packet, repositories, evidence)
            planner = settings.planning(options.planning_provider)
            if revision_chain is None:
                strategy = self.hierarchy.strategy_for(interpreted, strategy)
                while True:
                    parent_context_id = context.context_id
                    interpreted, context, packet, planning_input = self.evidence.resolve(
                        interpreted,
                        context,
                        packet,
                        planning_input,
                        repositories,
                        evidence,
                        planner,
                        strategy,
                        evidence_policy,
                    )
                    if context.context_id != parent_context_id:
                        continue
                    if strategy == PlanningStrategy.HIERARCHICAL:
                        directions, triage = self.hierarchy.direction_triage(interpreted, planning_input, planner)
                        interpreted, context, packet, planning_input = self.evidence.resolve(
                            interpreted,
                            context,
                            packet,
                            planning_input,
                            repositories,
                            evidence,
                            planner,
                            strategy,
                            evidence_policy,
                            direction_set=directions,
                            triage=triage,
                        )
                        if context.context_id != parent_context_id:
                            continue
                    break
            if revision_chain is not None:
                if (
                    revision_chain.planning_input_id != planning_input.planning_input_id
                    or revision_chain.planning_input_hash != planning_input.content_hash
                ):
                    return self.revision.block(
                        interpreted,
                        revision_chain,
                        "PLANNING_INPUT_CHANGED",
                        "trusted planning input changed; rebuild a new context before revision",
                    )
                compilation = self.revision.load_compilation(revision_chain, planning_input, evidence)
            else:
                if strategy == PlanningStrategy.HIERARCHICAL:
                    compilation = self.hierarchy.compile(
                        interpreted, planning_input, evidence, planner, directions, triage
                    )
                else:
                    compilation = self.planning.compile(
                        interpreted, planning_input, evidence, planner, max_plan_revisions
                    )
        except EvidenceResolutionBlocked as error:
            return self._evidence_block(interpreted, error)
        except HierarchicalPlanningBlocked as error:
            blocked = update_run(
                interpreted,
                status=ScientificProblemRunStatus.HIERARCHICAL_PLANNING_BLOCKED,
                last_successful_status=interpreted.status,
                failure=ScientificRunFailure(
                    stage="hierarchical_planning",
                    category=type(error).__name__,
                    message=sanitize_failure(error),
                    retryable=False,
                ),
            )
            self.runs.save(blocked)
            return WorkflowResult(blocked)
        except ProviderOutcomeUncertain as error:
            return fail_run(self.runs, interpreted, "planning", error, retryable=False)
        except (FileNotFoundError, OSError, ValueError, TypeError) as error:
            return fail_run(self.runs, interpreted, "planning", error, retryable=True)
        proposal = compilation.proposal_set
        if proposal is None:
            return fail_run(
                self.runs,
                interpreted,
                "planning",
                RuntimeError("grounded compiler returned no proposal set"),
                retryable=False,
            )
        planned, proposal_binding = self.planning.record_candidates(
            interpreted,
            planning_input,
            evidence,
            planner,
            compilation,
            selected_candidate_id,
            revision_chain,
            max_plan_revisions,
        )
        if revision_chain is None and max_plan_revisions > 0:
            try:
                planned, revision_chain = self.revision.start_chain(
                    planned, planning_input, context, proposal_binding, max_plan_revisions
                )
            except ValueError as error:
                return fail_run(self.runs, planned, "plan_revision", error, retryable=False)
            selected_candidate_id = planned.selected_candidate_id
        awaiting = update_run(
            planned,
            status=ScientificProblemRunStatus.AWAITING_APPROVAL,
            approval_review_id=None,
            approval_review_hash=None,
            approval_verdict_id=None,
            approval_verdict_hash=None,
            approval_receipt_id=None,
            approval_receipt_hash=None,
            gate_id=None,
            gate_hash=None,
            export_path=None,
            export_hash=None,
        )
        self.runs.save(awaiting)
        approved = self._review(
            awaiting,
            run,
            context,
            packet,
            planning_input,
            repositories,
            evidence,
            planner,
            revision_chain,
            selected_candidate_id,
            options.approval_provider,
            settings,
        )
        if isinstance(approved, WorkflowResult):
            return approved
        try:
            evidence_run = self.evidence.resolve_after_review(
                approved, context, packet, planning_input, repositories, evidence, planner, strategy, evidence_policy
            )
        except EvidenceResolutionBlocked as error:
            return self._evidence_block(approved, error)
        if evidence_run is not None:
            return self._advance(
                evidence_run,
                replace(
                    options,
                    selected_candidate_id=None,
                    max_plan_revisions=max_plan_revisions,
                    planning_strategy=strategy.value,
                ),
            )
        return WorkflowResult(approved)

    def _review(
        self,
        awaiting,
        run,
        context,
        packet,
        planning_input,
        repositories,
        evidence,
        planner,
        revision_chain,
        selected_candidate_id,
        approval_provider,
        settings,
    ) -> ScientificProblemRun | WorkflowResult:
        revision_approval = self.approval.revision_context(run.run_id, revision_chain)
        reviewed_round = revision_chain is not None and revision_chain.rounds[-1].approval_receipt is not None
        reusable = self.approval.reuse(
            awaiting,
            previous=run,
            context=context,
            packet=packet,
            planning_input=planning_input,
            repositories=repositories,
            evidence=evidence,
            selected_candidate_id=selected_candidate_id,
            revision_approval=revision_approval,
        )
        if approval_provider == "none":
            return WorkflowResult(reusable or awaiting)
        if reusable is not None:
            approved = reusable
        elif reviewed_round:
            return self.revision.block(
                awaiting, revision_chain, "STALE_APPROVAL_CHAIN", "stored revision approval chain failed validation"
            )
        else:
            approved = self.approval.approve(
                awaiting,
                context,
                packet,
                planning_input,
                repositories,
                evidence,
                approval_provider,
                selected_candidate_id,
                settings,
                artifact_prefix=self.runs.approval_prefix(run.run_id),
                revision_approval=revision_approval,
            )
        if revision_chain is not None:
            if not reviewed_round:
                if approved.approval_receipt_id is None:
                    return WorkflowResult(approved)
                revision_chain = self.revision.record_approval(revision_chain, approved)
            approved = self.revision.revise(
                approved,
                revision_chain,
                context,
                packet,
                planning_input,
                repositories,
                evidence,
                planner,
                approval_provider,
                settings,
            )
        return approved

    def _evidence_block(self, run, error: EvidenceResolutionBlocked) -> WorkflowResult:
        status = (
            ScientificProblemRunStatus.BLOCKED_SOURCE_CURATION
            if error.curation_items
            else ScientificProblemRunStatus.EVIDENCE_RESOLUTION_BLOCKED
        )
        blocked = update_run(
            run,
            status=status,
            last_successful_status=run.status,
            blocking_items=error.curation_items,
            failure=(
                None
                if error.curation_items
                else ScientificRunFailure(
                    stage="planning_evidence_resolution",
                    category=error.category,
                    message=sanitize_failure(error),
                    retryable=False,
                )
            ),
        )
        self.runs.save(blocked)
        return WorkflowResult(blocked)

    def export_downstream(
        self,
        run_id: str,
        output_root: Path,
        export_id: str,
        *,
        selected_candidate_id: str | None = None,
    ) -> Path:
        return export_downstream(
            self.runs, self.domain_loader, run_id, output_root, export_id, selected_candidate_id=selected_candidate_id
        )
