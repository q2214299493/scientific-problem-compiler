from __future__ import annotations

from dataclasses import dataclass

from ...compiler import CompilationResult, ScientificProblemCompiler
from ...models import (
    DirectionDisposition,
    DirectionTriageRecord,
    HierarchicalPlanningExpansion,
    PlanningEvidenceRequestSet,
    PlanningStrategy,
    ResearchDirectionSet,
    ScientificProblemRun,
)
from ...planning.hierarchical import (
    HierarchicalPlanningBlocked,
    build_direction_triage_record,
    build_hierarchical_expansion,
    build_hierarchical_stage_attempt,
    build_research_direction_set,
    unresolved_human_choice_items,
    validate_direction_triage,
    validate_hierarchical_expansion,
    validate_research_direction_set,
)
from ..repository import ScientificProblemRunRepository
from ..resume import reuse_compilation


@dataclass
class HierarchicalPlanningStage:
    """Bind direction generation, triage and expansion to the current planning input."""

    runs: ScientificProblemRunRepository

    def strategy_for(self, run: ScientificProblemRun, strategy: PlanningStrategy) -> PlanningStrategy:
        if self.runs.resolve_artifact_path(run.run_id, self.runs.hierarchical_root(run.run_id)).exists():
            return PlanningStrategy.HIERARCHICAL
        cycle_index = self.runs.evidence_attempt_count(run.run_id)
        if cycle_index:
            prefix = self.runs.evidence_cycle_prefix(cycle_index)
            try:
                requests = self.runs.load_artifact(run.run_id, f"{prefix}/request-set.yaml", PlanningEvidenceRequestSet)
            except FileNotFoundError:
                pass
            else:
                if requests.knowledge_snapshot_hash == run.knowledge_snapshot_hash:
                    return requests.planning_strategy
        if strategy == PlanningStrategy.HIERARCHICAL and run.planning_proposal_id is not None:
            raise HierarchicalPlanningBlocked(
                (
                    "this run already contains direct planning artifacts; use a new "
                    "state directory for a hierarchical comparison",
                )
            )
        return strategy

    def direction_triage(
        self,
        run: ScientificProblemRun,
        planning_input,
        planner,
    ) -> tuple[ResearchDirectionSet, DirectionTriageRecord]:
        root = self.runs.hierarchical_root(run.run_id)

        def existing(path: str) -> bool:
            return self.runs.resolve_artifact_path(run.run_id, path).exists()

        directions_path = f"{root}/directions.yaml"
        directions_attempt_path = f"{root}/attempts/directions.yaml"
        if existing(directions_path):
            directions = self.runs.load_artifact(run.run_id, directions_path, ResearchDirectionSet)
        else:
            if existing(directions_attempt_path):
                raise HierarchicalPlanningBlocked(
                    ("direction generation was started but no complete output was saved",)
                )
            attempt = build_hierarchical_stage_attempt("directions", planning_input, planner)
            self.runs.write_exclusive_artifact(
                run.run_id,
                directions_attempt_path,
                "hierarchical_planning_stage_attempt",
                attempt.attempt_id,
                attempt,
            )
            method = getattr(planner, "propose_directions", None)
            if not callable(method):
                raise HierarchicalPlanningBlocked(("planning provider does not support research direction generation",))
            directions = build_research_direction_set(planning_input, method(planning_input), planner)
            report = validate_research_direction_set(directions, planning_input)
            if not report.valid:
                raise HierarchicalPlanningBlocked(tuple(item.code for item in report.issues))
            self.runs.write_immutable_artifact(
                run.run_id,
                directions_path,
                "research_direction_set",
                directions.direction_set_id,
                directions,
            )
        direction_report = validate_research_direction_set(directions, planning_input)
        if not direction_report.valid:
            raise HierarchicalPlanningBlocked(tuple(item.code for item in direction_report.issues))

        triage_path = f"{root}/triage.yaml"
        triage_attempt_path = f"{root}/attempts/triage.yaml"
        if existing(triage_path):
            triage = self.runs.load_artifact(run.run_id, triage_path, DirectionTriageRecord)
        else:
            if existing(triage_attempt_path):
                raise HierarchicalPlanningBlocked(("direction triage was started but no complete output was saved",))
            attempt = build_hierarchical_stage_attempt(
                "triage",
                planning_input,
                planner,
                previous_stage_id=directions.direction_set_id,
                previous_stage_hash=directions.content_hash,
            )
            self.runs.write_exclusive_artifact(
                run.run_id,
                triage_attempt_path,
                "hierarchical_planning_stage_attempt",
                attempt.attempt_id,
                attempt,
            )
            method = getattr(planner, "triage_directions", None)
            if not callable(method):
                raise HierarchicalPlanningBlocked(("planning provider does not support direction triage",))
            triage = build_direction_triage_record(
                planning_input,
                directions,
                method(planning_input, directions),
                planner,
            )
            triage_report = validate_direction_triage(triage, directions, planning_input)
            if not triage_report.valid:
                raise HierarchicalPlanningBlocked(tuple(item.code for item in triage_report.issues))
            self.runs.write_immutable_artifact(
                run.run_id,
                triage_path,
                "direction_triage_record",
                triage.triage_id,
                triage,
            )
        triage_report = validate_direction_triage(triage, directions, planning_input)
        if not triage_report.valid:
            raise HierarchicalPlanningBlocked(tuple(item.code for item in triage_report.issues))
        human_choice_items = unresolved_human_choice_items(triage)
        if human_choice_items:
            raise HierarchicalPlanningBlocked(human_choice_items)
        if not any(item.disposition == DirectionDisposition.RETAIN for item in triage.dispositions):
            raise HierarchicalPlanningBlocked(("direction triage retained no direction for plan expansion",))
        return directions, triage

    def compile(
        self,
        run: ScientificProblemRun,
        planning_input,
        evidence,
        planner,
        directions: ResearchDirectionSet,
        triage: DirectionTriageRecord,
    ) -> CompilationResult:
        root = self.runs.hierarchical_root(run.run_id)

        def existing(path: str) -> bool:
            return self.runs.resolve_artifact_path(run.run_id, path).exists()

        expansion_path = f"{root}/expansion.yaml"
        expansion_attempt_path = f"{root}/attempts/expansion.yaml"
        if existing(expansion_path):
            expansion = self.runs.load_artifact(run.run_id, expansion_path, HierarchicalPlanningExpansion)
        else:
            if existing(expansion_attempt_path):
                raise HierarchicalPlanningBlocked(("plan expansion was started but no complete output was saved",))
            attempt = build_hierarchical_stage_attempt(
                "expansion",
                planning_input,
                planner,
                previous_stage_id=triage.triage_id,
                previous_stage_hash=triage.content_hash,
            )
            self.runs.write_exclusive_artifact(
                run.run_id,
                expansion_attempt_path,
                "hierarchical_planning_stage_attempt",
                attempt.attempt_id,
                attempt,
            )
            method = getattr(planner, "expand_directions", None)
            if not callable(method):
                raise HierarchicalPlanningBlocked(("planning provider does not support retained direction expansion",))
            expansion = build_hierarchical_expansion(
                planning_input,
                directions,
                triage,
                method(planning_input, directions, triage),
                planner,
            )
            self.runs.write_immutable_artifact(
                run.run_id,
                expansion_path,
                "hierarchical_planning_expansion",
                expansion.expansion_id,
                expansion,
            )
        expansion_report = validate_hierarchical_expansion(expansion, directions, triage, planning_input)
        if not expansion_report.valid:
            raise HierarchicalPlanningBlocked(tuple(item.code for item in expansion_report.issues))
        completed = reuse_compilation(self.runs, run, planning_input, evidence)
        if completed is not None:
            if completed.proposal_set != expansion.planning_proposal:
                raise HierarchicalPlanningBlocked(("saved candidates do not bind the hierarchical expansion",))
            return completed
        return ScientificProblemCompiler(planner, evidence_repository=evidence).compile_proposal(
            planning_input, expansion.planning_proposal
        )
