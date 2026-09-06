from __future__ import annotations

from ..models import ScientificQuestionPlan, ScientificTaskExecutionContext
from ..serialization import content_hash


class ExecutionContextError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ScientificTaskExecutionContextProjector:
    """Deterministically projects one task without exposing its source plan."""

    def project(
        self,
        plan: ScientificQuestionPlan,
        task_id: str,
    ) -> ScientificTaskExecutionContext:
        task = next((item for item in plan.tasks if item.task_id == task_id), None)
        if task is None:
            raise ExecutionContextError(
                "TASK_PLAN_MISMATCH",
                f"task {task_id!r} does not belong to the source plan",
            )
        plan_task_ids = {item.task_id for item in plan.tasks}
        unknown_dependencies = tuple(
            dependency
            for dependency in task.depends_on
            if dependency not in plan_task_ids
        )
        if unknown_dependencies:
            raise ExecutionContextError(
                "UNKNOWN_TASK_DEPENDENCY",
                "task dependencies are absent from the source plan: "
                + ", ".join(unknown_dependencies),
            )
        if task.task_id in task.depends_on:
            raise ExecutionContextError(
                "UNKNOWN_TASK_DEPENDENCY",
                "task cannot depend on itself",
            )
        evidence_by_id = {
            reference.evidence_id: reference for reference in plan.evidence_refs
        }
        grounded_statements = (
            plan.hypothesis.primary,
            plan.hypothesis.null,
            plan.model.description,
            *plan.model.parameters,
            *(observable.description for observable in plan.observables),
            *(baseline.description for baseline in plan.comparison_baselines),
        )
        context_evidence_ids = tuple(
            dict.fromkeys(
                (
                    *task.evidence_refs,
                    *(
                        evidence_id
                        for statement in grounded_statements
                        for evidence_id in statement.evidence_refs
                    ),
                    *plan.system_fingerprint.evidence_refs,
                    *plan.method_fingerprint.evidence_refs,
                )
            )
        )
        missing_evidence = tuple(
            evidence_id
            for evidence_id in context_evidence_ids
            if evidence_id not in evidence_by_id
        )
        if missing_evidence:
            raise ExecutionContextError(
                "TASK_EVIDENCE_MISMATCH",
                "task evidence is absent from the source plan: "
                + ", ".join(missing_evidence),
            )
        identity = {
            "source_plan_id": plan.plan_id,
            "source_plan_hash": content_hash(plan),
            "task_id": task.task_id,
            "scientific_objective": task.scientific_objective,
            "task_inputs": task.inputs,
            "hypothesis": plan.hypothesis,
            "model": plan.model,
            "observables": plan.observables,
            "comparison_baselines": plan.comparison_baselines,
            "intent_fingerprint_ref": plan.intent_fingerprint.fingerprint_id,
            "system_fingerprint_ref": plan.system_fingerprint.fingerprint_id,
            "method_fingerprint_ref": plan.method_fingerprint.fingerprint_id,
            "acceptance_criteria": plan.acceptance_criteria,
            "falsification_criteria": plan.falsification_criteria,
            "evidence_refs": tuple(
                evidence_by_id[evidence_id] for evidence_id in context_evidence_ids
            ),
            "success_criteria": task.success_criteria,
            "provenance_requirements": task.provenance_requirements,
            "depends_on_task_ids": task.depends_on,
            "source_query_manifest": plan.source_query_manifest,
        }
        context_id = f"task-execution-context-{content_hash(identity)[:24]}"
        payload = {"context_id": context_id, **identity}
        return ScientificTaskExecutionContext(
            **payload,
            content_hash=content_hash(payload),
        )
