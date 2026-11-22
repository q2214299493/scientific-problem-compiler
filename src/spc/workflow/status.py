from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domains import DomainPackLoader
from ..knowledge.trust import TrustedKnowledgeValidator
from ..models import (
    ApprovalVerdict,
    CurationStatus,
    DirectionDisposition,
    DirectionTriageRecord,
    EvidenceResolutionSet,
    EvidenceResolutionStatus,
    HierarchicalPlanningExpansion,
    PlanningEvidenceCyclePolicy,
    PlanningEvidenceRequestAttempt,
    PlanningEvidenceRequestSet,
    PlanningProposalSet,
    PlanRevisionAttemptStatus,
    ResearchDirectionSet,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificProblemRun,
    ScientificQuestionPlan,
)
from ..repositories import (
    CompositeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
)
from .knowledge_access import CURATED_TYPES
from .repository import ScientificProblemRunRepository
from .state import sanitize_failure


def dry_run_report(run, context):
    if context is None:
        knowledge_status = build_knowledge_status(
            Path(run.knowledge_dir), Path(run.state_dir), run.domain
        )
        trusted_literature_count = knowledge_status["literature"]["trusted_documents"]
        trusted_expert_opinion_count = knowledge_status["experts"]["trusted_opinions"]
        trusted_expert_case_count = knowledge_status["experts"]["trusted_cases"]
    else:
        # Context construction already verified these exact snapshot inventories.
        snapshot = context.knowledge_snapshot
        trusted_literature_count = len(snapshot.literature_document_hashes)
        trusted_expert_opinion_count = len(snapshot.expert_opinion_hashes)
        trusted_expert_case_count = len(snapshot.expert_case_hashes)
    hits = {
        "literature": len(context.literature_knowledge_hits) if context else 0,
        "expert_opinions": len(context.expert_opinion_hits) if context else 0,
        "expert_cases": len(context.expert_case_hits) if context else 0,
        "graph_expanded": len(context.graph_expanded_hits) if context else 0,
    }
    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "source_trust_ready": not run.blocking_items and run.failure is None,
        "trusted_literature_count": trusted_literature_count,
        "trusted_expert_opinion_count": trusted_expert_opinion_count,
        "trusted_expert_case_count": trusted_expert_case_count,
        "retrieval_hit_counts": hits,
        "blocking_curation_items": [item.model_dump(mode="json") for item in run.blocking_items],
        "available_providers": {
            "interpretation": ["mock"],
            "planning": ["mock", "llm"],
            "approval": ["none", "mock", "llm"],
        },
        "next_model_call": (
            None
            if run.blocking_items
            else "interpret ScientificContextPacket; dry-run performed no provider inference"
        ),
        "model_inference_performed": False,
    }


def build_knowledge_status(knowledge_dir: Path, state_dir: Path, domain: str = "base") -> dict[str, Any]:
    repositories = KnowledgeRepositories(knowledge_dir.resolve(), initialize=False)
    DomainPackLoader().load(domain)
    evidence = CompositeEvidenceStore(
        repositories.evidence_store,
        ProjectEvidenceStore(state_dir.resolve(), initialize=False),
    )
    validator = TrustedKnowledgeValidator(repositories, evidence)
    # Validation can add representation references. Report the initial inventory.
    record_items = sorted(validator.records.items())
    current = validator.resolve_current_curations()
    broken: list[str] = []
    try:
        trusted = validator.validate()
        trusted_keys = set(trusted.trusted_records)
    except (FileNotFoundError, OSError, ValueError) as error:
        trusted_keys = set()
        broken.append(sanitize_failure(error))
    linked_attributions = {
        item.attribution_id for item in repositories.expert_sources.list()
    }
    pending = tuple(
        f"{kind}:{record_id}"
        for (kind, record_id), record in record_items
        if (
            kind in CURATED_TYPES
            or (kind == "expert_case" and bool(getattr(record, "opinion_refs", ())))
            or (kind == "expert_attribution" and record_id in linked_attributions)
        )
        and (
            (kind, record_id) not in current
            or current[(kind, record_id)].status
            in {CurationStatus.MACHINE_EXTRACTED, CurationStatus.HUMAN_REVIEWED}
        )
    )
    return {
        "literature": {
            "documents": len(repositories.literature_documents.list()),
            "trusted_documents": sum(1 for kind, _ in trusted_keys if kind == "literature_document"),
            "machine_extracted_records": sum(
                1 for curation in current.values() if curation.status == CurationStatus.MACHINE_EXTRACTED
            ),
            "accepted_scientific_records": sum(
                1 for kind, _ in trusted_keys if kind in {"source_claim", "method_fact", "model_fact", "reported_result"}
            ),
        },
        "experts": {
            "profiles": len(repositories.expert_profiles.list()),
            "sources": len(repositories.expert_sources.list()),
            "accepted_attributions": sum(1 for kind, _ in trusted_keys if kind == "expert_attribution"),
            "opinions": len(repositories.expert_opinions.list()),
            "trusted_opinions": sum(1 for kind, _ in trusted_keys if kind == "expert_opinion"),
            "expert_cases": len(repositories.expert_cases.list()),
            "trusted_cases": sum(1 for kind, _ in trusted_keys if kind == "expert_case"),
        },
        "trust": {
            "broken_provenance_count": len(broken),
            "broken_provenance": broken,
            "pending_curation_count": len(pending),
            "pending_curation": pending,
        },
        "model_inference_performed": False,
    }


def scientific_run_status(
    run: ScientificProblemRun,
    repository: ScientificProblemRunRepository | None = None,
) -> dict[str, Any]:
    retrieval_summary: dict[str, int] = {}
    evidence_gaps: list[dict[str, Any]] = []
    revision_state: dict[str, Any] | None = None
    hierarchical_state: dict[str, Any] | None = None
    evidence_resolution_state: dict[str, Any] | None = None
    revision_chain = repository.load_revision_chain(run.run_id) if repository is not None else None
    if revision_chain is not None:
        attempt_states = [
            {
                "attempt_index": start.attempt_index,
                "attempt_id": start.attempt_id,
                "status": outcome.status.value if outcome is not None else PlanRevisionAttemptStatus.UNCERTAIN.value,
            }
            for start, outcome in repository.load_revision_attempts(run.run_id)
        ]
        revision_state = {
            "max_revisions": revision_chain.max_revisions,
            "revisions_used": revision_chain.revisions_used,
            "attempts_started": len(attempt_states),
            "attempts": attempt_states,
            "current_round": revision_chain.rounds[-1].round_index,
            "termination_reason": revision_chain.termination_reason,
            "round_plan_ids": [
                item.candidate_plan.artifact_id for item in revision_chain.rounds
            ],
        }
    if repository is not None:
        policy_path = repository.resolve_artifact_path(
            run.run_id, "planning-evidence/policy.yaml"
        )
        if policy_path.exists():
            policy = repository.load_artifact(
                run.run_id,
                "planning-evidence/policy.yaml",
                PlanningEvidenceCyclePolicy,
            )
            requests_generated = 0
            status_counts = {
                status: 0
                for status in EvidenceResolutionStatus
            }
            unresolved_requests: list[str] = []
            cycle_index = 1
            while True:
                attempt_path = repository.resolve_artifact_path(
                    run.run_id,
                    f"planning-evidence/cycle-{cycle_index}/request-attempt.yaml",
                )
                request_path = repository.resolve_artifact_path(
                    run.run_id,
                    f"planning-evidence/cycle-{cycle_index}/request-set.yaml",
                )
                if not attempt_path.exists() and not request_path.exists():
                    break
                if not request_path.exists():
                    attempt = repository.load_artifact(
                        run.run_id,
                        f"planning-evidence/cycle-{cycle_index}/request-attempt.yaml",
                        PlanningEvidenceRequestAttempt,
                    )
                    unresolved_requests.append(attempt.attempt_id)
                    cycle_index += 1
                    continue
                request_set = repository.load_artifact(
                    run.run_id,
                    f"planning-evidence/cycle-{cycle_index}/request-set.yaml",
                    PlanningEvidenceRequestSet,
                )
                requests_generated += len(request_set.requests)
                resolution_path = repository.resolve_artifact_path(
                    run.run_id,
                    f"planning-evidence/cycle-{cycle_index}/resolutions.yaml",
                )
                if resolution_path.exists():
                    resolutions = repository.load_artifact(
                        run.run_id,
                        f"planning-evidence/cycle-{cycle_index}/resolutions.yaml",
                        EvidenceResolutionSet,
                    )
                    if (resolutions.request_set_id, resolutions.request_set_hash) != (
                        request_set.request_set_id, request_set.content_hash
                    ):
                        raise ValueError("stored evidence resolution does not bind its request set")
                    for item in resolutions.records:
                        status_counts[item.status] += 1
                        if item.status != EvidenceResolutionStatus.RESOLVED_TRUSTED:
                            unresolved_requests.append(item.request_id)
                else:
                    unresolved_requests.extend(
                        item.request_id for item in request_set.requests
                    )
                cycle_index += 1
            evidence_resolution_state = {
                "max_cycles": policy.max_cycles,
                "cycles_used": cycle_index - 1,
                "requests_generated": requests_generated,
                "trusted_matches": status_counts[
                    EvidenceResolutionStatus.RESOLVED_TRUSTED
                ],
                "conflicting_matches": status_counts[
                    EvidenceResolutionStatus.CONFLICTING_EVIDENCE
                ],
                "no_matches": status_counts[EvidenceResolutionStatus.NO_MATCH],
                "curation_required": status_counts[
                    EvidenceResolutionStatus.REQUIRES_SOURCE_CURATION
                ],
                "unresolved_requests": sorted(set(unresolved_requests)),
            }
        hierarchy_root = repository.hierarchical_root(run.run_id)
        directions_path = repository.resolve_artifact_path(
            run.run_id, f"{hierarchy_root}/directions.yaml"
        )
        triage_path = repository.resolve_artifact_path(
            run.run_id, f"{hierarchy_root}/triage.yaml"
        )
        expansion_path = repository.resolve_artifact_path(
            run.run_id, f"{hierarchy_root}/expansion.yaml"
        )
        if directions_path.exists():
            directions = repository.load_artifact(
                run.run_id,
                f"{hierarchy_root}/directions.yaml",
                ResearchDirectionSet,
            )
            triage = (
                repository.load_artifact(
                    run.run_id,
                    f"{hierarchy_root}/triage.yaml",
                    DirectionTriageRecord,
                )
                if triage_path.exists()
                else None
            )
            expansion = (
                repository.load_artifact(
                    run.run_id,
                    f"{hierarchy_root}/expansion.yaml",
                    HierarchicalPlanningExpansion,
                )
                if expansion_path.exists()
                else None
            )
            if (directions.context_id, directions.context_hash) != (run.context_id, run.context_hash):
                raise ValueError("stored research directions do not bind the current context")
            if triage is not None and (
                triage.direction_set_id != directions.direction_set_id or triage.direction_set_hash != directions.content_hash
            ):
                raise ValueError("stored triage does not bind its direction set")
            if expansion is not None and (
                triage is None or expansion.triage_id != triage.triage_id or expansion.triage_hash != triage.content_hash
                or expansion.direction_set_id != directions.direction_set_id
                or expansion.direction_set_hash != directions.content_hash
            ):
                raise ValueError("stored expansion does not bind its direction set and triage")
            hierarchical_state = {
                "strategy": "hierarchical",
                "generated_direction_count": len(directions.directions),
                "retained_direction_count": (
                    sum(
                        item.disposition == DirectionDisposition.RETAIN
                        for item in triage.dispositions
                    )
                    if triage is not None
                    else 0
                ),
                "blocking_items": list(triage.blocking_items) if triage else [],
                "provenance_complete": expansion is not None,
                "direction_set_id": directions.direction_set_id,
                "triage_id": triage.triage_id if triage else None,
                "expansion_id": expansion.expansion_id if expansion else None,
            }
    if repository is not None and run.context_id is not None:
        context = repository.load_artifact(run.run_id, "context.yaml", ScientificContextPacket)
        retrieval_summary = {
            "literature_hits": len(context.literature_knowledge_hits),
            "expert_opinion_hits": len(context.expert_opinion_hits),
            "expert_case_hits": len(context.expert_case_hits),
            "workflow_pattern_hits": len(context.workflow_pattern_hits),
            "capability_hits": len(context.capability_hits),
            "graph_expanded_hits": len(context.graph_expanded_hits),
        }
    if repository is not None and run.evidence_packet_id is not None:
        packet = repository.load_artifact(
            run.run_id, "evidence-packet.yaml", ScientificEvidencePacket
        )
        evidence_gaps = [
            {
                "gap_id": gap.gap_id,
                "missing_evidence": gap.missing_evidence,
                "blocking": gap.blocking,
            }
            for gap in packet.evidence_gaps
        ]
    result_feedback = None
    if repository is not None:
        from ..result_feedback import result_feedback_status

        result_feedback = result_feedback_status(run.run_id, repository)
    return {
        "run_id": run.run_id,
        "request": run.original_request,
        "domain": run.domain,
        "current_stage": run.status.value,
        "blocking_items": [item.message for item in run.blocking_items],
        "knowledge_snapshot": run.knowledge_snapshot_id,
        "context_id": run.context_id,
        "evidence_packet_id": run.evidence_packet_id,
        "planning_input_id": run.planning_input_id,
        "retrieval_summary": retrieval_summary,
        "evidence_gaps": evidence_gaps,
        "candidate_count": len(run.candidate_plans),
        "candidate_ids": [item.artifact_id for item in run.candidate_plans],
        "approval_state": run.approval_verdict_id,
        "revision_state": revision_state,
        "hierarchical_planning": hierarchical_state,
        "evidence_resolution": evidence_resolution_state,
        "export_state": run.export_path,
        **(result_feedback or {}),
        "failure": run.failure.model_dump(mode="json") if run.failure else None,
    }


def render_scientific_run_markdown(run: ScientificProblemRun, repository: ScientificProblemRunRepository) -> str:
    knowledge = KnowledgeRepositories(Path(run.knowledge_dir), initialize=False)
    evidence_store = CompositeEvidenceStore(
        knowledge.evidence_store,
        ProjectEvidenceStore(Path(run.state_dir), initialize=False),
    )
    structured_locators: dict[str, list[str]] = {}
    record_index = TrustedKnowledgeValidator.record_index(knowledge)
    for locator in knowledge.structured_evidence_locators.list():
        details = [f"locator_id={locator.locator_id}"]
        if locator.page_number is not None:
            details.append(f"page={locator.page_number}")
        if locator.section_path:
            details.append(f"section={' > '.join(locator.section_path)}")
        if locator.table_id is not None:
            details.append(f"table={locator.table_id}")
        structured_locators.setdefault(locator.evidence_id, []).append(", ".join(details))

    def evidence_locator(evidence_id: str) -> str:
        try:
            evidence = evidence_store.get_evidence(evidence_id)
            evidence_store.verify_evidence_integrity(evidence)
        except (FileNotFoundError, OSError, ValueError):
            return "locator unavailable"
        locations = list(structured_locators.get(evidence_id, ()))
        if evidence.locator:
            locations.insert(0, evidence.locator)
        return "; ".join(locations) or (
            f"source offsets {evidence.start_offset}:{evidence.end_offset}"
        )

    def hit_statement(hit) -> str:
        record = record_index.get((hit.source_type.value, hit.record_id))
        if record is None:
            return hit.rationale
        for attribute in (
            "text",
            "statement",
            "latent_concern",
            "scientific_goal",
            "title",
        ):
            value = getattr(record, attribute, None)
            if isinstance(value, str) and value.strip():
                return value
        if hasattr(record, "quantity"):
            return f"{record.quantity} = {record.value} {record.unit}"
        return hit.rationale

    lines = [
        "# Scientific Problem Compiler run",
        "",
        f"- Run: `{run.run_id}`",
        f"- Status: `{run.status.value}`",
        f"- Domain: `{run.domain}`",
        f"- Knowledge snapshot: `{run.knowledge_snapshot_id or 'not built'}`",
        "",
        "## A. Original vague request",
        "",
        run.original_request,
    ]
    context = None
    packet = None
    proposal = None
    revision_chain = repository.load_revision_chain(run.run_id)
    if run.context_id is not None:
        context = repository.load_artifact(run.run_id, "context.yaml", ScientificContextPacket)
        lines.extend(["", "## B. Retrieved trusted context", ""])
        hit_groups = (
            ("Literature", context.literature_knowledge_hits),
            ("Expert opinions", context.expert_opinion_hits),
            ("Expert cases", context.expert_case_hits),
            ("Graph-expanded", context.graph_expanded_hits),
        )
        for label, hits in hit_groups:
            lines.append(f"### {label}")
            lines.append("")
            if not hits:
                lines.append("None retrieved.")
            for hit in hits:
                lines.append(
                    f"- `{hit.source_type.value}:{hit.record_id}` — {hit_statement(hit)}; "
                    f"authority: `{hit.authority_status or 'not stated'}`; "
                    f"evidence: {', '.join(hit.evidence_refs) or 'none'}"
                )
            lines.append("")
    if run.evidence_packet_id is not None:
        packet = repository.load_artifact(run.run_id, "evidence-packet.yaml", ScientificEvidencePacket)
        lines.extend(["## C. Interpreted scientific problem", ""])
        if packet.source_claims:
            lines.append("Source claims (source statements, not SPC-established facts):")
            lines.append("")
            quotes_by_id = {quote.quote_id: quote for quote in packet.source_quotes}
            for claim in packet.source_claims:
                lines.append(
                    f"- `{claim.claim_id}` [{claim.epistemic_status.value}]: {claim.text} "
                    f"(evidence: {', '.join(claim.evidence_refs)})"
                )
                for quote_id in claim.source_quote_refs:
                    quote = quotes_by_id.get(quote_id)
                    if quote is not None:
                        lines.append(
                            f"  - Exact quote `{quote.quote_id}` from "
                            f"`{quote.source_id}@{quote.source_version}` "
                            f"({evidence_locator(quote.evidence_ref)}): “{quote.text}”"
                        )
        else:
            lines.append("No source claims were extracted.")
        lines.extend(["", "Evidence gaps:", ""])
        if packet.evidence_gaps:
            for gap in packet.evidence_gaps:
                lines.append(f"- `{gap.gap_id}`: {gap.missing_evidence} (blocking: {gap.blocking})")
        else:
            lines.append("- None recorded.")
        lines.extend(["", "Unknowns and assumption candidates:", ""])
        for unknown in packet.unknowns:
            lines.append(f"- Unknown: {unknown}")
        for assumption in packet.assumption_candidates:
            lines.append(f"- Assumption candidate: {assumption}")
        if not packet.unknowns and not packet.assumption_candidates:
            lines.append("- None recorded.")
        lines.extend(["", "Conflicts:", ""])
        if packet.conflict_sets:
            for conflict in packet.conflict_sets:
                lines.append(f"- `{conflict.conflict_id}`: {conflict.topic} — {conflict.resolution_status}")
        else:
            lines.append("- None recorded.")
    if run.planning_proposal_id is not None:
        proposal_path = (
            revision_chain.rounds[-1].planning_proposal.relative_path
            if revision_chain is not None
            else "planning-proposal.yaml"
        )
        proposal = repository.load_artifact(
            run.run_id, proposal_path, PlanningProposalSet
        )
        lines.extend(
            [
                "",
                "### Intent interpretation",
                "",
                f"- Latent concern: {proposal.intent.latent_concern}",
                f"- Atomic questions: {', '.join(proposal.intent.atomic_questions)}",
                f"- Unresolved: {', '.join(proposal.intent.unresolved_points) or 'none'}",
                "",
                "## D. Candidate scientific plans/questions",
                "",
            ]
        )
        for binding in run.candidate_plans:
            plan = repository.load_artifact(run.run_id, binding.relative_path, ScientificQuestionPlan)
            lines.extend(
                [
                    f"### {plan.plan_id}",
                    "",
                    f"- Scientific question: {'; '.join(item.text for item in plan.atomic_questions)}",
                    f"- Primary hypothesis: {plan.hypothesis.primary.text}",
                    f"- Null hypothesis: {plan.hypothesis.null.text}",
                    f"- Scientific model: {plan.model.description.text}",
                    f"- Baselines: {'; '.join(item.description.text for item in plan.comparison_baselines)}",
                    f"- Capabilities: {', '.join(plan.scientific_capability_ids)}",
                    f"- Evidence: {', '.join(item.evidence_id for item in plan.evidence_refs)}",
                    f"- IntentFingerprint `{plan.intent_fingerprint.fingerprint_id}`: {plan.intent_fingerprint.model_dump(mode='json')}",
                    f"- SystemFingerprint `{plan.system_fingerprint.fingerprint_id}`: {plan.system_fingerprint.model_dump(mode='json')}",
                    f"- MethodFingerprint `{plan.method_fingerprint.fingerprint_id}`: {plan.method_fingerprint.model_dump(mode='json')}",
                    f"- Acceptance criteria: {'; '.join(item.statement for item in plan.acceptance_criteria)}",
                    f"- Falsification criteria: {'; '.join(item.statement for item in plan.falsification_criteria)}",
                    f"- Runnable tasks: {any(task.runnable for task in plan.tasks)}",
                    "",
                    "#### Model and observables",
                    "",
                ]
            )
            if plan.model.parameters:
                for parameter in plan.model.parameters:
                    lines.append(
                        f"- Model parameter [{parameter.classification.value}]: "
                        f"{parameter.text}"
                    )
            else:
                lines.append("- Model parameters: not specified; resolve before execution handoff.")
            for observable in plan.observables:
                lines.append(
                    f"- Observable `{observable.observable_id}`: "
                    f"{observable.description.text}; unit: {observable.unit or 'not specified'}"
                )
            lines.extend(["", "#### Assumptions, unknowns, and decisions", ""])
            for assumption in plan.assumptions:
                lines.append(f"- Assumption: {assumption.statement}; impact: {assumption.impact}")
            for unknown in plan.unknowns:
                lines.append(f"- Unknown: {unknown.statement}; resolution: {unknown.resolution}")
            for deviation in plan.proposed_deviations:
                lines.append(
                    f"- Proposed deviation `{deviation.deviation_id}`: "
                    f"{deviation.statement}; reason: {deviation.rationale}"
                )
            for decision in plan.required_human_decisions:
                lines.append(
                    f"- Human decision `{decision.decision_id}`: {decision.question}; "
                    f"options: {', '.join(decision.options)}; before: {decision.required_before}"
                )
            for limitation in plan.limitations:
                lines.append(f"- Limitation: {limitation}")
            if not (
                plan.assumptions
                or plan.unknowns
                or plan.proposed_deviations
                or plan.required_human_decisions
                or plan.limitations
            ):
                lines.append("- None recorded.")
            lines.extend(["", "#### Task handoff and acceptance checks", ""])
            for index, task in enumerate(plan.tasks, start=1):
                lines.extend(
                    [
                        f"##### Step {index}: `{task.task_id}`",
                        "",
                        f"- Objective: {task.scientific_objective}",
                        f"- Capability: `{task.capability_id}`",
                        f"- Depends on: {', '.join(task.depends_on) or 'none'}",
                        "- Inputs:",
                    ]
                )
                inputs = task.model_dump(mode="json")["inputs"]
                if inputs:
                    for key, value in sorted(inputs.items()):
                        lines.append(
                            f"  - `{key}`: {json.dumps(value, ensure_ascii=False, sort_keys=True)}"
                        )
                else:
                    lines.append("  - Not specified; define before execution handoff.")
                lines.extend(
                    [
                        f"- Deliverables: {'; '.join(task.outputs)}",
                        f"- Success checks: {'; '.join(task.success_criteria)}",
                        f"- Falsification relevance: {task.falsification_relevance}",
                        f"- Failure action: {task.failure_policy}",
                        f"- Release gates: {', '.join(task.release_gates) or 'none recorded'}",
                        f"- Evidence: {', '.join(task.evidence_refs) or 'none recorded'}",
                        f"- Provenance to retain: {'; '.join(task.provenance_requirements) or 'none recorded'}",
                        f"- Cost estimate: {task.cost_estimate}",
                        f"- Runnable: {task.runnable}",
                        "",
                    ]
                )
    lines.extend(["## E. Independent approval", ""])
    if run.approval_verdict_id is None:
        lines.append("Awaiting independent approval.")
    else:
        verdict_path = (
            revision_chain.rounds[-1].approval_verdict.relative_path
            if revision_chain is not None
            and revision_chain.rounds[-1].approval_verdict is not None
            else "approval/approval-verdict.yaml"
        )
        verdict = repository.load_artifact(
            run.run_id, verdict_path, ApprovalVerdict
        )
        lines.extend(
            [
                f"- Decision: `{verdict.decision.value}`",
                f"- Required fixes: {', '.join(item.description for item in verdict.required_fixes) or 'none'}",
                f"- Unresolved human choices: {', '.join(verdict.human_decisions_required) or 'none'}",
            ]
        )
    if revision_chain is not None:
        lines.extend(
            [
                "",
                "### Bounded plan revision",
                "",
                f"- Revision budget: {revision_chain.revisions_used}/{revision_chain.max_revisions}",
                f"- Termination: `{revision_chain.termination_reason or 'not terminated'}`",
            ]
        )
        for item in revision_chain.rounds:
            lines.append(
                f"- Round {item.round_index}: plan `{item.candidate_plan.artifact_id}`; "
                f"outcome `{item.outcome}`"
            )
    lines.extend(
        [
            "",
            "## F. Provenance",
            "",
            f"- Context: `{run.context_id or 'not built'}` / `{run.context_hash or 'not built'}`",
            f"- Evidence packet: `{run.evidence_packet_id or 'not built'}` / `{run.evidence_packet_hash or 'not built'}`",
            f"- Planning input: `{run.planning_input_id or 'not built'}` / `{run.planning_input_hash or 'not built'}`",
            f"- Planning proposal: `{run.planning_proposal_id or 'not built'}` / `{run.planning_proposal_hash or 'not built'}`",
            f"- Approval receipt: `{run.approval_receipt_id or 'not available'}` / `{run.approval_receipt_hash or 'not available'}`",
            "",
            "SPC preserves evidence and decision boundaries; it does not replace scientific judgment.",
            "",
        ]
    )
    return "\n".join(lines)
