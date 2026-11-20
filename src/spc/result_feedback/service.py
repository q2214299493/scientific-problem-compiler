from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable

from ..approval import (
    ApprovalContextResolver,
    IndependentApprovalService,
    MockApprovalProvider,
    bind_gate_verdict,
)
from ..compiler import CompilationResult, ScientificProblemCompiler
from ..domains import DomainPackLoader
from ..interpretation.validators import validate_evidence_packet_integrity
from ..models import (
    ApprovalMode,
    ApprovalReviewRecord,
    ApprovalVerdict,
    CurationStatus,
    ComparisonConstraint,
    EvidenceClassification,
    EvidenceGap,
    EvidenceSpan,
    GateVerdict,
    GroundedStatement,
    IndependentApprovalReceipt,
    KnowledgeCurationRecord,
    KnowledgeSnapshot,
    MethodFact,
    ModelFact,
    PlanCompilationReceipt,
    PlanValidationRecord,
    PlanningProposalSet,
    ProjectTrustPolicy,
    ReportedResult,
    ReportedObservation,
    ResultEvidenceAssessment,
    ResultEvidenceIntakeStatus,
    ResultEvidenceMaterializationReceipt,
    ResultEvidenceSubmission,
    ResultEvidenceType,
    ResearchUpdate,
    RetrievalHit,
    RetrievalManifest,
    RetrievalQuery,
    RetrievalSourceType,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificQuestionPlan,
    SuccessorParentBinding,
    SuccessorCandidateSelection,
    SuccessorPlanningInvocation,
    SuccessorPlanningCycle,
    SuccessorPlanningStatus,
    scientific_context_semantic_hash,
    parse_approval_review_input,
)
from ..planning import MockPlanningProvider, PlanningContextResolver
from ..planning.hierarchical import planning_provider_config
from ..planning.mock_provider import build_proposal_set
from ..planning.validators import validate_planning_proposal_set
from ..repositories import KnowledgeRepositories
from ..serialization import content_hash, load_data
from ..validators import (
    build_plan_validation_record,
    validate_approved_plan_authority,
    validate_plan_compilation_receipt,
    validate_plan_validation_record,
    validate_question_plan,
)
from ..workflow.repository import ScientificProblemRunRepository
from .submission import (
    RESULT_FEEDBACK_VERSION,
    ResultFeedbackError,
    build_content_bound,
    build_result_evidence_submission,
)
from .intake import ResultEvidenceIntakeService


class SuccessorPlanningService(ResultEvidenceIntakeService):
    def _validate_reusable_cycle(
        self,
        *,
        parent_run_id: str,
        prefix: str,
        cycle: SuccessorPlanningCycle,
        invocation: SuccessorPlanningInvocation,
    ) -> None:
        try:
            binding = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/parent-binding.yaml",
                SuccessorParentBinding,
            )
            update = self.runs.load_artifact(
                parent_run_id, f"{prefix}/research-update.yaml", ResearchUpdate
            )
            context = self.runs.load_artifact(
                parent_run_id, f"{prefix}/context.yaml", ScientificContextPacket
            )
            packet = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/evidence-packet.yaml",
                ScientificEvidencePacket,
            )
            planning_input = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/planning-input.yaml",
                ScientificPlanningInput,
            )
            if (
                binding.parent_run_id,
                binding.parent_plan_id,
                binding.parent_plan_version,
                binding.parent_plan_hash,
                binding.result_submission_ids,
                binding.result_submission_hashes,
                binding.materialization_receipt_ids,
                binding.materialization_receipt_hashes,
            ) != (
                invocation.parent_run_id,
                invocation.parent_plan_id,
                invocation.parent_plan_version,
                invocation.parent_plan_hash,
                invocation.result_submission_ids,
                invocation.result_submission_hashes,
                invocation.materialization_receipt_ids,
                invocation.materialization_receipt_hashes,
            ):
                raise ValueError("successor parent binding does not match invocation")
            if (
                cycle.parent_binding_id,
                cycle.parent_binding_hash,
                cycle.research_update_id,
                cycle.research_update_hash,
                cycle.context_id,
                cycle.context_hash,
                cycle.evidence_packet_id,
                cycle.evidence_packet_hash,
                cycle.planning_input_id,
                cycle.planning_input_hash,
            ) != (
                binding.binding_id,
                binding.content_hash,
                update.update_id,
                update.content_hash,
                context.context_id,
                context.content_hash,
                packet.packet_id,
                packet.content_hash,
                planning_input.planning_input_id,
                planning_input.content_hash,
            ):
                raise ValueError("successor cycle core artifact binding is invalid")
            if (
                update.parent_binding_id,
                update.parent_binding_hash,
            ) != (binding.binding_id, binding.content_hash):
                raise ValueError("research update does not bind the successor parent")
            if not validate_evidence_packet_integrity(
                packet, context, self.composite_evidence
            ).valid:
                raise ValueError("successor evidence packet failed integrity validation")

            plans: tuple[ScientificQuestionPlan, ...] = ()
            if cycle.planning_proposal_id is not None:
                proposal = self.runs.load_artifact(
                    parent_run_id,
                    f"{prefix}/planning-proposal.yaml",
                    PlanningProposalSet,
                )
                if (
                    cycle.planning_proposal_id,
                    cycle.planning_proposal_hash,
                ) != (proposal.proposal_id, content_hash(proposal)):
                    raise ValueError("successor proposal binding is invalid")
                if not validate_planning_proposal_set(
                    proposal, planning_input
                ).valid:
                    raise ValueError("stored successor proposal is invalid")
                plans = tuple(
                    self.runs.load_artifact(
                        parent_run_id,
                        f"{prefix}/candidates/{plan_id}.yaml",
                        ScientificQuestionPlan,
                    )
                    for plan_id in cycle.candidate_plan_ids
                )
                if tuple(content_hash(item) for item in plans) != (
                    cycle.candidate_plan_hashes
                ):
                    raise ValueError("successor candidate binding is invalid")

            run_root = self.runs.run_dir(parent_run_id)
            receipt_records = tuple(
                self.runs.load_artifact(
                    parent_run_id,
                    path.relative_to(run_root).as_posix(),
                    PlanCompilationReceipt,
                )
                for path in sorted(
                    self.runs.resolve_artifact_path(
                        parent_run_id, f"{prefix}/compilation-receipts"
                    ).glob("*.yaml")
                )
            )
            validation_records = tuple(
                self.runs.load_artifact(
                    parent_run_id,
                    path.relative_to(run_root).as_posix(),
                    PlanValidationRecord,
                )
                for path in sorted(
                    self.runs.resolve_artifact_path(
                        parent_run_id, f"{prefix}/validation"
                    ).glob("*.yaml")
                )
            )
            receipts = {item.plan_id: item for item in receipt_records}
            validations = {item.plan_id: item for item in validation_records}
            plan_ids = {item.plan_id for item in plans}
            if (
                len(receipts) != len(receipt_records)
                or len(validations) != len(validation_records)
                or set(receipts) != plan_ids
                or set(validations) != plan_ids
            ):
                raise ValueError(
                    "successor candidates require exact receipt and validation inventories"
                )
            current_reports = {
                plan.plan_id: validate_question_plan(
                    plan,
                    planning_input.scientific_capabilities,
                    self.composite_evidence,
                )
                for plan in plans
            }
            for plan in plans:
                if not validate_plan_compilation_receipt(
                    plan, receipts[plan.plan_id]
                ).valid or not validate_plan_validation_record(
                    plan,
                    validations[plan.plan_id],
                    current_reports[plan.plan_id],
                ).valid:
                    raise ValueError(
                        "stored successor candidate lineage failed validation"
                    )

            if cycle.selected_plan_id is None:
                return
            selected = next(
                item for item in plans if item.plan_id == cycle.selected_plan_id
            )
            if content_hash(selected) != cycle.selected_plan_hash:
                raise ValueError("selected successor plan binding is invalid")
            if cycle.selection_id is not None:
                selection = self.runs.load_artifact(
                    parent_run_id,
                    f"{prefix}/selection.yaml",
                    SuccessorCandidateSelection,
                )
                if (
                    cycle.selection_id,
                    cycle.selection_hash,
                    selection.invocation_id,
                    selection.invocation_hash,
                    selection.proposal_id,
                    selection.proposal_hash,
                    selection.candidate_id,
                    selection.candidate_hash,
                ) != (
                    selection.selection_id,
                    selection.content_hash,
                    invocation.invocation_id,
                    invocation.content_hash,
                    cycle.planning_proposal_id,
                    cycle.planning_proposal_hash,
                    selected.plan_id,
                    content_hash(selected),
                ):
                    raise ValueError("successor candidate selection binding is invalid")
            approval_prefix = f"{prefix}/approval"
            review_input = parse_approval_review_input(
                load_data(
                    self.runs.resolve_artifact_path(
                        parent_run_id,
                        f"{approval_prefix}/approval-review-input.yaml",
                    )
                )
            )
            review = self.runs.load_artifact(
                parent_run_id,
                f"{approval_prefix}/approval-review.yaml",
                ApprovalReviewRecord,
            )
            verdict = self.runs.load_artifact(
                parent_run_id,
                f"{approval_prefix}/approval-verdict.yaml",
                ApprovalVerdict,
            )
            approval_receipt = self.runs.load_artifact(
                parent_run_id,
                f"{approval_prefix}/independent-approval-receipt.yaml",
                IndependentApprovalReceipt,
            )
            gate = self.runs.load_artifact(
                parent_run_id, f"{approval_prefix}/plan-gate.yaml", GateVerdict
            )
            policy = self.runs.load_artifact(
                parent_run_id,
                f"{approval_prefix}/project-trust-policy.yaml",
                ProjectTrustPolicy,
            )
            if (
                cycle.approval_review_id,
                cycle.approval_review_hash,
                cycle.approval_verdict_id,
                cycle.approval_verdict_hash,
                cycle.approval_receipt_id,
                cycle.approval_receipt_hash,
                cycle.gate_id,
                cycle.gate_hash,
            ) != (
                review.review_id,
                review.content_hash,
                verdict.verdict_id,
                content_hash(verdict),
                approval_receipt.receipt_id,
                approval_receipt.content_hash,
                gate.gate_id,
                content_hash(gate),
            ):
                raise ValueError("successor cycle approval binding is invalid")
            authority = validate_approved_plan_authority(
                selected,
                validations[selected.plan_id],
                review_input,
                review,
                verdict,
                approval_receipt,
                gate,
                policy,
                receipts[selected.plan_id],
                current_report=current_reports[selected.plan_id],
                require_passed=cycle.status == SuccessorPlanningStatus.APPROVED,
            )
            if not authority.valid:
                raise ValueError(
                    "successor approval authority is invalid: "
                    + ", ".join(item.code for item in authority.issues)
                )
        except (FileNotFoundError, OSError, StopIteration, ValueError) as error:
            raise ResultFeedbackError(
                "SUCCESSOR_CYCLE_INTEGRITY_INVALID", str(error)
            ) from error

    def _load_complete_planning_stage(
        self,
        *,
        parent_run_id: str,
        prefix: str,
        invocation: SuccessorPlanningInvocation,
        planner: Any,
    ) -> tuple[CompilationResult, tuple[PlanValidationRecord, ...]]:
        """Load and verify a locally complete planning outcome without inference."""

        try:
            binding = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/parent-binding.yaml",
                SuccessorParentBinding,
            )
            update = self.runs.load_artifact(
                parent_run_id, f"{prefix}/research-update.yaml", ResearchUpdate
            )
            context = self.runs.load_artifact(
                parent_run_id, f"{prefix}/context.yaml", ScientificContextPacket
            )
            packet = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/evidence-packet.yaml",
                ScientificEvidencePacket,
            )
            planning_input = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/planning-input.yaml",
                ScientificPlanningInput,
            )
            proposal = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/planning-proposal.yaml",
                PlanningProposalSet,
            )
            expected = ScientificProblemCompiler(
                planner, evidence_repository=self.composite_evidence
            ).compile_proposal(planning_input, proposal)
            plans = tuple(
                self.runs.load_artifact(
                    parent_run_id,
                    f"{prefix}/candidates/{plan.plan_id}.yaml",
                    ScientificQuestionPlan,
                )
                for plan in expected.candidates
            )
            receipts = tuple(
                self.runs.load_artifact(
                    parent_run_id,
                    f"{prefix}/compilation-receipts/{receipt.receipt_id}.yaml",
                    PlanCompilationReceipt,
                )
                for receipt in expected.compilation_receipts
            )
            validations = tuple(
                self.runs.load_artifact(
                    parent_run_id,
                    f"{prefix}/validation/validation-{content_hash(plan)[:24]}.yaml",
                    PlanValidationRecord,
                )
                for plan in plans
            )
            if (
                tuple(content_hash(item) for item in plans)
                != tuple(content_hash(item) for item in expected.candidates)
                or tuple(content_hash(item) for item in receipts)
                != tuple(
                    content_hash(item) for item in expected.compilation_receipts
                )
            ):
                raise ValueError("stored planning output differs from materialization")
            payload = {
                "invocation_id": invocation.invocation_id,
                "invocation_hash": invocation.content_hash,
                "parent_binding_id": binding.binding_id,
                "parent_binding_hash": binding.content_hash,
                "research_update_id": update.update_id,
                "research_update_hash": update.content_hash,
                "context_id": context.context_id,
                "context_hash": context.content_hash,
                "evidence_packet_id": packet.packet_id,
                "evidence_packet_hash": packet.content_hash,
                "planning_input_id": planning_input.planning_input_id,
                "planning_input_hash": planning_input.content_hash,
                "planning_proposal_id": proposal.proposal_id,
                "planning_proposal_hash": content_hash(proposal),
                "candidate_plan_ids": tuple(item.plan_id for item in plans),
                "candidate_plan_hashes": tuple(content_hash(item) for item in plans),
                "status": SuccessorPlanningStatus.HUMAN_SCIENTIFIC_DECISION_REQUIRED,
            }
            planning_outcome = build_content_bound(
                SuccessorPlanningCycle, "successor-cycle", payload
            )
            self._validate_reusable_cycle(
                parent_run_id=parent_run_id,
                prefix=prefix,
                cycle=planning_outcome,
                invocation=invocation,
            )
            return (
                CompilationResult(
                    candidates=plans,
                    reports=expected.reports,
                    proposal_set=proposal,
                    compilation_receipts=receipts,
                ),
                validations,
            )
        except ResultFeedbackError:
            raise
        except (FileNotFoundError, OSError, ValueError) as error:
            raise ResultFeedbackError(
                "SUCCESSOR_PLANNING_OUTPUT_INCOMPLETE", str(error)
            ) from error

    def _rebuild_complete_approval_cycle(
        self,
        *,
        parent_run_id: str,
        prefix: str,
        invocation: SuccessorPlanningInvocation,
        compilation: CompilationResult,
        selection: SuccessorCandidateSelection,
    ) -> SuccessorPlanningCycle:
        """Reconstruct the final cycle only from a complete local approval chain."""

        try:
            binding = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/parent-binding.yaml",
                SuccessorParentBinding,
            )
            update = self.runs.load_artifact(
                parent_run_id, f"{prefix}/research-update.yaml", ResearchUpdate
            )
            context = self.runs.load_artifact(
                parent_run_id, f"{prefix}/context.yaml", ScientificContextPacket
            )
            packet = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/evidence-packet.yaml",
                ScientificEvidencePacket,
            )
            planning_input = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/planning-input.yaml",
                ScientificPlanningInput,
            )
            proposal = compilation.proposal_set
            if proposal is None:
                raise ValueError("successor planning proposal is missing")
            selected = next(
                item
                for item in compilation.candidates
                if item.plan_id == selection.candidate_id
            )
            review = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/approval/approval-review.yaml",
                ApprovalReviewRecord,
            )
            verdict = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/approval/approval-verdict.yaml",
                ApprovalVerdict,
            )
            approval_receipt = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/approval/independent-approval-receipt.yaml",
                IndependentApprovalReceipt,
            )
            gate = self.runs.load_artifact(
                parent_run_id,
                f"{prefix}/approval/plan-gate.yaml",
                GateVerdict,
            )
            status = (
                SuccessorPlanningStatus.APPROVED
                if gate.passed
                else SuccessorPlanningStatus.REJECTED
            )
            payload = {
                "invocation_id": invocation.invocation_id,
                "invocation_hash": invocation.content_hash,
                "parent_binding_id": binding.binding_id,
                "parent_binding_hash": binding.content_hash,
                "research_update_id": update.update_id,
                "research_update_hash": update.content_hash,
                "context_id": context.context_id,
                "context_hash": context.content_hash,
                "evidence_packet_id": packet.packet_id,
                "evidence_packet_hash": packet.content_hash,
                "planning_input_id": planning_input.planning_input_id,
                "planning_input_hash": planning_input.content_hash,
                "planning_proposal_id": proposal.proposal_id,
                "planning_proposal_hash": content_hash(proposal),
                "candidate_plan_ids": tuple(
                    item.plan_id for item in compilation.candidates
                ),
                "candidate_plan_hashes": tuple(
                    content_hash(item) for item in compilation.candidates
                ),
                "selection_id": selection.selection_id,
                "selection_hash": selection.content_hash,
                "selected_plan_id": selected.plan_id,
                "selected_plan_hash": content_hash(selected),
                "approval_review_id": review.review_id,
                "approval_review_hash": review.content_hash,
                "approval_verdict_id": verdict.verdict_id,
                "approval_verdict_hash": content_hash(verdict),
                "approval_receipt_id": approval_receipt.receipt_id,
                "approval_receipt_hash": approval_receipt.content_hash,
                "gate_id": gate.gate_id,
                "gate_hash": content_hash(gate),
                "status": status,
            }
            cycle = build_content_bound(
                SuccessorPlanningCycle, "successor-cycle", payload
            )
            self._validate_reusable_cycle(
                parent_run_id=parent_run_id,
                prefix=prefix,
                cycle=cycle,
                invocation=invocation,
            )
            return cycle
        except ResultFeedbackError:
            raise
        except (FileNotFoundError, OSError, StopIteration, ValueError) as error:
            raise ResultFeedbackError(
                "SUCCESSOR_APPROVAL_OUTPUT_INCOMPLETE", str(error)
            ) from error

    def _load_materialized_records(
        self,
        run_id: str,
        submission_id: str,
        receipt: ResultEvidenceMaterializationReceipt,
    ) -> tuple[
        tuple[ReportedResult, ...],
        tuple[ReportedObservation, ...],
        tuple[MethodFact, ...],
        tuple[ModelFact, ...],
    ]:
        prefix = self.artifacts.accepted_prefix(submission_id)
        results = tuple(
            self.runs.load_artifact(
                run_id,
                f"{prefix}/r/{record_id[-16:]}.yaml",
                ReportedResult,
            )
            for record_id in receipt.reported_result_hashes
        )
        observations = tuple(
            self.runs.load_artifact(
                run_id,
                f"{prefix}/o/{record_id[-16:]}.yaml",
                ReportedObservation,
            )
            for record_id in receipt.reported_observation_hashes
        )
        methods = tuple(
            self.runs.load_artifact(
                run_id,
                f"{prefix}/m/{record_id[-16:]}.yaml",
                MethodFact,
            )
            for record_id in receipt.method_fact_hashes
        )
        models = tuple(
            self.runs.load_artifact(
                run_id,
                f"{prefix}/d/{record_id[-16:]}.yaml",
                ModelFact,
            )
            for record_id in receipt.model_fact_hashes
        )
        if (
            {item.result_id: content_hash(item) for item in results}
            != receipt.reported_result_hashes
            or {
                item.observation_id: content_hash(item) for item in observations
            }
            != receipt.reported_observation_hashes
            or {item.fact_id: content_hash(item) for item in methods}
            != receipt.method_fact_hashes
            or {item.fact_id: content_hash(item) for item in models}
            != receipt.model_fact_hashes
        ):
            raise ResultFeedbackError(
                "MATERIALIZED_RESULT_TAMPERED",
                "materialized scientific records differ from their receipt",
            )
        return results, observations, methods, models

    @staticmethod
    def _snapshot_with_results(
        parent: KnowledgeSnapshot,
        evidence: Iterable[EvidenceSpan],
        sources: Iterable[Any],
        curations: Iterable[KnowledgeCurationRecord],
        records: Iterable[Any],
    ) -> KnowledgeSnapshot:
        payload = parent.model_dump(
            mode="python", exclude={"snapshot_id", "created_at"}
        )
        payload["evidence_span_hashes"] = {
            **payload["evidence_span_hashes"],
            **{item.evidence_id: content_hash(item) for item in evidence},
        }
        payload["evidence_source_versions"] = {
            **payload["evidence_source_versions"],
            **{
                f"{item.source_id}@{item.version}": item.content_sha256
                for item in sources
            },
        }
        payload["curation_record_hashes"] = {
            **payload.get("curation_record_hashes", {}),
            **{item.curation_id: item.content_hash for item in curations},
        }
        trusted = dict(payload.get("trusted_record_hashes", {}))
        for item in records:
            if isinstance(item, ReportedResult):
                key = f"reported_result:{item.result_id}"
            elif isinstance(item, ReportedObservation):
                key = f"reported_observation:{item.observation_id}"
            elif isinstance(item, MethodFact):
                key = f"method_fact:{item.fact_id}"
            elif isinstance(item, ModelFact):
                key = f"model_fact:{item.fact_id}"
            else:
                continue
            trusted[key] = content_hash(item)
        payload["trusted_record_hashes"] = trusted
        identity = KnowledgeSnapshot.model_construct(
            snapshot_id="pending",
            created_at=datetime.now(timezone.utc),
            **payload,
        ).model_dump(
            mode="json",
            exclude={"snapshot_id", "created_at"},
            exclude_defaults=True,
        )
        return KnowledgeSnapshot(
            snapshot_id=f"snapshot-{content_hash(identity)[:24]}",
            created_at=datetime.now(timezone.utc),
            **payload,
        )

    def _successor_context(
        self,
        parent_context: ScientificContextPacket,
        binding: SuccessorParentBinding,
        accepted_evidence: tuple[EvidenceSpan, ...],
        curations: tuple[KnowledgeCurationRecord, ...],
        records: tuple[Any, ...],
        request: str,
    ) -> ScientificContextPacket:
        sources = tuple(
            self.composite_evidence.get_source(item.source_id, item.source_version)
            for item in accepted_evidence
        )
        snapshot = self._snapshot_with_results(
            parent_context.knowledge_snapshot,
            accepted_evidence,
            sources,
            curations,
            records,
        )
        query_payload = parent_context.retrieval_query.model_dump(
            mode="python", exclude={"query_id"}
        )
        query_payload["raw_request"] = request
        query = RetrievalQuery(
            query_id=f"query-{content_hash(query_payload)[:24]}",
            **query_payload,
        )
        def normalized_hits(hits: Iterable[RetrievalHit]) -> tuple[RetrievalHit, ...]:
            return tuple(
                item.model_copy(update={"retriever_version": RESULT_FEEDBACK_VERSION})
                for item in hits
            )

        parent_evidence_hits = normalized_hits(parent_context.evidence_hits)
        literature_hits = normalized_hits(parent_context.literature_knowledge_hits)
        expert_opinion_hits = normalized_hits(parent_context.expert_opinion_hits)
        expert_case_hits = normalized_hits(parent_context.expert_case_hits)
        workflow_hits = normalized_hits(parent_context.workflow_pattern_hits)
        capability_hits = normalized_hits(parent_context.capability_hits)
        graph_hits = normalized_hits(parent_context.graph_expanded_hits)
        existing_hit_ids = {item.hit_id for item in parent_evidence_hits}
        result_hits: list[RetrievalHit] = []
        for evidence in accepted_evidence:
            hit_payload = {
                "source_type": RetrievalSourceType.EVIDENCE_SPAN,
                "record_id": evidence.evidence_id,
                "score": 1.0,
                "matched_terms": ("accepted downstream result",),
                "rationale": (
                    "Explicitly curated project result evidence bound to an approved parent task."
                ),
                "evidence_refs": (evidence.evidence_id,),
                "retriever_version": RESULT_FEEDBACK_VERSION,
                "record_hash": content_hash(evidence),
                "source_class": "project_result_evidence",
                "source_refs": (
                    f"{evidence.source_id}@{evidence.source_version}",
                ),
                "authority_status": "accepted_project_result",
            }
            hit_id = f"hit-{content_hash(hit_payload)[:24]}"
            if hit_id not in existing_hit_ids:
                result_hits.append(RetrievalHit(hit_id=hit_id, **hit_payload))
        evidence_hits = (*parent_evidence_hits, *result_hits)
        categorized = (
            evidence_hits,
            literature_hits,
            expert_opinion_hits,
            expert_case_hits,
            workflow_hits,
            capability_hits,
            graph_hits,
        )
        result_ids = tuple(item.hit_id for group in categorized for item in group)
        result_hashes = tuple(content_hash(item) for group in categorized for item in group)
        manifest_identity = {
            "query_hash": content_hash(query),
            "domain_pack_id": parent_context.domain,
            "domain_pack_version": parent_context.retrieval_manifest.domain_pack_version,
            "knowledge_snapshot_id": snapshot.snapshot_id,
            "retriever_version": RESULT_FEEDBACK_VERSION,
            "result_ids": result_ids,
            "result_hashes": result_hashes,
            "retrieval_policy_hash": binding.content_hash,
        }
        manifest = RetrievalManifest(
            retrieval_id=f"retrieval-{content_hash(manifest_identity)[:24]}",
            timestamp=datetime.now(timezone.utc),
            **manifest_identity,
        )
        statements = list(parent_context.retrieved_statements)
        known_refs = {
            ref for statement in statements for ref in statement.evidence_refs
        }
        for evidence in accepted_evidence:
            if evidence.evidence_id in known_refs:
                continue
            statements.append(
                GroundedStatement(
                    statement_id=(
                        "result-statement-"
                        + content_hash(
                            {
                                "evidence_id": evidence.evidence_id,
                                "text": evidence.text,
                            }
                        )[:24]
                    ),
                    text=evidence.text,
                    classification=EvidenceClassification.EVIDENCE,
                    evidence_refs=(evidence.evidence_id,),
                )
            )
        payload = {
            "context_id": f"context-{manifest.retrieval_id.removeprefix('retrieval-')}",
            "original_request": request,
            "domain": parent_context.domain,
            "retrieval_query": query,
            "evidence_hits": evidence_hits,
            "literature_knowledge_hits": literature_hits,
            "expert_opinion_hits": expert_opinion_hits,
            "expert_case_hits": expert_case_hits,
            "workflow_pattern_hits": workflow_hits,
            "capability_hits": capability_hits,
            "graph_expanded_hits": graph_hits,
            "knowledge_retrieval_context": None,
            "retrieved_statements": tuple(statements),
            "assumptions": parent_context.assumptions,
            "conflicting_evidence": parent_context.conflicting_evidence,
            "unknowns": parent_context.unknowns,
            "retrieval_manifest": manifest,
            "knowledge_snapshot": snapshot,
        }
        return ScientificContextPacket(
            **payload,
            content_hash=scientific_context_semantic_hash(payload),
        )

    @staticmethod
    def _research_update(
        binding: SuccessorParentBinding,
        plan: ScientificQuestionPlan,
        submissions: tuple[ResultEvidenceSubmission, ...],
        receipts: tuple[ResultEvidenceMaterializationReceipt, ...],
        results: tuple[ReportedResult, ...],
        observations: tuple[ReportedObservation, ...],
        follow_up_request: str | None,
    ) -> ResearchUpdate:
        reported_observations = tuple(
            f"{item.quantity} = {item.value} {item.unit}" for item in results
        ) + tuple(
            f"{item.quantity} = {item.qualitative_value}" for item in observations
        )
        observable_by_id = {
            item.observable_id: item.description.text for item in plan.observables
        }

        def covered(criterion) -> bool:
            description = observable_by_id.get(criterion.observable_id, "").casefold()
            return any(
                result.quantity.casefold() in description
                or description in result.quantity.casefold()
                for result in (*results, *observations)
                if description
            )

        affected_acceptance = tuple(
            item.criterion_id for item in plan.acceptance_criteria if covered(item)
        )
        affected_falsification = tuple(
            item.criterion_id for item in plan.falsification_criteria if covered(item)
        )
        all_criteria = (*plan.acceptance_criteria, *plan.falsification_criteria)
        unresolved = tuple(item.criterion_id for item in all_criteria if not covered(item))
        gaps = tuple(
            f"Missing task output: {output}"
            for receipt in receipts
            for output in receipt.missing_observables
        )
        failed = all(
            item.result_type == ResultEvidenceType.FAILED_EXECUTION
            for item in submissions
        )
        if failed and not results and not observations:
            status = SuccessorPlanningStatus.INSUFFICIENT_RESULT_EVIDENCE
            why = (
                "The downstream execution failed; diagnostic evidence is preserved but "
                "cannot be treated as hypothesis falsification."
            )
        elif follow_up_request is not None or unresolved or gaps:
            status = SuccessorPlanningStatus.FOLLOW_UP_PLAN_REQUIRED
            why = follow_up_request or (
                "Reported observations do not cover all declared criteria or required outputs."
            )
        else:
            status = SuccessorPlanningStatus.HUMAN_SCIENTIFIC_DECISION_REQUIRED
            why = (
                "All declared observables are structurally present, but interpreting whether "
                "they resolve the scientific question requires an explicit human decision."
            )
        payload = {
            "parent_binding_id": binding.binding_id,
            "parent_binding_hash": binding.content_hash,
            "previous_questions": tuple(item.text for item in plan.atomic_questions),
            "previous_hypotheses": (
                plan.hypothesis.primary.text,
                plan.hypothesis.null.text,
            ),
            "new_reported_observations": reported_observations,
            "affected_acceptance_criteria": affected_acceptance,
            "affected_falsification_criteria": affected_falsification,
            "unresolved_criteria": unresolved,
            "new_conflicts": (),
            "new_evidence_gaps": gaps,
            "why_successor_plan_is_needed": why,
            "recommended_status": status,
        }
        return build_content_bound(ResearchUpdate, "research-update", payload)

    def _successor_packet(
        self,
        parent_packet: ScientificEvidencePacket,
        context: ScientificContextPacket,
        binding: SuccessorParentBinding,
        results: tuple[ReportedResult, ...],
        observations: tuple[ReportedObservation, ...],
        methods: tuple[MethodFact, ...],
        models: tuple[ModelFact, ...],
        receipts: tuple[ResultEvidenceMaterializationReceipt, ...],
        capability_id: str,
    ) -> ScientificEvidencePacket:
        evidence_refs = tuple(
            dict.fromkeys(
                evidence_id
                for receipt in receipts
                for evidence_id in receipt.accepted_evidence_ids
            )
        )
        gaps = tuple(
            EvidenceGap(
                gap_id=(
                    "evidence-gap-"
                    + content_hash(
                        {
                            "receipt_id": receipt.receipt_id,
                            "missing_output": output,
                        }
                    )[:24]
                ),
                scientific_question=(
                    "How should the missing downstream output be obtained or reviewed?"
                ),
                missing_evidence=output,
                why_it_matters="The parent task declared this output as required.",
                blocking=True,
                candidate_capabilities=(capability_id,),
                evidence_refs=evidence_refs,
            )
            for receipt in receipts
            for output in receipt.missing_observables
        )
        all_results = (*parent_packet.reported_results, *results)
        existing_targets = {
            item.comparison_target for item in parent_packet.comparison_constraints
        }
        result_constraints: list[ComparisonConstraint] = []
        for left, right in combinations(all_results, 2):
            if left.quantity != right.quantity:
                continue
            left_fields = {
                **{
                    f"system_context.{key}": value
                    for key, value in left.system_context.items()
                },
                **{
                    f"method_context.{key}": value
                    for key, value in left.method_context.items()
                },
            }
            right_fields = {
                **{
                    f"system_context.{key}": value
                    for key, value in right.system_context.items()
                },
                **{
                    f"method_context.{key}": value
                    for key, value in right.method_context.items()
                },
            }
            mismatches = tuple(
                sorted(
                    field
                    for field in set(left_fields) | set(right_fields)
                    if left_fields.get(field) != right_fields.get(field)
                )
            )
            if left.unit != right.unit:
                mismatches = (*mismatches, "unit")
            target = f"{left.result_id} vs {right.result_id}"
            if not mismatches or target in existing_targets:
                continue
            constraint_payload = {
                "comparison_target": target,
                "must_match_fields": mismatches,
                "may_vary_fields": (),
                "disclosure_required_fields": mismatches,
                "rationale": (
                    "Sequential reported results have differing contexts and must not "
                    "be treated as directly comparable without explicit reconciliation."
                ),
                "evidence_refs": tuple(
                    dict.fromkeys((*left.evidence_refs, *right.evidence_refs))
                ),
            }
            result_constraints.append(
                ComparisonConstraint(
                    constraint_id=(
                        "comparison-constraint-"
                        + content_hash(constraint_payload)[:24]
                    ),
                    **constraint_payload,
                )
            )
        manifest = dict(parent_packet.provenance_manifest)
        manifest.update(
            {
                "context_id": context.context_id,
                "context_hash": context.content_hash,
                "retrieval_id": context.retrieval_manifest.retrieval_id,
                "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
                "successor_parent_binding_id": binding.binding_id,
                "successor_parent_binding_hash": binding.content_hash,
                "result_submission_ids": binding.result_submission_ids,
                "result_materialization_receipt_ids": (
                    binding.materialization_receipt_ids
                ),
                "result_record_lineage": {
                    record_id: next(
                        receipt.submission_id
                        for receipt in receipts
                        if record_id in {
                            **receipt.reported_result_hashes,
                            **receipt.reported_observation_hashes,
                        }
                    )
                    for record_id in (
                        *(result.result_id for result in results),
                        *(item.observation_id for item in observations),
                    )
                },
            }
        )
        identity = {
            "context_id": context.context_id,
            "context_hash": context.content_hash,
            "source_quotes": parent_packet.source_quotes,
            "source_claims": parent_packet.source_claims,
            "reported_results": all_results,
            **(
                {
                    "reported_observations": (
                        *parent_packet.reported_observations,
                        *observations,
                    )
                }
                if parent_packet.reported_observations or observations
                else {}
            ),
            "method_facts": (*parent_packet.method_facts, *methods),
            "model_facts": (*parent_packet.model_facts, *models),
            "evidence_assessments": parent_packet.evidence_assessments,
            "conflict_sets": parent_packet.conflict_sets,
            "comparison_constraints": (
                *parent_packet.comparison_constraints,
                *result_constraints,
            ),
            "evidence_gaps": (*parent_packet.evidence_gaps, *gaps),
            "unknowns": parent_packet.unknowns,
            "assumption_candidates": parent_packet.assumption_candidates,
            "capability_candidates": parent_packet.capability_candidates,
            "provenance_manifest": manifest,
        }
        packet_id = f"evidence-packet-{content_hash(identity)[:24]}"
        packet_payload = {"packet_id": packet_id, **identity}
        packet = ScientificEvidencePacket(
            **packet_payload, content_hash=content_hash(packet_payload)
        )
        report = validate_evidence_packet_integrity(
            packet, context, self.composite_evidence
        )
        if not report.valid:
            raise ResultFeedbackError(
                "SUCCESSOR_EVIDENCE_PACKET_INVALID",
                ", ".join(item.code for item in report.issues),
            )
        return packet

    @staticmethod
    def _bind_successor_proposal(
        proposal: PlanningProposalSet,
        planning_input: ScientificPlanningInput,
        binding: SuccessorParentBinding,
    ) -> PlanningProposalSet:
        result_evidence = tuple(binding.accepted_evidence_ids)

        def with_result_evidence(values: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(dict.fromkeys((*values, *result_evidence)))

        candidates = tuple(
            candidate.model_copy(
                update={
                    "evidence_refs": with_result_evidence(candidate.evidence_refs),
                    "observables": tuple(
                        item.model_copy(
                            update={
                                "evidence_refs": with_result_evidence(
                                    item.evidence_refs
                                )
                            }
                        )
                        for item in candidate.observables
                    ),
                    "comparison_baselines": tuple(
                        item.model_copy(
                            update={
                                "evidence_refs": with_result_evidence(
                                    item.evidence_refs
                                )
                            }
                        )
                        for item in candidate.comparison_baselines
                    ),
                    "task_drafts": tuple(
                        item.model_copy(
                            update={
                                "evidence_refs": with_result_evidence(
                                    item.evidence_refs
                                )
                            }
                        )
                        for item in candidate.task_drafts
                    ),
                }
            )
            for candidate in proposal.candidates
        )
        config = {
            **dict(proposal.provider_config),
            "successor_parent_plan_id": binding.parent_plan_id,
            "successor_parent_binding_id": binding.binding_id,
            "successor_parent_binding_hash": binding.content_hash,
            "successor_cycle_index": binding.successor_cycle_index,
            "result_submission_ids": binding.result_submission_ids,
        }
        return build_proposal_set(
            planning_input,
            provider_id=proposal.provider_id,
            provider_version=proposal.provider_version,
            provider_config=config,
            intent=proposal.intent,
            ambiguity_assessment=proposal.ambiguity_assessment,
            candidates=candidates,
        )

    def compile(
        self,
        *,
        parent_run_id: str,
        submission_ids: tuple[str, ...],
        follow_up_request: str | None = None,
        planning_provider: Any | None = None,
        approval_provider: Any | None = None,
        selected_candidate_id: str | None = None,
    ) -> SuccessorPlanningCycle:
        if not submission_ids or len(set(submission_ids)) != len(submission_ids):
            raise ResultFeedbackError(
                "INVALID_SUCCESSOR_SUBMISSIONS",
                "successor planning requires unique accepted submission IDs",
            )
        with self.runs.acquire_run_lock(parent_run_id):
            submissions = tuple(
                self.artifacts.load_submission(parent_run_id, item) for item in submission_ids
            )
            if any(item.parent_run_id != parent_run_id for item in submissions):
                raise ResultFeedbackError(
                    "CROSS_RUN_RESULT",
                    "all successor results must belong to the requested parent run",
                )
            parent = self.parent_authority.load_parent(submissions[0])
            if any(
                (
                    item.parent_plan_id,
                    item.parent_plan_version,
                    item.parent_plan_hash,
                )
                != (
                    parent.plan.plan_id,
                    parent.plan.version,
                    content_hash(parent.plan),
                )
                for item in submissions
            ):
                raise ResultFeedbackError(
                    "CROSS_PLAN_RESULT",
                    "successor results must bind the same exact parent plan version",
                )
            receipts = tuple(
                self.artifacts.load_receipt(parent_run_id, item) for item in submission_ids
            )
            curations = tuple(
                self.artifacts.current_curation(parent_run_id, item) for item in submission_ids
            )
            if any(item is None or item.status != CurationStatus.ACCEPTED for item in curations):
                raise ResultFeedbackError(
                    "RESULT_REQUIRES_CURATION",
                    "every successor result requires explicit ACCEPTED curation",
                )
            accepted_curations = tuple(item for item in curations if item is not None)
            for submission, receipt, curation in zip(
                submissions, receipts, accepted_curations, strict=True
            ):
                if (
                    receipt.submission_id != submission.submission_id
                    or receipt.submission_hash != submission.content_hash
                    or receipt.curation_id != curation.curation_id
                    or receipt.curation_hash != curation.content_hash
                    or receipt.parent_plan_hash != content_hash(parent.plan)
                ):
                    raise ResultFeedbackError(
                        "RESULT_MATERIALIZATION_BINDING_INVALID",
                        "accepted result receipt does not bind its exact submission/curation/plan",
                    )
                self.artifacts.verify_archived_artifacts(submission)

            bound_results = tuple(
                sorted(
                    zip(submissions, receipts, accepted_curations, strict=True),
                    key=lambda item: (
                        item[0].submission_id,
                        item[0].content_hash,
                        item[1].receipt_id,
                        item[1].content_hash,
                    ),
                )
            )
            submissions = tuple(item[0] for item in bound_results)
            receipts = tuple(item[1] for item in bound_results)
            accepted_curations = tuple(item[2] for item in bound_results)

            planner = planning_provider or MockPlanningProvider()
            approver = approval_provider or MockApprovalProvider()
            invocation_payload = {
                "parent_run_id": parent_run_id,
                "parent_plan_id": parent.plan.plan_id,
                "parent_plan_version": parent.plan.version,
                "parent_plan_hash": content_hash(parent.plan),
                "result_submission_ids": tuple(
                    item.submission_id for item in submissions
                ),
                "result_submission_hashes": tuple(
                    item.content_hash for item in submissions
                ),
                "materialization_receipt_ids": tuple(
                    item.receipt_id for item in receipts
                ),
                "materialization_receipt_hashes": tuple(
                    item.content_hash for item in receipts
                ),
                "follow_up_request": follow_up_request,
                "planning_provider_id": planner.provider_id,
                "planning_provider_version": planner.provider_version,
                "planning_provider_config_hash": content_hash(
                    planning_provider_config(planner)
                ),
            }
            invocation = build_content_bound(
                SuccessorPlanningInvocation,
                "successor-invocation",
                invocation_payload,
            )
            successor_root = self.runs.resolve_artifact_path(
                parent_run_id, "successor-planning"
            )
            cycle_indexes: list[int] = []
            matching_prefix: str | None = None
            matching_cycle: SuccessorPlanningCycle | None = None
            if successor_root.exists():
                seen_cycle_indexes: set[int] = set()
                for directory in successor_root.glob("cycle-*"):
                    if directory.is_symlink():
                        raise ResultFeedbackError(
                            "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                            f"symlinked successor cycle directory: {directory.name}",
                        )
                    if not directory.is_dir():
                        continue
                    try:
                        suffix = directory.name.removeprefix("cycle-")
                        cycle_index_value = int(suffix)
                    except ValueError as error:
                        raise ResultFeedbackError(
                            "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                            f"invalid successor cycle directory: {directory.name}",
                        ) from error
                    if (
                        cycle_index_value < 1
                        or suffix != str(cycle_index_value)
                        or cycle_index_value in seen_cycle_indexes
                    ):
                        raise ResultFeedbackError(
                            "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                            f"non-canonical successor cycle directory: {directory.name}",
                        )
                    seen_cycle_indexes.add(cycle_index_value)
                    cycle_indexes.append(cycle_index_value)
                    lineage_path = directory / "parent-binding.yaml"
                    if lineage_path.is_symlink():
                        raise ResultFeedbackError(
                            "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                            f"symlinked successor parent binding: {lineage_path.name}",
                        )
                    if lineage_path.exists():
                        lineage = self.runs.load_artifact(
                            parent_run_id,
                            lineage_path.relative_to(
                                self.runs.run_dir(parent_run_id)
                            ).as_posix(),
                            SuccessorParentBinding,
                        )
                        if lineage.successor_cycle_index != cycle_index_value:
                            raise ResultFeedbackError(
                                "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                                "successor cycle directory index does not match "
                                "its parent-binding lineage index",
                            )
                    invocation_path = directory / "invocation.yaml"
                    if not invocation_path.exists():
                        continue
                    relative = invocation_path.relative_to(
                        self.runs.run_dir(parent_run_id)
                    ).as_posix()
                    stored_invocation = self.runs.load_artifact(
                        parent_run_id, relative, SuccessorPlanningInvocation
                    )
                    if stored_invocation.invocation_id != invocation.invocation_id:
                        continue
                    if matching_prefix is not None:
                        raise ResultFeedbackError(
                            "AMBIGUOUS_SUCCESSOR_INVOCATION",
                            "the same successor invocation is bound to multiple cycles",
                        )
                    matching_prefix = (
                        directory.relative_to(self.runs.run_dir(parent_run_id)).as_posix()
                    )
                    cycle_path = directory / "cycle.yaml"
                    if cycle_path.exists():
                        matching_cycle = self.runs.load_artifact(
                            parent_run_id,
                            cycle_path.relative_to(
                                self.runs.run_dir(parent_run_id)
                            ).as_posix(),
                            SuccessorPlanningCycle,
                        )
                        if lineage_path.exists() and (
                            matching_cycle.parent_binding_id,
                            matching_cycle.parent_binding_hash,
                        ) != (lineage.binding_id, lineage.content_hash):
                            raise ResultFeedbackError(
                                "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                                "completed successor cycle does not bind the "
                                "parent-binding stored in its directory",
                            )
            if matching_cycle is not None:
                if (
                    matching_cycle.invocation_id != invocation.invocation_id
                    or matching_cycle.invocation_hash != invocation.content_hash
                ):
                    raise ResultFeedbackError(
                        "SUCCESSOR_INVOCATION_BINDING_INVALID",
                        "stored successor cycle does not bind its exact invocation",
                    )
                if matching_prefix is None:
                    raise ResultFeedbackError(
                        "SUCCESSOR_CYCLE_INTEGRITY_INVALID",
                        "completed successor cycle has no invocation directory",
                    )
                self._validate_reusable_cycle(
                    parent_run_id=parent_run_id,
                    prefix=matching_prefix,
                    cycle=matching_cycle,
                    invocation=invocation,
                )
                if (
                    selected_candidate_id is not None
                    and matching_cycle.selected_plan_id != selected_candidate_id
                ):
                    raise ResultFeedbackError(
                        "SUCCESSOR_SELECTION_CONFLICT",
                        "the completed invocation is bound to another candidate",
                    )
                return matching_cycle
            resuming_existing = matching_prefix is not None
            if matching_prefix is not None:
                prefix = matching_prefix
                cycle_index = int(prefix.rsplit("-", 1)[1])
            else:
                cycle_index = max(cycle_indexes, default=0) + 1
                prefix = f"successor-planning/cycle-{cycle_index}"
                self.runs.write_exclusive_artifact(
                    parent_run_id,
                    f"{prefix}/invocation.yaml",
                    "successor_planning_invocation",
                    invocation.invocation_id,
                    invocation,
                )
            binding_payload = {
                "parent_run_id": parent_run_id,
                "parent_plan_id": parent.plan.plan_id,
                "parent_plan_version": parent.plan.version,
                "parent_plan_hash": content_hash(parent.plan),
                "result_submission_ids": tuple(item.submission_id for item in submissions),
                "result_submission_hashes": tuple(item.content_hash for item in submissions),
                "materialization_receipt_ids": tuple(item.receipt_id for item in receipts),
                "materialization_receipt_hashes": tuple(item.content_hash for item in receipts),
                "accepted_evidence_ids": tuple(
                    sorted({
                        evidence_id
                        for receipt in receipts
                        for evidence_id in receipt.accepted_evidence_ids
                    })
                ),
                "successor_cycle_index": cycle_index,
            }
            binding = build_content_bound(
                SuccessorParentBinding,
                "successor-parent-binding",
                binding_payload,
            )
            self.runs.write_immutable_artifact(
                parent_run_id,
                f"{prefix}/parent-binding.yaml",
                "successor_parent_binding",
                binding.binding_id,
                binding,
            )

            records_by_submission = tuple(
                self._load_materialized_records(
                    parent_run_id, submission.submission_id, receipt
                )
                for submission, receipt in zip(submissions, receipts, strict=True)
            )
            results = tuple(item for group in records_by_submission for item in group[0])
            observations = tuple(
                item for group in records_by_submission for item in group[1]
            )
            methods = tuple(item for group in records_by_submission for item in group[2])
            models = tuple(item for group in records_by_submission for item in group[3])
            update = self._research_update(
                binding,
                parent.plan,
                submissions,
                receipts,
                results,
                observations,
                follow_up_request,
            )
            self.runs.write_immutable_artifact(
                parent_run_id,
                f"{prefix}/research-update.yaml",
                "research_update",
                update.update_id,
                update,
            )
            repositories = KnowledgeRepositories(self.knowledge_dir)
            pack = DomainPackLoader().load(parent.plan.domain)
            repositories.load_expert_cases(pack.expert_cases)
            repositories.load_workflow_patterns(pack.workflow_patterns)
            repositories.load_capabilities(pack.capabilities)
            if resuming_existing:
                context = self.runs.load_artifact(
                    parent_run_id,
                    f"{prefix}/context.yaml",
                    ScientificContextPacket,
                )
                packet = self.runs.load_artifact(
                    parent_run_id,
                    f"{prefix}/evidence-packet.yaml",
                    ScientificEvidencePacket,
                )
                planning_input = self.runs.load_artifact(
                    parent_run_id,
                    f"{prefix}/planning-input.yaml",
                    ScientificPlanningInput,
                )
                if (
                    packet.context_id != context.context_id
                    or packet.context_hash != context.content_hash
                    or planning_input.context_id != context.context_id
                    or planning_input.context_hash != context.content_hash
                    or planning_input.evidence_packet_id != packet.packet_id
                    or planning_input.evidence_packet_hash != packet.content_hash
                    or not validate_evidence_packet_integrity(
                        packet, context, self.composite_evidence
                    ).valid
                ):
                    raise ResultFeedbackError(
                        "SUCCESSOR_CYCLE_INTEGRITY_INVALID",
                        "stored successor core artifacts failed binding validation",
                    )
            else:
                parent_context = self.runs.load_artifact(
                    parent_run_id,
                    parent.context_relative_path,
                    ScientificContextPacket,
                )
                parent_packet = self.runs.load_artifact(
                    parent_run_id,
                    parent.evidence_packet_relative_path,
                    ScientificEvidencePacket,
                )
                accepted_evidence = tuple(
                    self.composite_evidence.get_evidence(item)
                    for item in binding.accepted_evidence_ids
                )
                request = parent_context.original_request
                if follow_up_request is not None:
                    request = f"{request}\nFollow-up request: {follow_up_request}"
                context = self._successor_context(
                    parent_context,
                    binding,
                    accepted_evidence,
                    accepted_curations,
                    (*results, *observations, *methods, *models),
                    request,
                )
                packet = self._successor_packet(
                    parent_packet,
                    context,
                    binding,
                    results,
                    observations,
                    methods,
                    models,
                    receipts,
                    parent.task.capability_id,
                )
                planning_input = PlanningContextResolver().resolve(
                    context, packet, repositories, self.composite_evidence
                )
                for name, artifact_type, identifier, value in (
                    (
                        "context.yaml",
                        "scientific_context_packet",
                        context.context_id,
                        context,
                    ),
                    (
                        "evidence-packet.yaml",
                        "scientific_evidence_packet",
                        packet.packet_id,
                        packet,
                    ),
                    (
                        "planning-input.yaml",
                        "scientific_planning_input",
                        planning_input.planning_input_id,
                        planning_input,
                    ),
                ):
                    self.runs.write_immutable_artifact(
                        parent_run_id,
                        f"{prefix}/{name}",
                        artifact_type,
                        identifier,
                        value,
                    )

            if update.recommended_status != SuccessorPlanningStatus.FOLLOW_UP_PLAN_REQUIRED:
                cycle_payload = {
                    "invocation_id": invocation.invocation_id,
                    "invocation_hash": invocation.content_hash,
                    "parent_binding_id": binding.binding_id,
                    "parent_binding_hash": binding.content_hash,
                    "research_update_id": update.update_id,
                    "research_update_hash": update.content_hash,
                    "context_id": context.context_id,
                    "context_hash": context.content_hash,
                    "evidence_packet_id": packet.packet_id,
                    "evidence_packet_hash": packet.content_hash,
                    "planning_input_id": planning_input.planning_input_id,
                    "planning_input_hash": planning_input.content_hash,
                    "status": update.recommended_status,
                }
                cycle = build_content_bound(
                    SuccessorPlanningCycle, "successor-cycle", cycle_payload
                )
                self.runs.write_immutable_artifact(
                    parent_run_id,
                    f"{prefix}/cycle.yaml",
                    "successor_planning_cycle",
                    cycle.cycle_id,
                    cycle,
                )
                return cycle

            planning_marker = self.runs.resolve_artifact_path(
                parent_run_id, f"{prefix}/provider-calls/planning.yaml"
            )
            if planning_marker.exists():
                try:
                    compilation, loaded_validations = (
                        self._load_complete_planning_stage(
                            parent_run_id=parent_run_id,
                            prefix=prefix,
                            invocation=invocation,
                            planner=planner,
                        )
                    )
                except ResultFeedbackError as error:
                    if error.code != "SUCCESSOR_PLANNING_OUTPUT_INCOMPLETE":
                        raise
                    raise ResultFeedbackError(
                        "SUCCESSOR_PLANNING_OUTCOME_UNCERTAIN",
                        "a planning provider call was claimed without a complete "
                        "verified local planning outcome; ordinary retry is disabled",
                    ) from error
                validation_records = list(loaded_validations)
                proposal = compilation.proposal_set
                if proposal is None:
                    raise ResultFeedbackError(
                        "SUCCESSOR_PLANNING_OUTPUT_INCOMPLETE",
                        "stored successor planning proposal is missing",
                    )
            else:
                if self.runs.resolve_artifact_path(
                    parent_run_id, f"{prefix}/planning-proposal.yaml"
                ).exists():
                    raise ResultFeedbackError(
                        "SUCCESSOR_PLANNING_STAGE_INVALID",
                        "planning output exists without its provider call claim",
                    )
                self.runs.write_exclusive_artifact(
                    parent_run_id,
                    f"{prefix}/provider-calls/planning.yaml",
                    "successor_planning_invocation",
                    invocation.invocation_id,
                    invocation,
                )
                raw_proposal = planner.propose(planning_input)
                proposal = self._bind_successor_proposal(
                    raw_proposal, planning_input, binding
                )
                if not validate_planning_proposal_set(proposal, planning_input).valid:
                    raise ResultFeedbackError(
                        "SUCCESSOR_PLANNING_PROPOSAL_INVALID",
                        "successor proposal failed deterministic validation",
                    )
                compilation = ScientificProblemCompiler(
                    planner, evidence_repository=self.composite_evidence
                ).compile_proposal(planning_input, proposal)
                if not all(report.valid for report in compilation.reports):
                    issue_codes = sorted(
                        {
                            issue.code
                            for report in compilation.reports
                            for issue in report.issues
                        }
                    )
                    raise ResultFeedbackError(
                        "SUCCESSOR_PLAN_INVALID",
                        "successor candidate failed deterministic plan validation: "
                        + ", ".join(issue_codes),
                    )
                self.runs.write_immutable_artifact(
                    parent_run_id,
                    f"{prefix}/planning-proposal.yaml",
                    "planning_proposal_set",
                    proposal.proposal_id,
                    proposal,
                )
                validation_records = []
                for plan, receipt, report in zip(
                    compilation.candidates,
                    compilation.compilation_receipts,
                    compilation.reports[1 : 1 + len(compilation.candidates)],
                    strict=True,
                ):
                    validation = build_plan_validation_record(
                        plan,
                        report,
                        validation_id=f"validation-{content_hash(plan)[:24]}",
                    )
                    validation_records.append(validation)
                    self.runs.write_immutable_artifact(
                        parent_run_id,
                        f"{prefix}/candidates/{plan.plan_id}.yaml",
                        "scientific_question_plan",
                        plan.plan_id,
                        plan,
                    )
                    self.runs.write_immutable_artifact(
                        parent_run_id,
                        f"{prefix}/compilation-receipts/{receipt.receipt_id}.yaml",
                        "plan_compilation_receipt",
                        receipt.receipt_id,
                        receipt,
                    )
                    self.runs.write_immutable_artifact(
                        parent_run_id,
                        f"{prefix}/validation/{validation.validation_id}.yaml",
                        "plan_validation_record",
                        validation.validation_id,
                        validation,
                    )
            candidate_ids = tuple(item.plan_id for item in compilation.candidates)
            selection_path = self.runs.resolve_artifact_path(
                parent_run_id, f"{prefix}/selection.yaml"
            )
            if selection_path.exists():
                stored_selection = self.runs.load_artifact(
                    parent_run_id,
                    f"{prefix}/selection.yaml",
                    SuccessorCandidateSelection,
                )
                selected_candidate = next(
                    (
                        item
                        for item in compilation.candidates
                        if item.plan_id == stored_selection.candidate_id
                    ),
                    None,
                )
                if (
                    stored_selection.invocation_id != invocation.invocation_id
                    or stored_selection.invocation_hash != invocation.content_hash
                    or stored_selection.proposal_id != proposal.proposal_id
                    or stored_selection.proposal_hash != content_hash(proposal)
                    or selected_candidate is None
                    or stored_selection.candidate_hash
                    != content_hash(selected_candidate)
                ):
                    raise ResultFeedbackError(
                        "SUCCESSOR_SELECTION_BINDING_INVALID",
                        "stored selection does not bind this invocation, proposal "
                        "and immutable candidate set",
                    )
                if (
                    selected_candidate_id is not None
                    and selected_candidate_id != stored_selection.candidate_id
                ):
                    raise ResultFeedbackError(
                        "SUCCESSOR_SELECTION_CONFLICT",
                        "requested candidate differs from the stored immutable selection",
                    )
                selected_candidate_id = stored_selection.candidate_id
            if selected_candidate_id is None:
                if len(compilation.candidates) != 1:
                    status = SuccessorPlanningStatus.HUMAN_SCIENTIFIC_DECISION_REQUIRED
                    cycle_payload = {
                        "invocation_id": invocation.invocation_id,
                        "invocation_hash": invocation.content_hash,
                        "parent_binding_id": binding.binding_id,
                        "parent_binding_hash": binding.content_hash,
                        "research_update_id": update.update_id,
                        "research_update_hash": update.content_hash,
                        "context_id": context.context_id,
                        "context_hash": context.content_hash,
                        "evidence_packet_id": packet.packet_id,
                        "evidence_packet_hash": packet.content_hash,
                        "planning_input_id": planning_input.planning_input_id,
                        "planning_input_hash": planning_input.content_hash,
                        "planning_proposal_id": proposal.proposal_id,
                        "planning_proposal_hash": content_hash(proposal),
                        "candidate_plan_ids": tuple(item.plan_id for item in compilation.candidates),
                        "candidate_plan_hashes": tuple(content_hash(item) for item in compilation.candidates),
                        "status": status,
                    }
                    cycle = build_content_bound(
                        SuccessorPlanningCycle, "successor-cycle", cycle_payload
                    )
                    self.runs.write_immutable_artifact(
                        parent_run_id,
                        f"{prefix}/planning-outcome.yaml",
                        "successor_planning_outcome",
                        cycle.cycle_id,
                        cycle,
                    )
                    return cycle
                selected_candidate_id = compilation.candidates[0].plan_id
            if selected_candidate_id not in candidate_ids:
                raise ResultFeedbackError(
                    "UNKNOWN_SUCCESSOR_CANDIDATE",
                    "selected successor candidate is not part of this compilation",
                )
            selected_index = candidate_ids.index(selected_candidate_id)
            selected = compilation.candidates[selected_index]
            validation = validation_records[selected_index]
            compilation_receipt = compilation.compilation_receipts[selected_index]
            selection_payload = {
                "invocation_id": invocation.invocation_id,
                "invocation_hash": invocation.content_hash,
                "proposal_id": proposal.proposal_id,
                "proposal_hash": content_hash(proposal),
                "candidate_id": selected.plan_id,
                "candidate_hash": content_hash(selected),
                "selection_provenance": (
                    "explicit_user_candidate_selection"
                    if len(compilation.candidates) != 1
                    else "deterministic_single_candidate"
                ),
            }
            selection = build_content_bound(
                SuccessorCandidateSelection,
                "successor-selection",
                selection_payload,
            )
            if selection_path.exists():
                selection = stored_selection
            else:
                self.runs.write_exclusive_artifact(
                    parent_run_id,
                    f"{prefix}/selection.yaml",
                    "successor_candidate_selection",
                    selection.selection_id,
                    selection,
                )
            review_input = ApprovalContextResolver().resolve(
                context,
                packet,
                planning_input,
                selected,
                validation,
                repositories,
                self.composite_evidence,
            )
            approval_marker = self.runs.resolve_artifact_path(
                parent_run_id, f"{prefix}/provider-calls/approval.yaml"
            )
            if approval_marker.exists():
                try:
                    cycle = self._rebuild_complete_approval_cycle(
                        parent_run_id=parent_run_id,
                        prefix=prefix,
                        invocation=invocation,
                        compilation=compilation,
                        selection=selection,
                    )
                except ResultFeedbackError as error:
                    if error.code != "SUCCESSOR_APPROVAL_OUTPUT_INCOMPLETE":
                        raise
                    raise ResultFeedbackError(
                        "SUCCESSOR_APPROVAL_OUTCOME_UNCERTAIN",
                        "an approval provider call was claimed without a complete "
                        "verified local authority chain; ordinary retry is disabled",
                    ) from error
                self.runs.write_immutable_artifact(
                    parent_run_id,
                    f"{prefix}/cycle.yaml",
                    "successor_planning_cycle",
                    cycle.cycle_id,
                    cycle,
                )
                return cycle
            self.runs.write_exclusive_artifact(
                parent_run_id,
                f"{prefix}/provider-calls/approval.yaml",
                "successor_candidate_selection",
                selection.selection_id,
                selection,
            )
            approval = IndependentApprovalService(
                approver, approver_id="independent-scientific-approver"
            ).review(review_input)
            policy = ProjectTrustPolicy(
                approval_mode=ApprovalMode.INDEPENDENT_REQUIRED,
                policy_version="1.0.0",
            )
            passed = approval.verdict.decision.value == "approve"
            gate = bind_gate_verdict(
                selected,
                approval.verdict,
                validation,
                trust_policy=policy,
                gate_id=f"plan-gate-{content_hash(approval.receipt)[:24]}",
                passed=passed,
                reasons=approval.review.policy_reasons,
                review_input=review_input,
                review=approval.review,
                receipt=approval.receipt,
                compilation_receipt=compilation_receipt,
            )
            approval_values = (
                ("approval-review-input.yaml", "approval_review_input", review_input.review_input_id, review_input),
                ("approval-review.yaml", "approval_review_record", approval.review.review_id, approval.review),
                ("approval-verdict.yaml", "approval_verdict", approval.verdict.verdict_id, approval.verdict),
                ("independent-approval-receipt.yaml", "independent_approval_receipt", approval.receipt.receipt_id, approval.receipt),
                ("plan-gate.yaml", "gate_verdict", gate.gate_id, gate),
                ("project-trust-policy.yaml", "project_trust_policy", "project-trust-policy", policy),
            )
            for name, artifact_type, identifier, value in approval_values:
                self.runs.write_immutable_artifact(
                    parent_run_id,
                    f"{prefix}/approval/{name}",
                    artifact_type,
                    identifier,
                    value,
                )
            cycle = self._rebuild_complete_approval_cycle(
                parent_run_id=parent_run_id,
                prefix=prefix,
                invocation=invocation,
                compilation=compilation,
                selection=selection,
            )
            self.runs.write_immutable_artifact(
                parent_run_id,
                f"{prefix}/cycle.yaml",
                "successor_planning_cycle",
                cycle.cycle_id,
                cycle,
            )
            return cycle


def result_feedback_status(
    parent_run_id: str,
    repository: ScientificProblemRunRepository,
) -> dict[str, Any]:
    run_root = repository.run_dir(parent_run_id)
    submissions: list[ResultEvidenceSubmission] = []
    assessments: list[ResultEvidenceAssessment] = []
    receipts: list[ResultEvidenceMaterializationReceipt] = []
    curations: list[KnowledgeCurationRecord] = []
    for path in sorted(run_root.glob("result-evidence/submissions/*/submission.yaml")):
        submissions.append(
            repository.load_artifact(
                parent_run_id,
                path.relative_to(run_root).as_posix(),
                ResultEvidenceSubmission,
            )
        )
    for path in sorted(run_root.glob("result-evidence/submissions/*/assessment.yaml")):
        assessments.append(
            repository.load_artifact(
                parent_run_id,
                path.relative_to(run_root).as_posix(),
                ResultEvidenceAssessment,
            )
        )
    for path in sorted(run_root.glob("result-evidence/accepted/*/receipt.yaml")):
        receipts.append(
            repository.load_artifact(
                parent_run_id,
                path.relative_to(run_root).as_posix(),
                ResultEvidenceMaterializationReceipt,
            )
        )
    for path in sorted(run_root.glob("result-evidence/curations/*/*.yaml")):
        curations.append(
            repository.load_artifact(
                parent_run_id,
                path.relative_to(run_root).as_posix(),
                KnowledgeCurationRecord,
            )
        )
    cycles_by_index: dict[int, SuccessorPlanningCycle] = {}
    attempted_indexes: set[int] = set()
    incomplete_cycles: list[dict[str, Any]] = []
    bindings_by_index: dict[int, SuccessorParentBinding] = {}
    successor_root = run_root / "successor-planning"
    if successor_root.exists():
        for directory in successor_root.glob("cycle-*"):
            if directory.is_symlink():
                raise ResultFeedbackError(
                    "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                    f"symlinked successor cycle directory: {directory.name}",
                )
            if not directory.is_dir():
                continue
            suffix = directory.name.removeprefix("cycle-")
            try:
                index = int(suffix)
            except ValueError as error:
                raise ResultFeedbackError(
                    "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                    f"invalid successor cycle directory: {directory.name}",
                ) from error
            if index < 1 or suffix != str(index) or index in attempted_indexes:
                raise ResultFeedbackError(
                    "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                    f"non-canonical successor cycle directory: {directory.name}",
                )
            attempted_indexes.add(index)
            binding_path = directory / "parent-binding.yaml"
            if binding_path.is_symlink():
                raise ResultFeedbackError(
                    "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                    f"symlinked successor parent binding: {binding_path.name}",
                )
            if binding_path.exists():
                binding = repository.load_artifact(
                    parent_run_id,
                    binding_path.relative_to(run_root).as_posix(),
                    SuccessorParentBinding,
                )
                if binding.successor_cycle_index != index:
                    raise ResultFeedbackError(
                        "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                        "successor cycle directory index does not match "
                        "its parent-binding lineage index",
                    )
                bindings_by_index[index] = binding
            cycle_path = directory / "cycle.yaml"
            if cycle_path.exists():
                cycle = repository.load_artifact(
                    parent_run_id,
                    cycle_path.relative_to(run_root).as_posix(),
                    SuccessorPlanningCycle,
                )
                binding = bindings_by_index.get(index)
                if binding is None or (
                    cycle.parent_binding_id,
                    cycle.parent_binding_hash,
                ) != (binding.binding_id, binding.content_hash):
                    raise ResultFeedbackError(
                        "INVALID_SUCCESSOR_CYCLE_DIRECTORY",
                        "completed successor cycle does not bind the "
                        "parent-binding stored in its directory",
                    )
                cycles_by_index[index] = cycle
                continue
            planning_outcome_path = directory / "planning-outcome.yaml"
            if planning_outcome_path.exists():
                planning_outcome = repository.load_artifact(
                    parent_run_id,
                    planning_outcome_path.relative_to(run_root).as_posix(),
                    SuccessorPlanningCycle,
                )
                incomplete_cycles.append(
                    {"cycle": index, "status": planning_outcome.status.value}
                )
                continue
            approval_claimed = (directory / "provider-calls" / "approval.yaml").exists()
            planning_claimed = (directory / "provider-calls" / "planning.yaml").exists()
            if approval_claimed:
                status = "SUCCESSOR_APPROVAL_OUTCOME_UNCERTAIN"
            elif planning_claimed and (directory / "planning-proposal.yaml").exists():
                status = "PLANNING_COMPLETE_AWAITING_SELECTION_OR_APPROVAL"
            elif planning_claimed:
                status = "SUCCESSOR_PLANNING_OUTCOME_UNCERTAIN"
            else:
                status = "SUCCESSOR_INVOCATION_RESERVED"
            incomplete_cycles.append({"cycle": index, "status": status})
    cycles = [cycles_by_index[index] for index in sorted(cycles_by_index)]
    assessment_by_submission = {item.submission_id: item for item in assessments}
    materialized_ids = {item.submission_id for item in receipts}
    superseded_curation_ids = {
        item.supersedes_curation_id
        for item in curations
        if item.supersedes_curation_id is not None
    }
    current_curation_by_submission = {
        item.target_id: item
        for item in curations
        if item.curation_id not in superseded_curation_ids
    }
    accepted_ids = {
        submission_id
        for submission_id, curation in current_curation_by_submission.items()
        if curation.status == CurationStatus.ACCEPTED
        and submission_id in materialized_ids
    }
    blocked_statuses = {
        ResultEvidenceIntakeStatus.BLOCKED_CONTENT_BINDING,
        ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING,
        ResultEvidenceIntakeStatus.BLOCKED_CONTEXT_COMPATIBILITY,
        ResultEvidenceIntakeStatus.BLOCKED_SCIENTIFIC_ADMISSIBILITY,
    }
    latest_complete_index = max(cycles_by_index, default=None)
    latest_attempted_index = max(attempted_indexes, default=None)
    incomplete_by_index = {item["cycle"]: item for item in incomplete_cycles}
    if latest_attempted_index in cycles_by_index:
        latest_status = cycles_by_index[latest_attempted_index].status.value
    elif latest_attempted_index in incomplete_by_index:
        latest_status = incomplete_by_index[latest_attempted_index]["status"]
    else:
        latest_status = None
    binding_index = latest_attempted_index
    triggering_result_ids: list[str] = []
    latest_binding = (
        bindings_by_index.get(binding_index)
        if binding_index is not None
        else None
    )
    if latest_binding is not None:
        triggering_result_ids = list(latest_binding.result_submission_ids)
    return {
        "result_evidence": {
            "submissions": len(submissions),
            "accepted": len(accepted_ids),
            "rejected_or_blocked": sum(
                1
                for item in submissions
                if (
                    assessment_by_submission.get(item.submission_id) is not None
                    and assessment_by_submission[item.submission_id].status
                    in blocked_statuses
                )
                or (
                    current_curation_by_submission.get(item.submission_id)
                    is not None
                    and current_curation_by_submission[item.submission_id].status
                    == CurationStatus.REJECTED
                )
            ),
            "pending_curation": sum(
                1
                for item in submissions
                if item.submission_id not in accepted_ids
                and assessment_by_submission.get(item.submission_id) is not None
                and assessment_by_submission[item.submission_id].status
                == ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
                and current_curation_by_submission.get(item.submission_id) is None
            ),
            "failed_execution": sum(
                item.result_type == ResultEvidenceType.FAILED_EXECUTION
                for item in submissions
            ),
            "partial_results": sum(
                item.result_type == ResultEvidenceType.PARTIAL_RESULT
                for item in submissions
            ),
            "parent_plan_binding": [
                {
                    "submission_id": item.submission_id,
                    "parent_plan_id": item.parent_plan_id,
                    "parent_plan_hash": item.parent_plan_hash,
                    "task_id": item.task_id,
                }
                for item in submissions
            ],
        },
        "successor_planning": {
            "cycles": len(cycles),
            "latest_status": latest_status,
            "cycle": latest_complete_index,
            "latest_complete_cycle": latest_complete_index,
            "latest_attempted_cycle": latest_attempted_index,
            "incomplete_cycles": sorted(
                incomplete_cycles, key=lambda item: item["cycle"]
            ),
            "parent_run": parent_run_id,
            "parent_plan": (
                latest_binding.parent_plan_id
                if latest_binding is not None
                else submissions[-1].parent_plan_id if submissions else None
            ),
            "triggering_result_ids": triggering_result_ids,
        },
    }


__all__ = [
    "RESULT_FEEDBACK_VERSION",
    "ResultEvidenceIntakeService",
    "ResultFeedbackError",
    "SuccessorPlanningService",
    "build_result_evidence_submission",
    "result_feedback_status",
]
