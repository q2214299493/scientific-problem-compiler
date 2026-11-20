from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...compiler import CompilationResult, ScientificProblemCompiler
from ...models import (
    ApprovalDecision,
    ApprovalReviewRecord,
    ApprovalVerdict,
    GateVerdict,
    IndependentApprovalReceipt,
    PlanCompilationReceipt,
    PlanningProposalSet,
    PlanRevisionAttemptOutcome,
    PlanRevisionAttemptStart,
    PlanRevisionAttemptStatus,
    PlanRevisionChain,
    PlanRevisionInput,
    PlanRevisionRecord,
    PlanRevisionRound,
    PlanValidationRecord,
    ProjectTrustPolicy,
    RevisionApprovalLLMResponse,
    RevisionApprovalReviewInput,
    RevisionIssueStatus,
    ScientificProblemRun,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
    ScientificRunArtifactBinding,
    ScientificRunFailure,
    parse_approval_review_input,
)
from ...planning import (
    PlanMaterializer,
    automatic_revision_block_reason,
    build_plan_revision_input,
)
from ...planning.validators import validate_planning_proposal_set
from ...serialization import content_hash, load_data
from ...validators import (
    build_plan_validation_record,
    validate_approved_plan_authority,
    validate_plan_compilation_receipt,
    validate_question_plan,
)
from ..providers import bind_provider, replace_provider
from ..repository import ScientificProblemRunRepository
from ..state import WorkflowResult, sanitize_failure, snapshot_hash, update_run
from .approval import ApprovalStage


def _make_revision_chain(payload: dict[str, Any]) -> PlanRevisionChain:
    normalized_payload = dict(payload)
    normalized_payload["rounds"] = tuple(
        item if isinstance(item, PlanRevisionRound) else PlanRevisionRound.model_validate(item)
        for item in payload["rounds"]
    )
    draft = PlanRevisionChain.model_construct(**normalized_payload, content_hash="0" * 64)
    normalized = draft.model_dump(mode="json", exclude={"content_hash"})
    return PlanRevisionChain(**normalized, content_hash=content_hash(normalized))


@dataclass
class RevisionStage:
    """Track global attempts and validate each new plan against its independent review."""

    runs: ScientificProblemRunRepository
    approval: ApprovalStage

    @staticmethod
    def needs_evidence(chain: PlanRevisionChain) -> bool:
        return chain.rounds[-1].outcome == ApprovalDecision.INSUFFICIENT_EVIDENCE.value

    def start_chain(self, planned, planning_input, context, proposal_binding, max_plan_revisions):
        selected_candidate_id = planned.selected_candidate_id
        candidate_bindings = planned.candidate_plans
        receipt_bindings = planned.candidate_compilation_receipts
        validation_bindings = planned.plan_validation_records
        if selected_candidate_id is None:
            if len(candidate_bindings) != 1:
                raise ValueError("bounded revision with multiple candidates requires --candidate-id")
            selected_candidate_id = candidate_bindings[0].artifact_id
        selected_ids = [item.artifact_id for item in candidate_bindings]
        if selected_candidate_id not in selected_ids:
            raise ValueError("selected candidate ID is not part of this run")
        selected_index = selected_ids.index(selected_candidate_id)
        revision_chain = _make_revision_chain(
            {
                "chain_id": ("plan-revision-chain-" + content_hash({"run_id": planned.run_id})[:24]),
                "run_id": planned.run_id,
                "planning_input_id": planning_input.planning_input_id,
                "planning_input_hash": planning_input.content_hash,
                "context_id": context.context_id,
                "context_hash": context.content_hash,
                "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
                "knowledge_snapshot_hash": snapshot_hash(context.knowledge_snapshot),
                "max_revisions": max_plan_revisions,
                "revisions_used": 0,
                "rounds": (
                    PlanRevisionRound(
                        round_index=0,
                        planning_proposal=proposal_binding,
                        candidate_plan=candidate_bindings[selected_index],
                        compilation_receipt=receipt_bindings[selected_index],
                        validation_record=validation_bindings[selected_index],
                    ),
                ),
                "termination_reason": None,
            }
        )
        self.runs.save_revision_chain(revision_chain)
        planned = update_run(
            planned,
            candidate_plans=(candidate_bindings[selected_index],),
            candidate_compilation_receipts=(receipt_bindings[selected_index],),
            plan_validation_records=(validation_bindings[selected_index],),
            selected_candidate_id=selected_candidate_id,
        )
        self.runs.save(planned)
        return planned, revision_chain

    def load_compilation(
        self,
        chain: PlanRevisionChain,
        planning_input,
        evidence,
    ) -> CompilationResult:
        current = chain.rounds[-1]
        proposal = self.runs.load_bound_artifact(chain.run_id, current.planning_proposal, PlanningProposalSet)
        plan = self.runs.load_bound_artifact(chain.run_id, current.candidate_plan, ScientificQuestionPlan)
        receipt = self.runs.load_bound_artifact(chain.run_id, current.compilation_receipt, PlanCompilationReceipt)
        validation = self.runs.load_bound_artifact(chain.run_id, current.validation_record, PlanValidationRecord)
        proposal_report = validate_planning_proposal_set(proposal, planning_input)
        plan_report = validate_question_plan(plan, planning_input.scientific_capabilities, evidence)
        if not proposal_report.valid:
            raise ValueError("stored revision planning proposal failed validation")
        if not validate_plan_compilation_receipt(plan, receipt).valid:
            raise ValueError("stored revision compilation receipt failed validation")
        if validation.plan_id != plan.plan_id or validation.plan_content_hash != content_hash(plan):
            raise ValueError("stored revision validation record is stale")
        return CompilationResult(
            candidates=(plan,),
            reports=(proposal_report, plan_report),
            proposal_set=proposal,
            compilation_receipts=(receipt,),
        )

    def record_approval(
        self,
        chain: PlanRevisionChain,
        run: ScientificProblemRun,
    ) -> PlanRevisionChain:
        current = chain.rounds[-1]
        prefix = f"{self.runs.round_prefix(chain.run_id, current.round_index)}/approval"
        values = (
            (
                "approval-review-input.yaml",
                "approval_review_input",
                None,
                run.approval_review_id,
            ),
            (
                "approval-review.yaml",
                "approval_review_record",
                ApprovalReviewRecord,
                run.approval_review_id,
            ),
            (
                "approval-verdict.yaml",
                "approval_verdict",
                ApprovalVerdict,
                run.approval_verdict_id,
            ),
            (
                "independent-approval-receipt.yaml",
                "independent_approval_receipt",
                IndependentApprovalReceipt,
                run.approval_receipt_id,
            ),
            ("plan-gate.yaml", "gate_verdict", GateVerdict, run.gate_id),
            (
                "project-trust-policy.yaml",
                "project_trust_policy",
                ProjectTrustPolicy,
                "project-trust-policy",
            ),
        )
        bindings: list[ScientificRunArtifactBinding] = []
        loaded: list[object] = []
        for filename, artifact_type, model_type, expected_id in values:
            path = f"{prefix}/{filename}"
            value = (
                parse_approval_review_input(load_data(self.runs.resolve_artifact_path(chain.run_id, path)))
                if model_type is None
                else self.runs.load_artifact(chain.run_id, path, model_type)
            )
            identifier = expected_id
            if identifier is None:
                raise ValueError("approval run is missing a required artifact identifier")
            if artifact_type == "approval_review_input":
                identifier = value.review_input_id
            bindings.append(
                ScientificRunArtifactBinding(
                    artifact_type=artifact_type,
                    artifact_id=identifier,
                    artifact_hash=content_hash(value),
                    relative_path=path,
                )
            )
            loaded.append(value)
        verdict = loaded[2]
        updated_round = current.model_copy(
            update={
                "approval_review_input": bindings[0],
                "approval_review_record": bindings[1],
                "approval_verdict": bindings[2],
                "approval_receipt": bindings[3],
                "gate": bindings[4],
                "trust_policy": bindings[5],
                "outcome": verdict.decision.value,
            }
        )
        termination_reason = (
            "approved"
            if verdict.decision == ApprovalDecision.APPROVE
            else (
                None
                if verdict.decision == ApprovalDecision.REQUEST_REVISION
                else f"non_revisable_decision:{verdict.decision.value}"
            )
        )
        payload = chain.model_dump(mode="python", exclude={"content_hash"})
        payload.update(
            rounds=(*chain.rounds[:-1], updated_round),
            termination_reason=termination_reason,
        )
        updated = _make_revision_chain(payload)
        self.runs.save_revision_chain(updated)
        return updated

    def block(
        self,
        run: ScientificProblemRun,
        chain: PlanRevisionChain,
        category: str,
        message: str,
    ) -> WorkflowResult:
        payload = chain.model_dump(mode="python", exclude={"content_hash"})
        payload["termination_reason"] = f"{category}:{message}"
        updated_chain = _make_revision_chain(payload)
        self.runs.save_revision_chain(updated_chain)
        blocked = update_run(
            run,
            status=ScientificProblemRunStatus.REVISION_BLOCKED,
            last_successful_status=run.status,
            failure=ScientificRunFailure(
                stage="plan_revision",
                category=category,
                message=message,
                retryable=False,
            ),
        )
        self.runs.save(blocked)
        return WorkflowResult(blocked)

    def revise(
        self,
        run: ScientificProblemRun,
        chain: PlanRevisionChain,
        context,
        packet,
        planning_input,
        repositories,
        evidence,
        planner,
        approval_provider,
        settings,
    ) -> ScientificProblemRun:
        current_run = run
        current_chain = chain
        while True:
            if current_run.status == ScientificProblemRunStatus.REQUIRES_REPLANNING:
                return current_run
            current_round = current_chain.rounds[-1]
            if current_round.approval_verdict is None:
                return current_run
            plan = self.runs.load_bound_artifact(
                current_run.run_id,
                current_round.candidate_plan,
                ScientificQuestionPlan,
            )
            validation = self.runs.load_bound_artifact(
                current_run.run_id,
                current_round.validation_record,
                PlanValidationRecord,
            )
            proposal = self.runs.load_bound_artifact(
                current_run.run_id,
                current_round.planning_proposal,
                PlanningProposalSet,
            )
            compilation_receipt = self.runs.load_bound_artifact(
                current_run.run_id,
                current_round.compilation_receipt,
                PlanCompilationReceipt,
            )
            review_input = self.runs.load_approval_input_binding(
                current_run.run_id,
                current_round.approval_review_input,
            )
            review = self.runs.load_bound_artifact(
                current_run.run_id,
                current_round.approval_review_record,
                ApprovalReviewRecord,
            )
            verdict = self.runs.load_bound_artifact(
                current_run.run_id,
                current_round.approval_verdict,
                ApprovalVerdict,
            )
            receipt = self.runs.load_bound_artifact(
                current_run.run_id,
                current_round.approval_receipt,
                IndependentApprovalReceipt,
            )
            gate = self.runs.load_bound_artifact(current_run.run_id, current_round.gate, GateVerdict)
            policy = self.runs.load_bound_artifact(
                current_run.run_id,
                current_round.trust_policy,
                ProjectTrustPolicy,
            )
            trigger_report = validate_question_plan(plan, planning_input.scientific_capabilities, evidence)
            if not validate_approved_plan_authority(
                plan,
                validation,
                review_input,
                review,
                verdict,
                receipt,
                gate,
                policy,
                compilation_receipt,
                current_report=trigger_report,
                require_passed=False,
            ).valid:
                return self.block(
                    current_run,
                    current_chain,
                    "INVALID_TRIGGER_APPROVAL_CHAIN",
                    "revision trigger approval chain is stale, cross-plan, or tampered",
                ).run
            if verdict.decision != ApprovalDecision.REQUEST_REVISION:
                return current_run
            attempts = self.runs.load_revision_attempts(current_run.run_id)
            if attempts and attempts[-1][1] is None:
                start = attempts[-1][0]
                completed_round = next(
                    (
                        item
                        for item in current_chain.rounds
                        if item.revision_input is not None
                        and item.revision_input.artifact_id == start.revision_input_id
                        and item.revision_record is not None
                    ),
                    None,
                )
                if completed_round is None:
                    return self.block(
                        current_run,
                        current_chain,
                        "REVISION_ATTEMPT_UNCERTAIN",
                        (
                            "a revision provider call was claimed but has no complete local "
                            "outcome; ordinary resume will not invoke it again"
                        ),
                    ).run
                completed_record = self.runs.load_bound_artifact(
                    current_run.run_id,
                    completed_round.revision_record,
                    PlanRevisionRecord,
                )
                outcome_payload = {
                    "attempt_id": start.attempt_id,
                    "attempt_start_hash": start.content_hash,
                    "status": PlanRevisionAttemptStatus.COMPLETED,
                    "revision_record_id": completed_record.revision_id,
                    "revision_record_hash": completed_record.content_hash,
                    "round_index": completed_round.round_index,
                    "failure_category": None,
                    "failure_message": None,
                }
                outcome = PlanRevisionAttemptOutcome(
                    **outcome_payload,
                    content_hash=content_hash(outcome_payload),
                )
                self.runs.write_immutable_artifact(
                    current_run.run_id,
                    f"{self.runs.attempt_prefix(run.run_id, start.attempt_index)}/outcome.yaml",
                    "plan_revision_attempt_outcome",
                    start.attempt_id,
                    outcome,
                )
                attempts = (*attempts[:-1], (start, outcome))
            if len(attempts) >= current_chain.max_revisions:
                return self.block(
                    current_run,
                    current_chain,
                    "REVISION_BUDGET_EXHAUSTED",
                    "independent review still requests revision after all started attempts",
                ).run
            block_reason = automatic_revision_block_reason(plan, review, verdict, validation)
            if block_reason is not None:
                return self.block(
                    current_run,
                    current_chain,
                    "REVISION_REQUIRES_EXTERNAL_ACTION",
                    block_reason,
                ).run
            validation_report = validate_question_plan(plan, planning_input.scientific_capabilities, evidence)
            candidate_key = next(
                candidate.candidate_key
                for candidate in proposal.candidates
                if PlanMaterializer().materialize_candidate(candidate, proposal, planning_input).plan_id == plan.plan_id
            )
            attempt: PlanRevisionAttemptStart | None = None
            attempt_claimed = False
            try:
                carried_feedback = ()
                if isinstance(review_input, RevisionApprovalReviewInput) and isinstance(
                    review.response, RevisionApprovalLLMResponse
                ):
                    unresolved_ids = {
                        item.feedback_id
                        for item in review.response.issue_assessments
                        if item.status
                        in {
                            RevisionIssueStatus.UNRESOLVED,
                            RevisionIssueStatus.NEEDS_HUMAN_DECISION,
                        }
                    }
                    carried_feedback = tuple(
                        item
                        for item in review_input.revision_context.tracked_feedback
                        if item.feedback_id in unresolved_ids
                    )
                revision_input = build_plan_revision_input(
                    revision_index=current_chain.revisions_used + 1,
                    planning_input=planning_input,
                    parent_proposal=proposal,
                    parent_candidate_key=candidate_key,
                    parent_plan=plan,
                    parent_compilation_receipt=compilation_receipt,
                    plan_validation_record=validation,
                    validation_report=validation_report,
                    approval_review_input=review_input,
                    approval_review_record=review,
                    approval_verdict=verdict,
                    approval_receipt=receipt,
                    carried_feedback=carried_feedback,
                )
                round_index = revision_input.revision_index
                prefix = self.runs.round_prefix(current_run.run_id, round_index)
                input_binding = self.runs.write_immutable_artifact(
                    current_run.run_id,
                    f"{prefix}/revision-input.yaml",
                    "plan_revision_input",
                    revision_input.revision_input_id,
                    revision_input,
                )
                attempt_index = len(attempts) + 1
                provider_binding = bind_provider("planning", planner)
                attempt_identity = {
                    "chain_id": current_chain.chain_id,
                    "attempt_index": attempt_index,
                    "revision_input_id": revision_input.revision_input_id,
                    "revision_input_hash": revision_input.content_hash,
                    "parent_plan_id": plan.plan_id,
                    "parent_plan_hash": content_hash(plan),
                    "trigger_review_id": review.review_id,
                    "trigger_review_hash": review.content_hash,
                    "provider_id": planner.provider_id,
                    "provider_version": planner.provider_version,
                    "provider_config_hash": provider_binding.configuration_hash,
                }
                attempt_id = f"plan-revision-attempt-{content_hash(attempt_identity)[:24]}"
                attempt_payload = {"attempt_id": attempt_id, **attempt_identity}
                attempt = PlanRevisionAttemptStart(
                    **attempt_payload,
                    content_hash=content_hash(attempt_payload),
                )
                self.runs.write_exclusive_artifact(
                    current_run.run_id,
                    f"{self.runs.attempt_prefix(current_run.run_id, attempt_index)}/start.yaml",
                    "plan_revision_attempt_start",
                    attempt.attempt_id,
                    attempt,
                )
                attempt_claimed = True
                revision = ScientificProblemCompiler(planner, evidence_repository=evidence).revise(revision_input)
            except (FileNotFoundError, OSError, ValueError, TypeError) as error:
                if attempt is not None and attempt_claimed:
                    outcome_payload = {
                        "attempt_id": attempt.attempt_id,
                        "attempt_start_hash": attempt.content_hash,
                        "status": PlanRevisionAttemptStatus.TERMINATED,
                        "revision_record_id": None,
                        "revision_record_hash": None,
                        "round_index": None,
                        "failure_category": type(error).__name__,
                        "failure_message": sanitize_failure(error),
                    }
                    outcome = PlanRevisionAttemptOutcome(
                        **outcome_payload,
                        content_hash=content_hash(outcome_payload),
                    )
                    self.runs.write_immutable_artifact(
                        current_run.run_id,
                        f"{self.runs.attempt_prefix(current_run.run_id, attempt.attempt_index)}/outcome.yaml",
                        "plan_revision_attempt_outcome",
                        attempt.attempt_id,
                        outcome,
                    )
                return self.block(
                    current_run,
                    current_chain,
                    "REVISION_OUTPUT_REJECTED",
                    sanitize_failure(error),
                ).run

            revised_plan = revision.plan
            validation_report = validate_question_plan(revised_plan, planning_input.scientific_capabilities, evidence)
            validation_record = build_plan_validation_record(
                revised_plan,
                validation_report,
                validation_id=f"plan-validation-{content_hash(revised_plan)[:24]}",
            )
            record_identity = {
                "revision_index": revision_input.revision_index,
                "revision_input_id": revision_input.revision_input_id,
                "revision_input_hash": revision_input.content_hash,
                "parent_plan_id": plan.plan_id,
                "parent_plan_hash": content_hash(plan),
                "trigger_review_id": review.review_id,
                "trigger_review_hash": review.content_hash,
                "provider_id": planner.provider_id,
                "provider_version": planner.provider_version,
                "provider_config": dict(revision.proposal_set.provider_config),
                "response": revision.response,
                "planning_proposal_id": revision.proposal_set.proposal_id,
                "planning_proposal_hash": content_hash(revision.proposal_set),
                "revised_plan_id": revised_plan.plan_id,
                "revised_plan_hash": content_hash(revised_plan),
                "validation_id": validation_record.validation_id,
                "validation_hash": content_hash(validation_record),
                "substantive_change": True,
            }
            revision_id = f"plan-revision-{content_hash(record_identity)[:24]}"
            record_payload = {"revision_id": revision_id, **record_identity}
            revision_record = PlanRevisionRecord(
                **record_payload,
                content_hash=content_hash(record_payload),
            )
            proposal_binding = self.runs.write_immutable_artifact(
                current_run.run_id,
                f"{prefix}/planning-proposal.yaml",
                "planning_proposal_set",
                revision.proposal_set.proposal_id,
                revision.proposal_set,
            )
            plan_binding = self.runs.write_immutable_artifact(
                current_run.run_id,
                f"{prefix}/candidates/{revised_plan.plan_id}.yaml",
                "scientific_question_plan",
                revised_plan.plan_id,
                revised_plan,
            )
            compilation_binding = self.runs.write_immutable_artifact(
                current_run.run_id,
                f"{prefix}/compilation-receipts/{revision.compilation_receipt.receipt_id}.yaml",
                "plan_compilation_receipt",
                revision.compilation_receipt.receipt_id,
                revision.compilation_receipt,
            )
            validation_binding = self.runs.write_immutable_artifact(
                current_run.run_id,
                f"{prefix}/validation-records/{validation_record.validation_id}.yaml",
                "plan_validation_record",
                validation_record.validation_id,
                validation_record,
            )
            record_binding = self.runs.write_immutable_artifact(
                current_run.run_id,
                f"{prefix}/revision-record.yaml",
                "plan_revision_record",
                revision_record.revision_id,
                revision_record,
            )
            next_round = PlanRevisionRound(
                round_index=round_index,
                parent_plan_id=plan.plan_id,
                revision_input=input_binding,
                revision_record=record_binding,
                planning_proposal=proposal_binding,
                candidate_plan=plan_binding,
                compilation_receipt=compilation_binding,
                validation_record=validation_binding,
            )
            chain_payload = current_chain.model_dump(mode="python", exclude={"content_hash"})
            chain_payload.update(
                revisions_used=round_index,
                rounds=(*current_chain.rounds, next_round),
                termination_reason=None,
            )
            current_chain = _make_revision_chain(chain_payload)
            self.runs.save_revision_chain(current_chain)
            outcome_payload = {
                "attempt_id": attempt.attempt_id,
                "attempt_start_hash": attempt.content_hash,
                "status": PlanRevisionAttemptStatus.COMPLETED,
                "revision_record_id": revision_record.revision_id,
                "revision_record_hash": revision_record.content_hash,
                "round_index": round_index,
                "failure_category": None,
                "failure_message": None,
            }
            outcome = PlanRevisionAttemptOutcome(
                **outcome_payload,
                content_hash=content_hash(outcome_payload),
            )
            self.runs.write_immutable_artifact(
                current_run.run_id,
                f"{self.runs.attempt_prefix(current_run.run_id, attempt.attempt_index)}/outcome.yaml",
                "plan_revision_attempt_outcome",
                attempt.attempt_id,
                outcome,
            )
            current_run = update_run(
                current_run,
                status=ScientificProblemRunStatus.AWAITING_APPROVAL,
                planning_proposal_id=revision.proposal_set.proposal_id,
                planning_proposal_hash=content_hash(revision.proposal_set),
                candidate_plans=(plan_binding,),
                candidate_compilation_receipts=(compilation_binding,),
                plan_validation_records=(validation_binding,),
                selected_candidate_id=revised_plan.plan_id,
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
                failure=None,
                providers=replace_provider(current_run.providers, bind_provider("planning", planner)),
            )
            self.runs.save(current_run)
            current_run = self.approval.approve(
                current_run,
                context,
                packet,
                planning_input,
                repositories,
                evidence,
                approval_provider,
                revised_plan.plan_id,
                settings,
                artifact_prefix=f"{prefix}/approval",
                revision_approval=(revision_input, revision_record, review),
            )
            if current_run.approval_receipt_id is None:
                return current_run
            current_chain = self.record_approval(current_chain, current_run)

    def validate_chain(self, chain: PlanRevisionChain) -> None:
        attempts = self.runs.load_revision_attempts(chain.run_id)
        if attempts and len(attempts) < chain.revisions_used:
            raise ValueError("revision chain has more versions than started attempts")
        for round_record in chain.rounds:
            plan = self.runs.load_bound_artifact(chain.run_id, round_record.candidate_plan, ScientificQuestionPlan)
            receipt = self.runs.load_bound_artifact(
                chain.run_id,
                round_record.compilation_receipt,
                PlanCompilationReceipt,
            )
            validation = self.runs.load_bound_artifact(
                chain.run_id,
                round_record.validation_record,
                PlanValidationRecord,
            )
            self.runs.load_bound_artifact(chain.run_id, round_record.planning_proposal, PlanningProposalSet)
            if not validate_plan_compilation_receipt(plan, receipt).valid:
                raise ValueError("revision compilation receipt failed integrity validation")
            if validation.plan_id != plan.plan_id or validation.plan_content_hash != content_hash(plan):
                raise ValueError("revision validation record does not bind its plan")
            if round_record.round_index > 0:
                self.runs.load_bound_artifact(chain.run_id, round_record.revision_input, PlanRevisionInput)
                self.runs.load_bound_artifact(chain.run_id, round_record.revision_record, PlanRevisionRecord)
            if round_record.approval_receipt is None:
                continue
            review_input = self.runs.load_approval_input_binding(chain.run_id, round_record.approval_review_input)
            review = self.runs.load_bound_artifact(
                chain.run_id,
                round_record.approval_review_record,
                ApprovalReviewRecord,
            )
            verdict = self.runs.load_bound_artifact(chain.run_id, round_record.approval_verdict, ApprovalVerdict)
            approval_receipt = self.runs.load_bound_artifact(
                chain.run_id,
                round_record.approval_receipt,
                IndependentApprovalReceipt,
            )
            gate = self.runs.load_bound_artifact(chain.run_id, round_record.gate, GateVerdict)
            policy = self.runs.load_bound_artifact(chain.run_id, round_record.trust_policy, ProjectTrustPolicy)
            if not validate_approved_plan_authority(
                plan,
                validation,
                review_input,
                review,
                verdict,
                approval_receipt,
                gate,
                policy,
                receipt,
                require_passed=False,
            ).valid:
                raise ValueError("revision approval authority failed integrity validation")
