from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import (
    ApprovalReviewRecord,
    ApprovalVerdict,
    GateVerdict,
    IndependentApprovalReceipt,
    PlanCompilationReceipt,
    PlanValidationRecord,
    PlanningProposalSet,
    ProjectTrustPolicy,
    ResultEvidenceSubmission,
    ResearchUpdate,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificQuestionPlan,
    SuccessorParentBinding,
    SuccessorPlanningCycle,
    SuccessorPlanningStatus,
    parse_approval_review_input,
)
from ..repositories import CompositeEvidenceStore
from ..serialization import content_hash, load_data
from ..validators import validate_approved_plan_authority, validate_question_plan
from ..workflow.repository import ScientificProblemRunRepository
from .artifacts import ResultArtifacts
from .submission import ResultFeedbackError


@dataclass(frozen=True)
class _ParentAuthorityCandidate:
    run: Any
    plan: ScientificQuestionPlan
    compilation_receipt: PlanCompilationReceipt
    validation_record: PlanValidationRecord
    review_input: Any
    review: ApprovalReviewRecord
    verdict: ApprovalVerdict
    approval_receipt: IndependentApprovalReceipt
    gate: GateVerdict
    policy: ProjectTrustPolicy
    origin: str
    origin_ref: str
    context_relative_path: str
    evidence_packet_relative_path: str
    planning_input_relative_path: str


@dataclass(frozen=True)
class ApprovedParent(_ParentAuthorityCandidate):
    task: Any


class ParentAuthority:
    def __init__(
        self,
        runs: ScientificProblemRunRepository,
        composite_evidence: CompositeEvidenceStore,
        artifacts: ResultArtifacts,
    ) -> None:
        self.runs = runs
        self.composite_evidence = composite_evidence
        self.artifacts = artifacts

    def _verify_parent_candidate(
        self,
        candidate: _ParentAuthorityCandidate,
        submission: ResultEvidenceSubmission,
    ) -> ApprovedParent:
        plan = candidate.plan
        planning_input = self.runs.load_artifact(
            candidate.run.run_id,
            candidate.planning_input_relative_path,
            ScientificPlanningInput,
        )
        current_report = validate_question_plan(
            plan,
            planning_input.scientific_capabilities,
            self.composite_evidence,
        )
        authority = validate_approved_plan_authority(
            plan,
            candidate.validation_record,
            candidate.review_input,
            candidate.review,
            candidate.verdict,
            candidate.approval_receipt,
            candidate.gate,
            candidate.policy,
            candidate.compilation_receipt,
            current_report=current_report,
        )
        if not authority.valid:
            issue_codes = {item.code for item in authority.issues}
            if issue_codes & {
                "TAMPERED_PLAN_COMPILATION_RECEIPT",
                "PLAN_COMPILATION_RECEIPT_MISMATCH",
                "INVALID_PLAN_COMPILATION_ORIGIN",
            }:
                category = "PARENT_COMPILATION_RECEIPT_INVALID"
            elif issue_codes & {
                "STALE_PLAN_VALIDATION",
                "PLAN_VALIDATION_RECORD_MISMATCH",
                "PLAN_VALIDATION_FAILED",
            }:
                category = "PARENT_VALIDATION_RECORD_INVALID"
            elif issue_codes & {
                "STALE_APPROVAL_REVIEW_INPUT",
                "STALE_APPROVAL_REVIEW_RECORD",
                "STALE_INDEPENDENT_APPROVAL_RECEIPT",
                "INDEPENDENT_REVIEW_INPUT_HASH_MISMATCH",
                "INDEPENDENT_REVIEW_HASH_MISMATCH",
                "INDEPENDENT_VERDICT_HASH_MISMATCH",
                "INDEPENDENT_RECEIPT_CANDIDATE_MISMATCH",
                "INDEPENDENT_PROVIDER_BINDING_MISMATCH",
                "INDEPENDENT_APPROVER_BINDING_MISMATCH",
                "INDEPENDENT_VERDICT_REVIEW_MISMATCH",
            }:
                category = "PARENT_APPROVAL_CHAIN_INVALID"
            else:
                category = "PARENT_GATE_INVALID"
            raise ResultFeedbackError(
                category,
                "parent plan approval authority failed validation: "
                + ", ".join(item.code for item in authority.issues),
            )
        task = next(
            (item for item in plan.tasks if item.task_id == submission.task_id), None
        )
        if task is None:
            raise ResultFeedbackError(
                "FABRICATED_TASK_ID",
                "result task does not belong to the exact parent plan",
            )
        if task.capability_id != submission.capability_id:
            raise ResultFeedbackError(
                "WRONG_CAPABILITY",
                "result capability does not match the approved parent task",
            )
        return ApprovedParent(**candidate.__dict__, task=task)

    def _successor_parent_candidates(
        self,
        run: Any,
        expected: tuple[str, str, str],
    ) -> list[_ParentAuthorityCandidate]:
        candidates: list[_ParentAuthorityCandidate] = []
        root = self.runs.resolve_artifact_path(run.run_id, "successor-planning")
        if not root.exists():
            return candidates
        for cycle_path in sorted(root.glob("cycle-*/cycle.yaml")):
            prefix = cycle_path.parent.relative_to(self.runs.run_dir(run.run_id)).as_posix()
            cycle = self.runs.load_artifact(
                run.run_id, f"{prefix}/cycle.yaml", SuccessorPlanningCycle
            )
            if cycle.selected_plan_id != expected[0]:
                continue
            plan = self.runs.load_artifact(
                run.run_id,
                f"{prefix}/candidates/{cycle.selected_plan_id}.yaml",
                ScientificQuestionPlan,
            )
            if (plan.plan_id, plan.version, content_hash(plan)) != expected:
                continue
            if cycle.status != SuccessorPlanningStatus.APPROVED:
                raise ResultFeedbackError(
                    "PARENT_SUCCESSOR_NOT_APPROVED",
                    "the exact successor cycle did not pass independent approval",
                )
            binding = self.runs.load_artifact(
                run.run_id,
                f"{prefix}/parent-binding.yaml",
                SuccessorParentBinding,
            )
            update = self.runs.load_artifact(
                run.run_id,
                f"{prefix}/research-update.yaml",
                ResearchUpdate,
            )
            try:
                cycle_index = int(prefix.rsplit("cycle-", 1)[1])
            except ValueError as error:
                raise ResultFeedbackError(
                    "SUCCESSOR_PARENT_AUTHORITY_INVALID",
                    "successor cycle directory has an invalid index",
                ) from error
            context = self.runs.load_artifact(
                run.run_id, f"{prefix}/context.yaml", ScientificContextPacket
            )
            packet = self.runs.load_artifact(
                run.run_id,
                f"{prefix}/evidence-packet.yaml",
                ScientificEvidencePacket,
            )
            planning_input = self.runs.load_artifact(
                run.run_id,
                f"{prefix}/planning-input.yaml",
                ScientificPlanningInput,
            )
            proposal = self.runs.load_artifact(
                run.run_id,
                f"{prefix}/planning-proposal.yaml",
                PlanningProposalSet,
            )
            receipt_paths = tuple(
                cycle_path.parent.joinpath("compilation-receipts").glob("*.yaml")
            )
            receipts = tuple(
                self.runs.load_artifact(
                    run.run_id,
                    path.relative_to(self.runs.run_dir(run.run_id)).as_posix(),
                    PlanCompilationReceipt,
                )
                for path in receipt_paths
            )
            matching_receipts = tuple(
                item
                for item in receipts
                if (item.plan_id, item.plan_hash) == (expected[0], expected[2])
            )
            validation_paths = tuple(
                cycle_path.parent.joinpath("validation").glob("*.yaml")
            )
            validations = tuple(
                self.runs.load_artifact(
                    run.run_id,
                    path.relative_to(self.runs.run_dir(run.run_id)).as_posix(),
                    PlanValidationRecord,
                )
                for path in validation_paths
            )
            matching_validations = tuple(
                item
                for item in validations
                if (item.plan_id, item.plan_version, item.plan_content_hash) == expected
            )
            if len(matching_receipts) != 1 or len(matching_validations) != 1:
                raise ResultFeedbackError(
                    "SUCCESSOR_PARENT_AUTHORITY_INVALID",
                    "successor plan must bind one compilation receipt and validation record",
                )
            if (
                binding.parent_run_id != run.run_id
                or binding.successor_cycle_index != cycle_index
                or (cycle.parent_binding_id, cycle.parent_binding_hash)
                != (binding.binding_id, binding.content_hash)
                or (cycle.research_update_id, cycle.research_update_hash)
                != (update.update_id, update.content_hash)
                or (update.parent_binding_id, update.parent_binding_hash)
                != (binding.binding_id, binding.content_hash)
                or plan.follow_up_of != binding.parent_plan_id
                or (cycle.context_id, cycle.context_hash)
                != (context.context_id, context.content_hash)
                or (cycle.evidence_packet_id, cycle.evidence_packet_hash)
                != (packet.packet_id, packet.content_hash)
                or (cycle.planning_input_id, cycle.planning_input_hash)
                != (planning_input.planning_input_id, planning_input.content_hash)
                or (cycle.planning_proposal_id, cycle.planning_proposal_hash)
                != (proposal.proposal_id, content_hash(proposal))
                or (cycle.selected_plan_id, cycle.selected_plan_hash)
                != (plan.plan_id, content_hash(plan))
                or plan.plan_id not in cycle.candidate_plan_ids
                or cycle.candidate_plan_hashes[
                    cycle.candidate_plan_ids.index(plan.plan_id)
                ]
                != content_hash(plan)
            ):
                raise ResultFeedbackError(
                    "SUCCESSOR_PARENT_AUTHORITY_INVALID",
                    "successor cycle lineage or artifact binding is invalid",
                )
            for submission_id, submission_hash, receipt_id, receipt_hash in zip(
                binding.result_submission_ids,
                binding.result_submission_hashes,
                binding.materialization_receipt_ids,
                binding.materialization_receipt_hashes,
                strict=True,
            ):
                source_submission = self.artifacts.load_submission(run.run_id, submission_id)
                source_receipt = self.artifacts.load_receipt(run.run_id, submission_id)
                source_parent = self.load_parent(source_submission)
                if (
                    source_submission.content_hash != submission_hash
                    or source_receipt.receipt_id != receipt_id
                    or source_receipt.content_hash != receipt_hash
                    or source_receipt.submission_id != source_submission.submission_id
                    or source_receipt.submission_hash != source_submission.content_hash
                    or (
                        source_submission.parent_plan_id,
                        source_submission.parent_plan_version,
                        source_submission.parent_plan_hash,
                    )
                    != (
                        binding.parent_plan_id,
                        binding.parent_plan_version,
                        binding.parent_plan_hash,
                    )
                    or (
                        source_parent.plan.plan_id,
                        source_parent.plan.version,
                        content_hash(source_parent.plan),
                    )
                    != (
                        binding.parent_plan_id,
                        binding.parent_plan_version,
                        binding.parent_plan_hash,
                    )
                ):
                    raise ResultFeedbackError(
                        "SUCCESSOR_PARENT_AUTHORITY_INVALID",
                        "successor parent binding does not match its source results",
                    )
            approval_prefix = f"{prefix}/approval"
            review_input = parse_approval_review_input(
                load_data(
                    self.runs.resolve_artifact_path(
                        run.run_id, f"{approval_prefix}/approval-review-input.yaml"
                    )
                )
            )
            review = self.runs.load_artifact(
                run.run_id,
                f"{approval_prefix}/approval-review.yaml",
                ApprovalReviewRecord,
            )
            verdict = self.runs.load_artifact(
                run.run_id,
                f"{approval_prefix}/approval-verdict.yaml",
                ApprovalVerdict,
            )
            approval_receipt = self.runs.load_artifact(
                run.run_id,
                f"{approval_prefix}/independent-approval-receipt.yaml",
                IndependentApprovalReceipt,
            )
            gate = self.runs.load_artifact(
                run.run_id, f"{approval_prefix}/plan-gate.yaml", GateVerdict
            )
            policy = self.runs.load_artifact(
                run.run_id,
                f"{approval_prefix}/project-trust-policy.yaml",
                ProjectTrustPolicy,
            )
            if (
                (cycle.approval_review_id, cycle.approval_review_hash)
                != (review.review_id, review.content_hash)
                or (cycle.approval_verdict_id, cycle.approval_verdict_hash)
                != (verdict.verdict_id, content_hash(verdict))
                or (cycle.approval_receipt_id, cycle.approval_receipt_hash)
                != (approval_receipt.receipt_id, approval_receipt.content_hash)
                or (cycle.gate_id, cycle.gate_hash)
                != (gate.gate_id, content_hash(gate))
            ):
                raise ResultFeedbackError(
                    "SUCCESSOR_PARENT_AUTHORITY_INVALID",
                    "successor cycle does not bind its independent approval artifacts",
                )
            candidates.append(
                _ParentAuthorityCandidate(
                    run=run,
                    plan=plan,
                    compilation_receipt=matching_receipts[0],
                    validation_record=matching_validations[0],
                    review_input=review_input,
                    review=review,
                    verdict=verdict,
                    approval_receipt=approval_receipt,
                    gate=gate,
                    policy=policy,
                    origin="successor_cycle",
                    origin_ref=cycle.cycle_id,
                    context_relative_path=f"{prefix}/context.yaml",
                    evidence_packet_relative_path=f"{prefix}/evidence-packet.yaml",
                    planning_input_relative_path=f"{prefix}/planning-input.yaml",
                )
            )
        return candidates

    def load_parent(self, submission: ResultEvidenceSubmission) -> ApprovedParent:
        try:
            run = self.runs.get(submission.parent_run_id)
        except FileNotFoundError as error:
            raise ResultFeedbackError(
                "PARENT_RUN_NOT_FOUND", submission.parent_run_id
            ) from error
        expected = (
            submission.parent_plan_id,
            submission.parent_plan_version,
            submission.parent_plan_hash,
        )
        candidates: list[_ParentAuthorityCandidate] = []
        chain = self.runs.load_revision_chain(run.run_id)
        if chain is not None:
            for round_record in chain.rounds:
                if any(
                    value is None
                    for value in (
                        round_record.approval_review_input,
                        round_record.approval_review_record,
                        round_record.approval_verdict,
                        round_record.approval_receipt,
                        round_record.gate,
                        round_record.trust_policy,
                    )
                ):
                    continue
                plan = self.runs.load_artifact(
                    run.run_id,
                    round_record.candidate_plan.relative_path,
                    ScientificQuestionPlan,
                )
                if (plan.plan_id, plan.version, content_hash(plan)) != expected:
                    continue
                candidates.append(
                    _ParentAuthorityCandidate(
                        run=run,
                        plan=plan,
                        compilation_receipt=self.runs.load_artifact(
                            run.run_id,
                            round_record.compilation_receipt.relative_path,
                            PlanCompilationReceipt,
                        ),
                        validation_record=self.runs.load_artifact(
                            run.run_id,
                            round_record.validation_record.relative_path,
                            PlanValidationRecord,
                        ),
                        review_input=parse_approval_review_input(
                            load_data(
                                self.runs.resolve_artifact_path(
                                    run.run_id,
                                    round_record.approval_review_input.relative_path,
                                )
                            )
                        ),
                        review=self.runs.load_artifact(
                            run.run_id,
                            round_record.approval_review_record.relative_path,
                            ApprovalReviewRecord,
                        ),
                        verdict=self.runs.load_artifact(
                            run.run_id,
                            round_record.approval_verdict.relative_path,
                            ApprovalVerdict,
                        ),
                        approval_receipt=self.runs.load_artifact(
                            run.run_id,
                            round_record.approval_receipt.relative_path,
                            IndependentApprovalReceipt,
                        ),
                        gate=self.runs.load_artifact(
                            run.run_id,
                            round_record.gate.relative_path,
                            GateVerdict,
                        ),
                        policy=self.runs.load_artifact(
                            run.run_id,
                            round_record.trust_policy.relative_path,
                            ProjectTrustPolicy,
                        ),
                        origin=(
                            "base_run_plan"
                            if round_record.round_index == 0
                            else "revision_round"
                        ),
                        origin_ref=f"revision-round-{round_record.round_index}",
                        context_relative_path="context.yaml",
                        evidence_packet_relative_path="evidence-packet.yaml",
                        planning_input_relative_path="planning-input.yaml",
                    )
                )
        else:
            for index, binding in enumerate(run.candidate_plans):
                plan = self.runs.load_artifact(
                    run.run_id, binding.relative_path, ScientificQuestionPlan
                )
                if (plan.plan_id, plan.version, content_hash(plan)) != expected:
                    continue
                try:
                    candidates.append(
                        _ParentAuthorityCandidate(
                            run=run,
                            plan=plan,
                            compilation_receipt=self.runs.load_artifact(
                                run.run_id,
                                run.candidate_compilation_receipts[index].relative_path,
                                PlanCompilationReceipt,
                            ),
                            validation_record=self.runs.load_artifact(
                                run.run_id,
                                run.plan_validation_records[index].relative_path,
                                PlanValidationRecord,
                            ),
                            review_input=parse_approval_review_input(
                                load_data(
                                    self.runs.resolve_artifact_path(
                                        run.run_id,
                                        "approval/approval-review-input.yaml",
                                    )
                                )
                            ),
                            review=self.runs.load_artifact(
                                run.run_id,
                                "approval/approval-review.yaml",
                                ApprovalReviewRecord,
                            ),
                            verdict=self.runs.load_artifact(
                                run.run_id,
                                "approval/approval-verdict.yaml",
                                ApprovalVerdict,
                            ),
                            approval_receipt=self.runs.load_artifact(
                                run.run_id,
                                "approval/independent-approval-receipt.yaml",
                                IndependentApprovalReceipt,
                            ),
                            gate=self.runs.load_artifact(
                                run.run_id,
                                "approval/plan-gate.yaml",
                                GateVerdict,
                            ),
                            policy=self.runs.load_artifact(
                                run.run_id,
                                "approval/project-trust-policy.yaml",
                                ProjectTrustPolicy,
                            ),
                            origin="base_run_plan",
                            origin_ref=binding.artifact_id,
                            context_relative_path="context.yaml",
                            evidence_packet_relative_path="evidence-packet.yaml",
                            planning_input_relative_path="planning-input.yaml",
                        )
                    )
                except FileNotFoundError:
                    continue
        candidates.extend(self._successor_parent_candidates(run, expected))
        if len(candidates) != 1:
            raise ResultFeedbackError(
                "PARENT_PLAN_BINDING_INVALID",
                "result must bind exactly one approved plan version in the parent run",
            )
        return self._verify_parent_candidate(candidates[0], submission)
