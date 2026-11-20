from __future__ import annotations

from dataclasses import dataclass

from ...approval import (
    ApprovalContextResolver,
    IndependentApprovalService,
    bind_gate_verdict,
)
from ...domains import DomainPackLoader
from ...models import (
    ApprovalDecision,
    ApprovalMode,
    ApprovalReviewRecord,
    ApprovalVerdict,
    GateVerdict,
    IndependentApprovalReceipt,
    PlanCompilationReceipt,
    PlanRevisionInput,
    PlanRevisionRecord,
    PlanValidationRecord,
    ProjectTrustPolicy,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificProblemRun,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
    parse_approval_review_input,
)
from ...serialization import content_hash, load_data
from ...validators import (
    validate_approved_plan_authority,
    validate_question_plan,
)
from ..providers import bind_provider, replace_provider
from ..repository import ScientificProblemRunRepository
from ..resume import ProviderOutcomeUncertain
from ..state import fail_run, update_run


@dataclass
class ApprovalStage:
    """Obtain independent review and verify every component before reusing a receipt."""

    runs: ScientificProblemRunRepository
    domain_loader: DomainPackLoader

    def revision_context(self, run_id, chain):
        if chain is None or chain.rounds[-1].round_index == 0:
            return None
        current_round = chain.rounds[-1]
        revision_input = self.runs.load_bound_artifact(run_id, current_round.revision_input, PlanRevisionInput)
        revision_record = self.runs.load_bound_artifact(run_id, current_round.revision_record, PlanRevisionRecord)
        return revision_input, revision_record, revision_input.approval_review_record

    def _review_input(
        self, context, packet, planning_input, plan, validation, repositories, evidence, revision_approval
    ):
        resolver = ApprovalContextResolver(self.domain_loader)
        return (
            resolver.resolve_revision(
                context,
                packet,
                planning_input,
                plan,
                validation,
                repositories,
                evidence,
                revision_input=revision_approval[0],
                revision_record=revision_approval[1],
                trigger_review=revision_approval[2],
            )
            if revision_approval is not None
            else resolver.resolve(
                context,
                packet,
                planning_input,
                plan,
                validation,
                repositories,
                evidence,
            )
        )

    def reuse(
        self,
        run: ScientificProblemRun,
        *,
        previous: ScientificProblemRun,
        context: ScientificContextPacket,
        packet: ScientificEvidencePacket,
        planning_input,
        repositories,
        evidence,
        selected_candidate_id: str | None,
        revision_approval=None,
    ) -> ScientificProblemRun | None:
        if previous.selected_candidate_id is None:
            return None
        if selected_candidate_id not in {None, previous.selected_candidate_id}:
            return None
        candidate_ids = tuple(item.artifact_id for item in run.candidate_plans)
        if previous.selected_candidate_id not in candidate_ids:
            return None
        index = candidate_ids.index(previous.selected_candidate_id)
        approval_prefix = self.runs.approval_prefix(run.run_id)
        try:
            plan = self.runs.load_artifact(run.run_id, run.candidate_plans[index].relative_path, ScientificQuestionPlan)
            validation = self.runs.load_artifact(
                run.run_id,
                run.plan_validation_records[index].relative_path,
                PlanValidationRecord,
            )
            expected_review_input = self._review_input(
                context, packet, planning_input, plan, validation, repositories, evidence, revision_approval
            )
            review_input = parse_approval_review_input(
                load_data(
                    self.runs.resolve_artifact_path(
                        run.run_id,
                        f"{approval_prefix}/approval-review-input.yaml",
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
            receipt = self.runs.load_artifact(
                run.run_id,
                f"{approval_prefix}/independent-approval-receipt.yaml",
                IndependentApprovalReceipt,
            )
            gate = self.runs.load_artifact(run.run_id, f"{approval_prefix}/plan-gate.yaml", GateVerdict)
            policy = self.runs.load_artifact(
                run.run_id,
                f"{approval_prefix}/project-trust-policy.yaml",
                ProjectTrustPolicy,
            )
            compilation_receipt = self.runs.load_artifact(
                run.run_id,
                run.candidate_compilation_receipts[index].relative_path,
                PlanCompilationReceipt,
            )
            current_report = validate_question_plan(plan, planning_input.scientific_capabilities, evidence)
            if content_hash(expected_review_input) != content_hash(review_input):
                return None
            if previous.approval_receipt_id is not None and (
                previous.approval_receipt_id != receipt.receipt_id
                or previous.approval_receipt_hash != receipt.content_hash
                or previous.approval_review_id != review.review_id
                or previous.approval_review_hash != review.content_hash
                or previous.approval_verdict_id != verdict.verdict_id
                or previous.approval_verdict_hash != content_hash(verdict)
                or previous.gate_id != gate.gate_id
                or previous.gate_hash != content_hash(gate)
            ):
                return None
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
                current_report=current_report,
                require_passed=False,
            ).valid:
                return None
        except (FileNotFoundError, OSError, ValueError):
            return None
        if "REQUIRES_REPLANNING" in verdict.hard_red_flags:
            status = ScientificProblemRunStatus.REQUIRES_REPLANNING
        elif gate.passed and previous.export_path is not None:
            status = ScientificProblemRunStatus.EXPORTED
        elif gate.passed:
            status = ScientificProblemRunStatus.APPROVED
        elif verdict.decision in {
            ApprovalDecision.REJECT,
            ApprovalDecision.REQUEST_REVISION,
            ApprovalDecision.INSUFFICIENT_EVIDENCE,
        }:
            status = ScientificProblemRunStatus.REJECTED
        else:
            status = ScientificProblemRunStatus.AWAITING_APPROVAL
        approval_binding = next((item for item in previous.providers if item.stage == "approval"), None)
        if approval_binding is None or (
            approval_binding.provider_id != receipt.provider_id
            or approval_binding.provider_version != receipt.provider_version
        ):
            return None
        providers = replace_provider(run.providers, approval_binding)
        reused = update_run(
            run,
            status=status,
            selected_candidate_id=previous.selected_candidate_id,
            approval_review_id=review.review_id,
            approval_review_hash=review.content_hash,
            approval_verdict_id=verdict.verdict_id,
            approval_verdict_hash=content_hash(verdict),
            approval_receipt_id=receipt.receipt_id,
            approval_receipt_hash=receipt.content_hash,
            gate_id=gate.gate_id,
            gate_hash=content_hash(gate),
            export_path=previous.export_path,
            export_hash=previous.export_hash,
            providers=providers,
        )
        self.runs.save(reused)
        return reused

    def approve(
        self,
        run,
        context,
        packet,
        planning_input,
        repositories,
        evidence,
        provider_name,
        selected_candidate_id,
        settings,
        artifact_prefix="approval",
        revision_approval=None,
    ) -> ScientificProblemRun:
        if selected_candidate_id is None:
            if len(run.candidate_plans) != 1:
                return update_run(run, status=ScientificProblemRunStatus.AWAITING_APPROVAL)
            selected_candidate_id = run.candidate_plans[0].artifact_id
        candidates = {binding.artifact_id: binding for binding in run.candidate_plans}
        if selected_candidate_id not in candidates:
            raise ValueError("selected candidate ID is not part of this run")
        candidate_binding = candidates[selected_candidate_id]
        plan = self.runs.load_artifact(run.run_id, candidate_binding.relative_path, ScientificQuestionPlan)
        index = tuple(binding.artifact_id for binding in run.candidate_plans).index(selected_candidate_id)
        validation_binding = run.plan_validation_records[index]
        receipt_binding = run.candidate_compilation_receipts[index]
        validation = self.runs.load_artifact(run.run_id, validation_binding.relative_path, PlanValidationRecord)
        compilation_receipt = self.runs.load_artifact(run.run_id, receipt_binding.relative_path, PlanCompilationReceipt)
        review_input = self._review_input(
            context, packet, planning_input, plan, validation, repositories, evidence, revision_approval
        )
        provider = settings.approval(provider_name)
        # The bound review input is the durable claim. An interrupted call is
        # recoverable only from a complete, independently validated receipt.
        input_path = f"{artifact_prefix}/approval-review-input.yaml"
        try:
            claimed = parse_approval_review_input(load_data(self.runs.resolve_artifact_path(run.run_id, input_path)))
        except FileNotFoundError:
            claimed = None
        if claimed is not None and claimed.content_hash == review_input.content_hash:
            error = ProviderOutcomeUncertain("the claimed approval has no reusable independent receipt")
            return fail_run(self.runs, run, "approval", error, retryable=False).run
        writer = self.runs.write_immutable_artifact if artifact_prefix != "approval" else self.runs.write_artifact
        run = update_run(
            run,
            selected_candidate_id=selected_candidate_id,
            providers=replace_provider(run.providers, bind_provider("approval", provider)),
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
        self.runs.save(run)
        writer(run.run_id, input_path, "approval_review_input", review_input.review_input_id, review_input)
        try:
            result = IndependentApprovalService(provider, approver_id="independent-scientific-approver").review(
                review_input
            )
            passed = result.verdict.decision == ApprovalDecision.APPROVE
            policy = ProjectTrustPolicy(
                approval_mode=ApprovalMode.INDEPENDENT_REQUIRED,
                policy_version="1.0.0",
            )
            gate = bind_gate_verdict(
                plan,
                result.verdict,
                validation,
                trust_policy=policy,
                gate_id=f"plan-gate-{content_hash(result.receipt)[:24]}",
                passed=passed,
                reasons=result.review.policy_reasons,
                review_input=review_input,
                review=result.review,
                receipt=result.receipt,
                compilation_receipt=compilation_receipt,
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            return fail_run(self.runs, run, "approval", error, retryable=False).run
        artifacts = (
            ("approval-review.yaml", "approval_review_record", result.review.review_id, result.review),
            ("approval-verdict.yaml", "approval_verdict", result.verdict.verdict_id, result.verdict),
            (
                "independent-approval-receipt.yaml",
                "independent_approval_receipt",
                result.receipt.receipt_id,
                result.receipt,
            ),
            ("plan-gate.yaml", "gate_verdict", gate.gate_id, gate),
            ("project-trust-policy.yaml", "project_trust_policy", "project-trust-policy", policy),
        )
        for path, kind, identifier, value in artifacts:
            writer(
                run.run_id,
                f"{artifact_prefix}/{path}",
                kind,
                identifier,
                value,
            )
        if "REQUIRES_REPLANNING" in result.verdict.hard_red_flags:
            status = ScientificProblemRunStatus.REQUIRES_REPLANNING
        elif passed:
            status = ScientificProblemRunStatus.APPROVED
        elif result.verdict.decision in {
            ApprovalDecision.REJECT,
            ApprovalDecision.REQUEST_REVISION,
            ApprovalDecision.INSUFFICIENT_EVIDENCE,
        }:
            status = ScientificProblemRunStatus.REJECTED
        else:
            status = ScientificProblemRunStatus.AWAITING_APPROVAL
        approved = update_run(
            run,
            status=status,
            selected_candidate_id=selected_candidate_id,
            approval_review_id=result.review.review_id,
            approval_review_hash=result.review.content_hash,
            approval_verdict_id=result.verdict.verdict_id,
            approval_verdict_hash=content_hash(result.verdict),
            approval_receipt_id=result.receipt.receipt_id,
            approval_receipt_hash=result.receipt.content_hash,
            gate_id=gate.gate_id,
            gate_hash=content_hash(gate),
            providers=replace_provider(run.providers, bind_provider("approval", provider)),
        )
        self.runs.save(approved)
        return approved
