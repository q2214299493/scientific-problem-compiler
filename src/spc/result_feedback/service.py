from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable

from ..approval import (
    ApprovalContextResolver,
    IndependentApprovalService,
    MockApprovalProvider,
    bind_gate_verdict,
)
from ..compiler import ScientificProblemCompiler
from ..domains import DomainPackLoader
from ..downstream import DownstreamImportError, DownstreamImportValidator
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
    ExecutionProposal,
    FingerprintDifference,
    GateVerdict,
    GroundedStatement,
    IndependentApprovalReceipt,
    KnowledgeCurationRecord,
    KnowledgeSnapshot,
    MethodFact,
    ModelFact,
    PlanCompilationReceipt,
    PlanValidationRecord,
    PlanRevisionChain,
    PlanningProposalSet,
    ProjectTrustPolicy,
    ReportedResult,
    ReportedObservation,
    ResultContext,
    ResultContextComparison,
    ResultContextComparisonStatus,
    ResultEvidenceAssessment,
    ResultEvidenceIntakeStatus,
    ResultEvidenceMaterializationReceipt,
    ResultEvidenceSubmission,
    ResultEvidenceType,
    ResultExecutionOutcome,
    ResultIntakeIssue,
    ResearchUpdate,
    RetrievalHit,
    RetrievalManifest,
    RetrievalQuery,
    RetrievalSourceType,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificQuestionPlan,
    SourceRole,
    SourceType,
    SuccessorParentBinding,
    SuccessorPlanningCycle,
    SuccessorPlanningStatus,
    scientific_context_semantic_hash,
    parse_approval_review_input,
)
from ..planning import MockPlanningProvider, PlanningContextResolver
from ..planning.mock_provider import build_proposal_set
from ..planning.validators import validate_planning_proposal_set
from ..repositories import (
    CompositeEvidenceStore,
    KnowledgeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
)
from ..serialization import content_hash, file_sha256, load_data, to_primitive
from ..validators import (
    build_plan_validation_record,
    validate_independent_approval_chain,
    validate_plan_compilation_receipt,
    validate_plan_validation_record,
    validate_question_plan,
)
from ..workflow.repository import ScientificProblemRunRepository


RESULT_FEEDBACK_VERSION = "result-feedback-1.0.0"


class ResultFeedbackError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


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
class _ApprovedParent(_ParentAuthorityCandidate):
    task: Any


def _build_content_bound(model_type, prefix: str, payload: dict[str, Any]):
    draft = model_type.model_construct(
        **payload,
        **{
            next(
                name
                for name in model_type.model_fields
                if name.endswith("_id") and name not in payload
            ): "pending",
            "content_hash": "0" * 64,
        },
    )
    normalized = draft.model_dump(
        mode="json",
        exclude={"content_hash", *(
            name
            for name in model_type.model_fields
            if name.endswith("_id") and name not in payload
        )},
        exclude_none=True,
    )
    id_field = next(
        name
        for name in model_type.model_fields
        if name.endswith("_id") and name not in payload
    )
    identifier = f"{prefix}-{content_hash(normalized)[:24]}"
    value = {id_field: identifier, **normalized}
    return model_type(**value, content_hash=content_hash(value))


def build_result_evidence_submission(**values: Any) -> ResultEvidenceSubmission:
    """Build the immutable external-declaration envelope without trusting it."""

    payload = ResultEvidenceSubmission.model_construct(
        submission_id="pending",
        content_hash="0" * 64,
        **values,
    ).model_dump(
        mode="json",
        exclude={"submission_id", "content_hash"},
        exclude_none=True,
    )
    submission_id = f"result-submission-{content_hash(payload)[:24]}"
    return ResultEvidenceSubmission(
        submission_id=submission_id,
        **payload,
        content_hash=content_hash({"submission_id": submission_id, **payload}),
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) or hasattr(value, "items"):
        flattened: dict[str, Any] = {}
        for key, item in sorted(dict(value).items()):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(item, path))
        return flattened
    return {prefix: value}


def _normalized_difference_field(field: str) -> str:
    parts = field.split(".", 1)
    if len(parts) == 1:
        return field
    prefix, suffix = parts
    if prefix in {"system", "system_context"}:
        return f"system.{suffix}"
    if prefix in {"method", "method_context"}:
        return f"method.{suffix}"
    return field


def _canonical_value(value: Any) -> str:
    return content_hash(to_primitive(value))


def _same_difference(
    actual: FingerprintDifference,
    declared: FingerprintDifference,
) -> bool:
    actual_field = _normalized_difference_field(actual.field)
    declared_field = _normalized_difference_field(declared.field)
    field_matches = declared_field == actual_field or (
        "." not in declared.field and declared.field == actual_field.split(".", 1)[-1]
    )
    return (
        declared.disclosed_deviation
        and field_matches
        and _canonical_value(actual.left) == _canonical_value(declared.left)
        and _canonical_value(actual.right) == _canonical_value(declared.right)
    )


def _safe_artifact_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ResultFeedbackError(
            "UNSAFE_RESULT_ARTIFACT_PATH",
            f"result artifact path is unsafe: {relative_path}",
        )
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ResultFeedbackError(
            "UNSAFE_RESULT_ARTIFACT_PATH",
            f"result artifact escapes its declared root: {relative_path}",
        )
    return path


class ResultEvidenceIntakeService:
    def __init__(self, *, state_dir: Path, knowledge_dir: Path) -> None:
        self.state_dir = state_dir.resolve()
        self.knowledge_dir = knowledge_dir.resolve()
        self.runs = ScientificProblemRunRepository(self.state_dir)
        self.project_evidence = ProjectEvidenceStore(self.state_dir)
        self.composite_evidence = CompositeEvidenceStore(
            KnowledgeEvidenceStore(self.knowledge_dir),
            self.project_evidence,
        )

    @staticmethod
    def _submission_prefix(submission_id: str) -> str:
        return f"result-evidence/submissions/{submission_id}"

    @staticmethod
    def _accepted_prefix(submission_id: str) -> str:
        return f"result-evidence/accepted/s-{submission_id[-12:]}"

    def _load_revision_chain(self, run_id: str) -> PlanRevisionChain | None:
        try:
            return self.runs.load_artifact(
                run_id, "plan-revisions/revision-chain.yaml", PlanRevisionChain
            )
        except FileNotFoundError:
            return None

    def _verify_parent_candidate(
        self,
        candidate: _ParentAuthorityCandidate,
        submission: ResultEvidenceSubmission,
    ) -> _ApprovedParent:
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
        if not validate_plan_compilation_receipt(
            plan, candidate.compilation_receipt
        ).valid:
            raise ResultFeedbackError(
                "PARENT_COMPILATION_RECEIPT_INVALID",
                "parent plan compilation receipt failed validation",
            )
        if not validate_plan_validation_record(
            plan, candidate.validation_record, current_report
        ).valid:
            raise ResultFeedbackError(
                "PARENT_VALIDATION_RECORD_INVALID",
                "parent validation record is stale or no longer reproducible",
            )
        if not validate_independent_approval_chain(
            plan,
            candidate.verdict,
            candidate.review_input,
            candidate.review,
            candidate.approval_receipt,
        ).valid:
            raise ResultFeedbackError(
                "PARENT_APPROVAL_CHAIN_INVALID",
                "parent independent approval chain failed validation",
            )
        plan_hash = content_hash(plan)
        gate = candidate.gate
        if (
            not gate.passed
            or candidate.policy.approval_mode != ApprovalMode.INDEPENDENT_REQUIRED
            or (gate.candidate_id, gate.candidate_version, gate.candidate_content_hash)
            != (plan.plan_id, plan.version, plan_hash)
            or (gate.plan_validation_id, gate.plan_validation_hash)
            != (
                candidate.validation_record.validation_id,
                content_hash(candidate.validation_record),
            )
            or (gate.approval_verdict_id, gate.approval_verdict_hash)
            != (candidate.verdict.verdict_id, content_hash(candidate.verdict))
            or (
                gate.independent_approval_receipt_id,
                gate.independent_approval_receipt_hash,
            )
            != (
                candidate.approval_receipt.receipt_id,
                candidate.approval_receipt.content_hash,
            )
            or (gate.plan_compilation_receipt_id, gate.plan_compilation_receipt_hash)
            != (
                candidate.compilation_receipt.receipt_id,
                candidate.compilation_receipt.content_hash,
            )
            or (gate.trust_policy_version, gate.trust_policy_hash)
            != (candidate.policy.policy_version, content_hash(candidate.policy))
        ):
            raise ResultFeedbackError(
                "PARENT_GATE_INVALID",
                "parent plan lacks a valid independently approved gate",
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
        return _ApprovedParent(**candidate.__dict__, task=task)

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
                source_submission = self.runs.load_artifact(
                    run.run_id,
                    f"{self._submission_prefix(submission_id)}/submission.yaml",
                    ResultEvidenceSubmission,
                )
                source_receipt = self.runs.load_artifact(
                    run.run_id,
                    f"{self._accepted_prefix(submission_id)}/receipt.yaml",
                    ResultEvidenceMaterializationReceipt,
                )
                source_parent = self._load_parent(source_submission)
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

    def _load_parent(self, submission: ResultEvidenceSubmission) -> _ApprovedParent:
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
        chain = self._load_revision_chain(run.run_id)
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

    def _verify_and_archive_artifacts(
        self,
        submission: ResultEvidenceSubmission,
        artifact_root: Path,
    ) -> tuple[ResultIntakeIssue, ...]:
        issues: list[ResultIntakeIssue] = []
        root = artifact_root.resolve()
        for artifact in submission.source_artifact_manifest:
            try:
                path = _safe_artifact_path(root, artifact.relative_path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path.stat().st_size != artifact.size_bytes:
                    raise ValueError("artifact byte size does not match manifest")
                if file_sha256(path) != artifact.sha256:
                    raise ValueError("artifact checksum does not match manifest")
                destination = self.runs.resolve_artifact_path(
                    submission.parent_run_id,
                    f"{self._submission_prefix(submission.submission_id)}/raw/"
                    f"{artifact.relative_path}",
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = path.read_bytes()
                if destination.exists():
                    if destination.is_symlink() or destination.read_bytes() != content:
                        raise ValueError("archived result artifact conflicts with existing bytes")
                else:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=".result-artifact-",
                        dir=destination.parent,
                        delete=False,
                    ) as handle:
                        handle.write(content)
                        temporary = Path(handle.name)
                    os.replace(temporary, destination)
                    os.chmod(destination, 0o444)
                if file_sha256(destination) != artifact.sha256:
                    raise ValueError("archived result artifact failed checksum verification")
            except (FileNotFoundError, OSError, ValueError, ResultFeedbackError) as error:
                issues.append(
                    ResultIntakeIssue(
                        code="RESULT_ARTIFACT_INTEGRITY_FAILURE",
                        message=f"{artifact.relative_path}: {error}",
                    )
                )
        return tuple(issues)

    @staticmethod
    def _comparison(
        submission: ResultEvidenceSubmission,
        plan: ScientificQuestionPlan,
    ) -> ResultContextComparison:
        differences: list[FingerprintDifference] = []
        insufficient = False
        for category, expected, actual in (
            (
                "system",
                _flatten(plan.system_fingerprint.attributes),
                _flatten(submission.system_fingerprint.attributes),
            ),
            (
                "method",
                _flatten(plan.method_fingerprint.attributes),
                _flatten(submission.method_fingerprint.attributes),
            ),
        ):
            for field in sorted(set(expected) | set(actual)):
                expected_value = expected.get(field)
                actual_value = actual.get(field)
                if field not in actual:
                    insufficient = True
                    differences.append(
                        FingerprintDifference(
                            field=f"{category}.{field}",
                            left=expected_value,
                            right=None,
                            disclosed_deviation=False,
                        )
                    )
                elif field not in expected or actual_value != expected_value:
                    differences.append(
                        FingerprintDifference(
                            field=f"{category}.{field}",
                            left=expected_value,
                            right=actual_value,
                            disclosed_deviation=False,
                        )
                    )
        approved = tuple(
            item
            for item in differences
            if any(
                _same_difference(item, declared)
                for declared in submission.declared_deviations
            )
            and any(
                _same_difference(item, allowed)
                for allowed in plan.fingerprint_differences
            )
        )
        if not differences:
            status = ResultContextComparisonStatus.MATCHED
        elif insufficient:
            status = ResultContextComparisonStatus.INSUFFICIENT_CONTEXT
        elif len(approved) == len(differences):
            status = ResultContextComparisonStatus.EXPLICIT_APPROVED_DEVIATION
        elif any(item.field.startswith("system.") for item in differences):
            status = ResultContextComparisonStatus.UNDECLARED_SYSTEM_CHANGE
        else:
            status = ResultContextComparisonStatus.UNDECLARED_METHOD_CHANGE
        payload = {
            "submission_id": submission.submission_id,
            "submission_hash": submission.content_hash,
            "parent_plan_id": plan.plan_id,
            "parent_plan_hash": content_hash(plan),
            "status": status,
            "differences": tuple(differences),
            "approved_deviation_refs": tuple(
                f"fingerprint-difference:{content_hash(item)[:24]}"
                for item in approved
            ),
        }
        return _build_content_bound(
            ResultContextComparison,
            "result-context-comparison",
            payload,
        )

    @staticmethod
    def _admissibility_issues(
        submission: ResultEvidenceSubmission,
        parent: _ApprovedParent,
    ) -> tuple[ResultIntakeIssue, ...]:
        issues: list[ResultIntakeIssue] = []
        observed = {
            item.observable_key for item in submission.reported_observables
        } | {item.quantity for item in submission.reported_observables}
        missing = tuple(output for output in parent.task.outputs if output not in observed)
        if (
            submission.execution_outcome == ResultExecutionOutcome.COMPLETED
            and missing
        ):
            issues.append(
                ResultIntakeIssue(
                    code="REQUIRED_TASK_OUTPUT_MISSING",
                    message="completed result omits required task outputs: "
                    + ", ".join(missing),
                )
            )
        if submission.result_type in {
            ResultEvidenceType.COMPUTED_RESULT,
            ResultEvidenceType.PARTIAL_RESULT,
            ResultEvidenceType.NULL_OR_NEGATIVE_RESULT,
        }:
            converged = submission.convergence_metadata.get("converged")
            if submission.result_type == ResultEvidenceType.COMPUTED_RESULT and converged is not True:
                issues.append(
                    ResultIntakeIssue(
                        code="CONVERGENCE_METADATA_INSUFFICIENT",
                        message=(
                            "computed_result requires an explicit converged=true declaration; "
                            "this declaration remains subject to curation"
                        ),
                    )
                )
        expected_status = {
            ResultEvidenceType.COMPUTED_RESULT: "computed_reported",
            ResultEvidenceType.EXPERIMENTAL_RESULT: "experimental_reported",
        }.get(submission.result_type)
        if expected_status is not None and any(
            item.result_status.value != expected_status
            for item in submission.reported_observables
        ):
            issues.append(
                ResultIntakeIssue(
                    code="RESULT_ORIGIN_STATUS_MISMATCH",
                    message="reported observable origin does not match result_type",
                )
            )
        return tuple(issues)

    def intake(
        self,
        submission: ResultEvidenceSubmission,
        *,
        artifact_root: Path,
        execution_proposal: ExecutionProposal | None = None,
    ) -> ResultEvidenceAssessment:
        try:
            self.runs.get(submission.parent_run_id)
        except FileNotFoundError as error:
            raise ResultFeedbackError(
                "PARENT_RUN_NOT_FOUND", submission.parent_run_id
            ) from error
        prefix = self._submission_prefix(submission.submission_id)
        with self.runs.acquire_run_lock(submission.parent_run_id):
            self.runs.write_immutable_artifact(
                submission.parent_run_id,
                f"{prefix}/submission.yaml",
                "result_evidence_submission",
                submission.submission_id,
                submission,
            )
            content_issues = list(
                self._verify_and_archive_artifacts(submission, artifact_root)
            )
            parent: _ApprovedParent | None = None
            plan_issues: list[ResultIntakeIssue] = []
            if not content_issues:
                try:
                    parent = self._load_parent(submission)
                    if submission.execution_proposal_id is not None:
                        if execution_proposal is None:
                            raise ResultFeedbackError(
                                "EXECUTION_PROPOSAL_MISSING",
                                "submission names an execution proposal but none was supplied",
                            )
                        if (
                            execution_proposal.proposal_id
                            != submission.execution_proposal_id
                            or content_hash(execution_proposal)
                            != submission.execution_proposal_hash
                            or execution_proposal.source_plan_id != parent.plan.plan_id
                            or execution_proposal.source_plan_hash
                            != content_hash(parent.plan)
                            or execution_proposal.task_id != parent.task.task_id
                            or execution_proposal.scientific_capability_id
                            != parent.task.capability_id
                        ):
                            raise ResultFeedbackError(
                                "EXECUTION_PROPOSAL_BINDING_INVALID",
                                "execution proposal does not bind the exact plan/task/capability",
                            )
                    if submission.export_id is not None:
                        if parent.run.export_path is None:
                            raise ResultFeedbackError(
                                "EXPORT_BINDING_MISSING",
                                "submission names an export but the parent run has none",
                            )
                        package = DownstreamImportValidator().import_package(
                            Path(parent.run.export_path)
                        )
                        manifest = package.export_manifest
                        if (
                            manifest.export_id != submission.export_id
                            or content_hash(manifest)
                            != submission.export_manifest_hash
                            or package.source_plan.plan_id != parent.plan.plan_id
                            or content_hash(package.source_plan)
                            != content_hash(parent.plan)
                            or not any(
                                item.scientific_capability_id
                                == parent.task.capability_id
                                for item in package.capability_bindings
                            )
                        ):
                            raise ResultFeedbackError(
                                "EXPORT_BINDING_INVALID",
                                "submission export binding does not match the parent export",
                            )
                except (
                    DownstreamImportError,
                    FileNotFoundError,
                    OSError,
                    ValueError,
                ) as error:
                    plan_issues.append(
                        ResultIntakeIssue(
                            code=getattr(error, "code", "PARENT_PLAN_BINDING_INVALID"),
                            message=str(error),
                        )
                    )

            comparison = self._comparison(submission, parent.plan) if parent else None
            context_compatible = comparison is not None and comparison.status in {
                ResultContextComparisonStatus.MATCHED,
                ResultContextComparisonStatus.EXPLICIT_APPROVED_DEVIATION,
            }
            context_issues = (
                ()
                if context_compatible or comparison is None
                else (
                    ResultIntakeIssue(
                        code=comparison.status.value,
                        message="result system/method context is not compatible with the parent task",
                    ),
                )
            )
            admissibility_issues = (
                self._admissibility_issues(submission, parent)
                if parent is not None and context_compatible
                else ()
            )
            issues = tuple(
                (*content_issues, *plan_issues, *context_issues, *admissibility_issues)
            )
            content_bound = not content_issues
            plan_bound = parent is not None and not plan_issues
            scientifically_admissible = (
                plan_bound and context_compatible and not admissibility_issues
            )
            if content_issues:
                status = ResultEvidenceIntakeStatus.BLOCKED_CONTENT_BINDING
            elif plan_issues:
                status = ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING
            elif context_issues:
                status = ResultEvidenceIntakeStatus.BLOCKED_CONTEXT_COMPATIBILITY
            elif admissibility_issues:
                status = ResultEvidenceIntakeStatus.BLOCKED_SCIENTIFIC_ADMISSIBILITY
            else:
                status = ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
            payload = {
                "submission_id": submission.submission_id,
                "submission_hash": submission.content_hash,
                "parent_run_id": submission.parent_run_id,
                "parent_plan_id": submission.parent_plan_id,
                "parent_plan_hash": submission.parent_plan_hash,
                "task_id": submission.task_id,
                "capability_id": submission.capability_id,
                "parent_authority_origin": parent.origin if parent else None,
                "parent_authority_ref": parent.origin_ref if parent else None,
                "content_bound": content_bound,
                "plan_bound": plan_bound,
                "context_compatible": context_compatible,
                "scientifically_admissible": scientifically_admissible,
                "context_comparison": comparison,
                "issues": issues,
                "status": status,
            }
            assessment = _build_content_bound(
                ResultEvidenceAssessment, "result-assessment", payload
            )
            self.runs.write_immutable_artifact(
                submission.parent_run_id,
                f"{prefix}/assessment.yaml",
                "result_evidence_assessment",
                assessment.assessment_id,
                assessment,
            )
            if comparison is not None:
                self.runs.write_immutable_artifact(
                    submission.parent_run_id,
                    f"{prefix}/context-comparison.yaml",
                    "result_context_comparison",
                    comparison.comparison_id,
                    comparison,
                )
            return assessment

    def _load_submission(self, run_id: str, submission_id: str) -> ResultEvidenceSubmission:
        return self.runs.load_artifact(
            run_id,
            f"{self._submission_prefix(submission_id)}/submission.yaml",
            ResultEvidenceSubmission,
        )

    def _load_assessment(self, run_id: str, submission_id: str) -> ResultEvidenceAssessment:
        return self.runs.load_artifact(
            run_id,
            f"{self._submission_prefix(submission_id)}/assessment.yaml",
            ResultEvidenceAssessment,
        )

    def _verify_archived_artifacts(
        self, submission: ResultEvidenceSubmission
    ) -> None:
        for artifact in submission.source_artifact_manifest:
            archived = self.runs.resolve_artifact_path(
                submission.parent_run_id,
                f"{self._submission_prefix(submission.submission_id)}/raw/"
                f"{artifact.relative_path}",
            )
            if (
                archived.is_symlink()
                or not archived.is_file()
                or archived.stat().st_size != artifact.size_bytes
                or file_sha256(archived) != artifact.sha256
            ):
                raise ResultFeedbackError(
                    "TAMPERED_ACCEPTED_RESULT_ARTIFACT",
                    f"archived result changed: {artifact.relative_path}",
                )

    def _current_curation(
        self, run_id: str, submission_id: str
    ) -> KnowledgeCurationRecord | None:
        root = self.runs.resolve_artifact_path(
            run_id, f"result-evidence/curations/{submission_id}"
        )
        if not root.exists():
            return None
        records = tuple(
            self.runs.load_artifact(
                run_id,
                f"result-evidence/curations/{submission_id}/{path.name}",
                KnowledgeCurationRecord,
            )
            for path in sorted(root.glob("*.yaml"))
        )
        superseded = {
            item.supersedes_curation_id
            for item in records
            if item.supersedes_curation_id is not None
        }
        current = tuple(item for item in records if item.curation_id not in superseded)
        if len(current) != 1:
            raise ResultFeedbackError(
                "AMBIGUOUS_RESULT_CURATION",
                "result curation history must have exactly one current record",
            )
        return current[0]

    @staticmethod
    def _curation_record(
        submission: ResultEvidenceSubmission,
        *,
        status: CurationStatus,
        curator_id: str,
        rationale: str,
        supersedes: KnowledgeCurationRecord | None,
    ) -> KnowledgeCurationRecord:
        identity = {
            "target_type": "result_evidence_submission",
            "target_id": submission.submission_id,
            "target_hash": submission.content_hash,
            "status": status,
            "curator_id": curator_id,
            "rationale": rationale,
            "evidence_refs": (),
            "supersedes_curation_id": (
                supersedes.curation_id if supersedes is not None else None
            ),
        }
        normalized = KnowledgeCurationRecord.model_construct(
            curation_id="pending",
            **identity,
            content_hash="0" * 64,
        ).model_dump(
            mode="json",
            exclude={"curation_id", "content_hash"},
            exclude_none=True,
        )
        curation_id = f"knowledge-curation-{content_hash(normalized)[:24]}"
        payload = {"curation_id": curation_id, **normalized}
        return KnowledgeCurationRecord(
            **payload, content_hash=content_hash(payload)
        )

    def curate(
        self,
        *,
        parent_run_id: str,
        submission_id: str,
        status: CurationStatus,
        curator_id: str,
        rationale: str,
    ) -> tuple[KnowledgeCurationRecord, ResultEvidenceMaterializationReceipt | None]:
        if status not in {CurationStatus.ACCEPTED, CurationStatus.REJECTED}:
            raise ResultFeedbackError(
                "INVALID_RESULT_CURATION_STATUS",
                "result intake may only be explicitly accepted or rejected",
            )
        with self.runs.acquire_run_lock(parent_run_id):
            submission = self._load_submission(parent_run_id, submission_id)
            assessment = self._load_assessment(parent_run_id, submission_id)
            if (
                assessment.submission_hash != submission.content_hash
                or assessment.parent_run_id != parent_run_id
            ):
                raise ResultFeedbackError(
                    "RESULT_ASSESSMENT_BINDING_INVALID",
                    "assessment does not bind the requested submission and run",
                )
            if (
                status == CurationStatus.ACCEPTED
                and assessment.status
                != ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
            ):
                raise ResultFeedbackError(
                    "RESULT_NOT_ADMISSIBLE_FOR_ACCEPTANCE",
                    "blocked result intake cannot be accepted",
                )
            if status == CurationStatus.ACCEPTED:
                self._verify_archived_artifacts(submission)
            current = self._current_curation(parent_run_id, submission_id)
            if current is not None and current.status == status:
                receipt = (
                    self.runs.load_artifact(
                        parent_run_id,
                        f"{self._accepted_prefix(submission_id)}/receipt.yaml",
                        ResultEvidenceMaterializationReceipt,
                    )
                    if status == CurationStatus.ACCEPTED
                    else None
                )
                return current, receipt
            curation = self._curation_record(
                submission,
                status=status,
                curator_id=curator_id,
                rationale=rationale,
                supersedes=current,
            )
            self.runs.write_immutable_artifact(
                parent_run_id,
                f"result-evidence/curations/{submission_id}/"
                f"c-{curation.content_hash[:16]}.yaml",
                "knowledge_curation_record",
                curation.curation_id,
                curation,
            )
            if status == CurationStatus.REJECTED:
                return curation, None
            return curation, self._materialize(
                submission, assessment, curation
            )

    def _materialize(
        self,
        submission: ResultEvidenceSubmission,
        assessment: ResultEvidenceAssessment,
        curation: KnowledgeCurationRecord,
    ) -> ResultEvidenceMaterializationReceipt:
        parent = self._load_parent(submission)
        method_text = "method_context = " + json.dumps(
            to_primitive(submission.method_fingerprint.attributes),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system_text = "system_context = " + json.dumps(
            to_primitive(submission.system_fingerprint.attributes),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        outcome_text = (
            f"execution_outcome = {submission.execution_outcome.value}; "
            f"result_type = {submission.result_type.value}"
        )
        observable_lines = tuple(
            (
                f"{item.quantity} = {item.value} {item.unit}"
                if item.value is not None
                else f"{item.quantity} = {item.qualitative_value}"
            )
            for item in submission.reported_observables
        )
        lines = (system_text, method_text, outcome_text, *observable_lines)
        text = "\n".join(lines) + "\n"
        run_dir = self.runs.run_dir(submission.parent_run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=".result-summary-",
                suffix=".txt",
                dir=run_dir,
                delete=False,
            ) as handle:
                handle.write(text)
                temporary_path = Path(handle.name)
            source_id = f"result-source-{submission.submission_id.removeprefix('result-submission-')}"
            source_version = submission.content_hash[:24]
            source = self.project_evidence.ingest_source(
                temporary_path,
                source_id,
                source_version,
                title=f"Accepted downstream result {submission.submission_id}",
                source_role=SourceRole.SYSTEM,
                source_type=SourceType.CALCULATION_ARCHIVE,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        evidence_by_label: dict[str, EvidenceSpan] = {}
        offset = 0
        labels = ("system", "method", "outcome", *(
            f"observable:{item.observable_key}" for item in submission.reported_observables
        ))
        for label, line in zip(labels, lines, strict=True):
            evidence_id = (
                "result-evidence-"
                + content_hash(
                    {"submission_id": submission.submission_id, "label": label}
                )[:24]
            )
            evidence = EvidenceSpan(
                evidence_id=evidence_id,
                source_id=source.source_id,
                source_version=source.version,
                content_sha256=source.content_sha256,
                start_offset=offset,
                end_offset=offset + len(line),
                text=line,
                locator=f"result submission {submission.submission_id}; {label}",
            )
            self.project_evidence.add_evidence(evidence)
            evidence_by_label[label] = evidence
            offset += len(line) + 1

        method_evidence = evidence_by_label["method"].evidence_id
        system_evidence = evidence_by_label["system"].evidence_id
        method_payload = {
            "text": method_text,
            "attributes": submission.method_fingerprint.attributes,
            "evidence_refs": (method_evidence,),
        }
        method_fact = MethodFact(
            fact_id=f"method-fact-{content_hash(method_payload)[:24]}",
            **method_payload,
        )
        model_payload = {
            "text": system_text,
            "attributes": submission.system_fingerprint.attributes,
            "evidence_refs": (system_evidence,),
        }
        model_fact = ModelFact(
            fact_id=f"model-fact-{content_hash(model_payload)[:24]}",
            **model_payload,
        )
        results: list[ReportedResult] = []
        observations: list[ReportedObservation] = []
        if submission.result_type in {
            ResultEvidenceType.COMPUTED_RESULT,
            ResultEvidenceType.EXPERIMENTAL_RESULT,
            ResultEvidenceType.PARTIAL_RESULT,
            ResultEvidenceType.NULL_OR_NEGATIVE_RESULT,
        }:
            for observable in submission.reported_observables:
                observable_evidence = evidence_by_label[
                    f"observable:{observable.observable_key}"
                ].evidence_id
                result_context_payload = {
                    "system_context": submission.system_fingerprint.attributes,
                    "method_context": submission.method_fingerprint.attributes,
                    "method_fact_refs": (method_fact.fact_id,),
                    "model_fact_refs": (model_fact.fact_id,),
                }
                result_context = ResultContext(
                    context_id=(
                        "result-context-"
                        + content_hash(result_context_payload)[:24]
                    ),
                    **result_context_payload,
                )
                shared_payload = {
                    "quantity": observable.quantity,
                    "system_context": submission.system_fingerprint.attributes,
                    "method_context": submission.method_fingerprint.attributes,
                    "result_context": result_context,
                    "evidence_refs": (
                        observable_evidence,
                        method_evidence,
                        system_evidence,
                    ),
                    "result_status": observable.result_status,
                }
                if observable.value is not None:
                    result_payload = {
                        **shared_payload,
                        "value": observable.value,
                        "unit": observable.unit,
                    }
                    results.append(
                        ReportedResult(
                            result_id=(
                                "reported-result-"
                                + content_hash(result_payload)[:24]
                            ),
                            **result_payload,
                        )
                    )
                else:
                    observation_payload = {
                        **shared_payload,
                        "qualitative_value": observable.qualitative_value,
                    }
                    observations.append(
                        ReportedObservation(
                            observation_id=(
                                "reported-observation-"
                                + content_hash(observation_payload)[:24]
                            ),
                            **observation_payload,
                        )
                    )

        prefix = self._accepted_prefix(submission.submission_id)
        self.runs.write_immutable_artifact(
            submission.parent_run_id,
            f"{prefix}/m/{method_fact.fact_id[-16:]}.yaml",
            "method_fact",
            method_fact.fact_id,
            method_fact,
        )
        self.runs.write_immutable_artifact(
            submission.parent_run_id,
            f"{prefix}/d/{model_fact.fact_id[-16:]}.yaml",
            "model_fact",
            model_fact.fact_id,
            model_fact,
        )
        for result in results:
            self.runs.write_immutable_artifact(
                submission.parent_run_id,
                f"{prefix}/r/{result.result_id[-16:]}.yaml",
                "reported_result",
                result.result_id,
                result,
            )
        for observation in observations:
            self.runs.write_immutable_artifact(
                submission.parent_run_id,
                f"{prefix}/o/{observation.observation_id[-16:]}.yaml",
                "reported_observation",
                observation.observation_id,
                observation,
            )
        observed = {
            item.observable_key for item in submission.reported_observables
        } | {item.quantity for item in submission.reported_observables}
        missing = tuple(output for output in parent.task.outputs if output not in observed)
        payload = {
            "submission_id": submission.submission_id,
            "submission_hash": submission.content_hash,
            "assessment_id": assessment.assessment_id,
            "assessment_hash": assessment.content_hash,
            "curation_id": curation.curation_id,
            "curation_hash": curation.content_hash,
            "parent_run_id": submission.parent_run_id,
            "parent_plan_id": submission.parent_plan_id,
            "parent_plan_hash": submission.parent_plan_hash,
            "task_id": submission.task_id,
            "parent_authority_origin": parent.origin,
            "parent_authority_ref": parent.origin_ref,
            "source_id": source.source_id,
            "source_version": source.version,
            "accepted_evidence_ids": tuple(
                evidence.evidence_id for evidence in evidence_by_label.values()
            ),
            "reported_result_hashes": {
                item.result_id: content_hash(item) for item in results
            },
            "reported_observation_hashes": {
                item.observation_id: content_hash(item) for item in observations
            },
            "method_fact_hashes": {method_fact.fact_id: content_hash(method_fact)},
            "model_fact_hashes": {model_fact.fact_id: content_hash(model_fact)},
            "missing_observables": missing,
        }
        receipt = _build_content_bound(
            ResultEvidenceMaterializationReceipt,
            "result-materialization",
            payload,
        )
        self.runs.write_immutable_artifact(
            submission.parent_run_id,
            f"{prefix}/receipt.yaml",
            "result_evidence_materialization_receipt",
            receipt.receipt_id,
            receipt,
        )
        return receipt


class SuccessorPlanningService(ResultEvidenceIntakeService):
    def _load_receipt(
        self, run_id: str, submission_id: str
    ) -> ResultEvidenceMaterializationReceipt:
        return self.runs.load_artifact(
            run_id,
            f"{self._accepted_prefix(submission_id)}/receipt.yaml",
            ResultEvidenceMaterializationReceipt,
        )

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
        prefix = self._accepted_prefix(submission_id)
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
        return _build_content_bound(ResearchUpdate, "research-update", payload)

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
                self._load_submission(parent_run_id, item) for item in submission_ids
            )
            if any(item.parent_run_id != parent_run_id for item in submissions):
                raise ResultFeedbackError(
                    "CROSS_RUN_RESULT",
                    "all successor results must belong to the requested parent run",
                )
            parent = self._load_parent(submissions[0])
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
                self._load_receipt(parent_run_id, item) for item in submission_ids
            )
            curations = tuple(
                self._current_curation(parent_run_id, item) for item in submission_ids
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
                self._verify_archived_artifacts(submission)

            existing_cycles: list[SuccessorPlanningCycle] = []
            successor_root = self.runs.resolve_artifact_path(
                parent_run_id, "successor-planning"
            )
            if successor_root.exists():
                for path in sorted(successor_root.glob("cycle-*/cycle.yaml")):
                    relative = path.relative_to(self.runs.run_dir(parent_run_id)).as_posix()
                    existing_cycles.append(
                        self.runs.load_artifact(
                            parent_run_id, relative, SuccessorPlanningCycle
                        )
                    )
            cycle_index = len(existing_cycles) + 1
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
                    dict.fromkeys(
                        evidence_id
                        for receipt in receipts
                        for evidence_id in receipt.accepted_evidence_ids
                    )
                ),
                "successor_cycle_index": cycle_index,
            }
            binding = _build_content_bound(
                SuccessorParentBinding,
                "successor-parent-binding",
                binding_payload,
            )
            prefix = f"successor-planning/cycle-{cycle_index}"
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
            repositories = KnowledgeRepositories(self.knowledge_dir)
            pack = DomainPackLoader().load(parent.plan.domain)
            repositories.load_expert_cases(pack.expert_cases)
            repositories.load_workflow_patterns(pack.workflow_patterns)
            repositories.load_capabilities(pack.capabilities)
            planning_input = PlanningContextResolver().resolve(
                context, packet, repositories, self.composite_evidence
            )
            for name, artifact_type, identifier, value in (
                ("context.yaml", "scientific_context_packet", context.context_id, context),
                ("evidence-packet.yaml", "scientific_evidence_packet", packet.packet_id, packet),
                ("planning-input.yaml", "scientific_planning_input", planning_input.planning_input_id, planning_input),
            ):
                self.runs.write_immutable_artifact(
                    parent_run_id, f"{prefix}/{name}", artifact_type, identifier, value
                )

            if update.recommended_status != SuccessorPlanningStatus.FOLLOW_UP_PLAN_REQUIRED:
                cycle_payload = {
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
                cycle = _build_content_bound(
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

            planner = planning_provider or MockPlanningProvider()
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
            if selected_candidate_id is None:
                if len(compilation.candidates) != 1:
                    status = SuccessorPlanningStatus.HUMAN_SCIENTIFIC_DECISION_REQUIRED
                    cycle_payload = {
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
                    cycle = _build_content_bound(
                        SuccessorPlanningCycle, "successor-cycle", cycle_payload
                    )
                    self.runs.write_immutable_artifact(
                        parent_run_id, f"{prefix}/cycle.yaml", "successor_planning_cycle", cycle.cycle_id, cycle
                    )
                    return cycle
                selected_candidate_id = compilation.candidates[0].plan_id
            candidate_ids = tuple(item.plan_id for item in compilation.candidates)
            if selected_candidate_id not in candidate_ids:
                raise ResultFeedbackError(
                    "UNKNOWN_SUCCESSOR_CANDIDATE",
                    "selected successor candidate is not part of this compilation",
                )
            selected_index = candidate_ids.index(selected_candidate_id)
            selected = compilation.candidates[selected_index]
            validation = validation_records[selected_index]
            compilation_receipt = compilation.compilation_receipts[selected_index]
            review_input = ApprovalContextResolver().resolve(
                context,
                packet,
                planning_input,
                selected,
                validation,
                repositories,
                self.composite_evidence,
            )
            approver = approval_provider or MockApprovalProvider()
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
            status = (
                SuccessorPlanningStatus.APPROVED
                if passed
                else SuccessorPlanningStatus.REJECTED
            )
            cycle_payload = {
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
                "candidate_plan_ids": candidate_ids,
                "candidate_plan_hashes": tuple(content_hash(item) for item in compilation.candidates),
                "selected_plan_id": selected.plan_id,
                "selected_plan_hash": content_hash(selected),
                "approval_review_id": approval.review.review_id,
                "approval_review_hash": approval.review.content_hash,
                "approval_verdict_id": approval.verdict.verdict_id,
                "approval_verdict_hash": content_hash(approval.verdict),
                "approval_receipt_id": approval.receipt.receipt_id,
                "approval_receipt_hash": approval.receipt.content_hash,
                "gate_id": gate.gate_id,
                "gate_hash": content_hash(gate),
                "status": status,
            }
            cycle = _build_content_bound(
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
    cycles: list[SuccessorPlanningCycle] = []
    for path in sorted(run_root.glob("successor-planning/cycle-*/cycle.yaml")):
        cycles.append(
            repository.load_artifact(
                parent_run_id,
                path.relative_to(run_root).as_posix(),
                SuccessorPlanningCycle,
            )
        )
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
            "latest_status": cycles[-1].status.value if cycles else None,
            "cycle": len(cycles),
            "parent_run": parent_run_id,
            "parent_plan": (
                submissions[-1].parent_plan_id if submissions else None
            ),
            "triggering_result_ids": (
                list(
                    repository.load_artifact(
                        parent_run_id,
                        f"successor-planning/cycle-{len(cycles)}/parent-binding.yaml",
                        SuccessorParentBinding,
                    ).result_submission_ids
                )
                if cycles
                else []
            ),
        },
    }
