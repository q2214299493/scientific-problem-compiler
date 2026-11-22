from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...domains import DomainPackLoader
from ...interpretation import ScientificEvidencePacketBuilder
from ...interpretation.validators import validate_evidence_packet_integrity
from ...models import (
    ApprovalDecision,
    ApprovalReviewInput,
    ApprovalReviewRecord,
    ApprovalVerdict,
    DirectionTriageRecord,
    EvidenceGap,
    EvidenceGapResolutionKind,
    EvidenceResolutionSet,
    EvidenceResolutionStatus,
    GateVerdict,
    IndependentApprovalReceipt,
    PlanCompilationReceipt,
    PlanningEvidenceCyclePolicy,
    PlanningEvidenceCycleRecord,
    PlanningEvidenceRequestAttempt,
    PlanningEvidenceRequestSet,
    PlanningProposalSet,
    PlanningStrategy,
    PlanValidationRecord,
    ProjectTrustPolicy,
    ResearchDirectionSet,
    RevisionApprovalReviewInput,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificProblemRun,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
    ScientificRunBlockingItem,
    parse_approval_review_input,
)
from ...planning import (
    PlanningContextResolver,
    PlanningEvidenceError,
    augment_scientific_context,
    build_evidence_resolution_set,
    build_planning_evidence_request_set,
    resolve_planning_evidence_requests,
    retrieval_resolvable_gaps,
)
from ...repositories import (
    CompositeEvidenceStore,
    KnowledgeRepositories,
)
from ...serialization import content_hash, load_data
from ...validators import (
    validate_approved_plan_authority,
)
from ..providers import bind_provider, interpretation_provider
from ..repository import ScientificProblemRunRepository
from ..state import snapshot_hash, update_run


class EvidenceResolutionBlocked(ValueError):
    def __init__(
        self,
        category: str,
        message: str,
        *,
        curation_items: tuple[ScientificRunBlockingItem, ...] = (),
    ) -> None:
        self.category = category
        self.curation_items = curation_items
        super().__init__(message)


def _make_evidence_cycle_policy(
    run_id: str,
    max_cycles: int,
    source_acquisition_allowed: bool,
) -> PlanningEvidenceCyclePolicy:
    identity = {
        "run_id": run_id,
        "max_cycles": max_cycles,
        "source_acquisition_allowed": source_acquisition_allowed,
    }
    policy_id = f"planning-evidence-policy-{content_hash(identity)[:24]}"
    return PlanningEvidenceCyclePolicy(
        policy_id=policy_id,
        **identity,
        content_hash=content_hash({"policy_id": policy_id, **identity}),
    )


def _make_evidence_cycle_record(
    payload: dict[str, Any],
) -> PlanningEvidenceCycleRecord:
    payload = PlanningEvidenceCycleRecord.model_construct(
        cycle_id="pending",
        **payload,
        content_hash="0" * 64,
    ).model_dump(mode="json", exclude={"cycle_id", "content_hash"})
    cycle_id = f"planning-evidence-cycle-{content_hash(payload)[:24]}"
    return PlanningEvidenceCycleRecord(
        cycle_id=cycle_id,
        **payload,
        content_hash=content_hash({"cycle_id": cycle_id, **payload}),
    )


def _make_evidence_request_attempt(
    payload: dict[str, Any],
) -> PlanningEvidenceRequestAttempt:
    payload = PlanningEvidenceRequestAttempt.model_construct(
        attempt_id="pending",
        **payload,
        content_hash="0" * 64,
    ).model_dump(mode="json", exclude={"attempt_id", "content_hash"})
    attempt_id = f"planning-evidence-attempt-{content_hash(payload)[:24]}"
    return PlanningEvidenceRequestAttempt(
        attempt_id=attempt_id,
        **payload,
        content_hash=content_hash({"attempt_id": attempt_id, **payload}),
    )


def _carry_unresolved_evidence_gaps(
    packet: ScientificEvidencePacket,
    parent_input: ScientificPlanningInput,
    request_set: PlanningEvidenceRequestSet,
    resolutions: EvidenceResolutionSet,
) -> ScientificEvidencePacket:
    requests = {item.request_id: item for item in request_set.requests}
    parent_gaps = {item.gap_id: item for item in parent_input.evidence_gaps}
    carried: list[EvidenceGap] = []
    for resolution in resolutions.records:
        if resolution.status == EvidenceResolutionStatus.RESOLVED_TRUSTED:
            continue
        request = requests[resolution.request_id]
        if request.gap_id is not None and request.gap_id in parent_gaps:
            carried.append(parent_gaps[request.gap_id])
            continue
        carried.append(
            EvidenceGap(
                gap_id=("planning-evidence-gap-" + request.request_id.removeprefix("planning-evidence-request-")),
                scientific_question=request.scientific_question,
                missing_evidence="; ".join(
                    resolution.unsatisfied_conditions
                    or request.comparison_conditions
                    or (request.triggering_question or request.why_needed,)
                ),
                why_it_matters=request.why_needed,
                blocking=request.blocking,
                evidence_refs=resolution.evidence_refs,
            )
        )
    if not carried:
        return packet
    gaps = {item.gap_id: item for item in (*packet.evidence_gaps, *carried)}
    identity = packet.model_dump(mode="python", exclude={"packet_id", "content_hash"})
    identity["evidence_gaps"] = tuple(gaps[key] for key in sorted(gaps))
    identity["provenance_manifest"] = {
        **dict(packet.provenance_manifest),
        "evidence_resolution_set_id": resolutions.resolution_set_id,
        "evidence_resolution_set_hash": resolutions.content_hash,
    }
    packet_id = f"evidence-packet-{content_hash(identity)[:24]}"
    payload = {"packet_id": packet_id, **identity}
    return ScientificEvidencePacket(**payload, content_hash=content_hash(payload))


@dataclass
class EvidenceResolutionStage:
    """Coordinate bounded evidence requests and persist their replayable records."""

    runs: ScientificProblemRunRepository
    domain_loader: DomainPackLoader

    @staticmethod
    def _evidence_policy_path() -> str:
        return "planning-evidence/policy.yaml"

    def _load_evidence_policy(self, run_id: str) -> PlanningEvidenceCyclePolicy | None:
        try:
            return self.runs.load_artifact(
                run_id,
                self._evidence_policy_path(),
                PlanningEvidenceCyclePolicy,
            )
        except FileNotFoundError:
            return None

    def ensure_policy(
        self,
        run_id: str,
        max_cycles: int,
        source_acquisition_allowed: bool,
    ) -> PlanningEvidenceCyclePolicy | None:
        persisted = self._load_evidence_policy(run_id)
        if persisted is not None:
            if max_cycles not in {0, persisted.max_cycles}:
                raise ValueError("persisted evidence-resolution budget cannot be changed during resume")
            if max_cycles != 0 and source_acquisition_allowed != persisted.source_acquisition_allowed:
                raise ValueError("persisted evidence acquisition policy cannot be changed during resume")
            return persisted
        if max_cycles == 0:
            return None
        policy = _make_evidence_cycle_policy(run_id, max_cycles, source_acquisition_allowed)
        self.runs.write_immutable_artifact(
            run_id,
            self._evidence_policy_path(),
            "planning_evidence_cycle_policy",
            policy.policy_id,
            policy,
        )
        return policy

    def reuse_context(
        self,
        run_id: str,
        base_context: ScientificContextPacket,
    ) -> ScientificContextPacket:
        cycles = self.runs.load_evidence_cycles(run_id)
        if not cycles:
            return base_context
        latest = cycles[-1]
        if latest.child_context_id is None:
            return base_context
        if (
            latest.knowledge_snapshot_id != base_context.knowledge_snapshot.snapshot_id
            or latest.knowledge_snapshot_hash != snapshot_hash(base_context.knowledge_snapshot)
        ):
            return base_context
        child = self.runs.load_artifact(
            run_id,
            f"{self.runs.evidence_cycle_prefix(latest.cycle_index)}/context.yaml",
            ScientificContextPacket,
        )
        if child.context_id != latest.child_context_id or child.content_hash != latest.child_context_hash:
            raise ValueError("stored planning evidence child context binding is invalid")
        return child

    def _record_evidence_cycle(
        self,
        run_id: str,
        payload: dict[str, Any],
    ) -> PlanningEvidenceCycleRecord:
        record = _make_evidence_cycle_record(payload)
        self.runs.write_immutable_artifact(
            run_id,
            f"{self.runs.evidence_cycle_prefix(record.cycle_index)}/cycle.yaml",
            "planning_evidence_cycle_record",
            record.cycle_id,
            record,
        )
        return record

    def _archive_evidence_trigger_artifacts(
        self,
        run: ScientificProblemRun,
        cycle_index: int,
        approval_prefix: str,
    ) -> tuple[
        ApprovalReviewInput | RevisionApprovalReviewInput,
        ScientificQuestionPlan,
    ]:
        prefix = f"{self.runs.evidence_cycle_prefix(cycle_index)}/trigger"
        planning_input = self.runs.load_artifact(run.run_id, "planning-input.yaml", ScientificPlanningInput)
        self.runs.write_immutable_artifact(
            run.run_id,
            f"{prefix}/planning-input.yaml",
            "scientific_planning_input",
            planning_input.planning_input_id,
            planning_input,
        )
        chain = self.runs.load_revision_chain(run.run_id)
        proposal_path = (
            chain.rounds[-1].planning_proposal.relative_path if chain is not None else "planning-proposal.yaml"
        )
        proposal = self.runs.load_artifact(run.run_id, proposal_path, PlanningProposalSet)
        self.runs.write_immutable_artifact(
            run.run_id,
            f"{prefix}/planning-proposal.yaml",
            "planning_proposal_set",
            proposal.proposal_id,
            proposal,
        )
        review_input = parse_approval_review_input(
            load_data(self.runs.resolve_artifact_path(run.run_id, f"{approval_prefix}/approval-review-input.yaml"))
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
        gate = self.runs.load_artifact(run.run_id, f"{approval_prefix}/plan-gate.yaml", GateVerdict)
        policy = self.runs.load_artifact(
            run.run_id,
            f"{approval_prefix}/project-trust-policy.yaml",
            ProjectTrustPolicy,
        )
        candidate_ids = tuple(item.artifact_id for item in run.candidate_plans)
        if review_input.candidate_plan.plan_id not in candidate_ids:
            raise ValueError("evidence trigger review candidate is not part of the run")
        candidate_index = candidate_ids.index(review_input.candidate_plan.plan_id)
        candidate = self.runs.load_bound_artifact(
            run.run_id,
            run.candidate_plans[candidate_index],
            ScientificQuestionPlan,
        )
        validation = self.runs.load_bound_artifact(
            run.run_id,
            run.plan_validation_records[candidate_index],
            PlanValidationRecord,
        )
        compilation_receipt = self.runs.load_bound_artifact(
            run.run_id,
            run.candidate_compilation_receipts[candidate_index],
            PlanCompilationReceipt,
        )
        authority = validate_approved_plan_authority(
            candidate,
            validation,
            review_input,
            review,
            verdict,
            approval_receipt,
            gate,
            policy,
            compilation_receipt,
            require_passed=False,
        )
        if not authority.valid:
            raise ValueError(
                "evidence trigger approval authority is invalid: " + ", ".join(item.code for item in authority.issues)
            )
        if (
            gate.candidate_id != candidate.plan_id
            or gate.candidate_content_hash != content_hash(candidate)
            or gate.plan_validation_id != validation.validation_id
            or gate.plan_validation_hash != content_hash(validation)
            or gate.approval_verdict_id != verdict.verdict_id
            or gate.approval_verdict_hash != content_hash(verdict)
            or gate.independent_approval_receipt_id != approval_receipt.receipt_id
            or gate.independent_approval_receipt_hash != approval_receipt.content_hash
            or gate.plan_compilation_receipt_id != compilation_receipt.receipt_id
            or gate.plan_compilation_receipt_hash != compilation_receipt.content_hash
            or gate.trust_policy_version != policy.policy_version
            or gate.trust_policy_hash != content_hash(policy)
        ):
            raise ValueError("evidence trigger gate does not bind its approval chain")
        artifacts = (
            (
                "approval-review-input.yaml",
                "approval_review_input",
                review_input.review_input_id,
                review_input,
            ),
            (
                "approval-review.yaml",
                "approval_review_record",
                review.review_id,
                review,
            ),
            (
                "approval-verdict.yaml",
                "approval_verdict",
                verdict.verdict_id,
                verdict,
            ),
            (
                "independent-approval-receipt.yaml",
                "independent_approval_receipt",
                approval_receipt.receipt_id,
                approval_receipt,
            ),
            ("plan-gate.yaml", "gate_verdict", gate.gate_id, gate),
            (
                "project-trust-policy.yaml",
                "project_trust_policy",
                "project-trust-policy",
                policy,
            ),
            (
                "candidate-plan.yaml",
                "scientific_question_plan",
                candidate.plan_id,
                candidate,
            ),
            (
                "plan-validation-record.yaml",
                "plan_validation_record",
                validation.validation_id,
                validation,
            ),
            (
                "plan-compilation-receipt.yaml",
                "plan_compilation_receipt",
                compilation_receipt.receipt_id,
                compilation_receipt,
            ),
        )
        for filename, artifact_type, artifact_id, value in artifacts:
            self.runs.write_immutable_artifact(
                run.run_id,
                f"{prefix}/{filename}",
                artifact_type,
                artifact_id,
                value,
            )
        return review_input, candidate

    def validate_trigger_archive(
        self,
        run_id: str,
        cycle_index: int,
        request_set: PlanningEvidenceRequestSet,
    ) -> tuple[
        ApprovalReviewInput | RevisionApprovalReviewInput,
        ApprovalReviewRecord,
        ScientificQuestionPlan,
    ]:
        prefix = f"{self.runs.evidence_cycle_prefix(cycle_index)}/trigger"
        review_input = parse_approval_review_input(
            load_data(self.runs.resolve_artifact_path(run_id, f"{prefix}/approval-review-input.yaml"))
        )
        review = self.runs.load_artifact(run_id, f"{prefix}/approval-review.yaml", ApprovalReviewRecord)
        verdict = self.runs.load_artifact(run_id, f"{prefix}/approval-verdict.yaml", ApprovalVerdict)
        receipt = self.runs.load_artifact(
            run_id,
            f"{prefix}/independent-approval-receipt.yaml",
            IndependentApprovalReceipt,
        )
        gate = self.runs.load_artifact(run_id, f"{prefix}/plan-gate.yaml", GateVerdict)
        policy = self.runs.load_artifact(run_id, f"{prefix}/project-trust-policy.yaml", ProjectTrustPolicy)
        candidate = self.runs.load_artifact(run_id, f"{prefix}/candidate-plan.yaml", ScientificQuestionPlan)
        validation = self.runs.load_artifact(
            run_id,
            f"{prefix}/plan-validation-record.yaml",
            PlanValidationRecord,
        )
        compilation_receipt = self.runs.load_artifact(
            run_id,
            f"{prefix}/plan-compilation-receipt.yaml",
            PlanCompilationReceipt,
        )
        expected_bindings = (
            request_set.triggering_review_input_id,
            request_set.triggering_review_input_hash,
            request_set.triggering_review_id,
            request_set.triggering_review_hash,
            request_set.triggering_candidate_id,
            request_set.triggering_candidate_hash,
        )
        actual_bindings = (
            review_input.review_input_id,
            review_input.content_hash,
            review.review_id,
            review.content_hash,
            candidate.plan_id,
            content_hash(candidate),
        )
        if expected_bindings != actual_bindings:
            raise ValueError("evidence request does not bind its archived approval trigger")
        authority = validate_approved_plan_authority(
            candidate,
            validation,
            review_input,
            review,
            verdict,
            receipt,
            gate,
            policy,
            compilation_receipt,
            require_passed=False,
        )
        if not authority.valid:
            raise ValueError(
                "archived evidence trigger approval authority is invalid: "
                + ", ".join(item.code for item in authority.issues)
            )
        if (
            review_input.candidate_plan != candidate
            or validation.plan_id != candidate.plan_id
            or validation.plan_content_hash != content_hash(candidate)
            or gate.candidate_id != candidate.plan_id
            or gate.candidate_content_hash != content_hash(candidate)
            or gate.plan_validation_id != validation.validation_id
            or gate.plan_validation_hash != content_hash(validation)
            or gate.approval_verdict_id != verdict.verdict_id
            or gate.approval_verdict_hash != content_hash(verdict)
            or gate.independent_approval_receipt_id != receipt.receipt_id
            or gate.independent_approval_receipt_hash != receipt.content_hash
            or gate.plan_compilation_receipt_id != compilation_receipt.receipt_id
            or gate.plan_compilation_receipt_hash != compilation_receipt.content_hash
            or gate.trust_policy_version != policy.policy_version
            or gate.trust_policy_hash != content_hash(policy)
        ):
            raise ValueError("archived evidence trigger gate binding is invalid")
        return review_input, review, candidate

    def resolve(
        self,
        run: ScientificProblemRun,
        context: ScientificContextPacket,
        packet: ScientificEvidencePacket,
        planning_input: ScientificPlanningInput,
        repositories: KnowledgeRepositories,
        evidence: CompositeEvidenceStore,
        planner: Any,
        strategy: PlanningStrategy,
        policy: PlanningEvidenceCyclePolicy | None,
        triggering_review: ApprovalReviewRecord | None = None,
        triggering_review_input: ApprovalReviewInput | RevisionApprovalReviewInput | None = None,
        triggering_candidate: ScientificQuestionPlan | None = None,
        direction_set: ResearchDirectionSet | None = None,
        triage: DirectionTriageRecord | None = None,
    ) -> tuple[
        ScientificProblemRun,
        ScientificContextPacket,
        ScientificEvidencePacket,
        ScientificPlanningInput,
    ]:
        if policy is None:
            return run, context, packet, planning_input
        completed = self.runs.load_evidence_cycles(run.run_id)
        if completed:
            latest = completed[-1]
            same_snapshot = (
                latest.knowledge_snapshot_id == context.knowledge_snapshot.snapshot_id
                and latest.knowledge_snapshot_hash == snapshot_hash(context.knowledge_snapshot)
            )
            if same_snapshot and latest.terminal_reason is not None:
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_RESOLUTION_TERMINATED",
                    latest.terminal_reason,
                )
        blocking_gaps = tuple(gap for gap in planning_input.evidence_gaps if gap.blocking)
        direction_needs = tuple(
            need
            for disposition in (triage.dispositions if triage is not None else ())
            for need in disposition.evidence_needs
            if need.resolution_kind == EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE
        )
        if not blocking_gaps and triggering_review is None and not direction_needs:
            return run, context, packet, planning_input
        attempt_count = self.runs.evidence_attempt_count(run.run_id)
        if attempt_count:
            previous_prefix = self.runs.evidence_cycle_prefix(attempt_count)
            previous_attempt = self.runs.resolve_artifact_path(run.run_id, f"{previous_prefix}/request-attempt.yaml")
            previous_request = self.runs.resolve_artifact_path(run.run_id, f"{previous_prefix}/request-set.yaml")
            previous_resolution = self.runs.resolve_artifact_path(run.run_id, f"{previous_prefix}/resolutions.yaml")
            if previous_attempt.exists() and not previous_request.exists():
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_REQUEST_OUTCOME_UNCERTAIN",
                    "an evidence-request provider call was claimed without a complete request set",
                )
            if previous_request.exists() and not previous_resolution.exists():
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_RETRIEVAL_OUTCOME_UNCERTAIN",
                    "evidence retrieval was started but no complete resolution was saved",
                )
        # A saved request and resolution can outlive an interrupted cycle commit.
        # Only a committed cycle advances the next index; the claim above still counts against the budget.
        cycle_index = completed[-1].cycle_index + 1 if completed else 1
        if cycle_index > policy.max_cycles:
            raise EvidenceResolutionBlocked(
                "EVIDENCE_RESOLUTION_BUDGET_EXHAUSTED",
                "planning evidence-resolution cycle budget is exhausted",
            )
        prefix = self.runs.evidence_cycle_prefix(cycle_index)
        request_path = self.runs.resolve_artifact_path(run.run_id, f"{prefix}/request-set.yaml")
        attempt_path = self.runs.resolve_artifact_path(run.run_id, f"{prefix}/request-attempt.yaml")
        resolution_path = self.runs.resolve_artifact_path(run.run_id, f"{prefix}/resolutions.yaml")
        provider_block_reason: str | None = None
        if request_path.exists():
            request_set = self.runs.load_artifact(
                run.run_id,
                f"{prefix}/request-set.yaml",
                PlanningEvidenceRequestSet,
            )
            if (
                request_set.planning_input_hash != planning_input.content_hash
                or request_set.context_hash != context.content_hash
                or request_set.knowledge_snapshot_hash != snapshot_hash(context.knowledge_snapshot)
                or request_set.source_acquisition_allowed != policy.source_acquisition_allowed
            ):
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_REQUEST_MISMATCH", "saved request has different inputs or policy"
                )
            if request_set.triggering_review_id is not None:
                (
                    archived_review_input,
                    archived_review,
                    archived_candidate,
                ) = self.validate_trigger_archive(run.run_id, cycle_index, request_set)
                if triggering_review is not None and (
                    triggering_review.review_id != archived_review.review_id
                    or triggering_review.content_hash != archived_review.content_hash
                ):
                    raise EvidenceResolutionBlocked(
                        "EVIDENCE_TRIGGER_REVIEW_MISMATCH",
                        "current review does not match the archived evidence trigger",
                    )
                triggering_review = archived_review
                triggering_review_input = archived_review_input
                triggering_candidate = archived_candidate
            if request_set.direction_set_id is not None:
                if (
                    direction_set is None
                    or triage is None
                    or (
                        request_set.direction_set_id != direction_set.direction_set_id
                        or request_set.direction_set_hash != direction_set.content_hash
                        or request_set.triage_id != triage.triage_id
                        or request_set.triage_hash != triage.content_hash
                    )
                ):
                    raise EvidenceResolutionBlocked(
                        "EVIDENCE_DIRECTION_CONTEXT_MISMATCH",
                        "saved evidence request does not bind the active direction and triage",
                    )
            if not resolution_path.exists():
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_RETRIEVAL_OUTCOME_UNCERTAIN",
                    "evidence retrieval was started but no complete resolution was saved",
                )
            resolutions = self.runs.load_artifact(
                run.run_id,
                f"{prefix}/resolutions.yaml",
                EvidenceResolutionSet,
            )
            if (resolutions.request_set_id, resolutions.request_set_hash) != (
                request_set.request_set_id,
                request_set.content_hash,
            ):
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_RESOLUTION_MISMATCH", "saved resolution does not bind the request"
                )
        else:
            if attempt_path.exists():
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_REQUEST_OUTCOME_UNCERTAIN",
                    "an evidence-request provider call was claimed without a complete request set",
                )
            provider_binding = bind_provider("planning_evidence_request", planner)
            attempt = _make_evidence_request_attempt(
                {
                    "run_id": run.run_id,
                    "cycle_index": cycle_index,
                    "planning_input_id": planning_input.planning_input_id,
                    "planning_input_hash": planning_input.content_hash,
                    "context_id": context.context_id,
                    "context_hash": context.content_hash,
                    "provider_id": planner.provider_id,
                    "provider_version": planner.provider_version,
                    "provider_config_hash": provider_binding.configuration_hash,
                    "triggering_review_id": (triggering_review.review_id if triggering_review else None),
                    "triggering_review_hash": (triggering_review.content_hash if triggering_review else None),
                    "triggering_review_input_id": (
                        triggering_review_input.review_input_id if triggering_review_input is not None else None
                    ),
                    "triggering_review_input_hash": (
                        triggering_review_input.content_hash if triggering_review_input is not None else None
                    ),
                    "triggering_candidate_id": (
                        triggering_candidate.plan_id if triggering_candidate is not None else None
                    ),
                    "triggering_candidate_hash": (
                        content_hash(triggering_candidate) if triggering_candidate is not None else None
                    ),
                    "direction_set_id": (direction_set.direction_set_id if direction_set is not None else None),
                    "direction_set_hash": (direction_set.content_hash if direction_set is not None else None),
                    "triage_id": triage.triage_id if triage is not None else None,
                    "triage_hash": triage.content_hash if triage is not None else None,
                }
            )
            try:
                self.runs.write_exclusive_artifact(
                    run.run_id,
                    f"{prefix}/request-attempt.yaml",
                    "planning_evidence_request_attempt",
                    attempt.attempt_id,
                    attempt,
                )
            except FileExistsError as error:
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_REQUEST_OUTCOME_UNCERTAIN",
                    "another process already claimed this evidence-request cycle",
                ) from error
            retrievable = retrieval_resolvable_gaps(planning_input)
            response = None
            if direction_set is not None and triage is not None:
                self.runs.write_immutable_artifact(
                    run.run_id,
                    f"{prefix}/trigger-directions.yaml",
                    "research_direction_set",
                    direction_set.direction_set_id,
                    direction_set,
                )
                self.runs.write_immutable_artifact(
                    run.run_id,
                    f"{prefix}/trigger-triage.yaml",
                    "direction_triage_record",
                    triage.triage_id,
                    triage,
                )
            if (
                triggering_review is not None
                or direction_needs
                or (retrievable and len(retrievable) == len(blocking_gaps))
            ):
                method = getattr(planner, "propose_evidence_requests", None)
                if not callable(method):
                    raise EvidenceResolutionBlocked(
                        "EVIDENCE_REQUEST_PROVIDER_UNAVAILABLE",
                        "planning provider cannot produce bounded evidence requests",
                    )
                try:
                    if triggering_review is not None:
                        response = method(
                            planning_input,
                            triggering_review,
                            triggering_review_input=triggering_review_input,
                        )
                    elif direction_needs:
                        response = method(
                            planning_input,
                            directions=direction_set,
                            triage=triage,
                        )
                    else:
                        response = method(planning_input)
                except (PlanningEvidenceError, ValueError) as error:
                    provider_block_reason = str(error)
            request_set = build_planning_evidence_request_set(
                planning_input,
                response,
                planner,
                planning_strategy=strategy,
                knowledge_snapshot_hash=snapshot_hash(context.knowledge_snapshot),
                source_acquisition_allowed=policy.source_acquisition_allowed,
                direction_set=direction_set,
                triage=triage,
                triggering_review=triggering_review,
                triggering_review_id=(triggering_review.review_id if triggering_review else None),
                triggering_review_hash=(triggering_review.content_hash if triggering_review else None),
                triggering_review_input=triggering_review_input,
                triggering_candidate=triggering_candidate,
            )
            for prior in completed:
                if prior.knowledge_snapshot_id != request_set.knowledge_snapshot_id:
                    continue
                prior_requests = self.runs.load_artifact(
                    run.run_id,
                    f"{self.runs.evidence_cycle_prefix(prior.cycle_index)}/request-set.yaml",
                    PlanningEvidenceRequestSet,
                )
                if tuple(item.request_id for item in prior_requests.requests) == tuple(
                    item.request_id for item in request_set.requests
                ):
                    raise EvidenceResolutionBlocked(
                        "EQUIVALENT_EVIDENCE_REQUEST_REPEATED",
                        "an equivalent request already ran against this knowledge snapshot",
                    )
            self.runs.write_exclusive_artifact(
                run.run_id,
                f"{prefix}/request-set.yaml",
                "planning_evidence_request_set",
                request_set.request_set_id,
                request_set,
            )
            nonretrieval = tuple(
                item
                for item in request_set.gap_classifications
                if item.resolution_kind != EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE
            )
            if nonretrieval or provider_block_reason is not None:
                resolutions = build_evidence_resolution_set(request_set)
            else:
                pack = self.domain_loader.load(run.domain)
                resolutions = resolve_planning_evidence_requests(
                    request_set,
                    repositories,
                    evidence,
                    pack.profile,
                    context.knowledge_snapshot,
                )
            self.runs.write_immutable_artifact(
                run.run_id,
                f"{prefix}/resolutions.yaml",
                "evidence_resolution_set",
                resolutions.resolution_set_id,
                resolutions,
            )
            for acquisition in resolutions.acquisition_proposals:
                self.runs.write_immutable_artifact(
                    run.run_id,
                    f"{prefix}/acquisition/{acquisition.proposal_id}.yaml",
                    "evidence_acquisition_proposal",
                    acquisition.proposal_id,
                    acquisition,
                )
        nonretrieval = tuple(
            item
            for item in request_set.gap_classifications
            if item.resolution_kind != EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE
        )
        terminal_reason = None
        if provider_block_reason is None and triggering_review is not None and not request_set.requests:
            provider_block_reason = "the triggering review does not identify a retrieval-resolvable source gap"
        if provider_block_reason is not None:
            terminal_reason = provider_block_reason
            category = "NOT_RETRIEVAL_RESOLVABLE"
            curation_items = ()
        elif nonretrieval:
            human = tuple(
                item
                for item in nonretrieval
                if item.resolution_kind == EvidenceGapResolutionKind.HUMAN_DECISION_REQUIRED
            )
            terminal_reason = "; ".join(
                f"{item.gap_id}:{item.resolution_kind.value}:{item.rationale}" for item in nonretrieval
            )
            category = "HUMAN_DECISION_REQUIRED" if human else "NOT_RETRIEVAL_RESOLVABLE"
        else:
            curation_records = tuple(
                record
                for record in resolutions.records
                if record.status == EvidenceResolutionStatus.REQUIRES_SOURCE_CURATION
            )
            if curation_records:
                curation_items = tuple(
                    ScientificRunBlockingItem(
                        target_type=hit.source_type.value,
                        target_id=hit.record_id,
                        current_status=hit.curation_status,
                        message=(
                            f"{hit.source_type.value} {hit.record_id} requires ACCEPTED curation before planning use"
                        ),
                    )
                    for record in curation_records
                    for hit in record.matched_hits
                )
                terminal_reason = "matching evidence requires source curation"
                category = "REQUIRES_SOURCE_CURATION"
            else:
                curation_items = ()
                no_matches = tuple(
                    record for record in resolutions.records if record.status == EvidenceResolutionStatus.NO_MATCH
                )
                if no_matches:
                    terminal_reason = "; ".join(record.resolution_rationale for record in no_matches)
                    category = (
                        "NO_TRUSTED_MATCH"
                        if all(
                            record.retrieval_status is None or record.retrieval_status.value == "no_trusted_match"
                            for record in no_matches
                        )
                        else "TRUSTED_MATCH_DID_NOT_RESOLVE_GAP"
                    )
        if terminal_reason is not None:
            cycle = self._record_evidence_cycle(
                run.run_id,
                {
                    "cycle_index": cycle_index,
                    "parent_context_id": context.context_id,
                    "parent_context_hash": context.content_hash,
                    "parent_planning_input_id": planning_input.planning_input_id,
                    "parent_planning_input_hash": planning_input.content_hash,
                    "request_set_id": request_set.request_set_id,
                    "request_set_hash": request_set.content_hash,
                    "resolution_set_id": resolutions.resolution_set_id,
                    "resolution_set_hash": resolutions.content_hash,
                    "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
                    "knowledge_snapshot_hash": snapshot_hash(context.knowledge_snapshot),
                    "terminal_reason": terminal_reason,
                },
            )
            del cycle
            raise EvidenceResolutionBlocked(
                category,
                terminal_reason,
                curation_items=(curation_items if category == "REQUIRES_SOURCE_CURATION" else ()),
            )
        existing_hit_ids = {
            hit.hit_id
            for hits in (
                context.literature_knowledge_hits,
                context.expert_opinion_hits,
                context.expert_case_hits,
                context.graph_expanded_hits,
            )
            for hit in hits
        }
        new_hit_ids = {
            hit.hit_id
            for record in resolutions.records
            for hit in record.matched_hits
            if hit.authority_status == "trusted_current"
        } - existing_hit_ids
        if not new_hit_ids:
            terminal_reason = "trusted retrieval returned no new evidence beyond the current planning context"
            self._record_evidence_cycle(
                run.run_id,
                {
                    "cycle_index": cycle_index,
                    "parent_context_id": context.context_id,
                    "parent_context_hash": context.content_hash,
                    "parent_planning_input_id": planning_input.planning_input_id,
                    "parent_planning_input_hash": planning_input.content_hash,
                    "request_set_id": request_set.request_set_id,
                    "request_set_hash": request_set.content_hash,
                    "resolution_set_id": resolutions.resolution_set_id,
                    "resolution_set_hash": resolutions.content_hash,
                    "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
                    "knowledge_snapshot_hash": snapshot_hash(context.knowledge_snapshot),
                    "terminal_reason": terminal_reason,
                },
            )
            raise EvidenceResolutionBlocked("NO_NEW_TRUSTED_EVIDENCE", terminal_reason)
        expected_context = augment_scientific_context(context, resolutions, evidence)
        try:
            child_context = self.runs.load_artifact(run.run_id, f"{prefix}/context.yaml", ScientificContextPacket)
        except FileNotFoundError:
            child_context = expected_context
        if child_context.content_hash != expected_context.content_hash:
            raise EvidenceResolutionBlocked(
                "EVIDENCE_CONTEXT_MISMATCH", "saved child context does not bind the resolution"
            )
        try:
            child_packet = self.runs.load_artifact(
                run.run_id, f"{prefix}/evidence-packet.yaml", ScientificEvidencePacket
            )
        except FileNotFoundError:
            child_packet = ScientificEvidencePacketBuilder(interpretation_provider()).build(child_context, evidence)
            child_packet = _carry_unresolved_evidence_gaps(child_packet, planning_input, request_set, resolutions)
        else:
            if not validate_evidence_packet_integrity(child_packet, child_context, evidence).valid:
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_PACKET_MISMATCH", "saved packet does not bind the child context"
                )
        try:
            child_input = self.runs.load_artifact(run.run_id, f"{prefix}/planning-input.yaml", ScientificPlanningInput)
        except FileNotFoundError:
            child_input = PlanningContextResolver(self.domain_loader).resolve(
                child_context, child_packet, repositories, evidence
            )
        if (
            child_input.context_hash != child_context.content_hash
            or child_input.evidence_packet_hash != child_packet.content_hash
        ):
            raise EvidenceResolutionBlocked(
                "EVIDENCE_INPUT_MISMATCH", "saved planning input does not bind its child artifacts"
            )
        for filename, artifact_type, artifact_id, value in (
            (
                "context.yaml",
                "scientific_context_packet",
                child_context.context_id,
                child_context,
            ),
            (
                "evidence-packet.yaml",
                "scientific_evidence_packet",
                child_packet.packet_id,
                child_packet,
            ),
            (
                "planning-input.yaml",
                "scientific_planning_input",
                child_input.planning_input_id,
                child_input,
            ),
        ):
            self.runs.write_immutable_artifact(
                run.run_id,
                f"{prefix}/{filename}",
                artifact_type,
                artifact_id,
                value,
            )
        self._record_evidence_cycle(
            run.run_id,
            {
                "cycle_index": cycle_index,
                "parent_context_id": context.context_id,
                "parent_context_hash": context.content_hash,
                "parent_planning_input_id": planning_input.planning_input_id,
                "parent_planning_input_hash": planning_input.content_hash,
                "request_set_id": request_set.request_set_id,
                "request_set_hash": request_set.content_hash,
                "resolution_set_id": resolutions.resolution_set_id,
                "resolution_set_hash": resolutions.content_hash,
                "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
                "knowledge_snapshot_hash": snapshot_hash(context.knowledge_snapshot),
                "child_context_id": child_context.context_id,
                "child_context_hash": child_context.content_hash,
                "child_evidence_packet_id": child_packet.packet_id,
                "child_evidence_packet_hash": child_packet.content_hash,
                "child_planning_input_id": child_input.planning_input_id,
                "child_planning_input_hash": child_input.content_hash,
            },
        )
        updated = self._publish_child(run, child_context, child_packet)
        return updated, child_context, child_packet, child_input

    def recover_committed_child(self, run: ScientificProblemRun) -> ScientificProblemRun:
        """Finish a committed cycle whose root projections were interrupted."""
        cycles = self.runs.load_evidence_cycles(run.run_id)
        if not cycles:
            return run
        cycle = cycles[-1]
        if cycle.child_context_id is None or cycle.parent_context_hash != run.context_hash:
            return run
        prefix = self.runs.evidence_cycle_prefix(cycle.cycle_index)
        context = self.runs.load_artifact(run.run_id, f"{prefix}/context.yaml", ScientificContextPacket)
        packet = self.runs.load_artifact(run.run_id, f"{prefix}/evidence-packet.yaml", ScientificEvidencePacket)
        planning_input = self.runs.load_artifact(run.run_id, f"{prefix}/planning-input.yaml", ScientificPlanningInput)
        if (
            (context.context_id, context.content_hash) != (cycle.child_context_id, cycle.child_context_hash)
            or (packet.packet_id, packet.content_hash)
            != (cycle.child_evidence_packet_id, cycle.child_evidence_packet_hash)
            or (planning_input.planning_input_id, planning_input.content_hash)
            != (cycle.child_planning_input_id, cycle.child_planning_input_hash)
            or packet.context_hash != context.content_hash
            or planning_input.evidence_packet_hash != packet.content_hash
        ):
            raise ValueError("stored evidence cycle does not bind its completed child artifacts")
        return self._publish_child(run, context, packet)

    def _publish_child(self, run, child_context, child_packet):
        self.runs.write_artifact(
            run.run_id,
            "context.yaml",
            "scientific_context_packet",
            child_context.context_id,
            child_context,
        )
        self.runs.write_artifact(
            run.run_id,
            "evidence-packet.yaml",
            "scientific_evidence_packet",
            child_packet.packet_id,
            child_packet,
        )
        updated = update_run(
            run,
            status=ScientificProblemRunStatus.INTERPRETED,
            knowledge_snapshot_id=child_context.knowledge_snapshot.snapshot_id,
            knowledge_snapshot_hash=snapshot_hash(child_context.knowledge_snapshot),
            retrieval_context_id=child_context.retrieval_manifest.retrieval_id,
            retrieval_context_hash=content_hash(child_context.retrieval_manifest),
            context_id=child_context.context_id,
            context_hash=child_context.content_hash,
            evidence_packet_id=child_packet.packet_id,
            evidence_packet_hash=child_packet.content_hash,
            planning_input_id=None,
            planning_input_hash=None,
            planning_proposal_id=None,
            planning_proposal_hash=None,
            candidate_plans=(),
            candidate_compilation_receipts=(),
            plan_validation_records=(),
            selected_candidate_id=None,
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
        self.runs.save(updated)
        return updated

    def resolve_after_review(
        self, approved, context, packet, planning_input, repositories, evidence, planner, strategy, policy
    ) -> ScientificProblemRun | None:
        if approved.approval_verdict_id is None or policy is None:
            return None
        approval_prefix = self.runs.approval_prefix(approved.run_id)
        verdict = self.runs.load_artifact(
            approved.run_id,
            f"{approval_prefix}/approval-verdict.yaml",
            ApprovalVerdict,
        )
        if verdict.decision != ApprovalDecision.INSUFFICIENT_EVIDENCE:
            return None
        review = self.runs.load_artifact(
            approved.run_id,
            f"{approval_prefix}/approval-review.yaml",
            ApprovalReviewRecord,
        )
        completed = self.runs.load_evidence_cycles(approved.run_id)
        cycle_index = completed[-1].cycle_index + 1 if completed else 1
        if cycle_index > policy.max_cycles:
            raise EvidenceResolutionBlocked(
                "EVIDENCE_RESOLUTION_BUDGET_EXHAUSTED", "planning evidence-resolution cycle budget is exhausted"
            )
        trigger_input, candidate = self._archive_evidence_trigger_artifacts(approved, cycle_index, approval_prefix)
        updated, child, _, _ = self.resolve(
            approved,
            context,
            packet,
            planning_input,
            repositories,
            evidence,
            planner,
            strategy,
            policy,
            triggering_review=review,
            triggering_review_input=trigger_input,
            triggering_candidate=candidate,
        )
        return updated if child.context_id != context.context_id else None
