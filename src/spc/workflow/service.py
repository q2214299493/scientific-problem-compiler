from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

from ..adapters.ft_agent import FTAgentAdapter
from ..approval import (
    ApprovalContextResolver,
    IndependentApprovalService,
    MockApprovalProvider,
    StructuredLLMApprovalProvider,
    bind_gate_verdict,
)
from ..compiler import CompilationResult, ScientificProblemCompiler
from ..domains import DomainPackLoader
from ..export import GenericExportService
from ..interpretation import MockInterpretationProvider, ScientificEvidencePacketBuilder
from ..interpretation.validators import validate_evidence_packet_integrity
from ..knowledge.trust import TrustedKnowledgeValidator
from ..models import (
    ApprovalDecision,
    ApprovalMode,
    ApprovalReviewInput,
    ApprovalReviewRecord,
    ApprovalVerdict,
    CurationStatus,
    EvidenceGap,
    EvidenceGapResolutionKind,
    EvidenceResolutionSet,
    EvidenceResolutionStatus,
    GateVerdict,
    IndependentApprovalReceipt,
    KnowledgeSnapshot,
    PlanCompilationReceipt,
    PlanRevisionChain,
    PlanRevisionAttemptOutcome,
    PlanRevisionAttemptStart,
    PlanRevisionAttemptStatus,
    PlanRevisionInput,
    PlanRevisionRecord,
    PlanRevisionRound,
    PlanValidationRecord,
    PlanningStrategy,
    PlanningProposalSet,
    PlanningEvidenceCyclePolicy,
    PlanningEvidenceCycleRecord,
    PlanningEvidenceRequestAttempt,
    PlanningEvidenceRequestSet,
    ProjectTrustPolicy,
    RevisionApprovalReviewInput,
    RevisionApprovalLLMResponse,
    RevisionIssueStatus,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificProblemRun,
    ScientificProblemRunStatus,
    ScientificQuestionPlan,
    ScientificRunArtifactBinding,
    ScientificRunBlockingItem,
    ScientificRunFailure,
    ScientificRunProviderBinding,
    DirectionDisposition,
    DirectionTriageRecord,
    HierarchicalPlanningExpansion,
    ResearchDirectionSet,
    parse_approval_review_input,
)
from ..planning import (
    HTTPJSONLLMTransport,
    MockPlanningProvider,
    PlanMaterializer,
    PlanningContextResolver,
    PlanningEvidenceError,
    StructuredLLMPlanningProvider,
    automatic_revision_block_reason,
    augment_scientific_context,
    build_evidence_resolution_set,
    build_planning_evidence_request_set,
    build_plan_revision_input,
    resolve_planning_evidence_requests,
    retrieval_resolvable_gaps,
)
from ..planning.hierarchical import (
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
from ..planning.validators import validate_planning_proposal_set
from ..repositories import (
    CompositeEvidenceStore,
    KnowledgeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
)
from ..retrieval import ScientificContextBuilder
from ..serialization import content_hash, file_sha256, load_data
from ..validators import (
    build_plan_validation_record,
    validate_independent_approval_chain,
    validate_plan_compilation_receipt,
    validate_question_plan,
)
from .repository import ScientificProblemRunRepository


_CURATED_TYPES = frozenset(
    {
        "expert_source",
        "literature_document",
        "literature_representation_selection",
        "expert_opinion",
        "source_claim",
        "method_fact",
        "model_fact",
        "reported_result",
        "knowledge_relation",
    }
)
_SECRET = re.compile(
    r"(?i)(authorization|cookie|token|password|secret|api[_-]?key)\s*[:=]\s*\S+"
)


@dataclass(frozen=True)
class WorkflowResult:
    run: ScientificProblemRun
    dry_run_report: dict[str, Any] | None = None


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


def _snapshot_hash(snapshot: KnowledgeSnapshot) -> str:
    return content_hash(snapshot.model_dump(mode="json", exclude={"created_at"}))


def _make_run(payload: dict[str, Any]) -> ScientificProblemRun:
    draft = ScientificProblemRun.model_construct(**payload, content_hash="0" * 64)
    normalized = draft.model_dump(mode="json", exclude={"content_hash"})
    return ScientificProblemRun(**normalized, content_hash=content_hash(normalized))


def _update_run(run: ScientificProblemRun, **updates: Any) -> ScientificProblemRun:
    payload = {
        field_name: getattr(run, field_name)
        for field_name in ScientificProblemRun.model_fields
        if field_name != "content_hash"
    }
    payload.update(updates)
    return _make_run(payload)


def _make_revision_chain(payload: dict[str, Any]) -> PlanRevisionChain:
    normalized_payload = dict(payload)
    normalized_payload["rounds"] = tuple(
        item if isinstance(item, PlanRevisionRound) else PlanRevisionRound.model_validate(item)
        for item in payload["rounds"]
    )
    draft = PlanRevisionChain.model_construct(
        **normalized_payload, content_hash="0" * 64
    )
    normalized = draft.model_dump(mode="json", exclude={"content_hash"})
    return PlanRevisionChain(**normalized, content_hash=content_hash(normalized))


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
                gap_id=(
                    "planning-evidence-gap-"
                    + request.request_id.removeprefix("planning-evidence-request-")
                ),
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
    identity = packet.model_dump(
        mode="python", exclude={"packet_id", "content_hash"}
    )
    identity["evidence_gaps"] = tuple(gaps[key] for key in sorted(gaps))
    identity["provenance_manifest"] = {
        **dict(packet.provenance_manifest),
        "evidence_resolution_set_id": resolutions.resolution_set_id,
        "evidence_resolution_set_hash": resolutions.content_hash,
    }
    packet_id = f"evidence-packet-{content_hash(identity)[:24]}"
    payload = {"packet_id": packet_id, **identity}
    return ScientificEvidencePacket(**payload, content_hash=content_hash(payload))


def _provider_binding(stage: str, provider: Any) -> ScientificRunProviderBinding:
    config = dict(getattr(provider, "provider_config", {}))
    model_id = getattr(getattr(provider, "transport", None), "model_id", None)
    for attribute in ("temperature", "max_attempts"):
        value = getattr(provider, attribute, None)
        if value is not None:
            config[attribute] = value
    public_config = {
        "provider_id": provider.provider_id,
        "provider_version": provider.provider_version,
        "model_id": model_id,
        "configuration": config,
    }
    return ScientificRunProviderBinding(
        stage=stage,
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        model_id=model_id,
        configuration_hash=content_hash(public_config),
    )


def _sanitize_failure(error: Exception) -> str:
    return _SECRET.sub(lambda match: f"{match.group(1)}=<redacted>", str(error))[:2000]


class ScientificProblemWorkflow:
    """Thin orchestration over the frozen retrieval, interpretation, planning and approval layers."""

    def __init__(self, *, state_dir: Path, knowledge_dir: Path) -> None:
        self.state_dir = state_dir.resolve()
        self.knowledge_dir = knowledge_dir.resolve()
        self.runs = ScientificProblemRunRepository(self.state_dir)
        self.domain_loader = DomainPackLoader()

    @staticmethod
    def _revision_chain_path() -> str:
        return "plan-revisions/revision-chain.yaml"

    def _load_revision_chain(self, run_id: str) -> PlanRevisionChain | None:
        try:
            return self.runs.load_artifact(
                run_id, self._revision_chain_path(), PlanRevisionChain
            )
        except FileNotFoundError:
            return None

    def _save_revision_chain(self, chain: PlanRevisionChain) -> None:
        self.runs.write_immutable_artifact(
            chain.run_id,
            f"plan-revisions/history/{chain.content_hash}.yaml",
            "plan_revision_chain",
            chain.chain_id,
            chain,
        )
        self.runs.write_artifact(
            chain.run_id,
            self._revision_chain_path(),
            "plan_revision_chain",
            chain.chain_id,
            chain,
        )

    @staticmethod
    def _evidence_policy_path() -> str:
        return "planning-evidence/policy.yaml"

    @staticmethod
    def _evidence_cycle_prefix(cycle_index: int) -> str:
        return f"planning-evidence/cycle-{cycle_index}"

    def _load_evidence_policy(
        self, run_id: str
    ) -> PlanningEvidenceCyclePolicy | None:
        try:
            return self.runs.load_artifact(
                run_id,
                self._evidence_policy_path(),
                PlanningEvidenceCyclePolicy,
            )
        except FileNotFoundError:
            return None

    def _ensure_evidence_policy(
        self,
        run_id: str,
        max_cycles: int,
        source_acquisition_allowed: bool,
    ) -> PlanningEvidenceCyclePolicy | None:
        persisted = self._load_evidence_policy(run_id)
        if persisted is not None:
            if max_cycles not in {0, persisted.max_cycles}:
                raise ValueError(
                    "persisted evidence-resolution budget cannot be changed during resume"
                )
            if (
                max_cycles != 0
                and source_acquisition_allowed
                != persisted.source_acquisition_allowed
            ):
                raise ValueError(
                    "persisted evidence acquisition policy cannot be changed during resume"
                )
            return persisted
        if max_cycles == 0:
            return None
        policy = _make_evidence_cycle_policy(
            run_id, max_cycles, source_acquisition_allowed
        )
        self.runs.write_immutable_artifact(
            run_id,
            self._evidence_policy_path(),
            "planning_evidence_cycle_policy",
            policy.policy_id,
            policy,
        )
        return policy

    def _load_evidence_cycles(
        self, run_id: str
    ) -> tuple[PlanningEvidenceCycleRecord, ...]:
        root = self.runs.resolve_artifact_path(run_id, "planning-evidence")
        if not root.exists():
            return ()
        records: list[PlanningEvidenceCycleRecord] = []
        directories = sorted(
            root.glob("cycle-*"),
            key=lambda path: int(path.name.removeprefix("cycle-")),
        )
        for expected_index, directory in enumerate(directories, start=1):
            if directory.is_symlink() or directory.name != f"cycle-{expected_index}":
                raise ValueError("planning evidence cycle indices are not contiguous")
            cycle_path = directory / "cycle.yaml"
            if cycle_path.exists():
                record = self.runs.load_artifact(
                    run_id,
                    f"{self._evidence_cycle_prefix(expected_index)}/cycle.yaml",
                    PlanningEvidenceCycleRecord,
                )
                if record.cycle_index != expected_index:
                    raise ValueError("planning evidence cycle index is invalid")
                records.append(record)
        return tuple(records)

    def _evidence_attempt_count(self, run_id: str) -> int:
        root = self.runs.resolve_artifact_path(run_id, "planning-evidence")
        if not root.exists():
            return 0
        return sum(
            (directory / "request-attempt.yaml").exists()
            or (directory / "request-set.yaml").exists()
            for directory in root.glob("cycle-*")
            if directory.is_dir() and not directory.is_symlink()
        )

    def _reuse_evidence_cycle_context(
        self,
        run_id: str,
        base_context: ScientificContextPacket,
    ) -> ScientificContextPacket:
        cycles = self._load_evidence_cycles(run_id)
        if not cycles:
            return base_context
        latest = cycles[-1]
        if latest.child_context_id is None:
            return base_context
        if (
            latest.knowledge_snapshot_id
            != base_context.knowledge_snapshot.snapshot_id
            or latest.knowledge_snapshot_hash
            != _snapshot_hash(base_context.knowledge_snapshot)
        ):
            return base_context
        child = self.runs.load_artifact(
            run_id,
            f"{self._evidence_cycle_prefix(latest.cycle_index)}/context.yaml",
            ScientificContextPacket,
        )
        if (
            child.context_id != latest.child_context_id
            or child.content_hash != latest.child_context_hash
        ):
            raise ValueError("stored planning evidence child context binding is invalid")
        return child

    def _revision_artifact_root(self, run_id: str) -> str:
        cycles = self._load_evidence_cycles(run_id)
        if cycles and cycles[-1].child_context_id is not None:
            return f"evidence-replan-{cycles[-1].cycle_index}"
        return "plan-revisions"

    def _hierarchical_root(self, run_id: str) -> str:
        cycles = self._load_evidence_cycles(run_id)
        if cycles and cycles[-1].child_context_id is not None:
            return f"evidence-hierarchy-{cycles[-1].cycle_index}"
        return "hierarchical-planning"

    def _round_prefix(self, run_id: str, round_index: int) -> str:
        root = self._revision_artifact_root(run_id)
        name = f"round-{round_index}" if root == "plan-revisions" else f"r-{round_index}"
        return f"{root}/{name}"

    def _approval_prefix(self, run_id: str) -> str:
        chain = self._load_revision_chain(run_id)
        if chain is None:
            return "approval"
        return f"{self._round_prefix(run_id, chain.rounds[-1].round_index)}/approval"

    def _load_bound_artifact(self, run_id: str, binding, model_type):
        value = self.runs.load_artifact(run_id, binding.relative_path, model_type)
        if content_hash(value) != binding.artifact_hash:
            raise ValueError(
                f"workflow artifact hash mismatch: {binding.relative_path}"
            )
        return value

    def _load_approval_input_binding(
        self,
        run_id: str,
        binding: ScientificRunArtifactBinding,
    ) -> ApprovalReviewInput | RevisionApprovalReviewInput:
        value = parse_approval_review_input(
            load_data(self.runs.resolve_artifact_path(run_id, binding.relative_path))
        )
        if content_hash(value) != binding.artifact_hash:
            raise ValueError(
                f"workflow artifact hash mismatch: {binding.relative_path}"
            )
        return value

    def _attempt_prefix(self, run_id: str, attempt_index: int) -> str:
        root = self._revision_artifact_root(run_id)
        if root == "plan-revisions":
            return f"{root}/attempts/attempt-{attempt_index}"
        return f"{root}/a-{attempt_index}"

    def _load_revision_attempts(
        self, run_id: str
    ) -> tuple[tuple[PlanRevisionAttemptStart, PlanRevisionAttemptOutcome | None], ...]:
        root = self._revision_artifact_root(run_id)
        relative_root = f"{root}/attempts" if root == "plan-revisions" else root
        name_prefix = "attempt-" if root == "plan-revisions" else "a-"
        attempts_root = self.runs.resolve_artifact_path(
            run_id, relative_root
        )
        if not attempts_root.exists():
            return ()
        attempts: list[
            tuple[PlanRevisionAttemptStart, PlanRevisionAttemptOutcome | None]
        ] = []
        for expected_index, directory in enumerate(
            sorted(
                attempts_root.glob(f"{name_prefix}*"),
                key=lambda path: int(path.name.removeprefix(name_prefix)),
            ),
            start=1,
        ):
            if directory.is_symlink() or directory.name != f"{name_prefix}{expected_index}":
                raise ValueError("revision attempt indices are not contiguous")
            start = self.runs.load_artifact(
                run_id,
                f"{self._attempt_prefix(run_id, expected_index)}/start.yaml",
                PlanRevisionAttemptStart,
            )
            if start.attempt_index != expected_index:
                raise ValueError("revision attempt start index is invalid")
            outcome_path = directory / "outcome.yaml"
            outcome = (
                self.runs.load_artifact(
                    run_id,
                    f"{self._attempt_prefix(run_id, expected_index)}/outcome.yaml",
                    PlanRevisionAttemptOutcome,
                )
                if outcome_path.exists()
                else None
            )
            if outcome is not None and (
                outcome.attempt_id != start.attempt_id
                or outcome.attempt_start_hash != start.content_hash
            ):
                raise ValueError("revision attempt outcome does not bind its start")
            attempts.append((start, outcome))
        return tuple(attempts)

    def _validate_revision_chain_integrity(self, chain: PlanRevisionChain) -> None:
        attempts = self._load_revision_attempts(chain.run_id)
        if attempts and len(attempts) < chain.revisions_used:
            raise ValueError("revision chain has more versions than started attempts")
        for round_record in chain.rounds:
            plan = self._load_bound_artifact(
                chain.run_id, round_record.candidate_plan, ScientificQuestionPlan
            )
            receipt = self._load_bound_artifact(
                chain.run_id,
                round_record.compilation_receipt,
                PlanCompilationReceipt,
            )
            validation = self._load_bound_artifact(
                chain.run_id,
                round_record.validation_record,
                PlanValidationRecord,
            )
            self._load_bound_artifact(
                chain.run_id, round_record.planning_proposal, PlanningProposalSet
            )
            if not validate_plan_compilation_receipt(plan, receipt).valid:
                raise ValueError("revision compilation receipt failed integrity validation")
            if (
                validation.plan_id != plan.plan_id
                or validation.plan_content_hash != content_hash(plan)
            ):
                raise ValueError("revision validation record does not bind its plan")
            if round_record.round_index > 0:
                self._load_bound_artifact(
                    chain.run_id, round_record.revision_input, PlanRevisionInput
                )
                self._load_bound_artifact(
                    chain.run_id, round_record.revision_record, PlanRevisionRecord
                )
            if round_record.approval_receipt is None:
                continue
            review_input = self._load_approval_input_binding(
                chain.run_id, round_record.approval_review_input
            )
            review = self._load_bound_artifact(
                chain.run_id,
                round_record.approval_review_record,
                ApprovalReviewRecord,
            )
            verdict = self._load_bound_artifact(
                chain.run_id, round_record.approval_verdict, ApprovalVerdict
            )
            approval_receipt = self._load_bound_artifact(
                chain.run_id,
                round_record.approval_receipt,
                IndependentApprovalReceipt,
            )
            gate = self._load_bound_artifact(
                chain.run_id, round_record.gate, GateVerdict
            )
            if not validate_independent_approval_chain(
                plan, verdict, review_input, review, approval_receipt
            ).valid:
                raise ValueError("revision approval chain failed integrity validation")
            if (
                gate.candidate_id != plan.plan_id
                or gate.candidate_content_hash != content_hash(plan)
                or gate.approval_verdict_hash != content_hash(verdict)
                or gate.independent_approval_receipt_hash
                != approval_receipt.content_hash
            ):
                raise ValueError("revision gate does not bind its approval chain")

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
        try:
            run = self.runs.get(run_id)
        except FileNotFoundError:
            run = _make_run(
                {
                    "run_id": run_id,
                    **identity,
                    "status": ScientificProblemRunStatus.CREATED,
                }
            )
            self.runs.save(run)
        with self.runs.acquire_run_lock(run_id):
            return self._advance(
                run,
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

    def resume(self, run_id: str, **options: Any) -> WorkflowResult:
        with self.runs.acquire_run_lock(run_id):
            return self._advance(self.runs.get(run_id), **options)

    def _knowledge(self, domain: str) -> tuple[KnowledgeRepositories, CompositeEvidenceStore]:
        pack = self.domain_loader.load(domain)
        repositories = KnowledgeRepositories(self.knowledge_dir)
        repositories.load_expert_cases(pack.expert_cases)
        repositories.load_workflow_patterns(pack.workflow_patterns)
        repositories.load_capabilities(pack.capabilities)
        evidence = CompositeEvidenceStore(
            KnowledgeEvidenceStore(self.knowledge_dir),
            ProjectEvidenceStore(self.state_dir),
        )
        return repositories, evidence

    def _blocking_curations(
        self,
        repositories: KnowledgeRepositories,
        evidence: CompositeEvidenceStore,
        domain: str,
    ) -> tuple[ScientificRunBlockingItem, ...]:
        validator = TrustedKnowledgeValidator(repositories, evidence)
        records = validator.record_index(repositories)
        current = validator.resolve_current_curations()
        linked_attributions = {
            item.attribution_id for item in repositories.expert_sources.list()
        }
        pending: list[ScientificRunBlockingItem] = []
        for (record_type, record_id), record in sorted(records.items()):
            record_domain = getattr(record, "domain", domain)
            if record_domain not in {domain, "base"}:
                continue
            requires_curation = record_type in _CURATED_TYPES
            if record_type == "expert_case":
                requires_curation = bool(getattr(record, "opinion_refs", ()))
            elif record_type == "expert_attribution":
                requires_curation = record_id in linked_attributions
            if not requires_curation:
                continue
            curation = current.get((record_type, record_id))
            if curation is not None and curation.status in {
                CurationStatus.ACCEPTED,
                CurationStatus.REJECTED,
            }:
                continue
            status = curation.status if curation is not None else None
            pending.append(
                ScientificRunBlockingItem(
                    target_type=record_type,
                    target_id=record_id,
                    current_status=status,
                    message=(
                        f"{record_type} {record_id} requires ACCEPTED curation"
                    ),
                )
            )
        return tuple(pending)

    def _advance(
        self,
        run: ScientificProblemRun,
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
        try:
            strategy = PlanningStrategy(planning_strategy)
        except ValueError as error:
            raise ValueError("planning_strategy must be 'direct' or 'hierarchical'") from error
        if not 0 <= max_plan_revisions <= 10:
            raise ValueError("max_plan_revisions must be between zero and ten")
        if not 0 <= max_evidence_resolution_cycles <= 3:
            raise ValueError(
                "max_evidence_resolution_cycles must be between zero and three"
            )
        evidence_policy = self._ensure_evidence_policy(
            run.run_id,
            max_evidence_resolution_cycles,
            allow_evidence_source_acquisition,
        )
        revision_chain = self._load_revision_chain(run.run_id)
        if (
            revision_chain is not None
            and run.context_id is not None
            and run.context_id != revision_chain.context_id
        ):
            revision_chain = None
        if revision_chain is not None and revision_chain.termination_reason is not None:
            try:
                self._validate_revision_chain_integrity(revision_chain)
            except (FileNotFoundError, OSError, ValueError) as error:
                return self._fail(
                    run,
                    "plan_revision_integrity",
                    error,
                    retryable=False,
                )
            return WorkflowResult(run)
        repositories, evidence = self._knowledge(run.domain)
        try:
            blocking = self._blocking_curations(repositories, evidence, run.domain)
        except (FileNotFoundError, OSError, ValueError) as error:
            return self._fail(run, "knowledge_trust", error, retryable=False)
        if blocking:
            blocked = _update_run(
                run,
                status=ScientificProblemRunStatus.BLOCKED_SOURCE_CURATION,
                blocking_items=blocking,
                failure=None,
            )
            self.runs.save(blocked)
            return WorkflowResult(blocked, self._dry_run_report(blocked, repositories, None))

        ready = _update_run(
            run,
            status=ScientificProblemRunStatus.READY,
            last_successful_status=None,
            blocking_items=(),
            failure=None,
        )
        self.runs.save(ready)
        try:
            base_context = ScientificContextBuilder(self.domain_loader).build(
                ready.original_request,
                ready.domain,
                state_dir=self.state_dir,
                knowledge_dir=self.knowledge_dir,
            )
            context = self._reuse_evidence_cycle_context(
                ready.run_id, base_context
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            return self._fail(ready, "trusted_retrieval", error, retryable=False)
        if revision_chain is not None and (
            revision_chain.context_id != context.context_id
            or revision_chain.context_hash != context.content_hash
            or revision_chain.knowledge_snapshot_id
            != context.knowledge_snapshot.snapshot_id
            or revision_chain.knowledge_snapshot_hash
            != _snapshot_hash(context.knowledge_snapshot)
        ):
            return self._revision_block(
                ready,
                revision_chain,
                "TRUSTED_CONTEXT_CHANGED",
                "trusted context or knowledge authority changed; start a new compiled context",
            )
        self.runs.write_artifact(
            ready.run_id, "context.yaml", "scientific_context_packet", context.context_id, context
        )
        stale_context = (
            ready.context_hash is not None and ready.context_hash != context.content_hash
        )
        invalidated = {
            "evidence_packet_id": None,
            "evidence_packet_hash": None,
            "planning_input_id": None,
            "planning_input_hash": None,
            "planning_proposal_id": None,
            "planning_proposal_hash": None,
            "candidate_plans": (),
            "candidate_compilation_receipts": (),
            "plan_validation_records": (),
            "selected_candidate_id": None,
            "approval_review_id": None,
            "approval_review_hash": None,
            "approval_verdict_id": None,
            "approval_verdict_hash": None,
            "approval_receipt_id": None,
            "approval_receipt_hash": None,
            "gate_id": None,
            "gate_hash": None,
            "export_path": None,
            "export_hash": None,
        }
        context_run = _update_run(
            ready,
            **(invalidated if stale_context else {}),
            status=ScientificProblemRunStatus.CONTEXT_BUILT,
            knowledge_snapshot_id=context.knowledge_snapshot.snapshot_id,
            knowledge_snapshot_hash=_snapshot_hash(context.knowledge_snapshot),
            retrieval_context_id=(
                context.knowledge_retrieval_context.retrieval_context_id
                if context.knowledge_retrieval_context is not None
                else context.retrieval_manifest.retrieval_id
            ),
            retrieval_context_hash=(
                context.knowledge_retrieval_context.content_hash
                if context.knowledge_retrieval_context is not None
                else content_hash(context.retrieval_manifest)
            ),
            context_id=context.context_id,
            context_hash=context.content_hash,
        )
        self.runs.save(context_run)
        if dry_run:
            return WorkflowResult(
                context_run,
                self._dry_run_report(context_run, repositories, context),
            )

        if interpretation_provider != "mock":
            return self._fail(
                context_run,
                "interpretation",
                ValueError("available interpretation provider: mock"),
                retryable=False,
            )
        interpretation = MockInterpretationProvider()
        try:
            packet = self._reuse_evidence_packet(context_run, context, evidence)
            if packet is None:
                packet = ScientificEvidencePacketBuilder(interpretation).build(context, evidence)
        except (FileNotFoundError, OSError, ValueError) as error:
            return self._fail(context_run, "interpretation", error, retryable=True)
        self.runs.write_artifact(
            context_run.run_id,
            "evidence-packet.yaml",
            "scientific_evidence_packet",
            packet.packet_id,
            packet,
        )
        interpreted = _update_run(
            context_run,
            status=ScientificProblemRunStatus.INTERPRETED,
            evidence_packet_id=packet.packet_id,
            evidence_packet_hash=packet.content_hash,
            providers=self._replace_provider(
                context_run.providers, _provider_binding("interpretation", interpretation)
            ),
        )
        self.runs.save(interpreted)

        try:
            planning_input = PlanningContextResolver(self.domain_loader).resolve(
                context, packet, repositories, evidence
            )
            planner = self._planning_provider(
                planning_provider,
                llm_endpoint=llm_endpoint,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                temperature=temperature,
                max_attempts=max_attempts,
            )
            if revision_chain is None:
                while True:
                    parent_context_id = context.context_id
                    (
                        interpreted,
                        context,
                        packet,
                        planning_input,
                    ) = self._maybe_resolve_planning_evidence(
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
                        directions, triage = self._hierarchical_direction_triage(
                            interpreted, planning_input, planner
                        )
                        (
                            interpreted,
                            context,
                            packet,
                            planning_input,
                        ) = self._maybe_resolve_planning_evidence(
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
                if max_plan_revisions not in {0, revision_chain.max_revisions}:
                    raise ValueError(
                        "persisted revision budget cannot be changed during resume"
                    )
                if (
                    revision_chain.planning_input_id != planning_input.planning_input_id
                    or revision_chain.planning_input_hash != planning_input.content_hash
                ):
                    return self._revision_block(
                        interpreted,
                        revision_chain,
                        "PLANNING_INPUT_CHANGED",
                        "trusted planning input changed; rebuild a new context before revision",
                    )
                compilation = self._load_revision_compilation(
                    revision_chain, planning_input, evidence
                )
            else:
                hierarchical_root = self.runs.resolve_artifact_path(
                    interpreted.run_id,
                    self._hierarchical_root(interpreted.run_id),
                )
                if hierarchical_root.exists():
                    strategy = PlanningStrategy.HIERARCHICAL
                elif (
                    strategy == PlanningStrategy.HIERARCHICAL
                    and interpreted.planning_proposal_id is not None
                ):
                    raise HierarchicalPlanningBlocked(
                        (
                            "this run already contains direct planning artifacts; use a new "
                            "state directory for a hierarchical comparison",
                        )
                    )
                if strategy == PlanningStrategy.HIERARCHICAL:
                    compilation = self._hierarchical_compilation(
                        interpreted,
                        planning_input,
                        evidence,
                        planner,
                    )
                else:
                    compilation = self._reuse_compilation(
                        interpreted, planning_input, evidence
                    )
                    if compilation is None:
                        compilation = ScientificProblemCompiler(
                            planner, evidence_repository=evidence
                        ).compile(planning_input)
        except EvidenceResolutionBlocked as error:
            status = (
                ScientificProblemRunStatus.BLOCKED_SOURCE_CURATION
                if error.curation_items
                else ScientificProblemRunStatus.EVIDENCE_RESOLUTION_BLOCKED
            )
            blocked = _update_run(
                interpreted,
                status=status,
                last_successful_status=interpreted.status,
                blocking_items=error.curation_items,
                failure=(
                    None
                    if error.curation_items
                    else ScientificRunFailure(
                        stage="planning_evidence_resolution",
                        category=error.category,
                        message=_sanitize_failure(error),
                        retryable=False,
                    )
                ),
            )
            self.runs.save(blocked)
            return WorkflowResult(blocked)
        except HierarchicalPlanningBlocked as error:
            blocked = _update_run(
                interpreted,
                status=ScientificProblemRunStatus.HIERARCHICAL_PLANNING_BLOCKED,
                last_successful_status=interpreted.status,
                failure=ScientificRunFailure(
                    stage="hierarchical_planning",
                    category=type(error).__name__,
                    message=_sanitize_failure(error),
                    retryable=False,
                ),
            )
            self.runs.save(blocked)
            return WorkflowResult(blocked)
        except (FileNotFoundError, OSError, ValueError, TypeError) as error:
            return self._fail(interpreted, "planning", error, retryable=True)
        proposal = compilation.proposal_set
        if proposal is None:
            return self._fail(
                interpreted,
                "planning",
                RuntimeError("grounded compiler returned no proposal set"),
                retryable=False,
            )
        self.runs.write_artifact(
            run.run_id, "planning-input.yaml", "scientific_planning_input", planning_input.planning_input_id, planning_input
        )
        if revision_chain is not None:
            current_round = revision_chain.rounds[-1]
            proposal_binding = current_round.planning_proposal
            candidate_bindings = [current_round.candidate_plan]
            receipt_bindings = [current_round.compilation_receipt]
            validation_bindings = [current_round.validation_record]
        else:
            revision_enabled = max_plan_revisions > 0
            prefix = self._round_prefix(run.run_id, 0) if revision_enabled else ""
            proposal_path = (
                f"{prefix}/planning-proposal.yaml"
                if prefix
                else "planning-proposal.yaml"
            )
            writer = (
                self.runs.write_immutable_artifact
                if revision_enabled
                else self.runs.write_artifact
            )
            proposal_binding = writer(
                run.run_id,
                proposal_path,
                "planning_proposal_set",
                proposal.proposal_id,
                proposal,
            )
            candidate_bindings = []
            receipt_bindings = []
            validation_bindings = []
            for plan, receipt in zip(
                compilation.candidates,
                compilation.compilation_receipts,
                strict=True,
            ):
                candidate_path = (
                    f"{prefix}/candidates/{plan.plan_id}.yaml"
                    if prefix
                    else f"candidates/{plan.plan_id}.yaml"
                )
                receipt_path = (
                    f"{prefix}/compilation-receipts/{receipt.receipt_id}.yaml"
                    if prefix
                    else f"compilation-receipts/{receipt.receipt_id}.yaml"
                )
                candidate_bindings.append(
                    writer(
                        run.run_id,
                        candidate_path,
                        "scientific_question_plan",
                        plan.plan_id,
                        plan,
                    )
                )
                receipt_bindings.append(
                    writer(
                        run.run_id,
                        receipt_path,
                        "plan_compilation_receipt",
                        receipt.receipt_id,
                        receipt,
                    )
                )
                validation = build_plan_validation_record(
                    plan,
                    validate_question_plan(
                        plan, planning_input.scientific_capabilities, evidence
                    ),
                    validation_id=f"plan-validation-{content_hash(plan)[:24]}",
                )
                validation_path = (
                    f"{prefix}/validation-records/{validation.validation_id}.yaml"
                    if prefix
                    else f"validation-records/{validation.validation_id}.yaml"
                )
                validation_bindings.append(
                    writer(
                        run.run_id,
                        validation_path,
                        "plan_validation_record",
                        validation.validation_id,
                        validation,
                    )
                )
        planned = _update_run(
            interpreted,
            status=ScientificProblemRunStatus.PLANNED,
            planning_input_id=planning_input.planning_input_id,
            planning_input_hash=planning_input.content_hash,
            planning_proposal_id=proposal.proposal_id,
            planning_proposal_hash=content_hash(proposal),
            candidate_plans=tuple(candidate_bindings),
            candidate_compilation_receipts=tuple(receipt_bindings),
            plan_validation_records=tuple(validation_bindings),
            selected_candidate_id=selected_candidate_id,
            providers=self._replace_provider(
                interpreted.providers, _provider_binding("planning", planner)
            ),
        )
        self.runs.save(planned)
        if revision_chain is None and max_plan_revisions > 0:
            if selected_candidate_id is None:
                if len(candidate_bindings) != 1:
                    return self._fail(
                        planned,
                        "plan_revision",
                        ValueError(
                            "bounded revision with multiple candidates requires --candidate-id"
                        ),
                        retryable=False,
                    )
                selected_candidate_id = candidate_bindings[0].artifact_id
            selected_ids = [item.artifact_id for item in candidate_bindings]
            if selected_candidate_id not in selected_ids:
                return self._fail(
                    planned,
                    "plan_revision",
                    ValueError("selected candidate ID is not part of this run"),
                    retryable=False,
                )
            selected_index = selected_ids.index(selected_candidate_id)
            revision_chain = _make_revision_chain(
                {
                    "chain_id": (
                        "plan-revision-chain-"
                        + content_hash({"run_id": planned.run_id})[:24]
                    ),
                    "run_id": planned.run_id,
                    "planning_input_id": planning_input.planning_input_id,
                    "planning_input_hash": planning_input.content_hash,
                    "context_id": context.context_id,
                    "context_hash": context.content_hash,
                    "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
                    "knowledge_snapshot_hash": _snapshot_hash(
                        context.knowledge_snapshot
                    ),
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
            self._save_revision_chain(revision_chain)
            planned = _update_run(
                planned,
                candidate_plans=(candidate_bindings[selected_index],),
                candidate_compilation_receipts=(receipt_bindings[selected_index],),
                plan_validation_records=(validation_bindings[selected_index],),
                selected_candidate_id=selected_candidate_id,
            )
            self.runs.save(planned)
        awaiting = _update_run(
            planned,
            status=ScientificProblemRunStatus.AWAITING_APPROVAL,
        )
        self.runs.save(awaiting)
        if approval_provider == "none":
            reusable = self._reuse_approval(
                awaiting,
                previous=run,
                context=context,
                packet=packet,
                planning_input=planning_input,
                repositories=repositories,
                evidence=evidence,
                selected_candidate_id=selected_candidate_id,
            )
            return WorkflowResult(reusable or awaiting)
        if revision_chain is not None and revision_chain.rounds[-1].approval_receipt is not None:
            approved = self._reuse_approval(
                awaiting,
                previous=run,
                context=context,
                packet=packet,
                planning_input=planning_input,
                repositories=repositories,
                evidence=evidence,
                selected_candidate_id=selected_candidate_id,
            )
            if approved is None:
                return self._revision_block(
                    awaiting,
                    revision_chain,
                    "STALE_APPROVAL_CHAIN",
                    "stored revision approval chain failed validation",
                )
        else:
            revision_approval = None
            if (
                revision_chain is not None
                and revision_chain.rounds[-1].round_index > 0
            ):
                current_round = revision_chain.rounds[-1]
                revision_input = self._load_bound_artifact(
                    awaiting.run_id,
                    current_round.revision_input,
                    PlanRevisionInput,
                )
                revision_record = self._load_bound_artifact(
                    awaiting.run_id,
                    current_round.revision_record,
                    PlanRevisionRecord,
                )
                revision_approval = (
                    revision_input,
                    revision_record,
                    revision_input.approval_review_record,
                )
            approved = self._approve(
                awaiting,
                context,
                packet,
                planning_input,
                repositories,
                evidence,
                approval_provider,
                selected_candidate_id,
                llm_endpoint,
                llm_model,
                llm_api_key,
                temperature,
                max_attempts,
                artifact_prefix=(
                    f"{self._round_prefix(run.run_id, revision_chain.rounds[-1].round_index)}/approval"
                    if revision_chain is not None
                    else "approval"
                ),
                revision_approval=revision_approval,
            )
            if revision_chain is not None:
                if approved.approval_receipt_id is None:
                    return WorkflowResult(approved)
                revision_chain = self._record_round_approval(revision_chain, approved)
        if approved.approval_verdict_id is not None and evidence_policy is not None:
            approval_prefix = self._approval_prefix(approved.run_id)
            verdict = self.runs.load_artifact(
                approved.run_id,
                f"{approval_prefix}/approval-verdict.yaml",
                ApprovalVerdict,
            )
            if verdict.decision == ApprovalDecision.INSUFFICIENT_EVIDENCE:
                review = self.runs.load_artifact(
                    approved.run_id,
                    f"{approval_prefix}/approval-review.yaml",
                    ApprovalReviewRecord,
                )
                cycle_index = self._evidence_attempt_count(approved.run_id) + 1
                trigger_review_input, trigger_candidate = (
                    self._archive_evidence_trigger_artifacts(
                    approved, cycle_index, approval_prefix
                    )
                )
                try:
                    (
                        evidence_run,
                        child_context,
                        _child_packet,
                        _child_input,
                    ) = self._maybe_resolve_planning_evidence(
                        approved,
                        context,
                        packet,
                        planning_input,
                        repositories,
                        evidence,
                        planner,
                        strategy,
                        evidence_policy,
                        triggering_review=review,
                        triggering_review_input=trigger_review_input,
                        triggering_candidate=trigger_candidate,
                    )
                except EvidenceResolutionBlocked as error:
                    status = (
                        ScientificProblemRunStatus.BLOCKED_SOURCE_CURATION
                        if error.curation_items
                        else ScientificProblemRunStatus.EVIDENCE_RESOLUTION_BLOCKED
                    )
                    blocked = _update_run(
                        approved,
                        status=status,
                        last_successful_status=approved.status,
                        blocking_items=error.curation_items,
                        failure=(
                            None
                            if error.curation_items
                            else ScientificRunFailure(
                                stage="planning_evidence_resolution",
                                category=error.category,
                                message=_sanitize_failure(error),
                                retryable=False,
                            )
                        ),
                    )
                    self.runs.save(blocked)
                    return WorkflowResult(blocked)
                if child_context.context_id != context.context_id:
                    return self._advance(
                        evidence_run,
                        dry_run=False,
                        interpretation_provider=interpretation_provider,
                        planning_provider=planning_provider,
                        approval_provider=approval_provider,
                        selected_candidate_id=None,
                        llm_endpoint=llm_endpoint,
                        llm_model=llm_model,
                        llm_api_key=llm_api_key,
                        temperature=temperature,
                        max_attempts=max_attempts,
                        max_plan_revisions=max_plan_revisions,
                        max_evidence_resolution_cycles=max_evidence_resolution_cycles,
                        allow_evidence_source_acquisition=(
                            allow_evidence_source_acquisition
                        ),
                        planning_strategy=strategy.value,
                    )
        if revision_chain is None:
            return WorkflowResult(approved)
        return WorkflowResult(
            self._run_revision_cycle(
                approved,
                revision_chain,
                context,
                packet,
                planning_input,
                repositories,
                evidence,
                planner,
                approval_provider,
                llm_endpoint,
                llm_model,
                llm_api_key,
                temperature,
                max_attempts,
            )
        )

    def _reuse_approval(
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
    ) -> ScientificProblemRun | None:
        if previous.approval_receipt_id is None or previous.selected_candidate_id is None:
            return None
        if selected_candidate_id not in {None, previous.selected_candidate_id}:
            return None
        candidate_ids = tuple(item.artifact_id for item in run.candidate_plans)
        if previous.selected_candidate_id not in candidate_ids:
            return None
        index = candidate_ids.index(previous.selected_candidate_id)
        approval_prefix = self._approval_prefix(run.run_id)
        try:
            plan = self.runs.load_artifact(
                run.run_id, run.candidate_plans[index].relative_path, ScientificQuestionPlan
            )
            validation = self.runs.load_artifact(
                run.run_id,
                run.plan_validation_records[index].relative_path,
                PlanValidationRecord,
            )
            ApprovalContextResolver(self.domain_loader).resolve(
                context,
                packet,
                planning_input,
                plan,
                validation,
                repositories,
                evidence,
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
            gate = self.runs.load_artifact(
                run.run_id, f"{approval_prefix}/plan-gate.yaml", GateVerdict
            )
            if not validate_independent_approval_chain(
                plan, verdict, review_input, review, receipt
            ).valid:
                return None
            if (
                gate.candidate_id != plan.plan_id
                or gate.candidate_content_hash != content_hash(plan)
                or gate.approval_verdict_hash != content_hash(verdict)
                or gate.independent_approval_receipt_hash != receipt.content_hash
            ):
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
        approval_binding = next(
            (item for item in previous.providers if item.stage == "approval"), None
        )
        providers = run.providers
        if approval_binding is not None:
            providers = self._replace_provider(providers, approval_binding)
        reused = _update_run(
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

    def _reuse_evidence_packet(
        self,
        run: ScientificProblemRun,
        context: ScientificContextPacket,
        evidence: CompositeEvidenceStore,
    ) -> ScientificEvidencePacket | None:
        if run.context_hash != context.content_hash or run.evidence_packet_id is None:
            return None
        try:
            packet = self.runs.load_artifact(run.run_id, "evidence-packet.yaml", ScientificEvidencePacket)
        except (FileNotFoundError, OSError, ValueError):
            return None
        report = validate_evidence_packet_integrity(packet, context, evidence)
        return packet if report.valid and packet.content_hash == run.evidence_packet_hash else None

    def _reuse_compilation(self, run, planning_input, evidence):
        if run.planning_input_hash != planning_input.content_hash or not run.candidate_plans:
            return None
        try:
            proposal = self.runs.load_artifact(run.run_id, "planning-proposal.yaml", PlanningProposalSet)
            if not validate_planning_proposal_set(proposal, planning_input).valid:
                return None
            candidates = tuple(
                self.runs.load_artifact(run.run_id, binding.relative_path, ScientificQuestionPlan)
                for binding in run.candidate_plans
            )
            receipts = tuple(
                self.runs.load_artifact(run.run_id, binding.relative_path, PlanCompilationReceipt)
                for binding in run.candidate_compilation_receipts
            )
            reports = tuple(
                validate_question_plan(plan, planning_input.scientific_capabilities, evidence)
                for plan in candidates
            )
            if not all(report.valid for report in reports):
                return None
            if not all(
                validate_plan_compilation_receipt(plan, receipt).valid
                for plan, receipt in zip(candidates, receipts, strict=True)
            ):
                return None
            from ..compiler import CompilationResult

            return CompilationResult(candidates, reports, proposal, receipts)
        except (FileNotFoundError, OSError, ValueError):
            return None

    def _record_evidence_cycle(
        self,
        run_id: str,
        payload: dict[str, Any],
    ) -> PlanningEvidenceCycleRecord:
        record = _make_evidence_cycle_record(payload)
        self.runs.write_immutable_artifact(
            run_id,
            f"{self._evidence_cycle_prefix(record.cycle_index)}/cycle.yaml",
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
        prefix = f"{self._evidence_cycle_prefix(cycle_index)}/trigger"
        planning_input = self.runs.load_artifact(
            run.run_id, "planning-input.yaml", ScientificPlanningInput
        )
        self.runs.write_immutable_artifact(
            run.run_id,
            f"{prefix}/planning-input.yaml",
            "scientific_planning_input",
            planning_input.planning_input_id,
            planning_input,
        )
        chain = self._load_revision_chain(run.run_id)
        proposal_path = (
            chain.rounds[-1].planning_proposal.relative_path
            if chain is not None
            else "planning-proposal.yaml"
        )
        proposal = self.runs.load_artifact(
            run.run_id, proposal_path, PlanningProposalSet
        )
        self.runs.write_immutable_artifact(
            run.run_id,
            f"{prefix}/planning-proposal.yaml",
            "planning_proposal_set",
            proposal.proposal_id,
            proposal,
        )
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
        candidate_ids = tuple(item.artifact_id for item in run.candidate_plans)
        if review_input.candidate_plan.plan_id not in candidate_ids:
            raise ValueError("evidence trigger review candidate is not part of the run")
        candidate_index = candidate_ids.index(review_input.candidate_plan.plan_id)
        candidate = self._load_bound_artifact(
            run.run_id,
            run.candidate_plans[candidate_index],
            ScientificQuestionPlan,
        )
        validation = self._load_bound_artifact(
            run.run_id,
            run.plan_validation_records[candidate_index],
            PlanValidationRecord,
        )
        compilation_receipt = self._load_bound_artifact(
            run.run_id,
            run.candidate_compilation_receipts[candidate_index],
            PlanCompilationReceipt,
        )
        if not validate_plan_compilation_receipt(
            candidate, compilation_receipt
        ).valid:
            raise ValueError("evidence trigger compilation receipt is invalid")
        if not validate_independent_approval_chain(
            candidate,
            verdict,
            review_input,
            review,
            approval_receipt,
        ).valid:
            raise ValueError("evidence trigger independent approval chain is invalid")
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

    def _validate_evidence_trigger_archive(
        self,
        run_id: str,
        cycle_index: int,
        request_set: PlanningEvidenceRequestSet,
    ) -> tuple[
        ApprovalReviewInput | RevisionApprovalReviewInput,
        ApprovalReviewRecord,
        ScientificQuestionPlan,
    ]:
        prefix = f"{self._evidence_cycle_prefix(cycle_index)}/trigger"
        review_input = parse_approval_review_input(
            load_data(
                self.runs.resolve_artifact_path(
                    run_id, f"{prefix}/approval-review-input.yaml"
                )
            )
        )
        review = self.runs.load_artifact(
            run_id, f"{prefix}/approval-review.yaml", ApprovalReviewRecord
        )
        verdict = self.runs.load_artifact(
            run_id, f"{prefix}/approval-verdict.yaml", ApprovalVerdict
        )
        receipt = self.runs.load_artifact(
            run_id,
            f"{prefix}/independent-approval-receipt.yaml",
            IndependentApprovalReceipt,
        )
        gate = self.runs.load_artifact(
            run_id, f"{prefix}/plan-gate.yaml", GateVerdict
        )
        policy = self.runs.load_artifact(
            run_id, f"{prefix}/project-trust-policy.yaml", ProjectTrustPolicy
        )
        candidate = self.runs.load_artifact(
            run_id, f"{prefix}/candidate-plan.yaml", ScientificQuestionPlan
        )
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
        if not validate_plan_compilation_receipt(
            candidate, compilation_receipt
        ).valid or not validate_independent_approval_chain(
            candidate, verdict, review_input, review, receipt
        ).valid:
            raise ValueError("archived evidence trigger approval chain is invalid")
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

    def _maybe_resolve_planning_evidence(
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
        completed = self._load_evidence_cycles(run.run_id)
        if completed:
            latest = completed[-1]
            same_snapshot = (
                latest.knowledge_snapshot_id == context.knowledge_snapshot.snapshot_id
                and latest.knowledge_snapshot_hash
                == _snapshot_hash(context.knowledge_snapshot)
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
            if need.resolution_kind
            == EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE
        )
        if not blocking_gaps and triggering_review is None and not direction_needs:
            return run, context, packet, planning_input
        attempt_count = self._evidence_attempt_count(run.run_id)
        if attempt_count:
            previous_prefix = self._evidence_cycle_prefix(attempt_count)
            previous_attempt = self.runs.resolve_artifact_path(
                run.run_id, f"{previous_prefix}/request-attempt.yaml"
            )
            previous_request = self.runs.resolve_artifact_path(
                run.run_id, f"{previous_prefix}/request-set.yaml"
            )
            previous_resolution = self.runs.resolve_artifact_path(
                run.run_id, f"{previous_prefix}/resolutions.yaml"
            )
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
        cycle_index = attempt_count + 1
        if cycle_index > policy.max_cycles:
            raise EvidenceResolutionBlocked(
                "EVIDENCE_RESOLUTION_BUDGET_EXHAUSTED",
                "planning evidence-resolution cycle budget is exhausted",
            )
        prefix = self._evidence_cycle_prefix(cycle_index)
        request_path = self.runs.resolve_artifact_path(
            run.run_id, f"{prefix}/request-set.yaml"
        )
        attempt_path = self.runs.resolve_artifact_path(
            run.run_id, f"{prefix}/request-attempt.yaml"
        )
        resolution_path = self.runs.resolve_artifact_path(
            run.run_id, f"{prefix}/resolutions.yaml"
        )
        provider_block_reason: str | None = None
        if request_path.exists():
            request_set = self.runs.load_artifact(
                run.run_id,
                f"{prefix}/request-set.yaml",
                PlanningEvidenceRequestSet,
            )
            if request_set.triggering_review_id is not None:
                (
                    archived_review_input,
                    archived_review,
                    archived_candidate,
                ) = self._validate_evidence_trigger_archive(
                    run.run_id, cycle_index, request_set
                )
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
                if direction_set is None or triage is None or (
                    request_set.direction_set_id != direction_set.direction_set_id
                    or request_set.direction_set_hash != direction_set.content_hash
                    or request_set.triage_id != triage.triage_id
                    or request_set.triage_hash != triage.content_hash
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
        else:
            if attempt_path.exists():
                raise EvidenceResolutionBlocked(
                    "EVIDENCE_REQUEST_OUTCOME_UNCERTAIN",
                    "an evidence-request provider call was claimed without a complete request set",
                )
            provider_binding = _provider_binding("planning_evidence_request", planner)
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
                    "triggering_review_id": (
                        triggering_review.review_id if triggering_review else None
                    ),
                    "triggering_review_hash": (
                        triggering_review.content_hash if triggering_review else None
                    ),
                    "triggering_review_input_id": (
                        triggering_review_input.review_input_id
                        if triggering_review_input is not None
                        else None
                    ),
                    "triggering_review_input_hash": (
                        triggering_review_input.content_hash
                        if triggering_review_input is not None
                        else None
                    ),
                    "triggering_candidate_id": (
                        triggering_candidate.plan_id
                        if triggering_candidate is not None
                        else None
                    ),
                    "triggering_candidate_hash": (
                        content_hash(triggering_candidate)
                        if triggering_candidate is not None
                        else None
                    ),
                    "direction_set_id": (
                        direction_set.direction_set_id
                        if direction_set is not None
                        else None
                    ),
                    "direction_set_hash": (
                        direction_set.content_hash if direction_set is not None else None
                    ),
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
            if triggering_review is not None or direction_needs or (
                retrievable and len(retrievable) == len(blocking_gaps)
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
                knowledge_snapshot_hash=_snapshot_hash(
                    context.knowledge_snapshot
                ),
                source_acquisition_allowed=policy.source_acquisition_allowed,
                direction_set=direction_set,
                triage=triage,
                triggering_review=triggering_review,
                triggering_review_id=(
                    triggering_review.review_id if triggering_review else None
                ),
                triggering_review_hash=(
                    triggering_review.content_hash if triggering_review else None
                ),
                triggering_review_input=triggering_review_input,
                triggering_candidate=triggering_candidate,
            )
            for prior in completed:
                if prior.knowledge_snapshot_id != request_set.knowledge_snapshot_id:
                    continue
                prior_requests = self.runs.load_artifact(
                    run.run_id,
                    f"{self._evidence_cycle_prefix(prior.cycle_index)}/request-set.yaml",
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
                if item.resolution_kind
                != EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE
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
            if item.resolution_kind
            != EvidenceGapResolutionKind.RETRIEVAL_RESOLVABLE
        )
        terminal_reason = None
        if (
            provider_block_reason is None
            and triggering_review is not None
            and not request_set.requests
        ):
            provider_block_reason = (
                "the triggering review does not identify a retrieval-resolvable source gap"
            )
        if provider_block_reason is not None:
            terminal_reason = provider_block_reason
            category = "NOT_RETRIEVAL_RESOLVABLE"
            curation_items = ()
        elif nonretrieval:
            human = tuple(
                item
                for item in nonretrieval
                if item.resolution_kind
                == EvidenceGapResolutionKind.HUMAN_DECISION_REQUIRED
            )
            terminal_reason = "; ".join(
                f"{item.gap_id}:{item.resolution_kind.value}:{item.rationale}"
                for item in nonretrieval
            )
            category = (
                "HUMAN_DECISION_REQUIRED"
                if human
                else "NOT_RETRIEVAL_RESOLVABLE"
            )
        else:
            curation_records = tuple(
                record
                for record in resolutions.records
                if record.status
                == EvidenceResolutionStatus.REQUIRES_SOURCE_CURATION
            )
            if curation_records:
                curation_items = tuple(
                    ScientificRunBlockingItem(
                        target_type=hit.source_type.value,
                        target_id=hit.record_id,
                        current_status=hit.curation_status,
                        message=(
                            f"{hit.source_type.value} {hit.record_id} requires "
                            "ACCEPTED curation before planning use"
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
                    record
                    for record in resolutions.records
                    if record.status == EvidenceResolutionStatus.NO_MATCH
                )
                if no_matches:
                    terminal_reason = "; ".join(
                        record.resolution_rationale for record in no_matches
                    )
                    category = (
                        "NO_TRUSTED_MATCH"
                        if all(
                            record.retrieval_status is None
                            or record.retrieval_status.value == "no_trusted_match"
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
                    "knowledge_snapshot_hash": _snapshot_hash(
                        context.knowledge_snapshot
                    ),
                    "terminal_reason": terminal_reason,
                },
            )
            del cycle
            raise EvidenceResolutionBlocked(
                category,
                terminal_reason,
                curation_items=(
                    curation_items if category == "REQUIRES_SOURCE_CURATION" else ()
                ),
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
            terminal_reason = (
                "trusted retrieval returned no new evidence beyond the current planning context"
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
                    "knowledge_snapshot_hash": _snapshot_hash(
                        context.knowledge_snapshot
                    ),
                    "terminal_reason": terminal_reason,
                },
            )
            raise EvidenceResolutionBlocked(
                "NO_NEW_TRUSTED_EVIDENCE", terminal_reason
            )
        child_context = augment_scientific_context(context, resolutions, evidence)
        interpretation = MockInterpretationProvider()
        child_packet = ScientificEvidencePacketBuilder(interpretation).build(
            child_context, evidence
        )
        child_packet = _carry_unresolved_evidence_gaps(
            child_packet,
            planning_input,
            request_set,
            resolutions,
        )
        child_input = PlanningContextResolver(self.domain_loader).resolve(
            child_context,
            child_packet,
            repositories,
            evidence,
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
                "knowledge_snapshot_hash": _snapshot_hash(context.knowledge_snapshot),
                "child_context_id": child_context.context_id,
                "child_context_hash": child_context.content_hash,
                "child_evidence_packet_id": child_packet.packet_id,
                "child_evidence_packet_hash": child_packet.content_hash,
                "child_planning_input_id": child_input.planning_input_id,
                "child_planning_input_hash": child_input.content_hash,
            },
        )
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
        updated = _update_run(
            run,
            status=ScientificProblemRunStatus.INTERPRETED,
            knowledge_snapshot_id=child_context.knowledge_snapshot.snapshot_id,
            knowledge_snapshot_hash=_snapshot_hash(child_context.knowledge_snapshot),
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
        return updated, child_context, child_packet, child_input

    def _hierarchical_direction_triage(
        self,
        run: ScientificProblemRun,
        planning_input,
        planner,
    ) -> tuple[ResearchDirectionSet, DirectionTriageRecord]:
        root = self._hierarchical_root(run.run_id)

        def existing(path: str) -> bool:
            return self.runs.resolve_artifact_path(run.run_id, path).exists()

        directions_path = f"{root}/directions.yaml"
        directions_attempt_path = f"{root}/attempts/directions.yaml"
        if existing(directions_path):
            directions = self.runs.load_artifact(
                run.run_id, directions_path, ResearchDirectionSet
            )
        else:
            if existing(directions_attempt_path):
                raise HierarchicalPlanningBlocked(
                    ("direction generation was started but no complete output was saved",)
                )
            attempt = build_hierarchical_stage_attempt(
                "directions", planning_input, planner
            )
            self.runs.write_exclusive_artifact(
                run.run_id,
                directions_attempt_path,
                "hierarchical_planning_stage_attempt",
                attempt.attempt_id,
                attempt,
            )
            method = getattr(planner, "propose_directions", None)
            if not callable(method):
                raise HierarchicalPlanningBlocked(
                    ("planning provider does not support research direction generation",)
                )
            directions = build_research_direction_set(
                planning_input, method(planning_input), planner
            )
            report = validate_research_direction_set(directions, planning_input)
            if not report.valid:
                raise HierarchicalPlanningBlocked(
                    tuple(item.code for item in report.issues)
                )
            self.runs.write_immutable_artifact(
                run.run_id,
                directions_path,
                "research_direction_set",
                directions.direction_set_id,
                directions,
            )
        direction_report = validate_research_direction_set(directions, planning_input)
        if not direction_report.valid:
            raise HierarchicalPlanningBlocked(
                tuple(item.code for item in direction_report.issues)
            )

        triage_path = f"{root}/triage.yaml"
        triage_attempt_path = f"{root}/attempts/triage.yaml"
        if existing(triage_path):
            triage = self.runs.load_artifact(
                run.run_id, triage_path, DirectionTriageRecord
            )
        else:
            if existing(triage_attempt_path):
                raise HierarchicalPlanningBlocked(
                    ("direction triage was started but no complete output was saved",)
                )
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
                raise HierarchicalPlanningBlocked(
                    ("planning provider does not support direction triage",)
                )
            triage = build_direction_triage_record(
                planning_input,
                directions,
                method(planning_input, directions),
                planner,
            )
            triage_report = validate_direction_triage(
                triage, directions, planning_input
            )
            if not triage_report.valid:
                raise HierarchicalPlanningBlocked(
                    tuple(item.code for item in triage_report.issues)
                )
            self.runs.write_immutable_artifact(
                run.run_id,
                triage_path,
                "direction_triage_record",
                triage.triage_id,
                triage,
            )
        triage_report = validate_direction_triage(triage, directions, planning_input)
        if not triage_report.valid:
            raise HierarchicalPlanningBlocked(
                tuple(item.code for item in triage_report.issues)
            )
        human_choice_items = unresolved_human_choice_items(triage)
        if human_choice_items:
            raise HierarchicalPlanningBlocked(human_choice_items)
        if not any(
            item.disposition == DirectionDisposition.RETAIN
            for item in triage.dispositions
        ):
            raise HierarchicalPlanningBlocked(
                ("direction triage retained no direction for plan expansion",)
            )
        return directions, triage

    def _hierarchical_compilation(
        self,
        run: ScientificProblemRun,
        planning_input,
        evidence,
        planner,
    ) -> CompilationResult:
        root = self._hierarchical_root(run.run_id)

        def existing(path: str) -> bool:
            return self.runs.resolve_artifact_path(run.run_id, path).exists()

        directions, triage = self._hierarchical_direction_triage(
            run, planning_input, planner
        )

        expansion_path = f"{root}/expansion.yaml"
        expansion_attempt_path = f"{root}/attempts/expansion.yaml"
        if existing(expansion_path):
            expansion = self.runs.load_artifact(
                run.run_id, expansion_path, HierarchicalPlanningExpansion
            )
        else:
            if existing(expansion_attempt_path):
                raise HierarchicalPlanningBlocked(
                    ("plan expansion was started but no complete output was saved",)
                )
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
                raise HierarchicalPlanningBlocked(
                    ("planning provider does not support retained direction expansion",)
                )
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
        expansion_report = validate_hierarchical_expansion(
            expansion, directions, triage, planning_input
        )
        if not expansion_report.valid:
            raise HierarchicalPlanningBlocked(
                tuple(item.code for item in expansion_report.issues)
            )
        return ScientificProblemCompiler(
            planner, evidence_repository=evidence
        ).compile_proposal(planning_input, expansion.planning_proposal)

    def _load_revision_compilation(
        self,
        chain: PlanRevisionChain,
        planning_input,
        evidence,
    ) -> CompilationResult:
        current = chain.rounds[-1]
        proposal = self._load_bound_artifact(
            chain.run_id, current.planning_proposal, PlanningProposalSet
        )
        plan = self._load_bound_artifact(
            chain.run_id, current.candidate_plan, ScientificQuestionPlan
        )
        receipt = self._load_bound_artifact(
            chain.run_id, current.compilation_receipt, PlanCompilationReceipt
        )
        validation = self._load_bound_artifact(
            chain.run_id, current.validation_record, PlanValidationRecord
        )
        proposal_report = validate_planning_proposal_set(proposal, planning_input)
        plan_report = validate_question_plan(
            plan, planning_input.scientific_capabilities, evidence
        )
        if not proposal_report.valid:
            raise ValueError("stored revision planning proposal failed validation")
        if not validate_plan_compilation_receipt(plan, receipt).valid:
            raise ValueError("stored revision compilation receipt failed validation")
        if (
            validation.plan_id != plan.plan_id
            or validation.plan_content_hash != content_hash(plan)
        ):
            raise ValueError("stored revision validation record is stale")
        return CompilationResult(
            candidates=(plan,),
            reports=(proposal_report, plan_report),
            proposal_set=proposal,
            compilation_receipts=(receipt,),
        )

    def _record_round_approval(
        self,
        chain: PlanRevisionChain,
        run: ScientificProblemRun,
    ) -> PlanRevisionChain:
        current = chain.rounds[-1]
        prefix = f"{self._round_prefix(chain.run_id, current.round_index)}/approval"
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
                parse_approval_review_input(
                    load_data(self.runs.resolve_artifact_path(chain.run_id, path))
                )
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
        self._save_revision_chain(updated)
        return updated

    def _revision_block(
        self,
        run: ScientificProblemRun,
        chain: PlanRevisionChain,
        category: str,
        message: str,
    ) -> WorkflowResult:
        payload = chain.model_dump(mode="python", exclude={"content_hash"})
        payload["termination_reason"] = f"{category}:{message}"
        updated_chain = _make_revision_chain(payload)
        self._save_revision_chain(updated_chain)
        blocked = _update_run(
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

    def _run_revision_cycle(
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
        llm_endpoint,
        llm_model,
        llm_api_key,
        temperature,
        max_attempts,
    ) -> ScientificProblemRun:
        current_run = run
        current_chain = chain
        while True:
            if current_run.status == ScientificProblemRunStatus.REQUIRES_REPLANNING:
                return current_run
            current_round = current_chain.rounds[-1]
            if current_round.approval_verdict is None:
                return current_run
            plan = self._load_bound_artifact(
                current_run.run_id,
                current_round.candidate_plan,
                ScientificQuestionPlan,
            )
            validation = self._load_bound_artifact(
                current_run.run_id,
                current_round.validation_record,
                PlanValidationRecord,
            )
            proposal = self._load_bound_artifact(
                current_run.run_id,
                current_round.planning_proposal,
                PlanningProposalSet,
            )
            compilation_receipt = self._load_bound_artifact(
                current_run.run_id,
                current_round.compilation_receipt,
                PlanCompilationReceipt,
            )
            review_input = self._load_approval_input_binding(
                current_run.run_id,
                current_round.approval_review_input,
            )
            review = self._load_bound_artifact(
                current_run.run_id,
                current_round.approval_review_record,
                ApprovalReviewRecord,
            )
            verdict = self._load_bound_artifact(
                current_run.run_id,
                current_round.approval_verdict,
                ApprovalVerdict,
            )
            receipt = self._load_bound_artifact(
                current_run.run_id,
                current_round.approval_receipt,
                IndependentApprovalReceipt,
            )
            gate = self._load_bound_artifact(
                current_run.run_id, current_round.gate, GateVerdict
            )
            if not validate_independent_approval_chain(
                plan, verdict, review_input, review, receipt
            ).valid or (
                gate.candidate_id != plan.plan_id
                or gate.candidate_content_hash != content_hash(plan)
                or gate.approval_verdict_hash != content_hash(verdict)
                or gate.independent_approval_receipt_hash != receipt.content_hash
            ):
                return self._revision_block(
                    current_run,
                    current_chain,
                    "INVALID_TRIGGER_APPROVAL_CHAIN",
                    "revision trigger approval chain is stale, cross-plan, or tampered",
                ).run
            if verdict.decision != ApprovalDecision.REQUEST_REVISION:
                return current_run
            attempts = self._load_revision_attempts(current_run.run_id)
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
                    return self._revision_block(
                        current_run,
                        current_chain,
                        "REVISION_ATTEMPT_UNCERTAIN",
                        (
                            "a revision provider call was claimed but has no complete local "
                            "outcome; ordinary resume will not invoke it again"
                        ),
                    ).run
                completed_record = self._load_bound_artifact(
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
                    f"{self._attempt_prefix(run.run_id, start.attempt_index)}/outcome.yaml",
                    "plan_revision_attempt_outcome",
                    start.attempt_id,
                    outcome,
                )
                attempts = (*attempts[:-1], (start, outcome))
            if len(attempts) >= current_chain.max_revisions:
                return self._revision_block(
                    current_run,
                    current_chain,
                    "REVISION_BUDGET_EXHAUSTED",
                    "independent review still requests revision after all started attempts",
                ).run
            block_reason = automatic_revision_block_reason(
                plan, review, verdict, validation
            )
            if block_reason is not None:
                return self._revision_block(
                    current_run,
                    current_chain,
                    "REVISION_REQUIRES_EXTERNAL_ACTION",
                    block_reason,
                ).run
            validation_report = validate_question_plan(
                plan, planning_input.scientific_capabilities, evidence
            )
            candidate_key = next(
                candidate.candidate_key
                for candidate in proposal.candidates
                if PlanMaterializer().materialize_candidate(
                    candidate, proposal, planning_input
                ).plan_id
                == plan.plan_id
            )
            attempt: PlanRevisionAttemptStart | None = None
            attempt_claimed = False
            try:
                carried_feedback = ()
                if (
                    isinstance(review_input, RevisionApprovalReviewInput)
                    and isinstance(review.response, RevisionApprovalLLMResponse)
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
                prefix = self._round_prefix(current_run.run_id, round_index)
                input_binding = self.runs.write_immutable_artifact(
                    current_run.run_id,
                    f"{prefix}/revision-input.yaml",
                    "plan_revision_input",
                    revision_input.revision_input_id,
                    revision_input,
                )
                attempt_index = len(attempts) + 1
                provider_binding = _provider_binding("planning", planner)
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
                attempt_id = (
                    f"plan-revision-attempt-{content_hash(attempt_identity)[:24]}"
                )
                attempt_payload = {"attempt_id": attempt_id, **attempt_identity}
                attempt = PlanRevisionAttemptStart(
                    **attempt_payload,
                    content_hash=content_hash(attempt_payload),
                )
                self.runs.write_exclusive_artifact(
                    current_run.run_id,
                    f"{self._attempt_prefix(current_run.run_id, attempt_index)}/start.yaml",
                    "plan_revision_attempt_start",
                    attempt.attempt_id,
                    attempt,
                )
                attempt_claimed = True
                revision = ScientificProblemCompiler(
                    planner, evidence_repository=evidence
                ).revise(revision_input)
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
                        "failure_message": _sanitize_failure(error),
                    }
                    outcome = PlanRevisionAttemptOutcome(
                        **outcome_payload,
                        content_hash=content_hash(outcome_payload),
                    )
                    self.runs.write_immutable_artifact(
                        current_run.run_id,
                        f"{self._attempt_prefix(current_run.run_id, attempt.attempt_index)}/outcome.yaml",
                        "plan_revision_attempt_outcome",
                        attempt.attempt_id,
                        outcome,
                    )
                return self._revision_block(
                    current_run,
                    current_chain,
                    "REVISION_OUTPUT_REJECTED",
                    _sanitize_failure(error),
                ).run

            revised_plan = revision.plan
            validation_report = validate_question_plan(
                revised_plan, planning_input.scientific_capabilities, evidence
            )
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
            chain_payload = current_chain.model_dump(
                mode="python", exclude={"content_hash"}
            )
            chain_payload.update(
                revisions_used=round_index,
                rounds=(*current_chain.rounds, next_round),
                termination_reason=None,
            )
            current_chain = _make_revision_chain(chain_payload)
            self._save_revision_chain(current_chain)
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
                f"{self._attempt_prefix(current_run.run_id, attempt.attempt_index)}/outcome.yaml",
                "plan_revision_attempt_outcome",
                attempt.attempt_id,
                outcome,
            )
            current_run = _update_run(
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
                providers=self._replace_provider(
                    current_run.providers, _provider_binding("planning", planner)
                ),
            )
            self.runs.save(current_run)
            current_run = self._approve(
                current_run,
                context,
                packet,
                planning_input,
                repositories,
                evidence,
                approval_provider,
                revised_plan.plan_id,
                llm_endpoint,
                llm_model,
                llm_api_key,
                temperature,
                max_attempts,
                artifact_prefix=f"{prefix}/approval",
                revision_approval=(revision_input, revision_record, review),
            )
            if current_run.approval_receipt_id is None:
                return current_run
            current_chain = self._record_round_approval(current_chain, current_run)

    @staticmethod
    def _replace_provider(
        providers: tuple[ScientificRunProviderBinding, ...],
        binding: ScientificRunProviderBinding,
    ) -> tuple[ScientificRunProviderBinding, ...]:
        return tuple(item for item in providers if item.stage != binding.stage) + (binding,)

    @staticmethod
    def _planning_provider(name: str, **options: Any):
        if name == "mock":
            return MockPlanningProvider()
        if name != "llm":
            raise ValueError("planning provider must be 'mock' or 'llm'")
        endpoint = options["llm_endpoint"]
        model = options["llm_model"]
        if endpoint is None or model is None:
            raise ValueError("planning provider llm requires --llm-endpoint and --llm-model")
        return StructuredLLMPlanningProvider(
            HTTPJSONLLMTransport(endpoint, model, api_key=options["llm_api_key"]),
            temperature=options["temperature"],
            max_attempts=options["max_attempts"],
        )

    def _approve(
        self,
        run,
        context,
        packet,
        planning_input,
        repositories,
        evidence,
        provider_name,
        selected_candidate_id,
        llm_endpoint,
        llm_model,
        llm_api_key,
        temperature,
        max_attempts,
        artifact_prefix="approval",
        revision_approval=None,
    ) -> ScientificProblemRun:
        if selected_candidate_id is None:
            if len(run.candidate_plans) != 1:
                return _update_run(run, status=ScientificProblemRunStatus.AWAITING_APPROVAL)
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
        compilation_receipt = self.runs.load_artifact(
            run.run_id, receipt_binding.relative_path, PlanCompilationReceipt
        )
        resolver = ApprovalContextResolver(self.domain_loader)
        review_input = (
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
        if provider_name == "mock":
            provider = MockApprovalProvider()
        elif provider_name == "llm":
            if llm_endpoint is None or llm_model is None:
                raise ValueError("approval provider llm requires --llm-endpoint and --llm-model")
            provider = StructuredLLMApprovalProvider(
                HTTPJSONLLMTransport(llm_endpoint, llm_model, api_key=llm_api_key),
                temperature=temperature,
                max_attempts=max_attempts,
            )
        else:
            raise ValueError("approval provider must be 'none', 'mock', or 'llm'")
        try:
            result = IndependentApprovalService(
                provider, approver_id="independent-scientific-approver"
            ).review(review_input)
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
            return self._fail(run, "approval", error, retryable=True).run
        artifacts = (
            ("approval-review-input.yaml", "approval_review_input", review_input.review_input_id, review_input),
            ("approval-review.yaml", "approval_review_record", result.review.review_id, result.review),
            ("approval-verdict.yaml", "approval_verdict", result.verdict.verdict_id, result.verdict),
            ("independent-approval-receipt.yaml", "independent_approval_receipt", result.receipt.receipt_id, result.receipt),
            ("plan-gate.yaml", "gate_verdict", gate.gate_id, gate),
            ("project-trust-policy.yaml", "project_trust_policy", "project-trust-policy", policy),
        )
        for path, kind, identifier, value in artifacts:
            writer = (
                self.runs.write_immutable_artifact
                if artifact_prefix != "approval"
                else self.runs.write_artifact
            )
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
        approved = _update_run(
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
            providers=self._replace_provider(run.providers, _provider_binding("approval", provider)),
        )
        self.runs.save(approved)
        return approved

    def _fail(self, run, stage: str, error: Exception, *, retryable: bool) -> WorkflowResult:
        failed = _update_run(
            run,
            status=ScientificProblemRunStatus.FAILED,
            last_successful_status=run.status,
            failure=ScientificRunFailure(
                stage=stage,
                category=type(error).__name__,
                message=_sanitize_failure(error),
                retryable=retryable,
            ),
        )
        self.runs.save(failed)
        return WorkflowResult(failed)

    def _dry_run_report(self, run, repositories, context):
        knowledge_status = build_knowledge_status(self.knowledge_dir, self.state_dir, run.domain)
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
            "trusted_literature_count": knowledge_status["literature"]["trusted_documents"],
            "trusted_expert_opinion_count": knowledge_status["experts"]["trusted_opinions"],
            "trusted_expert_case_count": knowledge_status["experts"]["trusted_cases"],
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

    def export_downstream(
        self,
        run_id: str,
        output_root: Path,
        export_id: str,
        *,
        selected_candidate_id: str | None = None,
    ) -> Path:
        run = self.runs.get(run_id)
        if run.status not in {ScientificProblemRunStatus.APPROVED, ScientificProblemRunStatus.EXPORTED}:
            raise ValueError("downstream export requires an independently APPROVED run")
        if run.selected_candidate_id is None:
            raise ValueError("approved run has no selected candidate")
        revision_chain = self._load_revision_chain(run_id)
        if revision_chain is not None and revision_chain.revisions_used > 0:
            if selected_candidate_id != run.selected_candidate_id:
                raise ValueError(
                    "revised-plan export requires explicit selection of the final candidate ID"
                )
        elif selected_candidate_id not in {None, run.selected_candidate_id}:
            raise ValueError("selected export candidate does not match the approved plan")
        index = tuple(item.artifact_id for item in run.candidate_plans).index(run.selected_candidate_id)
        plan = self.runs.load_artifact(run_id, run.candidate_plans[index].relative_path, ScientificQuestionPlan)
        validation = self.runs.load_artifact(
            run_id, run.plan_validation_records[index].relative_path, PlanValidationRecord
        )
        compilation_receipt = self.runs.load_artifact(
            run_id, run.candidate_compilation_receipts[index].relative_path, PlanCompilationReceipt
        )
        approval_prefix = self._approval_prefix(run_id)
        review_input = parse_approval_review_input(
            load_data(
                self.runs.resolve_artifact_path(
                    run_id, f"{approval_prefix}/approval-review-input.yaml"
                )
            )
        )
        review = self.runs.load_artifact(
            run_id, f"{approval_prefix}/approval-review.yaml", ApprovalReviewRecord
        )
        verdict = self.runs.load_artifact(
            run_id, f"{approval_prefix}/approval-verdict.yaml", ApprovalVerdict
        )
        receipt = self.runs.load_artifact(
            run_id,
            f"{approval_prefix}/independent-approval-receipt.yaml",
            IndependentApprovalReceipt,
        )
        gate = self.runs.load_artifact(
            run_id, f"{approval_prefix}/plan-gate.yaml", GateVerdict
        )
        policy = self.runs.load_artifact(
            run_id, f"{approval_prefix}/project-trust-policy.yaml", ProjectTrustPolicy
        )
        _, evidence = self._knowledge(run.domain)
        path = GenericExportService(output_root, evidence).export(
            plan=plan,
            verdict=verdict,
            validation_record=validation,
            gate=gate,
            human_selected=True,
            adapter=FTAgentAdapter(),
            export_id=export_id,
            trust_policy=policy,
            compilation_receipt=compilation_receipt,
            review_input=review_input,
            review=review,
            receipt=receipt,
        )
        checksums = path / "checksums.json"
        updated = _update_run(
            run,
            status=ScientificProblemRunStatus.EXPORTED,
            export_path=str(path),
            export_hash=file_sha256(checksums),
        )
        self.runs.save(updated)
        return path


def build_knowledge_status(knowledge_dir: Path, state_dir: Path, domain: str = "base") -> dict[str, Any]:
    repositories = KnowledgeRepositories(knowledge_dir.resolve())
    pack = DomainPackLoader().load(domain)
    repositories.load_expert_cases(pack.expert_cases)
    repositories.load_workflow_patterns(pack.workflow_patterns)
    repositories.load_capabilities(pack.capabilities)
    evidence = CompositeEvidenceStore(
        KnowledgeEvidenceStore(knowledge_dir.resolve()),
        ProjectEvidenceStore(state_dir.resolve()),
    )
    validator = TrustedKnowledgeValidator(repositories, evidence)
    records = validator.record_index(repositories)
    current = validator.resolve_current_curations()
    broken: list[str] = []
    try:
        trusted = validator.validate()
        trusted_keys = set(trusted.trusted_records)
    except (FileNotFoundError, OSError, ValueError) as error:
        trusted_keys = set()
        broken.append(_sanitize_failure(error))
    linked_attributions = {
        item.attribution_id for item in repositories.expert_sources.list()
    }
    pending = tuple(
        f"{kind}:{record_id}"
        for (kind, record_id), record in sorted(records.items())
        if (
            kind in _CURATED_TYPES
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
    revision_chain: PlanRevisionChain | None = None
    if repository is not None:
        try:
            revision_chain = repository.load_artifact(
                run.run_id,
                "plan-revisions/revision-chain.yaml",
                PlanRevisionChain,
            )
        except FileNotFoundError:
            revision_chain = None
    if revision_chain is not None:
        attempt_states: list[dict[str, object]] = []
        run_root = repository.run_dir(run.run_id)
        evidence_replan_roots = sorted(
            run_root.glob("evidence-replan-*"),
            key=lambda path: int(path.name.removeprefix("evidence-replan-")),
        )
        revision_root = (
            evidence_replan_roots[-1].name
            if evidence_replan_roots
            and revision_chain.context_id == run.context_id
            else "plan-revisions"
        )
        attempts_relative_root = (
            f"{revision_root}/attempts"
            if revision_root == "plan-revisions"
            else revision_root
        )
        attempt_name_prefix = (
            "attempt-" if revision_root == "plan-revisions" else "a-"
        )
        attempts_root = repository.resolve_artifact_path(
            run.run_id, attempts_relative_root
        )
        if attempts_root.exists():
            attempt_directories = sorted(
                attempts_root.glob(f"{attempt_name_prefix}*"),
                key=lambda path: int(path.name.removeprefix(attempt_name_prefix)),
            )
            for expected_index, directory in enumerate(
                attempt_directories, start=1
            ):
                if directory.is_symlink() or directory.name != (
                    f"{attempt_name_prefix}{expected_index}"
                ):
                    raise ValueError("revision attempt indices are not contiguous")
                start = repository.load_artifact(
                    run.run_id,
                    f"{attempts_relative_root}/{attempt_name_prefix}{expected_index}"
                    "/start.yaml",
                    PlanRevisionAttemptStart,
                )
                outcome_path = directory / "outcome.yaml"
                outcome = (
                    repository.load_artifact(
                        run.run_id,
                        f"{attempts_relative_root}/{attempt_name_prefix}{expected_index}"
                        "/outcome.yaml",
                        PlanRevisionAttemptOutcome,
                    )
                    if outcome_path.exists()
                    else None
                )
                if outcome is not None and (
                    outcome.attempt_id != start.attempt_id
                    or outcome.attempt_start_hash != start.content_hash
                ):
                    raise ValueError("revision attempt outcome does not bind its start")
                attempt_states.append(
                    {
                        "attempt_index": expected_index,
                        "attempt_id": start.attempt_id,
                        "status": (
                            outcome.status.value
                            if outcome is not None
                            else PlanRevisionAttemptStatus.UNCERTAIN.value
                        ),
                    }
                )
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
        hierarchy_roots = ["hierarchical-planning"]
        run_root = repository.run_dir(run.run_id)
        hierarchy_roots.extend(
            path.name
            for path in sorted(
                run_root.glob("evidence-hierarchy-*"),
                key=lambda path: int(path.name.removeprefix("evidence-hierarchy-")),
            )
        )
        hierarchy_root = next(
            (
                root
                for root in reversed(hierarchy_roots)
                if repository.resolve_artifact_path(
                    run.run_id, f"{root}/directions.yaml"
                ).exists()
            ),
            hierarchy_roots[0],
        )
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
        "failure": run.failure.model_dump(mode="json") if run.failure else None,
    }


def render_scientific_run_markdown(run: ScientificProblemRun, repository: ScientificProblemRunRepository) -> str:
    knowledge = KnowledgeRepositories(Path(run.knowledge_dir))
    evidence_store = CompositeEvidenceStore(
        KnowledgeEvidenceStore(Path(run.knowledge_dir)),
        ProjectEvidenceStore(Path(run.state_dir)),
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
    try:
        revision_chain = repository.load_artifact(
            run.run_id,
            "plan-revisions/revision-chain.yaml",
            PlanRevisionChain,
        )
    except FileNotFoundError:
        revision_chain = None
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
