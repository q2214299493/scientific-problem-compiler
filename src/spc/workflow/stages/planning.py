from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...compiler import ScientificProblemCompiler
from ...domains import DomainPackLoader
from ...interpretation import ScientificEvidencePacketBuilder
from ...knowledge.trust import TrustedKnowledgeValidator
from ...models import (
    CurationStatus,
    PlanningProposalSet,
    ScientificProblemRunStatus,
    ScientificRunBlockingItem,
)
from ...planning import (
    PlanningContextResolver,
)
from ...repositories import (
    CompositeEvidenceStore,
    KnowledgeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
)
from ...retrieval import ScientificContextBuilder
from ...serialization import content_hash
from ...validators import (
    build_plan_validation_record,
    validate_question_plan,
)
from ..providers import bind_provider, interpretation_provider, replace_provider
from ..repository import ScientificProblemRunRepository
from ..resume import (
    ProviderOutcomeUncertain,
    reuse_compilation,
    reuse_context,
    reuse_evidence_packet,
    stored_planning_input,
)
from ..state import snapshot_hash, update_run

CURATED_TYPES = frozenset(
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


def open_knowledge(
    knowledge_dir: Path, state_dir: Path, domain: str, domain_loader: DomainPackLoader
) -> tuple[KnowledgeRepositories, CompositeEvidenceStore]:
    pack = domain_loader.load(domain)
    repositories = KnowledgeRepositories(knowledge_dir)
    repositories.load_expert_cases(pack.expert_cases)
    repositories.load_workflow_patterns(pack.workflow_patterns)
    repositories.load_capabilities(pack.capabilities)
    evidence = CompositeEvidenceStore(
        KnowledgeEvidenceStore(knowledge_dir),
        ProjectEvidenceStore(state_dir),
    )
    return repositories, evidence


def blocking_curations(
    repositories: KnowledgeRepositories,
    evidence: CompositeEvidenceStore,
    domain: str,
) -> tuple[ScientificRunBlockingItem, ...]:
    validator = TrustedKnowledgeValidator(repositories, evidence)
    records = validator.record_index(repositories)
    current = validator.resolve_current_curations()
    linked_attributions = {item.attribution_id for item in repositories.expert_sources.list()}
    pending: list[ScientificRunBlockingItem] = []
    for (record_type, record_id), record in sorted(records.items()):
        record_domain = getattr(record, "domain", domain)
        if record_domain not in {domain, "base"}:
            continue
        requires_curation = record_type in CURATED_TYPES
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
                message=(f"{record_type} {record_id} requires ACCEPTED curation"),
            )
        )
    return tuple(pending)


@dataclass
class PlanningStage:
    """Prepare trusted inputs, compile candidates and record their validation bindings."""

    runs: ScientificProblemRunRepository
    domain_loader: DomainPackLoader

    def retrieve(self, run, repositories, evidence):
        context = reuse_context(self.runs, run, repositories, evidence, self.domain_loader.load(run.domain).profile)
        if context is not None:
            return context
        return ScientificContextBuilder(self.domain_loader).build(
            run.original_request, run.domain, state_dir=Path(run.state_dir), knowledge_dir=Path(run.knowledge_dir)
        )

    def record_context(self, ready, context):
        if ready.context_hash != context.content_hash:
            self.runs.write_artifact(
                ready.run_id, "context.yaml", "scientific_context_packet", context.context_id, context
            )
        stale_context = ready.context_hash is not None and ready.context_hash != context.content_hash
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
        context_run = update_run(
            ready,
            **(invalidated if stale_context else {}),
            status=ScientificProblemRunStatus.CONTEXT_BUILT,
            knowledge_snapshot_id=context.knowledge_snapshot.snapshot_id,
            knowledge_snapshot_hash=snapshot_hash(context.knowledge_snapshot),
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
        return context_run

    def interpret(self, context_run, context, evidence):
        providers = context_run.providers
        packet = reuse_evidence_packet(self.runs, context_run, context, evidence)
        if packet is None:
            interpretation = interpretation_provider()
            providers = replace_provider(providers, bind_provider("interpretation", interpretation))
            packet = ScientificEvidencePacketBuilder(interpretation).build(context, evidence)
            self.runs.write_artifact(
                context_run.run_id, "evidence-packet.yaml", "scientific_evidence_packet", packet.packet_id, packet
            )
        interpreted = update_run(
            context_run,
            status=ScientificProblemRunStatus.INTERPRETED,
            evidence_packet_id=packet.packet_id,
            evidence_packet_hash=packet.content_hash,
            providers=providers,
        )
        self.runs.save(interpreted)
        return interpreted, packet

    def resolve_input(self, run, context, packet, repositories, evidence):
        if run.planning_input_hash is not None:
            saved = stored_planning_input(self.runs, run.run_id)
            if saved is None or (saved.planning_input_id, saved.content_hash) != (
                run.planning_input_id,
                run.planning_input_hash,
            ):
                raise ValueError("stored planning input does not match the run projection")
            if saved.context_hash == context.content_hash and saved.evidence_packet_hash == packet.content_hash:
                return saved
        return PlanningContextResolver(self.domain_loader).resolve(context, packet, repositories, evidence)

    def compile(self, run, planning_input, evidence, planner, max_revisions):
        completed = reuse_compilation(self.runs, run, planning_input, evidence)
        if completed is not None:
            return completed
        compiler = ScientificProblemCompiler(planner, evidence_repository=evidence)
        claimed = stored_planning_input(self.runs, run.run_id)
        if claimed is not None and claimed.content_hash == planning_input.content_hash:
            prefix = self.runs.round_prefix(run.run_id, 0) + "/" if max_revisions else ""
            try:
                proposal = self.runs.load_artifact(run.run_id, prefix + "planning-proposal.yaml", PlanningProposalSet)
            except FileNotFoundError as error:
                raise ProviderOutcomeUncertain("planning input was claimed without a complete proposal") from error
            if (proposal.provider_id, proposal.provider_version) != (planner.provider_id, planner.provider_version):
                raise ProviderOutcomeUncertain(
                    "stored proposal and configured planning provider have different identities"
                )
            return compiler.compile_proposal(planning_input, proposal)
        self.runs.write_artifact(
            run.run_id,
            "planning-input.yaml",
            "scientific_planning_input",
            planning_input.planning_input_id,
            planning_input,
        )
        return compiler.compile(planning_input)

    def record_candidates(
        self,
        interpreted,
        planning_input,
        evidence,
        planner,
        compilation,
        selected_candidate_id,
        revision_chain,
        max_plan_revisions,
    ):
        proposal = compilation.proposal_set
        if interpreted.planning_input_hash != planning_input.content_hash:
            self.runs.write_artifact(
                interpreted.run_id,
                "planning-input.yaml",
                "scientific_planning_input",
                planning_input.planning_input_id,
                planning_input,
            )
        if revision_chain is not None:
            current_round = revision_chain.rounds[-1]
            proposal_binding = current_round.planning_proposal
            candidate_bindings = [current_round.candidate_plan]
            receipt_bindings = [current_round.compilation_receipt]
            validation_bindings = [current_round.validation_record]
        elif interpreted.planning_proposal_hash == content_hash(proposal) and max_plan_revisions == 0:
            # A complete projection has already bound these exact compilation outputs.
            proposal_binding = None
            candidate_bindings = interpreted.candidate_plans
            receipt_bindings = interpreted.candidate_compilation_receipts
            validation_bindings = interpreted.plan_validation_records
        else:
            revision_enabled = max_plan_revisions > 0
            prefix = self.runs.round_prefix(interpreted.run_id, 0) if revision_enabled else ""
            proposal_path = f"{prefix}/planning-proposal.yaml" if prefix else "planning-proposal.yaml"
            writer = self.runs.write_immutable_artifact if revision_enabled else self.runs.write_artifact
            proposal_binding = writer(
                interpreted.run_id,
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
                    f"{prefix}/candidates/{plan.plan_id}.yaml" if prefix else f"candidates/{plan.plan_id}.yaml"
                )
                receipt_path = (
                    f"{prefix}/compilation-receipts/{receipt.receipt_id}.yaml"
                    if prefix
                    else f"compilation-receipts/{receipt.receipt_id}.yaml"
                )
                candidate_bindings.append(
                    writer(
                        interpreted.run_id,
                        candidate_path,
                        "scientific_question_plan",
                        plan.plan_id,
                        plan,
                    )
                )
                receipt_bindings.append(
                    writer(
                        interpreted.run_id,
                        receipt_path,
                        "plan_compilation_receipt",
                        receipt.receipt_id,
                        receipt,
                    )
                )
                validation = build_plan_validation_record(
                    plan,
                    validate_question_plan(plan, planning_input.scientific_capabilities, evidence),
                    validation_id=f"plan-validation-{content_hash(plan)[:24]}",
                )
                validation_path = (
                    f"{prefix}/validation-records/{validation.validation_id}.yaml"
                    if prefix
                    else f"validation-records/{validation.validation_id}.yaml"
                )
                validation_bindings.append(
                    writer(
                        interpreted.run_id,
                        validation_path,
                        "plan_validation_record",
                        validation.validation_id,
                        validation,
                    )
                )
        planned = update_run(
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
            providers=(
                interpreted.providers
                if interpreted.planning_proposal_hash == content_hash(proposal)
                else replace_provider(interpreted.providers, bind_provider("planning", planner))
            ),
        )
        self.runs.save(planned)
        return planned, proposal_binding
