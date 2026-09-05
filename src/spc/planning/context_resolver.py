from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from ..domains import DomainPackLoader
from ..models import (
    ExpertCase,
    LiteratureWorkflowPattern,
    RequiredHumanDecision,
    RetrievalHit,
    ScientificCapability,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
)
from ..repositories import KnowledgeRepositories, ModelRepository
from ..serialization import content_hash

RESOLVER_VERSION = "planning-context-resolver-1.0.0"
RecordT = TypeVar("RecordT", ExpertCase, LiteratureWorkflowPattern, ScientificCapability)


class PlanningContextError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _resolve_records(
    hits: Iterable[RetrievalHit],
    repository: ModelRepository[RecordT],
    snapshot_hashes: dict[str, str],
    *,
    domain: str,
    record_kind: str,
) -> tuple[RecordT, ...]:
    resolved: list[RecordT] = []
    for hit in hits:
        try:
            record = repository.get(hit.record_id)
        except (FileNotFoundError, KeyError, OSError, ValueError) as error:
            raise PlanningContextError(
                "KNOWLEDGE_RECORD_NOT_FOUND",
                f"{record_kind} {hit.record_id} cannot be resolved",
            ) from error
        expected_hash = snapshot_hashes.get(hit.record_id)
        if expected_hash is None:
            raise PlanningContextError(
                "KNOWLEDGE_RECORD_NOT_IN_SNAPSHOT",
                f"{record_kind} {hit.record_id} is absent from KnowledgeSnapshot",
            )
        if content_hash(record) != expected_hash:
            raise PlanningContextError(
                f"STALE_{record_kind.upper()}",
                f"{record_kind} {hit.record_id} no longer matches KnowledgeSnapshot",
            )
        if record.domain != domain:
            raise PlanningContextError(
                "KNOWLEDGE_DOMAIN_MISMATCH",
                f"{record_kind} {hit.record_id} has domain {record.domain}, expected {domain}",
            )
        resolved.append(record)
    return tuple(resolved)


class PlanningContextResolver:
    def __init__(self, domain_loader: DomainPackLoader | None = None) -> None:
        self.domain_loader = domain_loader or DomainPackLoader()

    def resolve(
        self,
        context: ScientificContextPacket,
        evidence_packet: ScientificEvidencePacket,
        knowledge: KnowledgeRepositories,
    ) -> ScientificPlanningInput:
        if (
            evidence_packet.context_id != context.context_id
            or evidence_packet.context_hash != context.content_hash
        ):
            raise PlanningContextError(
                "CONTEXT_EVIDENCE_PACKET_MISMATCH",
                "ScientificEvidencePacket is not bound to the supplied ScientificContextPacket",
            )
        pack = self.domain_loader.load(context.domain)
        manifest = context.retrieval_manifest
        if manifest.domain_pack_id != context.domain or pack.profile.domain_id != context.domain:
            raise PlanningContextError(
                "DOMAIN_MISMATCH", "context and current Domain Pack identify different domains"
            )
        if manifest.domain_pack_version != pack.profile.version:
            raise PlanningContextError(
                "DOMAIN_PACK_VERSION_MISMATCH",
                "context was created with a different Domain Pack version",
            )
        if context.knowledge_snapshot.domain_profile_hash != content_hash(pack.profile):
            raise PlanningContextError(
                "STALE_DOMAIN_PROFILE", "current Domain Pack profile does not match KnowledgeSnapshot"
            )

        snapshot = context.knowledge_snapshot
        expert_cases = _resolve_records(
            context.expert_case_hits,
            knowledge.expert_cases,
            snapshot.expert_case_hashes,
            domain=context.domain,
            record_kind="expert_case",
        )
        workflow_patterns = _resolve_records(
            context.workflow_pattern_hits,
            knowledge.workflow_patterns,
            snapshot.workflow_pattern_hashes,
            domain=context.domain,
            record_kind="workflow_pattern",
        )
        capabilities = _resolve_records(
            context.capability_hits,
            knowledge.capabilities,
            snapshot.capability_hashes,
            domain=context.domain,
            record_kind="capability",
        )

        allowed_evidence_ids = tuple(hit.record_id for hit in context.evidence_hits)
        allowed_claim_ids = tuple(claim.claim_id for claim in evidence_packet.source_claims)
        allowed_capability_ids = tuple(item.capability_id for item in capabilities)
        decisions = tuple(
            RequiredHumanDecision(
                decision_id=f"decision-{content_hash({'gap_id': gap.gap_id})[:24]}",
                question=(
                    f"How should blocking evidence gap {gap.gap_id} be handled before release?"
                ),
                options=(
                    "Address it with an allowed scientific capability",
                    "Retain it as an explicit blocking limitation",
                ),
                required_before="candidate release",
            )
            for gap in evidence_packet.evidence_gaps
            if gap.blocking
        )
        evidence_bindings = {
            quote.evidence_ref: {
                "source_id": quote.source_id,
                "source_version": quote.source_version,
            }
            for quote in evidence_packet.source_quotes
        }
        provenance_manifest = {
            "resolver_version": RESOLVER_VERSION,
            "retrieval_id": manifest.retrieval_id,
            "knowledge_snapshot_id": snapshot.snapshot_id,
            "context_id": context.context_id,
            "context_hash": context.content_hash,
            "evidence_packet_id": evidence_packet.packet_id,
            "evidence_packet_hash": evidence_packet.content_hash,
            "expert_case_hashes": {
                item.case_id: content_hash(item) for item in expert_cases
            },
            "workflow_pattern_hashes": {
                item.pattern_id: content_hash(item) for item in workflow_patterns
            },
            "capability_hashes": {
                item.capability_id: content_hash(item) for item in capabilities
            },
            "evidence_bindings": evidence_bindings,
        }
        identity = {
            "original_request": context.original_request,
            "domain": context.domain,
            "domain_pack_version": pack.profile.version,
            "context_id": context.context_id,
            "context_hash": context.content_hash,
            "evidence_packet_id": evidence_packet.packet_id,
            "evidence_packet_hash": evidence_packet.content_hash,
            "source_quotes": evidence_packet.source_quotes,
            "source_claims": evidence_packet.source_claims,
            "reported_results": evidence_packet.reported_results,
            "method_facts": evidence_packet.method_facts,
            "model_facts": evidence_packet.model_facts,
            "evidence_assessments": evidence_packet.evidence_assessments,
            "conflict_sets": evidence_packet.conflict_sets,
            "comparison_constraints": evidence_packet.comparison_constraints,
            "evidence_gaps": evidence_packet.evidence_gaps,
            "unknowns": evidence_packet.unknowns,
            "assumption_candidates": evidence_packet.assumption_candidates,
            "expert_cases": expert_cases,
            "workflow_patterns": workflow_patterns,
            "scientific_capabilities": capabilities,
            "allowed_evidence_ids": allowed_evidence_ids,
            "allowed_claim_ids": allowed_claim_ids,
            "allowed_capability_ids": allowed_capability_ids,
            "required_human_decisions": decisions,
            "provenance_manifest": provenance_manifest,
        }
        planning_input_id = f"planning-input-{content_hash(identity)[:24]}"
        payload = {"planning_input_id": planning_input_id, **identity}
        return ScientificPlanningInput(**payload, content_hash=content_hash(payload))
