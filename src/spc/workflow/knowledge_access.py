from __future__ import annotations

from pathlib import Path

from ..domains import DomainPackLoader
from ..knowledge.trust import TrustedKnowledgeValidator
from ..models import CurationStatus, ScientificRunBlockingItem
from ..repositories import (
    CompositeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
)

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
    repositories.evidence_store.prepare_directories(initialize=True)
    evidence = CompositeEvidenceStore(
        repositories.evidence_store,
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
