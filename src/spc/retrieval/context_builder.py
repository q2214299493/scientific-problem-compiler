from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..domains import DomainPackLoader
from ..models import (
    EvidenceClassification,
    GroundedStatement,
    RetrievalManifest,
    ScientificContextPacket,
)
from ..repositories import KnowledgeRepositories, SourceEvidenceStore, initialize_state
from ..serialization import content_hash
from .capability_retriever import retrieve_capabilities
from .evidence_retriever import retrieve_evidence_spans
from .expert_case_retriever import retrieve_expert_cases
from .query_builder import build_retrieval_query
from .ranker import RETRIEVER_VERSION
from .workflow_retriever import retrieve_workflow_patterns


class ScientificContextBuilder:
    def __init__(self, domain_loader: DomainPackLoader | None = None) -> None:
        self.domain_loader = domain_loader or DomainPackLoader()

    def build(
        self,
        raw_request: str,
        domain: str,
        *,
        state_dir: Path,
        knowledge_dir: Path,
    ) -> ScientificContextPacket:
        pack = self.domain_loader.load(domain)
        initialize_state(state_dir, domain=domain)
        evidence_store = SourceEvidenceStore(state_dir)
        knowledge = KnowledgeRepositories(knowledge_dir)
        knowledge.load_expert_cases(pack.expert_cases)
        knowledge.load_workflow_patterns(pack.workflow_patterns)
        knowledge.load_capabilities(pack.capabilities)

        query = build_retrieval_query(raw_request, domain, pack.profile)
        snapshot = knowledge.create_snapshot(evidence_store, pack.profile)
        evidence_hits = retrieve_evidence_spans(query, evidence_store, pack.profile)
        expert_case_hits = retrieve_expert_cases(query, knowledge.expert_cases, pack.profile)
        workflow_hits = retrieve_workflow_patterns(
            query, knowledge.workflow_patterns, pack.profile
        )
        capability_hits = retrieve_capabilities(query, knowledge.capabilities, pack.profile)
        categorized_hits = (
            evidence_hits,
            expert_case_hits,
            workflow_hits,
            capability_hits,
        )
        result_ids = tuple(hit.hit_id for hits in categorized_hits for hit in hits)
        query_hash = content_hash(query)
        retrieval_binding = {
            "query_hash": query_hash,
            "knowledge_snapshot_id": snapshot.snapshot_id,
            "domain_pack_id": pack.profile.domain_id,
            "domain_pack_version": pack.profile.version,
            "retriever_version": RETRIEVER_VERSION,
            "result_ids": result_ids,
        }
        retrieval_manifest = RetrievalManifest(
            retrieval_id=f"retrieval-{content_hash(retrieval_binding)[:24]}",
            timestamp=datetime.now(timezone.utc),
            **retrieval_binding,
        )
        retrieved_statements = tuple(
            GroundedStatement(
                statement_id=f"retrieved-statement-{index}",
                text=evidence_store.get(hit.record_id).text,
                classification=EvidenceClassification.EVIDENCE,
                evidence_refs=(hit.record_id,),
            )
            for index, hit in enumerate(evidence_hits, start=1)
        )
        packet_payload = {
            "context_id": f"context-{retrieval_manifest.retrieval_id.removeprefix('retrieval-')}",
            "original_request": raw_request,
            "domain": domain,
            "retrieval_query": query,
            "evidence_hits": evidence_hits,
            "expert_case_hits": expert_case_hits,
            "workflow_pattern_hits": workflow_hits,
            "capability_hits": capability_hits,
            "retrieved_statements": retrieved_statements,
            "assumptions": (),
            "conflicting_evidence": (),
            "unknowns": (),
            "retrieval_manifest": retrieval_manifest,
            "knowledge_snapshot": snapshot,
        }
        return ScientificContextPacket(
            **packet_payload,
            content_hash=content_hash(packet_payload),
        )
