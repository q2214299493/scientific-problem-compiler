from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..domains import DomainPackLoader
from ..models import (
    EvidenceClassification,
    GroundedStatement,
    KnowledgeViewMode,
    RetrievalManifest,
    ScientificContextPacket,
    scientific_context_semantic_hash,
)
from ..repositories import (
    CompositeEvidenceStore,
    KnowledgeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
    initialize_state,
)
from ..serialization import content_hash
from .capability_retriever import retrieve_capabilities
from .evidence_retriever import retrieve_evidence_spans
from .knowledge_retriever import PersistentKnowledgeRetriever
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
        retrieval_mode: KnowledgeViewMode | str = KnowledgeViewMode.TRUSTED,
        max_literature_hits: int = 20,
        max_expert_hits: int = 10,
        max_expert_cases: int = 5,
        graph_hops: int = 1,
        max_graph_hits: int = 40,
    ) -> ScientificContextPacket:
        pack = self.domain_loader.load(domain)
        initialize_state(state_dir, domain=domain)
        evidence_store = CompositeEvidenceStore(
            KnowledgeEvidenceStore(knowledge_dir),
            ProjectEvidenceStore(state_dir),
        )
        knowledge = KnowledgeRepositories(knowledge_dir)
        knowledge.load_expert_cases(pack.expert_cases)
        knowledge.load_workflow_patterns(pack.workflow_patterns)
        knowledge.load_capabilities(pack.capabilities)

        query = build_retrieval_query(raw_request, domain, pack.profile)
        all_evidence_hits = retrieve_evidence_spans(
            query,
            evidence_store,
            pack.profile,
        )
        snapshot = knowledge.create_snapshot(evidence_store, pack.profile)
        evidence_hits = tuple(
            hit
            for hit in all_evidence_hits
            if hit.record_id in snapshot.evidence_span_hashes
        )
        workflow_hits = retrieve_workflow_patterns(
            query, knowledge.workflow_patterns, pack.profile
        )
        capability_hits = retrieve_capabilities(query, knowledge.capabilities, pack.profile)
        knowledge_context = PersistentKnowledgeRetriever().retrieve(
            query,
            knowledge,
            evidence_store,
            pack.profile,
            snapshot,
            mode=retrieval_mode,
            max_literature_hits=max_literature_hits,
            max_expert_hits=max_expert_hits,
            max_expert_cases=max_expert_cases,
            graph_hops=graph_hops,
            max_graph_hits=max_graph_hits,
            excluded_hit_ids=frozenset(
                hit.hit_id for hit in (*workflow_hits, *capability_hits)
            ),
        )
        literature_hits = knowledge_context.initial_literature_hits
        expert_opinion_hits = tuple(
            hit
            for hit in knowledge_context.initial_expert_hits
            if hit.source_type.value == "expert_opinion"
        )
        expert_case_hits = tuple(
            hit
            for hit in knowledge_context.initial_expert_hits
            if hit.source_type.value == "expert_case"
        )
        categorized_hits = (
            evidence_hits,
            literature_hits,
            expert_opinion_hits,
            expert_case_hits,
            workflow_hits,
            capability_hits,
            knowledge_context.graph_expanded_hits,
        )
        result_ids = tuple(hit.hit_id for hits in categorized_hits for hit in hits)
        result_hashes = tuple(content_hash(hit) for hits in categorized_hits for hit in hits)
        query_hash = content_hash(query)
        retrieval_binding = {
            "query_hash": query_hash,
            "knowledge_snapshot_id": snapshot.snapshot_id,
            "domain_pack_id": pack.profile.domain_id,
            "domain_pack_version": pack.profile.version,
            "retriever_version": RETRIEVER_VERSION,
            "result_ids": result_ids,
            "result_hashes": result_hashes,
            "retrieval_policy_hash": knowledge_context.content_hash,
        }
        retrieval_manifest = RetrievalManifest(
            retrieval_id=f"retrieval-{content_hash(retrieval_binding)[:24]}",
            timestamp=datetime.now(timezone.utc),
            **retrieval_binding,
        )
        retrieved_statements = list(
            GroundedStatement(
                statement_id=f"retrieved-statement-{index}",
                text=evidence_store.get_evidence(hit.record_id).text,
                classification=EvidenceClassification.EVIDENCE,
                evidence_refs=(hit.record_id,),
            )
            for index, hit in enumerate(evidence_hits, start=1)
        )
        included_evidence = {hit.record_id for hit in evidence_hits}
        knowledge_hits = (
            *literature_hits,
            *expert_opinion_hits,
            *expert_case_hits,
            *knowledge_context.graph_expanded_hits,
        )
        for evidence_id in tuple(
            dict.fromkeys(
                evidence_id
                for hit in knowledge_hits
                for evidence_id in hit.evidence_refs
                if evidence_id not in included_evidence
            )
        ):
            evidence = evidence_store.get_evidence(evidence_id)
            evidence_store.verify_evidence_integrity(evidence)
            retrieved_statements.append(
                GroundedStatement(
                    statement_id=f"retrieved-statement-{len(retrieved_statements) + 1}",
                    text=evidence.text,
                    classification=EvidenceClassification.EVIDENCE,
                    evidence_refs=(evidence_id,),
                )
            )
        packet_payload = {
            "context_id": f"context-{retrieval_manifest.retrieval_id.removeprefix('retrieval-')}",
            "original_request": raw_request,
            "domain": domain,
            "retrieval_query": query,
            "evidence_hits": evidence_hits,
            "literature_knowledge_hits": literature_hits,
            "expert_opinion_hits": expert_opinion_hits,
            "expert_case_hits": expert_case_hits,
            "workflow_pattern_hits": workflow_hits,
            "capability_hits": capability_hits,
            "graph_expanded_hits": knowledge_context.graph_expanded_hits,
            "knowledge_retrieval_context": knowledge_context,
            "retrieved_statements": tuple(retrieved_statements),
            "assumptions": (),
            "conflicting_evidence": knowledge_context.conflict_relation_refs,
            "unknowns": (),
            "retrieval_manifest": retrieval_manifest,
            "knowledge_snapshot": snapshot,
        }
        return ScientificContextPacket(
            **packet_payload,
            content_hash=scientific_context_semantic_hash(packet_payload),
        )
