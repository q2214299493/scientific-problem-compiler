from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from ..knowledge.graph import KnowledgeGraphBuilder
from ..knowledge.k1f_authority import (
    K1FScientificAuthorityStatus,
    classify_k1f_scientific_authority,
)
from ..knowledge.trust import TrustedKnowledgeValidator
from ..models import (
    CurationStatus,
    DomainProfile,
    ExpertCase,
    ExpertOpinion,
    KnowledgeRetrievalContext,
    KnowledgeSnapshot,
    KnowledgeViewMode,
    LiteratureDocument,
    MethodFact,
    ModelFact,
    ReportedResult,
    RetrievalHit,
    RetrievalQuery,
    RetrievalSourceType,
    SourceClaim,
)
from ..repositories import EvidenceStore, KnowledgeRepositories
from ..serialization import canonical_json_bytes, content_hash
from .ranker import RETRIEVER_VERSION, score_text, source_type_allowed

RETRIEVAL_POLICY_VERSION = "trusted-knowledge-retrieval-1.0.0"

_LITERATURE_TYPES = frozenset(
    {
        "literature_document",
        "source_claim",
        "method_fact",
        "model_fact",
        "reported_result",
    }
)
_EXPERT_TYPES = frozenset({"expert_opinion", "expert_case"})
_SOURCE_TYPES = {
    "literature_document": RetrievalSourceType.LITERATURE_DOCUMENT,
    "source_claim": RetrievalSourceType.SOURCE_CLAIM,
    "method_fact": RetrievalSourceType.METHOD_FACT,
    "model_fact": RetrievalSourceType.MODEL_FACT,
    "reported_result": RetrievalSourceType.REPORTED_RESULT,
    "expert_profile": RetrievalSourceType.EXPERT_PROFILE,
    "expert_opinion": RetrievalSourceType.EXPERT_OPINION,
    "expert_case": RetrievalSourceType.EXPERT_CASE,
    "workflow_pattern": RetrievalSourceType.WORKFLOW_PATTERN,
    "scientific_capability": RetrievalSourceType.SCIENTIFIC_CAPABILITY,
}
_TYPE_WEIGHTS = {
    "literature_document": 1.0,
    "source_claim": 4.0,
    "method_fact": 3.0,
    "model_fact": 3.0,
    "reported_result": 4.0,
    "expert_opinion": 4.0,
    "expert_case": 6.0,
}


class KnowledgeRetrievalError(ValueError):
    pass


class PersistentKnowledgeRetriever:
    def retrieve(
        self,
        query: RetrievalQuery,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        profile: DomainProfile,
        snapshot: KnowledgeSnapshot,
        *,
        mode: KnowledgeViewMode | str = KnowledgeViewMode.TRUSTED,
        max_literature_hits: int = 20,
        max_expert_hits: int = 10,
        max_expert_cases: int = 5,
        graph_hops: int = 1,
        max_graph_hits: int = 40,
        excluded_hit_ids: frozenset[str] = frozenset(),
    ) -> KnowledgeRetrievalContext:
        retrieval_mode = KnowledgeViewMode(mode)
        if min(max_literature_hits, max_expert_hits, max_expert_cases, max_graph_hits) <= 0:
            raise KnowledgeRetrievalError("retrieval result caps must be positive")
        if graph_hops not in {0, 1, 2}:
            raise KnowledgeRetrievalError("graph_hops must be 0, 1, or 2")

        validator = TrustedKnowledgeValidator(repositories, evidence_store)
        trusted = validator.validate()
        all_records = validator.record_index(repositories)
        trusted_keys = set(trusted.trusted_records)
        current_curations = validator.resolve_current_curations()
        candidate_keys = trusted_keys if retrieval_mode == KnowledgeViewMode.TRUSTED else set(all_records)
        literature_metadata = self._literature_metadata(repositories)
        grounding_refs = self._grounding_refs(repositories, all_records)

        literature_hits = self._rank_records(
            query,
            profile,
            repositories,
            evidence_store,
            all_records,
            candidate_keys & {(kind, record_id) for kind, record_id in candidate_keys if kind in _LITERATURE_TYPES},
            trusted_keys,
            current_curations,
            literature_metadata,
            grounding_refs,
        )[:max_literature_hits]
        expert_ranked = self._rank_records(
            query,
            profile,
            repositories,
            evidence_store,
            all_records,
            candidate_keys & {(kind, record_id) for kind, record_id in candidate_keys if kind in _EXPERT_TYPES},
            trusted_keys,
            current_curations,
            literature_metadata,
            grounding_refs,
        )
        opinions = tuple(hit for hit in expert_ranked if hit.source_type == RetrievalSourceType.EXPERT_OPINION)[
            :max_expert_hits
        ]
        cases = tuple(hit for hit in expert_ranked if hit.source_type == RetrievalSourceType.EXPERT_CASE)[
            :max_expert_cases
        ]
        # Preserve a single deterministic cross-expert ranking after applying the
        # independent opinion/case caps.
        expert_hits = tuple(sorted((*opinions, *cases), key=lambda item: (-item.score, item.record_id)))

        graph = KnowledgeGraphBuilder().build(
            repositories,
            evidence_store,
            view_mode=retrieval_mode,
            domain=query.domain,
        )
        initial_hits = (*literature_hits, *expert_hits)
        expanded_hits = self._expand_graph(
            query,
            profile,
            repositories,
            evidence_store,
            graph,
            all_records,
            trusted_keys,
            current_curations,
            literature_metadata,
            grounding_refs,
            initial_hits,
            graph_hops=graph_hops,
            max_graph_hits=max_graph_hits,
            excluded_hit_ids=excluded_hit_ids,
        )
        retrieved_keys = {(hit.source_type.value, hit.record_id) for hit in (*initial_hits, *expanded_hits)}
        node_keys = {node.node_id: (node.record_type, node.record_id) for node in graph.nodes}
        conflict_relation_refs = tuple(
            sorted(
                edge.relation_id
                for edge in graph.edges
                if edge.predicate.value == "contradicts"
                and node_keys[edge.subject_node_id] in retrieved_keys
                and node_keys[edge.object_node_id] in retrieved_keys
            )
        )
        if retrieval_mode == KnowledgeViewMode.TRUSTED:
            self._validate_snapshot_binding(snapshot, (*literature_hits, *expert_hits, *expanded_hits))
        snapshot_hash = content_hash(snapshot.model_dump(mode="json", exclude={"created_at"}))
        identity = {
            "query": query,
            "knowledge_snapshot_id": snapshot.snapshot_id,
            "knowledge_snapshot_hash": snapshot_hash,
            "retrieval_mode": retrieval_mode,
            "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
            "max_literature_hits": max_literature_hits,
            "max_expert_hits": max_expert_hits,
            "max_expert_cases": max_expert_cases,
            "graph_hops": graph_hops,
            "max_graph_hits": max_graph_hits,
            "initial_literature_hits": literature_hits,
            "initial_expert_hits": expert_hits,
            "graph_expanded_hits": expanded_hits,
            "conflict_relation_refs": conflict_relation_refs,
        }
        context_id = f"knowledge-retrieval-{content_hash(identity)[:24]}"
        payload = {"retrieval_context_id": context_id, **identity}
        return KnowledgeRetrievalContext(**payload, content_hash=content_hash(payload))

    def _rank_records(
        self,
        query: RetrievalQuery,
        profile: DomainProfile,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        records: Mapping[tuple[str, str], BaseModel],
        keys: Iterable[tuple[str, str]],
        trusted_keys: set[tuple[str, str]],
        curations: Mapping[tuple[str, str], Any],
        literature_metadata: Mapping[str, tuple[tuple[str, str, str], ...]],
        grounding_refs: Mapping[str, tuple[str, ...]],
    ) -> tuple[RetrievalHit, ...]:
        hits = []
        for key in sorted(keys):
            record = records[key]
            source_type = _SOURCE_TYPES[key[0]]
            if not source_type_allowed(query, source_type):
                continue
            if not self._domain_matches(record, key, query.domain, literature_metadata):
                continue
            scored = score_text(
                query,
                self._search_text(record, key, literature_metadata),
                profile,
            )
            if scored is None:
                continue
            score, matched_terms, rationale = scored
            type_weight = _TYPE_WEIGHTS.get(key[0], 0.0)
            hits.append(
                self._hit(
                    key,
                    record,
                    evidence_store,
                    repositories,
                    score=score + type_weight,
                    matched_terms=matched_terms,
                    rationale=f"{rationale}; record type weight: {type_weight:g}",
                    trusted=key in trusted_keys,
                    curation=curations.get(key),
                    current_curations=curations,
                    grounding_refs=grounding_refs.get(key[1], ()),
                    score_components=(
                        f"lexical={score:g}",
                        f"record_type={type_weight:g}",
                    ),
                )
            )
        return tuple(sorted(hits, key=lambda item: (-item.score, item.record_id, item.hit_id)))

    def _expand_graph(
        self,
        query: RetrievalQuery,
        profile: DomainProfile,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        graph: Any,
        records: Mapping[tuple[str, str], BaseModel],
        trusted_keys: set[tuple[str, str]],
        curations: Mapping[tuple[str, str], Any],
        literature_metadata: Mapping[str, tuple[tuple[str, str, str], ...]],
        grounding_refs: Mapping[str, tuple[str, ...]],
        initial_hits: tuple[RetrievalHit, ...],
        *,
        graph_hops: int,
        max_graph_hits: int,
        excluded_hit_ids: frozenset[str],
    ) -> tuple[RetrievalHit, ...]:
        if graph_hops == 0 or not initial_hits:
            return ()
        node_keys = {node.node_id: (node.record_type, node.record_id) for node in graph.nodes}
        adjacency: dict[tuple[str, str], list[tuple[tuple[str, str], str, str]]] = defaultdict(list)
        for edge in graph.edges:
            subject = node_keys[edge.subject_node_id]
            obj = node_keys[edge.object_node_id]
            adjacency[subject].append((obj, edge.predicate.value, edge.relation_id))
            adjacency[obj].append((subject, f"inverse:{edge.predicate.value}", edge.relation_id))
        for neighbors in adjacency.values():
            neighbors.sort(key=lambda item: (item[1], item[0]))

        initial_by_key = {(hit.source_type.value, hit.record_id): hit for hit in initial_hits}
        excluded_keys = {tuple(hit_id.split(":", 1)) for hit_id in excluded_hit_ids if ":" in hit_id}
        visited = set(initial_by_key) | excluded_keys
        queue = deque((key, 0, hit.score, ()) for key, hit in sorted(initial_by_key.items(), key=lambda item: item[0]))
        expanded: list[RetrievalHit] = []
        while queue and len(expanded) < max_graph_hits:
            key, depth, parent_score, path = queue.popleft()
            if depth >= graph_hops:
                continue
            for neighbor, predicate, relation_id in adjacency.get(key, ()):
                if neighbor in visited or neighbor[0] not in _SOURCE_TYPES:
                    continue
                visited.add(neighbor)
                record = records[neighbor]
                if not self._domain_matches(record, neighbor, query.domain, literature_metadata):
                    continue
                next_path = (*path, f"{key[0]}:{key[1]} --{predicate}--> {neighbor[0]}:{neighbor[1]}")
                hit = self._hit(
                    neighbor,
                    record,
                    evidence_store,
                    repositories,
                    score=max(parent_score / 2.0, 0.000001),
                    matched_terms=(f"graph:{predicate}",),
                    rationale=f"trusted scientific graph expansion via {predicate}",
                    trusted=neighbor in trusted_keys,
                    curation=curations.get(neighbor),
                    current_curations=curations,
                    grounding_refs=grounding_refs.get(neighbor[1], ()),
                    score_components=(f"graph_depth={depth + 1}",),
                    graph_expanded=True,
                    expansion_path=next_path,
                    relation_refs=(relation_id,),
                )
                expanded.append(hit)
                queue.append((neighbor, depth + 1, hit.score, next_path))
                if len(expanded) >= max_graph_hits:
                    break
        return tuple(sorted(expanded, key=lambda item: (-item.score, item.record_id, item.hit_id)))

    def _hit(
        self,
        key: tuple[str, str],
        record: BaseModel,
        evidence_store: EvidenceStore,
        repositories: KnowledgeRepositories,
        *,
        score: float,
        matched_terms: tuple[str, ...],
        rationale: str,
        trusted: bool,
        curation: Any,
        current_curations: Mapping[tuple[str, str], Any],
        grounding_refs: tuple[str, ...],
        score_components: tuple[str, ...],
        graph_expanded: bool = False,
        expansion_path: tuple[str, ...] = (),
        relation_refs: tuple[str, ...] = (),
    ) -> RetrievalHit:
        evidence_refs = self._evidence_refs(record, repositories)
        source_refs: list[str] = []
        for evidence_id in evidence_refs:
            try:
                evidence = evidence_store.get_evidence(evidence_id)
                source = evidence_store.verify_evidence_integrity(evidence)
            except (FileNotFoundError, OSError, ValueError):
                if trusted:
                    raise KnowledgeRetrievalError(f"trusted retrieval evidence is invalid: {evidence_id}")
                continue
            source_refs.append(f"{source.source_id}@{source.version}")
        status = curation.status if curation is not None else None
        authority = "trusted_current" if trusted else (status.value if status is not None else "untrusted_or_uncurated")
        if not trusted and status == CurationStatus.ACCEPTED and key[0] in _LITERATURE_TYPES - {"literature_document"}:
            try:
                k1f_status = classify_k1f_scientific_authority(
                    key[0], key[1], repositories, evidence_store, current_curations
                )
            except (FileNotFoundError, OSError, ValueError):
                k1f_status = None
            if not trusted and k1f_status == K1FScientificAuthorityStatus.STALE:
                authority = "valid_but_stale"
            elif not trusted and k1f_status is None:
                authority = "invalid_provenance"
        return RetrievalHit(
            hit_id=f"{key[0]}:{key[1]}",
            source_type=_SOURCE_TYPES[key[0]],
            record_id=key[1],
            score=score,
            matched_terms=matched_terms,
            rationale=rationale,
            evidence_refs=evidence_refs,
            retriever_version=RETRIEVER_VERSION,
            record_hash=getattr(record, "content_hash", None) or content_hash(record),
            source_class=self._source_class(key[0]),
            source_refs=tuple(sorted(set(source_refs))),
            grounding_refs=tuple(sorted(set(grounding_refs))),
            relation_refs=tuple(sorted(set(relation_refs))),
            curation_status=status,
            authority_status=authority,
            score_components=score_components,
            graph_expanded=graph_expanded,
            expansion_path=expansion_path,
        )

    @staticmethod
    def _validate_snapshot_binding(
        snapshot: KnowledgeSnapshot,
        hits: tuple[RetrievalHit, ...],
    ) -> None:
        for hit in hits:
            expected = snapshot.trusted_record_hashes.get(f"{hit.source_type.value}:{hit.record_id}")
            if expected is None:
                typed_maps = {
                    RetrievalSourceType.EXPERT_CASE: snapshot.expert_case_hashes,
                    RetrievalSourceType.WORKFLOW_PATTERN: snapshot.workflow_pattern_hashes,
                    RetrievalSourceType.SCIENTIFIC_CAPABILITY: snapshot.capability_hashes,
                }
                expected = typed_maps.get(hit.source_type, {}).get(hit.record_id)
            if expected is None or expected != hit.record_hash:
                raise KnowledgeRetrievalError(
                    "trusted retrieval record is absent or stale in KnowledgeSnapshot: "
                    f"{hit.source_type.value}:{hit.record_id}"
                )

    @staticmethod
    def _evidence_refs(record: BaseModel, repositories: KnowledgeRepositories) -> tuple[str, ...]:
        refs = tuple(getattr(record, "evidence_refs", ()))
        if isinstance(record, ExpertCase) and record.opinion_refs:
            refs = (
                *refs,
                *(
                    evidence_id
                    for opinion_id in record.opinion_refs
                    for evidence_id in repositories.expert_opinions.get(opinion_id).evidence_refs
                ),
            )
        return tuple(sorted(set(refs)))

    @staticmethod
    def _source_class(record_type: str) -> str:
        if record_type == "expert_opinion":
            return "expert_opinion"
        if record_type == "expert_case":
            return "expert_case"
        if record_type in _LITERATURE_TYPES:
            return "literature_reported"
        return "supporting_context"

    @staticmethod
    def _search_text(
        record: BaseModel,
        key: tuple[str, str],
        literature_metadata: Mapping[str, tuple[tuple[str, str, str], ...]],
    ) -> str:
        if isinstance(record, LiteratureDocument):
            parts: Iterable[Any] = (
                record.title,
                *record.authors,
                record.journal or "",
                *record.topics,
                *record.keywords,
            )
        elif isinstance(record, SourceClaim):
            parts = (record.text, record.claim_type, record.epistemic_status.value)
        elif isinstance(record, MethodFact | ModelFact):
            parts = (record.text, canonical_json_bytes(record.attributes).decode("utf-8"))
        elif isinstance(record, ReportedResult):
            parts = (
                record.quantity,
                str(record.value),
                record.unit,
                canonical_json_bytes(record.system_context).decode("utf-8"),
                canonical_json_bytes(record.method_context).decode("utf-8"),
            )
        elif isinstance(record, ExpertOpinion):
            parts = (
                record.statement,
                record.topic,
                record.scope,
                record.rationale,
                *record.conditions,
            )
        elif isinstance(record, ExpertCase):
            parts = (
                record.vague_request,
                *record.translated_questions,
                record.rationale,
                record.original_wording or "",
                record.latent_concern or "",
                *record.atomic_questions,
                *record.good_question_formulations,
                record.resolution_pattern or "",
                *record.applicability,
            )
        else:
            parts = (canonical_json_bytes(record).decode("utf-8"),)
        titles = tuple(item[1] for item in literature_metadata.get(key[1], ()))
        return " ".join(str(item) for item in (*parts, *titles) if item)

    @staticmethod
    def _domain_matches(
        record: BaseModel,
        key: tuple[str, str],
        domain: str,
        literature_metadata: Mapping[str, tuple[tuple[str, str, str], ...]],
    ) -> bool:
        record_domain = getattr(record, "domain", None)
        if record_domain is not None:
            return record_domain == domain
        inferred = {item[2] for item in literature_metadata.get(key[1], ())}
        return not inferred or domain in inferred

    @staticmethod
    def _literature_metadata(
        repositories: KnowledgeRepositories,
    ) -> dict[str, tuple[tuple[str, str, str], ...]]:
        values: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        for compilation in repositories.literature_knowledge_compilations.list():
            compilation_input = repositories.literature_knowledge_inputs.get(compilation.compilation_input_id)
            document = repositories.literature_documents.get(compilation_input.literature_id)
            for record_id in (
                *compilation.source_claim_ids,
                *compilation.method_fact_ids,
                *compilation.model_fact_ids,
                *compilation.reported_result_ids,
            ):
                values[record_id].add((document.literature_id, document.title, document.domain))
        return {key: tuple(sorted(items)) for key, items in values.items()}

    @staticmethod
    def _grounding_refs(
        repositories: KnowledgeRepositories,
        records: Mapping[tuple[str, str], BaseModel],
    ) -> dict[str, tuple[str, ...]]:
        values: dict[str, set[str]] = defaultdict(set)
        for compilation in repositories.literature_knowledge_compilations.list():
            groundings = tuple(
                repositories.literature_knowledge_groundings.get(grounding_id)
                for grounding_id in compilation.grounding_hashes
            )
            for record_id in compilation.source_claim_ids:
                record = records.get(("source_claim", record_id))
                quote_ids = set(getattr(record, "source_quote_refs", ()))
                values[record_id].update(item.grounding_id for item in groundings if item.quote_id in quote_ids)
            for record_type, record_ids in (
                ("method_fact", compilation.method_fact_ids),
                ("model_fact", compilation.model_fact_ids),
                ("reported_result", compilation.reported_result_ids),
            ):
                for record_id in record_ids:
                    record = records.get((record_type, record_id))
                    evidence_ids = set(getattr(record, "evidence_refs", ()))
                    values[record_id].update(
                        item.grounding_id for item in groundings if item.evidence_id in evidence_ids
                    )
        expert_groundings = repositories.expert_knowledge_groundings.list()
        for opinion in repositories.expert_opinions.list():
            evidence_ids = set(opinion.evidence_refs)
            values[opinion.opinion_id].update(
                item.grounding_id for item in expert_groundings if item.evidence_id in evidence_ids
            )
        return {key: tuple(sorted(items)) for key, items in values.items()}


__all__ = [
    "KnowledgeRetrievalError",
    "PersistentKnowledgeRetriever",
    "RETRIEVAL_POLICY_VERSION",
]
