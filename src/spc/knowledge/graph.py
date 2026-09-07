from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from ..models import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..repositories import KnowledgeRepositories, ModelRepository
from ..serialization import canonical_json_bytes, content_hash


class KnowledgeGraphError(ValueError):
    pass


REPOSITORY_NODE_SPECS = (
    ("literature_document", "literature_documents", "literature_id"),
    ("expert_profile", "expert_profiles", "expert_id"),
    ("expert_opinion", "expert_opinions", "opinion_id"),
    ("expert_case", "expert_cases", "case_id"),
    ("workflow_pattern", "workflow_patterns", "pattern_id"),
    ("scientific_capability", "capabilities", "capability_id"),
    ("source_quote", "source_quotes", "quote_id"),
    ("source_claim", "source_claims", "claim_id"),
    ("reported_result", "reported_results", "result_id"),
    ("method_fact", "method_facts", "fact_id"),
    ("model_fact", "model_facts", "fact_id"),
)


class KnowledgeGraphBuilder:
    def build(
        self,
        repositories: KnowledgeRepositories,
        *,
        domain: str | None = None,
        topic: str | None = None,
    ) -> KnowledgeGraph:
        domain_filter = self._normalize_filter(domain, "domain")
        topic_filter = self._normalize_filter(topic, "topic")
        records = self._record_index(repositories)
        relations = repositories.relations.list()
        for relation in relations:
            for endpoint in (
                (relation.subject_type, relation.subject_id),
                (relation.object_type, relation.object_id),
            ):
                if endpoint not in records:
                    raise KnowledgeGraphError(
                        "knowledge relation references an unknown record: "
                        f"{endpoint[0]}:{endpoint[1]}"
                    )

        selected_relations = tuple(
            relation
            for relation in relations
            if self._relation_matches(
                relation,
                records,
                domain=domain_filter,
                topic=topic_filter,
            )
        )
        endpoint_keys = {
            endpoint
            for relation in selected_relations
            for endpoint in (
                (relation.subject_type, relation.subject_id),
                (relation.object_type, relation.object_id),
            )
        }
        selected_records = {
            key: record
            for key, record in records.items()
            if key in endpoint_keys
            or self._record_matches(
                record,
                domain=domain_filter,
                topic=topic_filter,
            )
        }
        nodes = tuple(
            self._node(record_type, record_id, record)
            for (record_type, record_id), record in sorted(selected_records.items())
        )
        node_ids = {
            (node.record_type, node.record_id): node.node_id for node in nodes
        }
        edges = tuple(
            KnowledgeGraphEdge(
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                subject_node_id=node_ids[
                    (relation.subject_type, relation.subject_id)
                ],
                predicate=relation.predicate,
                object_node_id=node_ids[(relation.object_type, relation.object_id)],
                evidence_refs=relation.evidence_refs,
            )
            for relation in sorted(
                selected_relations, key=lambda item: item.relation_id
            )
        )
        identity = {
            "domain_filter": domain_filter,
            "topic_filter": topic_filter,
            "nodes": nodes,
            "edges": edges,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        graph_id = f"knowledge-graph-{content_hash(identity)[:24]}"
        payload = {"graph_id": graph_id, **identity}
        return KnowledgeGraph(**payload, content_hash=content_hash(payload))

    @staticmethod
    def _record_index(
        repositories: KnowledgeRepositories,
    ) -> dict[tuple[str, str], BaseModel]:
        records: dict[tuple[str, str], BaseModel] = {}
        for record_type, repository_name, identity_field in REPOSITORY_NODE_SPECS:
            repository: ModelRepository[Any] = getattr(
                repositories, repository_name
            )
            for record in repository.list():
                key = (record_type, getattr(record, identity_field))
                if key in records:
                    raise KnowledgeGraphError(
                        f"duplicate knowledge graph record: {key[0]}:{key[1]}"
                    )
                records[key] = record
        return records

    @staticmethod
    def _node(
        record_type: str,
        record_id: str,
        record: BaseModel,
    ) -> KnowledgeGraphNode:
        identity = {"record_type": record_type, "record_id": record_id}
        record_hash = getattr(record, "content_hash", None) or content_hash(record)
        return KnowledgeGraphNode(
            node_id=f"knowledge-node-{content_hash(identity)[:24]}",
            record_type=record_type,
            record_id=record_id,
            record_hash=record_hash,
        )

    @classmethod
    def _relation_matches(
        cls,
        relation: BaseModel,
        records: dict[tuple[str, str], BaseModel],
        *,
        domain: str | None,
        topic: str | None,
    ) -> bool:
        if domain is not None and getattr(relation, "domain") != domain:
            return False
        if topic is None:
            return True
        endpoints: Iterable[BaseModel] = (
            records[(getattr(relation, "subject_type"), getattr(relation, "subject_id"))],
            records[(getattr(relation, "object_type"), getattr(relation, "object_id"))],
        )
        return cls._contains_topic(relation, topic) or any(
            cls._contains_topic(record, topic) for record in endpoints
        )

    @classmethod
    def _record_matches(
        cls,
        record: BaseModel,
        *,
        domain: str | None,
        topic: str | None,
    ) -> bool:
        if domain is not None:
            record_domain = getattr(record, "domain", None)
            expertise_domains = getattr(record, "expertise_domains", ())
            if record_domain != domain and domain not in expertise_domains:
                return False
        return topic is None or cls._contains_topic(record, topic)

    @staticmethod
    def _contains_topic(record: BaseModel, topic: str) -> bool:
        return topic.casefold() in canonical_json_bytes(record).decode("utf-8").casefold()

    @staticmethod
    def _normalize_filter(value: str | None, name: str) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{name} filter must not be blank")
        return normalized
