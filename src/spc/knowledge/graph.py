from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel

from ..models import (
    CurationStatus,
    KnowledgeCurationRecord,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    KnowledgeViewMode,
)
from ..repositories import KnowledgeRepositories, SourceEvidenceStore
from ..serialization import canonical_json_bytes, content_hash
from .trust import TrustedKnowledgeSelection, TrustedKnowledgeValidator


class KnowledgeGraphError(ValueError):
    pass


class KnowledgeGraphBuilder:
    def build(
        self,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore | None = None,
        *,
        view_mode: KnowledgeViewMode | str = KnowledgeViewMode.TRUSTED,
        domain: str | None = None,
        topic: str | None = None,
    ) -> KnowledgeGraph:
        mode = KnowledgeViewMode(view_mode)
        domain_filter = self._normalize_filter(domain, "domain")
        topic_filter = self._normalize_filter(topic, "topic")
        validator = TrustedKnowledgeValidator(repositories, evidence_store)
        all_records = validator.record_index(repositories)
        current = validator.resolve_current_curations()

        if mode == KnowledgeViewMode.TRUSTED:
            trusted = validator.validate()
            relations = trusted.relations
            allowed_keys = self._trusted_record_keys(trusted)
        else:
            allowed_keys = set(all_records)
            relations = repositories.relations.list()

        records = {
            key: record
            for key, record in all_records.items()
            if key in allowed_keys
        }
        self._validate_relation_endpoints(relations, records)
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
            or (
                key[0] != "knowledge_relation"
                and self._record_matches(
                    record,
                    domain=domain_filter,
                    topic=topic_filter,
                )
            )
        }
        nodes = tuple(
            self._node(
                record_type,
                record_id,
                record,
                current.get((record_type, record_id)),
            )
            for (record_type, record_id), record in sorted(selected_records.items())
        )
        node_ids = {
            (node.record_type, node.record_id): node.node_id for node in nodes
        }
        edges = tuple(
            KnowledgeGraphEdge(
                relation_id=relation.relation_id,
                relation_hash=relation.content_hash,
                subject_node_id=node_ids[(relation.subject_type, relation.subject_id)],
                predicate=relation.predicate,
                object_node_id=node_ids[(relation.object_type, relation.object_id)],
                evidence_refs=relation.evidence_refs,
                curation_status=self._curation_status(
                    current.get(("knowledge_relation", relation.relation_id))
                ),
            )
            for relation in sorted(selected_relations, key=lambda item: item.relation_id)
        )
        identity = {
            "view_mode": mode,
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
    def _trusted_record_keys(
        trusted: TrustedKnowledgeSelection,
    ) -> set[tuple[str, str]]:
        allowed = {
            ("literature_document", item.literature_id)
            for item in trusted.literature_documents
        }
        allowed.update(
            ("expert_profile", item.expert_id) for item in trusted.expert_profiles
        )
        allowed.update(
            ("expert_opinion", item.opinion_id)
            for item in trusted.expert_opinions
        )
        for relation in trusted.relations:
            allowed.add((relation.subject_type, relation.subject_id))
            allowed.add((relation.object_type, relation.object_id))
        return allowed

    @staticmethod
    def _validate_relation_endpoints(
        relations: tuple[BaseModel, ...],
        records: Mapping[tuple[str, str], BaseModel],
    ) -> None:
        for relation in relations:
            for endpoint in (
                (getattr(relation, "subject_type"), getattr(relation, "subject_id")),
                (getattr(relation, "object_type"), getattr(relation, "object_id")),
            ):
                if endpoint not in records:
                    raise KnowledgeGraphError(
                        "knowledge relation references an unavailable record: "
                        f"{endpoint[0]}:{endpoint[1]}"
                    )

    @staticmethod
    def _node(
        record_type: str,
        record_id: str,
        record: BaseModel,
        curation: KnowledgeCurationRecord | None,
    ) -> KnowledgeGraphNode:
        identity = {"record_type": record_type, "record_id": record_id}
        record_hash = getattr(record, "content_hash", None) or content_hash(record)
        return KnowledgeGraphNode(
            node_id=f"knowledge-node-{content_hash(identity)[:24]}",
            record_type=record_type,
            record_id=record_id,
            record_hash=record_hash,
            curation_status=KnowledgeGraphBuilder._curation_status(curation),
        )

    @classmethod
    def _relation_matches(
        cls,
        relation: BaseModel,
        records: Mapping[tuple[str, str], BaseModel],
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
    def _curation_status(
        curation: KnowledgeCurationRecord | None,
    ) -> CurationStatus | None:
        return curation.status if curation is not None else None

    @staticmethod
    def _normalize_filter(value: str | None, name: str) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{name} filter must not be blank")
        return normalized
