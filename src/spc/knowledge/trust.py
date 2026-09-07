from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import BaseModel

from ..models import (
    CurationStatus,
    ExpertOpinion,
    ExpertProfile,
    KnowledgeCurationRecord,
    KnowledgeRelation,
    LiteratureDocument,
)
from ..serialization import content_hash

if TYPE_CHECKING:
    from ..repositories import KnowledgeRepositories, SourceEvidenceStore


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
    ("knowledge_relation", "relations", "relation_id"),
)

CURATION_REQUIRED_TYPES = frozenset(
    {"literature_document", "expert_opinion", "knowledge_relation"}
)


class TrustedKnowledgeError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class TrustedKnowledgeSelection:
    literature_documents: tuple[LiteratureDocument, ...]
    expert_profiles: tuple[ExpertProfile, ...]
    expert_opinions: tuple[ExpertOpinion, ...]
    relations: tuple[KnowledgeRelation, ...]
    curations: tuple[KnowledgeCurationRecord, ...]
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord]


class TrustedKnowledgeValidator:
    def __init__(
        self,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore | None,
    ) -> None:
        self.repositories = repositories
        self.evidence_store = evidence_store
        self.records = self.record_index(repositories)

    @staticmethod
    def record_index(
        repositories: KnowledgeRepositories,
    ) -> dict[tuple[str, str], BaseModel]:
        records: dict[tuple[str, str], BaseModel] = {}
        for record_type, repository_name, identity_field in REPOSITORY_NODE_SPECS:
            repository: Any = getattr(repositories, repository_name)
            for record in repository.list():
                key = (record_type, getattr(record, identity_field))
                if key in records:
                    raise TrustedKnowledgeError(
                        "DUPLICATE_KNOWLEDGE_RECORD",
                        f"duplicate record {key[0]}:{key[1]}",
                    )
                records[key] = record
        return records

    def resolve_current_curations(
        self,
    ) -> Mapping[tuple[str, str], KnowledgeCurationRecord]:
        curations = self.repositories.curations.list()
        by_id = {item.curation_id: item for item in curations}
        groups: dict[tuple[str, str], list[KnowledgeCurationRecord]] = {}
        successors: dict[str, str] = {}
        for curation in curations:
            key = (curation.target_type, curation.target_id)
            target = self.records.get(key)
            if target is None:
                raise TrustedKnowledgeError(
                    "UNKNOWN_CURATION_TARGET",
                    f"curation target does not exist: {key[0]}:{key[1]}",
                )
            if curation.target_hash != self._record_hash(target):
                raise TrustedKnowledgeError(
                    "CURATION_TARGET_HASH_MISMATCH",
                    f"curation target hash is stale: {key[0]}:{key[1]}",
                )
            groups.setdefault(key, []).append(curation)
            predecessor_id = curation.supersedes_curation_id
            if predecessor_id is None:
                continue
            predecessor = by_id.get(predecessor_id)
            if predecessor is None:
                raise TrustedKnowledgeError(
                    "MISSING_CURATION_PREDECESSOR",
                    f"missing superseded curation {predecessor_id}",
                )
            if (predecessor.target_type, predecessor.target_id) != key:
                raise TrustedKnowledgeError(
                    "CURATION_TARGET_MISMATCH",
                    "a curation transition must retain the same target",
                )
            if predecessor_id in successors:
                raise TrustedKnowledgeError(
                    "AMBIGUOUS_CURATION_CHAIN",
                    f"curation {predecessor_id} has multiple successors",
                )
            successors[predecessor_id] = curation.curation_id

        current: dict[tuple[str, str], KnowledgeCurationRecord] = {}
        for key, group in groups.items():
            heads = [item for item in group if item.curation_id not in successors]
            if len(heads) != 1:
                raise TrustedKnowledgeError(
                    "AMBIGUOUS_CURATION_CHAIN",
                    f"expected one current curation for {key[0]}:{key[1]}",
                )
            self._assert_acyclic(heads[0], by_id)
            current[key] = heads[0]
        return MappingProxyType(current)

    def validate(self) -> TrustedKnowledgeSelection:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires a SourceEvidenceStore",
            )
        current = self.resolve_current_curations()
        accepted = {
            key: curation
            for key, curation in current.items()
            if curation.status == CurationStatus.ACCEPTED
        }
        for curation in accepted.values():
            self._verify_evidence_refs(curation.evidence_refs)

        documents = tuple(
            sorted(
                (
                    record
                    for record in self.repositories.literature_documents.list()
                    if ("literature_document", record.literature_id) in accepted
                ),
                key=lambda item: item.literature_id,
            )
        )
        opinions = tuple(
            sorted(
                (
                    record
                    for record in self.repositories.expert_opinions.list()
                    if ("expert_opinion", record.opinion_id) in accepted
                ),
                key=lambda item: item.opinion_id,
            )
        )
        relations = tuple(
            sorted(
                (
                    record
                    for record in self.repositories.relations.list()
                    if ("knowledge_relation", record.relation_id) in accepted
                ),
                key=lambda item: item.relation_id,
            )
        )

        for document in documents:
            self._validate_literature(document)
        for opinion in opinions:
            self._validate_opinion(opinion)
        for relation in relations:
            self._validate_relation(relation, current, set())

        profile_ids = {item.expert_id for item in opinions}
        for relation in relations:
            if relation.subject_type == "expert_profile":
                profile_ids.add(relation.subject_id)
            if relation.object_type == "expert_profile":
                profile_ids.add(relation.object_id)
        profiles = tuple(
            sorted(
                (
                    record
                    for record in self.repositories.expert_profiles.list()
                    if record.expert_id in profile_ids
                ),
                key=lambda item: item.expert_id,
            )
        )
        accepted_curations = tuple(
            sorted(accepted.values(), key=lambda item: item.curation_id)
        )
        return TrustedKnowledgeSelection(
            literature_documents=documents,
            expert_profiles=profiles,
            expert_opinions=opinions,
            relations=relations,
            curations=accepted_curations,
            current_curations=current,
        )

    def _validate_literature(self, document: LiteratureDocument) -> None:
        try:
            source = self.evidence_store.source_records.get(
                f"{document.source_id}--{document.source_version}"
            )
            if (source.source_id, source.version) != (
                document.source_id,
                document.source_version,
            ):
                raise ValueError("source identity mismatch")
            self.evidence_store.verify_source_integrity(source)
        except Exception as error:
            raise TrustedKnowledgeError(
                "INVALID_LITERATURE_SOURCE",
                f"literature source is unavailable or invalid: {document.literature_id}",
            ) from error

    def _validate_opinion(self, opinion: ExpertOpinion) -> None:
        self._require_record("expert_profile", opinion.expert_id)
        self._verify_evidence_refs(opinion.evidence_refs)
        for claim_id in opinion.related_claim_refs:
            self._require_record("source_claim", claim_id)
        for workflow_id in opinion.related_workflow_refs:
            self._require_record("workflow_pattern", workflow_id)
        if opinion.supersedes is not None:
            self._require_record("expert_opinion", opinion.supersedes)

    def _validate_relation(
        self,
        relation: KnowledgeRelation,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        seen: set[str],
    ) -> None:
        if relation.relation_id in seen:
            return
        seen.add(relation.relation_id)
        self._verify_evidence_refs(relation.evidence_refs)
        for key in (
            (relation.subject_type, relation.subject_id),
            (relation.object_type, relation.object_id),
        ):
            endpoint = self._require_record(*key)
            curation = current.get(key)
            if (
                key[0] in CURATION_REQUIRED_TYPES or curation is not None
            ) and (
                curation is None or curation.status != CurationStatus.ACCEPTED
            ):
                raise TrustedKnowledgeError(
                    "UNTRUSTED_RELATION_ENDPOINT",
                    f"accepted relation points to untrusted target {key[0]}:{key[1]}",
                )
            if isinstance(endpoint, LiteratureDocument):
                self._validate_literature(endpoint)
            elif isinstance(endpoint, ExpertOpinion):
                self._validate_opinion(endpoint)
            elif isinstance(endpoint, KnowledgeRelation):
                self._validate_relation(endpoint, current, seen)

    def _verify_evidence_refs(self, evidence_refs: tuple[str, ...]) -> None:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires a SourceEvidenceStore",
            )
        for evidence_id in evidence_refs:
            try:
                evidence = self.evidence_store.get(evidence_id)
                self.evidence_store.verify_evidence_integrity(evidence)
            except Exception as error:
                raise TrustedKnowledgeError(
                    "INVALID_KNOWLEDGE_EVIDENCE",
                    f"evidence is unavailable or invalid: {evidence_id}",
                ) from error

    def _require_record(self, record_type: str, record_id: str) -> BaseModel:
        record = self.records.get((record_type, record_id))
        if record is None:
            raise TrustedKnowledgeError(
                "UNKNOWN_KNOWLEDGE_ENDPOINT",
                f"record does not exist: {record_type}:{record_id}",
            )
        return record

    @staticmethod
    def _record_hash(record: BaseModel) -> str:
        return getattr(record, "content_hash", None) or content_hash(record)

    @staticmethod
    def _assert_acyclic(
        head: KnowledgeCurationRecord,
        by_id: Mapping[str, KnowledgeCurationRecord],
    ) -> None:
        seen: set[str] = set()
        current: KnowledgeCurationRecord | None = head
        while current is not None:
            if current.curation_id in seen:
                raise TrustedKnowledgeError(
                    "CURATION_CHAIN_CYCLE",
                    f"curation chain contains a cycle at {current.curation_id}",
                )
            seen.add(current.curation_id)
            predecessor_id = current.supersedes_curation_id
            current = by_id.get(predecessor_id) if predecessor_id is not None else None
