from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, Field, model_validator

from .base import NonBlankStr, Sha256Str, StrictModel
from .evidence import GroundedStatement
from .knowledge import CurationStatus, KnowledgeSnapshot, KnowledgeViewMode
from .scientific import AssumptionRecord, UnknownRecord


class RetrievalSourceType(StrEnum):
    EVIDENCE_SPAN = "evidence_span"
    LITERATURE_DOCUMENT = "literature_document"
    SOURCE_CLAIM = "source_claim"
    METHOD_FACT = "method_fact"
    MODEL_FACT = "model_fact"
    REPORTED_RESULT = "reported_result"
    EXPERT_OPINION = "expert_opinion"
    EXPERT_PROFILE = "expert_profile"
    EXPERT_CASE = "expert_case"
    WORKFLOW_PATTERN = "workflow_pattern"
    SCIENTIFIC_CAPABILITY = "scientific_capability"


class RetrievalQuery(StrictModel):
    query_id: NonBlankStr
    raw_request: NonBlankStr
    domain: NonBlankStr
    concepts: tuple[NonBlankStr, ...] = ()
    system_terms: tuple[NonBlankStr, ...] = ()
    method_terms: tuple[NonBlankStr, ...] = ()
    desired_observables: tuple[NonBlankStr, ...] = ()
    evidence_types: tuple[RetrievalSourceType, ...] = ()
    exclusions: tuple[NonBlankStr, ...] = ()

    @model_validator(mode="after")
    def validate_query_id(self) -> RetrievalQuery:
        from ..serialization import content_hash

        expected = f"query-{content_hash(self.model_dump(mode='json', exclude={'query_id'}))[:24]}"
        if self.query_id != expected:
            raise ValueError("RetrievalQuery query_id is not content-bound")
        return self


class RetrievalHit(StrictModel):
    hit_id: NonBlankStr
    source_type: RetrievalSourceType
    record_id: NonBlankStr
    score: float = Field(ge=0, allow_inf_nan=False)
    matched_terms: tuple[NonBlankStr, ...] = Field(min_length=1)
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    retriever_version: NonBlankStr
    record_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_class: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_refs: tuple[NonBlankStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    grounding_refs: tuple[NonBlankStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    relation_refs: tuple[NonBlankStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    curation_status: CurationStatus | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    authority_status: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    score_components: tuple[NonBlankStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    graph_expanded: bool = Field(default=False, exclude_if=lambda value: not value)
    expansion_path: tuple[NonBlankStr, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


class KnowledgeRetrievalContext(StrictModel):
    retrieval_context_id: NonBlankStr
    query: RetrievalQuery
    knowledge_snapshot_id: NonBlankStr
    knowledge_snapshot_hash: Sha256Str
    retrieval_mode: KnowledgeViewMode
    retrieval_policy_version: NonBlankStr
    max_literature_hits: int = Field(gt=0)
    max_expert_hits: int = Field(gt=0)
    max_expert_cases: int = Field(gt=0)
    graph_hops: int = Field(ge=0, le=2)
    max_graph_hits: int = Field(gt=0)
    initial_literature_hits: tuple[RetrievalHit, ...] = ()
    initial_expert_hits: tuple[RetrievalHit, ...] = ()
    graph_expanded_hits: tuple[RetrievalHit, ...] = ()
    conflict_relation_refs: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> KnowledgeRetrievalContext:
        from ..serialization import content_hash

        if len(self.initial_literature_hits) > self.max_literature_hits:
            raise ValueError("literature retrieval cap exceeded")
        literature_types = {
            RetrievalSourceType.LITERATURE_DOCUMENT,
            RetrievalSourceType.SOURCE_CLAIM,
            RetrievalSourceType.METHOD_FACT,
            RetrievalSourceType.MODEL_FACT,
            RetrievalSourceType.REPORTED_RESULT,
        }
        if any(
            hit.source_type not in literature_types
            for hit in self.initial_literature_hits
        ):
            raise ValueError("literature retrieval contains a non-literature hit")
        opinions = tuple(
            hit
            for hit in self.initial_expert_hits
            if hit.source_type == RetrievalSourceType.EXPERT_OPINION
        )
        cases = tuple(
            hit
            for hit in self.initial_expert_hits
            if hit.source_type == RetrievalSourceType.EXPERT_CASE
        )
        if len(opinions) > self.max_expert_hits or len(cases) > self.max_expert_cases:
            raise ValueError("expert retrieval cap exceeded")
        if len(opinions) + len(cases) != len(self.initial_expert_hits):
            raise ValueError("expert retrieval contains a non-expert hit")
        if len(self.graph_expanded_hits) > self.max_graph_hits:
            raise ValueError("graph retrieval cap exceeded")
        all_hits = (
            *self.initial_literature_hits,
            *self.initial_expert_hits,
            *self.graph_expanded_hits,
        )
        if len({hit.hit_id for hit in all_hits}) != len(all_hits):
            raise ValueError("KnowledgeRetrievalContext hit IDs must be unique")
        if self.conflict_relation_refs != tuple(sorted(set(self.conflict_relation_refs))):
            raise ValueError("conflict relation refs must be unique and sorted")
        if self.retrieval_mode == KnowledgeViewMode.TRUSTED and any(
            hit.authority_status != "trusted_current" for hit in all_hits
        ):
            raise ValueError("trusted retrieval contains a nontrusted knowledge hit")
        if any(
            not hit.graph_expanded or not hit.expansion_path
            for hit in self.graph_expanded_hits
        ):
            raise ValueError("graph-expanded hits must preserve their expansion path")
        identity = self.model_dump(
            mode="json", exclude={"retrieval_context_id", "content_hash"}
        )
        expected_id = f"knowledge-retrieval-{content_hash(identity)[:24]}"
        if self.retrieval_context_id != expected_id:
            raise ValueError("KnowledgeRetrievalContext ID is not content-bound")
        payload = {"retrieval_context_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("KnowledgeRetrievalContext content_hash is invalid")
        return self


class RetrievalManifest(StrictModel):
    retrieval_id: NonBlankStr
    query_hash: Sha256Str
    domain_pack_id: NonBlankStr
    domain_pack_version: NonBlankStr
    knowledge_snapshot_id: NonBlankStr
    retriever_version: NonBlankStr
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result_ids: tuple[NonBlankStr, ...]
    result_hashes: tuple[Sha256Str, ...]
    retrieval_policy_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_retrieval_id(self) -> RetrievalManifest:
        from ..serialization import content_hash

        identity = self.model_dump(mode="json", exclude={"retrieval_id", "timestamp"})
        current_id = f"retrieval-{content_hash(identity)[:24]}"
        if self.retrieval_id != current_id:
            raise ValueError("RetrievalManifest retrieval_id is not content-bound")
        if len(self.result_ids) != len(self.result_hashes):
            raise ValueError("RetrievalManifest result IDs and hashes must have equal length")
        return self


def scientific_context_semantic_hash(value: Any) -> str:
    from ..serialization import content_hash, to_primitive

    payload = to_primitive(value)
    if not isinstance(payload, dict):
        raise TypeError("ScientificContextPacket semantic identity requires an object")
    payload.pop("content_hash", None)
    for optional_field in (
        "literature_knowledge_hits",
        "expert_opinion_hits",
        "graph_expanded_hits",
        "knowledge_retrieval_context",
    ):
        if not payload.get(optional_field):
            payload.pop(optional_field, None)
    retrieval_manifest = payload.get("retrieval_manifest")
    if isinstance(retrieval_manifest, dict):
        retrieval_manifest.pop("timestamp", None)
    knowledge_snapshot = payload.get("knowledge_snapshot")
    if isinstance(knowledge_snapshot, dict):
        knowledge_snapshot.pop("created_at", None)
    return content_hash(payload)


class ScientificContextPacket(StrictModel):
    context_id: NonBlankStr
    original_request: NonBlankStr
    domain: NonBlankStr
    retrieval_query: RetrievalQuery
    evidence_hits: tuple[RetrievalHit, ...] = ()
    literature_knowledge_hits: tuple[RetrievalHit, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    expert_opinion_hits: tuple[RetrievalHit, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    expert_case_hits: tuple[RetrievalHit, ...] = ()
    workflow_pattern_hits: tuple[RetrievalHit, ...] = ()
    capability_hits: tuple[RetrievalHit, ...] = ()
    graph_expanded_hits: tuple[RetrievalHit, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    knowledge_retrieval_context: KnowledgeRetrievalContext | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    retrieved_statements: tuple[GroundedStatement, ...] = Field(
        default=(),
        validation_alias=AliasChoices("retrieved_statements", "known_facts"),
    )
    assumptions: tuple[AssumptionRecord, ...] = ()
    conflicting_evidence: tuple[NonBlankStr, ...] = ()
    unknowns: tuple[UnknownRecord, ...] = ()
    retrieval_manifest: RetrievalManifest
    knowledge_snapshot: KnowledgeSnapshot
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_retrieval_bindings(self) -> ScientificContextPacket:
        from ..serialization import content_hash

        categorized = (
            (self.evidence_hits, RetrievalSourceType.EVIDENCE_SPAN),
            (
                self.literature_knowledge_hits,
                {
                    RetrievalSourceType.LITERATURE_DOCUMENT,
                    RetrievalSourceType.SOURCE_CLAIM,
                    RetrievalSourceType.METHOD_FACT,
                    RetrievalSourceType.MODEL_FACT,
                    RetrievalSourceType.REPORTED_RESULT,
                },
            ),
            (self.expert_opinion_hits, RetrievalSourceType.EXPERT_OPINION),
            (self.expert_case_hits, RetrievalSourceType.EXPERT_CASE),
            (self.workflow_pattern_hits, RetrievalSourceType.WORKFLOW_PATTERN),
            (self.capability_hits, RetrievalSourceType.SCIENTIFIC_CAPABILITY),
            (
                self.graph_expanded_hits,
                {
                    RetrievalSourceType.LITERATURE_DOCUMENT,
                    RetrievalSourceType.SOURCE_CLAIM,
                    RetrievalSourceType.METHOD_FACT,
                    RetrievalSourceType.MODEL_FACT,
                    RetrievalSourceType.REPORTED_RESULT,
                    RetrievalSourceType.EXPERT_PROFILE,
                    RetrievalSourceType.EXPERT_OPINION,
                    RetrievalSourceType.EXPERT_CASE,
                    RetrievalSourceType.WORKFLOW_PATTERN,
                    RetrievalSourceType.SCIENTIFIC_CAPABILITY,
                },
            ),
        )
        for hits, expected_type in categorized:
            allowed_types = (
                expected_type if isinstance(expected_type, set) else {expected_type}
            )
            if any(hit.source_type not in allowed_types for hit in hits):
                raise ValueError(f"context hit category does not match {expected_type}")
        result_ids = tuple(hit.hit_id for hits, _ in categorized for hit in hits)
        if result_ids != self.retrieval_manifest.result_ids:
            raise ValueError("retrieval manifest result IDs do not match context hits")
        result_hashes = tuple(content_hash(hit) for hits, _ in categorized for hit in hits)
        if self.retrieval_manifest.result_hashes != result_hashes:
            raise ValueError("retrieval manifest result hashes do not match context hits")
        if self.retrieval_manifest.knowledge_snapshot_id != self.knowledge_snapshot.snapshot_id:
            raise ValueError("retrieval manifest does not bind the included knowledge snapshot")
        if self.retrieval_query.domain != self.domain:
            raise ValueError("retrieval query domain does not match context domain")
        if self.original_request != self.retrieval_query.raw_request:
            raise ValueError("context request does not match RetrievalQuery raw_request")
        if self.retrieval_manifest.query_hash != content_hash(self.retrieval_query):
            raise ValueError("retrieval manifest query hash does not match RetrievalQuery")
        if self.retrieval_manifest.domain_pack_id != self.domain:
            raise ValueError("retrieval manifest Domain Pack does not match context domain")
        if any(
            hit.retriever_version != self.retrieval_manifest.retriever_version
            for hits, _ in categorized
            for hit in hits
        ):
            raise ValueError("retrieval hit version does not match retrieval manifest")
        if self.knowledge_retrieval_context is not None:
            retrieval_context = self.knowledge_retrieval_context
            if retrieval_context.query != self.retrieval_query:
                raise ValueError("knowledge retrieval context query mismatch")
            if retrieval_context.knowledge_snapshot_id != self.knowledge_snapshot.snapshot_id:
                raise ValueError("knowledge retrieval context snapshot mismatch")
            snapshot_hash = content_hash(
                self.knowledge_snapshot.model_dump(
                    mode="json", exclude={"created_at"}
                )
            )
            if retrieval_context.knowledge_snapshot_hash != snapshot_hash:
                raise ValueError("knowledge retrieval context snapshot hash mismatch")
            if retrieval_context.initial_literature_hits != self.literature_knowledge_hits:
                raise ValueError("knowledge retrieval literature hits mismatch")
            expected_expert_hits = (
                *self.expert_opinion_hits,
                *tuple(
                    hit
                    for hit in self.expert_case_hits
                    if hit.record_hash is not None
                ),
            )
            if tuple(
                sorted(retrieval_context.initial_expert_hits, key=lambda hit: hit.hit_id)
            ) != tuple(sorted(expected_expert_hits, key=lambda hit: hit.hit_id)):
                raise ValueError("knowledge retrieval expert hits mismatch")
            if retrieval_context.graph_expanded_hits != self.graph_expanded_hits:
                raise ValueError("knowledge retrieval graph hits mismatch")
            if (
                self.retrieval_manifest.retrieval_policy_hash
                != retrieval_context.content_hash
            ):
                raise ValueError("retrieval manifest does not bind retrieval policy")
            for hit in (
                *self.literature_knowledge_hits,
                *self.expert_opinion_hits,
                *self.expert_case_hits,
                *self.graph_expanded_hits,
            ):
                if hit.authority_status != "trusted_current":
                    continue
                expected = self.knowledge_snapshot.trusted_record_hashes.get(
                    f"{hit.source_type.value}:{hit.record_id}"
                )
                if expected is None:
                    legacy_maps = {
                        RetrievalSourceType.EXPERT_CASE: (
                            self.knowledge_snapshot.expert_case_hashes
                        ),
                        RetrievalSourceType.WORKFLOW_PATTERN: (
                            self.knowledge_snapshot.workflow_pattern_hashes
                        ),
                        RetrievalSourceType.SCIENTIFIC_CAPABILITY: (
                            self.knowledge_snapshot.capability_hashes
                        ),
                    }
                    expected = legacy_maps.get(hit.source_type, {}).get(hit.record_id)
                if expected is None or expected != hit.record_hash:
                    raise ValueError("trusted retrieval hit is not bound to KnowledgeSnapshot")
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("retrieval result IDs must be unique")
        if self.context_id != f"context-{self.retrieval_manifest.retrieval_id.removeprefix('retrieval-')}":
            raise ValueError("context ID does not bind the retrieval manifest")
        expected_hash = scientific_context_semantic_hash(self)
        legacy_payload = self.model_dump(mode="json", exclude={"content_hash"})
        legacy_payload["retrieval_manifest"].pop("result_hashes", None)
        legacy_current_hash = content_hash(legacy_payload)
        legacy_payload["known_facts"] = legacy_payload.pop("retrieved_statements")
        legacy_known_facts_hash = content_hash(legacy_payload)
        if self.content_hash not in {
            expected_hash,
            legacy_current_hash,
            legacy_known_facts_hash,
        }:
            raise ValueError("ScientificContextPacket content hash is invalid")
        return self
