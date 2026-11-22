from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import Field, model_validator

from .base import (
    KnowledgeEntityType,
    NonBlankStr,
    Sha256Str,
    StrictModel,
)


class CurationStatus(StrEnum):
    MACHINE_EXTRACTED = "machine_extracted"
    HUMAN_REVIEWED = "human_reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class KnowledgeViewMode(StrEnum):
    TRUSTED = "trusted"
    AUDIT = "audit"


class KnowledgePredicate(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    USES_METHOD = "uses_method"
    USES_MODEL = "uses_model"
    REPORTS_RESULT = "reports_result"
    DERIVED_FROM = "derived_from"
    COMMENTS_ON = "comments_on"
    SUPERSEDES = "supersedes"
    APPLIES_TO = "applies_to"
    REQUIRES_CAPABILITY = "requires_capability"


class ExpertCase(StrictModel):
    case_id: str
    domain: str
    vague_request: NonBlankStr
    translated_questions: tuple[NonBlankStr, ...]
    positive: bool
    rationale: NonBlankStr
    evidence_refs: tuple[str, ...] = ()
    opinion_refs: tuple[NonBlankStr, ...] = ()
    original_wording: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    latent_concern: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    atomic_questions: tuple[NonBlankStr, ...] = ()
    good_question_formulations: tuple[NonBlankStr, ...] = ()
    wrong_formulations: tuple[NonBlankStr, ...] = ()
    answerability_conditions: tuple[NonBlankStr, ...] = ()
    required_evidence_types: tuple[NonBlankStr, ...] = ()
    baseline_guidance: tuple[NonBlankStr, ...] = ()
    common_misinterpretations: tuple[NonBlankStr, ...] = ()
    resolution_pattern: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    applicability: tuple[NonBlankStr, ...] = ()
    provenance_hash: Sha256Str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_compiled_case(self) -> ExpertCase:
        from ..serialization import content_hash

        tuple_fields = (
            "translated_questions",
            "evidence_refs",
            "opinion_refs",
            "atomic_questions",
            "good_question_formulations",
            "wrong_formulations",
            "answerability_conditions",
            "required_evidence_types",
            "baseline_guidance",
            "common_misinterpretations",
            "applicability",
        )
        for field_name in tuple_fields:
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"ExpertCase {field_name} must be unique")
        if not self.opinion_refs:
            return self
        required_values = (
            self.original_wording,
            self.latent_concern,
            self.atomic_questions,
            self.good_question_formulations,
            self.wrong_formulations,
            self.answerability_conditions,
            self.required_evidence_types,
            self.resolution_pattern,
            self.applicability,
            self.provenance_hash,
        )
        if any(not value for value in required_values):
            raise ValueError("compiled ExpertCase is missing reusable reasoning fields")
        identity = self.model_dump(
            mode="json",
            exclude={"case_id", "provenance_hash"},
            exclude_none=True,
        )
        expected_hash = content_hash(identity)
        if self.provenance_hash != expected_hash:
            raise ValueError("compiled ExpertCase provenance_hash is invalid")
        if self.case_id != f"expert-case-{expected_hash[:24]}":
            raise ValueError("compiled ExpertCase case_id is not content-bound")
        return self


class ExpertProfile(StrictModel):
    expert_id: NonBlankStr
    display_name: NonBlankStr
    affiliations: tuple[NonBlankStr, ...] = ()
    expertise_domains: tuple[NonBlankStr, ...] = ()
    expertise_topics: tuple[NonBlankStr, ...] = ()
    profile_source_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    organization: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    role: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_content_hash(self) -> ExpertProfile:
        from ..serialization import content_hash

        for field_name in (
            "affiliations",
            "expertise_domains",
            "expertise_topics",
            "profile_source_refs",
        ):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"ExpertProfile {field_name} must be unique")
        identity = self.model_dump(
            mode="json", exclude={"content_hash"}, exclude_none=True
        )
        if self.content_hash != content_hash(identity):
            raise ValueError("ExpertProfile content_hash is invalid")
        return self


class ExpertAttributionRecord(StrictModel):
    attribution_id: NonBlankStr
    expert_id: NonBlankStr
    source_id: NonBlankStr
    source_version: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    medium: NonBlankStr
    attribution_basis: NonBlankStr
    attribution_status: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    captured_at: datetime | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_provenance(self) -> ExpertAttributionRecord:
        from ..serialization import content_hash

        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("ExpertAttributionRecord evidence_refs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"attribution_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"expert-attribution-{content_hash(identity)[:24]}"
        if self.attribution_id != expected_id:
            raise ValueError("ExpertAttributionRecord attribution_id is not content-bound")
        payload = {"attribution_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExpertAttributionRecord content_hash is invalid")
        return self


class ExpertOpinion(StrictModel):
    opinion_id: NonBlankStr
    expert_id: NonBlankStr
    domain: NonBlankStr
    topic: NonBlankStr
    statement: NonBlankStr
    opinion_type: NonBlankStr
    scope: NonBlankStr
    rationale: NonBlankStr
    conditions: tuple[NonBlankStr, ...] = ()
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    attribution_refs: tuple[NonBlankStr, ...] = ()
    related_claim_refs: tuple[NonBlankStr, ...] = ()
    related_workflow_refs: tuple[NonBlankStr, ...] = ()
    supersedes: NonBlankStr | None = None
    authority_status: NonBlankStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_provenance(self) -> ExpertOpinion:
        from ..serialization import content_hash

        for field_name in (
            "conditions",
            "evidence_refs",
            "attribution_refs",
            "related_claim_refs",
            "related_workflow_refs",
        ):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"ExpertOpinion {field_name} must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"opinion_id", "content_hash"},
            exclude_none=True,
        )
        if not self.attribution_refs:
            identity.pop("attribution_refs")
        expected_id = f"expert-opinion-{content_hash(identity)[:24]}"
        if self.opinion_id != expected_id:
            raise ValueError("ExpertOpinion opinion_id is not content-bound")
        if self.supersedes == self.opinion_id:
            raise ValueError("ExpertOpinion cannot supersede itself")
        payload = {"opinion_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExpertOpinion content_hash is invalid")
        return self


class KnowledgeRelation(StrictModel):
    relation_id: NonBlankStr
    subject_type: KnowledgeEntityType
    subject_id: NonBlankStr
    predicate: KnowledgePredicate
    object_type: KnowledgeEntityType
    object_id: NonBlankStr
    domain: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    rationale: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_endpoints(self) -> KnowledgeRelation:
        from ..serialization import content_hash

        if (self.subject_type, self.subject_id) == (
            self.object_type,
            self.object_id,
        ):
            raise ValueError("KnowledgeRelation endpoints must be distinct")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("KnowledgeRelation evidence_refs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"relation_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"knowledge-relation-{content_hash(identity)[:24]}"
        if self.relation_id != expected_id:
            raise ValueError("KnowledgeRelation relation_id is not content-bound")
        payload = {"relation_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("KnowledgeRelation content_hash is invalid")
        return self


class KnowledgeCurationRecord(StrictModel):
    curation_id: NonBlankStr
    target_type: KnowledgeEntityType
    target_id: NonBlankStr
    target_hash: Sha256Str
    status: CurationStatus
    curator_id: NonBlankStr
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    supersedes_curation_id: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> KnowledgeCurationRecord:
        from ..serialization import content_hash

        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("KnowledgeCurationRecord evidence_refs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"curation_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"knowledge-curation-{content_hash(identity)[:24]}"
        if self.curation_id != expected_id:
            raise ValueError("KnowledgeCurationRecord curation_id is not content-bound")
        if self.supersedes_curation_id == self.curation_id:
            raise ValueError("KnowledgeCurationRecord cannot supersede itself")
        payload = {"curation_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("KnowledgeCurationRecord content_hash is invalid")
        return self


class KnowledgeSnapshot(StrictModel):
    snapshot_id: NonBlankStr
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    domain_profile_hash: Sha256Str
    expert_case_hashes: dict[NonBlankStr, Sha256Str]
    workflow_pattern_hashes: dict[NonBlankStr, Sha256Str]
    capability_hashes: dict[NonBlankStr, Sha256Str]
    evidence_span_hashes: dict[NonBlankStr, Sha256Str]
    evidence_source_versions: dict[NonBlankStr, Sha256Str]
    literature_document_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    raw_literature_artifact_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    canonical_text_artifact_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    literature_ingestion_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    literature_representation_selection_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    literature_representation_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    historical_evidence_authorization_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    expert_profile_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    expert_opinion_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    expert_attribution_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    knowledge_relation_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    curation_record_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    trusted_record_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    document_structure_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    document_structure_selection_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    structured_evidence_locator_hashes: dict[NonBlankStr, Sha256Str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_snapshot_id(self) -> KnowledgeSnapshot:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json",
            exclude={"snapshot_id", "created_at"},
            exclude_defaults=True,
        )
        if self.snapshot_id != f"snapshot-{content_hash(identity)[:24]}":
            raise ValueError("KnowledgeSnapshot snapshot_id is not content-bound")
        return self


class KnowledgeGraphNode(StrictModel):
    node_id: NonBlankStr
    record_type: KnowledgeEntityType
    record_id: NonBlankStr
    record_hash: Sha256Str
    curation_status: CurationStatus | None = None

    @model_validator(mode="after")
    def validate_node_id(self) -> KnowledgeGraphNode:
        from ..serialization import content_hash

        identity = {"record_type": self.record_type, "record_id": self.record_id}
        if self.node_id != f"knowledge-node-{content_hash(identity)[:24]}":
            raise ValueError("KnowledgeGraphNode node_id is not content-bound")
        return self


class KnowledgeGraphEdge(StrictModel):
    relation_id: NonBlankStr
    relation_hash: Sha256Str
    subject_node_id: NonBlankStr
    predicate: KnowledgePredicate
    object_node_id: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = ()
    curation_status: CurationStatus | None = None


class KnowledgeGraph(StrictModel):
    graph_id: NonBlankStr
    view_mode: KnowledgeViewMode
    domain_filter: NonBlankStr | None = None
    topic_filter: NonBlankStr | None = None
    nodes: tuple[KnowledgeGraphNode, ...]
    edges: tuple[KnowledgeGraphEdge, ...]
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity_and_references(self) -> KnowledgeGraph:
        from ..serialization import content_hash

        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("KnowledgeGraph node IDs must be unique")
        relation_ids = [edge.relation_id for edge in self.edges]
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("KnowledgeGraph relation IDs must be unique")
        known_nodes = set(node_ids)
        for edge in self.edges:
            if (
                edge.subject_node_id not in known_nodes
                or edge.object_node_id not in known_nodes
            ):
                raise ValueError("KnowledgeGraph edge references an unknown node")
        if self.view_mode == KnowledgeViewMode.TRUSTED:
            for node in self.nodes:
                if (
                    node.record_type
                    in {"literature_document", "expert_opinion", "knowledge_relation"}
                    and node.curation_status != CurationStatus.ACCEPTED
                ):
                    raise ValueError("trusted KnowledgeGraph contains an unaccepted entity")
            if any(
                edge.curation_status != CurationStatus.ACCEPTED
                for edge in self.edges
            ):
                raise ValueError("trusted KnowledgeGraph contains an unaccepted relation")
        if tuple(sorted(self.nodes, key=lambda item: (item.record_type, item.record_id))) != self.nodes:
            raise ValueError("KnowledgeGraph nodes must be deterministically ordered")
        if tuple(sorted(self.edges, key=lambda item: item.relation_id)) != self.edges:
            raise ValueError("KnowledgeGraph edges must be deterministically ordered")
        identity = self.model_dump(
            mode="json", exclude={"graph_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"knowledge-graph-{content_hash(identity)[:24]}"
        if self.graph_id != expected_id:
            raise ValueError("KnowledgeGraph graph_id is not content-bound")
        payload = {"graph_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("KnowledgeGraph content_hash is invalid")
        return self
