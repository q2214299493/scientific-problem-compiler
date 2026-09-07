from __future__ import annotations

from importlib.resources import as_file, files

import pytest
from pydantic import ValidationError

from spc.domains import DomainPackLoader
from spc.knowledge import (
    KnowledgeGraphBuilder,
    KnowledgeGraphError,
    knowledge_graph_to_mermaid,
)
from spc.knowledge.fixtures import load_knowledge_fixture
from spc.models import (
    CurationStatus,
    EpistemicStatus,
    ExpertOpinion,
    ExpertProfile,
    KnowledgePredicate,
    KnowledgeRelation,
    LiteratureDocument,
    LiteratureWorkflowPattern,
    SourceClaim,
    SourceRole,
)
from spc.repositories import KnowledgeRepositories, SourceEvidenceStore
from spc.serialization import content_hash


def make_literature(**updates) -> LiteratureDocument:
    identity = {
        "title": "Connectivity-preserving scientific workflow design",
        "authors": ("A. Researcher", "B. Curator"),
        "year": 2026,
        "journal": "Journal of Reproducible Science",
        "doi": "10.0000/example.k1a",
        "url": "https://example.org/literature/k1a",
        "domain": "base",
        "topics": ("mechanistic connectivity", "workflow validation"),
        "keywords": ("endpoint selection", "provenance"),
        "raw_artifact_ref": "artifacts/generic-paper.pdf",
        "canonical_text_ref": "texts/generic-paper.txt",
        "source_id": "source-generic-paper",
        "source_version": "v1",
        "citation_refs": ("citation-example-1",),
        "curation_status": CurationStatus.ACCEPTED,
    }
    identity.update(updates)
    literature_id = f"literature-{content_hash(identity)[:24]}"
    payload = {"literature_id": literature_id, **identity}
    return LiteratureDocument(**payload, content_hash=content_hash(payload))


def make_expert_profile(**updates) -> ExpertProfile:
    identity = {
        "expert_id": "expert-methodology-001",
        "display_name": "Methodology Expert",
        "affiliations": ("Generic Scientific Methods Group",),
        "expertise_domains": ("base",),
        "expertise_topics": ("mechanistic connectivity", "endpoint selection"),
        "profile_source_refs": ("profile-source-methodology-001",),
    }
    identity.update(updates)
    return ExpertProfile(**identity, content_hash=content_hash(identity))


def make_expert_opinion(**updates) -> ExpertOpinion:
    identity = {
        "expert_id": "expert-methodology-001",
        "domain": "base",
        "topic": "endpoint selection",
        "statement": "Endpoint selection should preserve mechanistic connectivity.",
        "opinion_type": "methodological_guidance",
        "scope": "Scientific pathway planning",
        "rationale": "Disconnected endpoints cannot support a coherent pathway comparison.",
        "conditions": ("Atom identity and boundary conditions remain consistent.",),
        "evidence_refs": ("ev-expert-guidance",),
        "related_claim_refs": ("claim-connectivity-001",),
        "related_workflow_refs": ("workflow-connectivity-001",),
        "status": CurationStatus.ACCEPTED,
        "supersedes": None,
    }
    identity.update(updates)
    clean_identity = {key: value for key, value in identity.items() if value is not None}
    opinion_id = f"expert-opinion-{content_hash(clean_identity)[:24]}"
    payload = {"opinion_id": opinion_id, **clean_identity}
    return ExpertOpinion(**payload, content_hash=content_hash(payload))


def make_relation(**updates) -> KnowledgeRelation:
    identity = {
        "subject_type": "literature_document",
        "subject_id": make_literature().literature_id,
        "predicate": KnowledgePredicate.SUPPORTS,
        "object_type": "source_claim",
        "object_id": "claim-connectivity-001",
        "domain": "base",
        "evidence_refs": ("ev-literature-connectivity",),
        "rationale": "The curated document is the provenance source for the claim.",
        "status": CurationStatus.ACCEPTED,
    }
    identity.update(updates)
    relation_id = f"knowledge-relation-{content_hash(identity)[:24]}"
    payload = {"relation_id": relation_id, **identity}
    return KnowledgeRelation(**payload, content_hash=content_hash(payload))


def populate_generic_knowledge(repositories: KnowledgeRepositories) -> None:
    literature = make_literature()
    profile = make_expert_profile()
    opinion = make_expert_opinion()
    claim = SourceClaim(
        claim_id="claim-connectivity-001",
        text="A pathway comparison requires connectivity-preserving endpoints.",
        claim_type="methodological_claim",
        source_role=SourceRole.AUTHOR,
        evidence_refs=("ev-literature-connectivity",),
        source_quote_refs=("quote-connectivity-001",),
        claim_strength="reported guidance",
        epistemic_status=EpistemicStatus.SOURCE_REPORTED,
    )
    workflow = LiteratureWorkflowPattern(
        pattern_id="workflow-connectivity-001",
        domain="base",
        trigger="A scientific comparison requires consistent endpoints.",
        workflow_capabilities=("comparative_analysis",),
        limitations=("Planning guidance only.",),
        evidence_refs=("ev-literature-connectivity",),
    )
    repositories.literature_documents.put(literature.literature_id, literature)
    repositories.expert_profiles.put(profile.expert_id, profile)
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    repositories.source_claims.put(claim.claim_id, claim)
    repositories.workflow_patterns.put(workflow.pattern_id, workflow)
    relations = (
        make_relation(),
        make_relation(
            subject_type="expert_opinion",
            subject_id=opinion.opinion_id,
            predicate=KnowledgePredicate.DERIVED_FROM,
            object_type="expert_profile",
            object_id=profile.expert_id,
            evidence_refs=("ev-expert-guidance",),
            rationale="The opinion is attributed to the persisted expert profile.",
        ),
        make_relation(
            predicate=KnowledgePredicate.APPLIES_TO,
            object_type="workflow_pattern",
            object_id=workflow.pattern_id,
            rationale="The literature guidance applies to this workflow pattern.",
        ),
    )
    repositories.load_relations(relations)


def test_literature_document_id_and_hash_are_deterministic() -> None:
    first = make_literature()
    second = make_literature()
    assert first == second
    assert first.literature_id == second.literature_id
    assert first.content_hash == second.content_hash
    with pytest.raises(ValidationError, match="literature_id is not content-bound"):
        first.model_copy(update={"title": "Changed title"})


def test_expert_opinion_retains_expert_and_provenance_refs() -> None:
    opinion = make_expert_opinion()
    assert opinion.expert_id == "expert-methodology-001"
    assert opinion.evidence_refs == ("ev-expert-guidance",)
    assert opinion.related_claim_refs == ("claim-connectivity-001",)
    assert opinion.related_workflow_refs == ("workflow-connectivity-001",)


def test_expert_opinion_cannot_become_source_claim_silently() -> None:
    opinion = make_expert_opinion()
    assert "epistemic_status" not in opinion.model_dump(mode="json")
    with pytest.raises(ValidationError):
        SourceClaim.model_validate(opinion.model_dump(mode="json"))


@pytest.mark.parametrize(
    "updates",
    (
        {"subject_id": " "},
        {"object_id": ""},
        {"subject_type": "Invalid Type"},
        {"object_type": "bad/type"},
    ),
)
def test_knowledge_relation_rejects_blank_or_invalid_endpoints(updates) -> None:
    with pytest.raises(ValidationError):
        make_relation(**updates)


def test_repository_rejects_conflicting_overwrite(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    first = make_expert_profile()
    changed = make_expert_profile(display_name="Different profile content")
    repositories.expert_profiles.put(first.expert_id, first)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        repositories.expert_profiles.put(changed.expert_id, changed)


def test_duplicate_relation_id_is_rejected(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    relation = make_relation()
    repositories.relations.put(relation.relation_id, relation)
    with pytest.raises(FileExistsError, match="duplicate knowledge relation ID"):
        repositories.relations.put(relation.relation_id, relation)


def test_snapshot_changes_when_accepted_literature_changes(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    evidence = SourceEvidenceStore(tmp_path / ".spc")
    profile = DomainPackLoader().load("base").profile
    first = repositories.create_snapshot(evidence, profile)
    literature = make_literature()
    repositories.literature_documents.put(literature.literature_id, literature)
    second = repositories.create_snapshot(evidence, profile)
    assert first.snapshot_id != second.snapshot_id
    assert second.literature_document_hashes[literature.literature_id] == literature.content_hash


def test_snapshot_changes_when_accepted_expert_opinion_changes(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    evidence = SourceEvidenceStore(tmp_path / ".spc")
    profile = DomainPackLoader().load("base").profile
    first = repositories.create_snapshot(evidence, profile)
    opinion = make_expert_opinion()
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    second = repositories.create_snapshot(evidence, profile)
    assert first.snapshot_id != second.snapshot_id
    assert second.expert_opinion_hashes[opinion.opinion_id] == opinion.content_hash


def test_unaccepted_records_do_not_enter_trusted_snapshot(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    evidence = SourceEvidenceStore(tmp_path / ".spc")
    profile = DomainPackLoader().load("base").profile
    before = repositories.create_snapshot(evidence, profile)
    literature = make_literature(curation_status=CurationStatus.MACHINE_EXTRACTED)
    opinion = make_expert_opinion(status=CurationStatus.HUMAN_REVIEWED)
    relation = make_relation(status=CurationStatus.REJECTED)
    repositories.literature_documents.put(literature.literature_id, literature)
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    repositories.relations.put(relation.relation_id, relation)
    after = repositories.create_snapshot(evidence, profile)
    assert after.snapshot_id == before.snapshot_id
    assert after.literature_document_hashes == {}
    assert after.expert_opinion_hashes == {}
    assert after.knowledge_relation_hashes == {}


def test_graph_and_mermaid_views_are_deterministic(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    populate_generic_knowledge(repositories)
    builder = KnowledgeGraphBuilder()
    first = builder.build(repositories, domain="base")
    second = builder.build(repositories, domain="base")
    assert first == second
    assert first.graph_id == second.graph_id
    assert knowledge_graph_to_mermaid(first) == knowledge_graph_to_mermaid(second)


def test_graph_edges_reference_valid_node_ids(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    populate_generic_knowledge(repositories)
    graph = KnowledgeGraphBuilder().build(
        repositories,
        domain="base",
        topic="connectivity",
    )
    node_ids = {node.node_id for node in graph.nodes}
    assert graph.edges
    assert all(
        edge.subject_node_id in node_ids and edge.object_node_id in node_ids
        for edge in graph.edges
    )
    assert {node.record_type for node in graph.nodes}.issuperset(
        {"literature_document", "source_claim", "expert_profile", "expert_opinion"}
    )


def test_graph_rejects_relation_with_unknown_record(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    relation = make_relation(object_id="claim-that-does-not-exist")
    repositories.relations.put(relation.relation_id, relation)
    with pytest.raises(KnowledgeGraphError, match="unknown record"):
        KnowledgeGraphBuilder().build(repositories)


def test_generic_non_ft_knowledge_fixture_works(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    fixture = files("spc.knowledge.fixtures").joinpath("generic.yaml")
    with as_file(fixture) as fixture_path:
        load_knowledge_fixture(fixture_path, repositories)
    graph = KnowledgeGraphBuilder().build(repositories, domain="base")
    assert graph.domain_filter == "base"
    assert all("fischer_tropsch" not in node.record_id for node in graph.nodes)
    assert len(graph.edges) == 3
