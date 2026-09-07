from __future__ import annotations

from importlib.resources import as_file, files

import pytest
from pydantic import ValidationError

from spc.domains import DomainPackLoader
from spc.knowledge import (
    KnowledgeGraphBuilder,
    KnowledgeGraphError,
    TrustedKnowledgeError,
    TrustedKnowledgeValidator,
    knowledge_graph_to_mermaid,
)
from spc.knowledge.fixtures import load_knowledge_fixture
from spc.models import (
    CurationStatus,
    EpistemicStatus,
    EvidenceSpan,
    ExpertAttributionRecord,
    ExpertOpinion,
    ExpertProfile,
    KnowledgeCurationRecord,
    KnowledgePredicate,
    KnowledgeRelation,
    KnowledgeViewMode,
    LiteratureDocument,
    LiteratureWorkflowPattern,
    MethodFact,
    ModelFact,
    ReportedResult,
    ResultContext,
    SourceClaim,
    SourceQuote,
    SourceRole,
    SourceType,
)
from spc.repositories import KnowledgeRepositories, SourceEvidenceStore
from spc.serialization import content_hash, dump_json


LITERATURE_TEXT = "A pathway comparison requires connectivity-preserving endpoints."
EXPERT_TEXT = "Endpoint selection should preserve mechanistic connectivity."
LITERATURE_SOURCE_TEXT = f"{LITERATURE_TEXT}\n"
EXPERT_SOURCE_TEXT = f"{EXPERT_TEXT}\n"


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalized_doi(value: str) -> str:
    normalized = _normalized(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix).strip()
    return normalized


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
    }
    identity.update(updates)
    if identity["doi"] is not None:
        stable_identity = {"doi": _normalized_doi(identity["doi"])}
    else:
        stable_identity = {
            "title": _normalized(identity["title"]),
            "authors": tuple(_normalized(item) for item in identity["authors"]),
            "year": identity["year"],
        }
    literature_id = f"literature-{content_hash(stable_identity)[:24]}"
    payload = {
        "literature_id": literature_id,
        **{key: value for key, value in identity.items() if value is not None},
    }
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


def make_attribution(**updates) -> ExpertAttributionRecord:
    identity = {
        "expert_id": "expert-methodology-001",
        "source_id": "source-expert-note",
        "source_version": "v1",
        "evidence_refs": ("ev-expert-guidance",),
        "medium": "internal_note",
        "attribution_basis": "Authenticated meeting note authored by the named expert.",
        "captured_at": None,
    }
    identity.update(updates)
    clean_identity = {key: value for key, value in identity.items() if value is not None}
    attribution_id = f"expert-attribution-{content_hash(clean_identity)[:24]}"
    payload = {"attribution_id": attribution_id, **clean_identity}
    return ExpertAttributionRecord(**payload, content_hash=content_hash(payload))


def make_expert_opinion(**updates) -> ExpertOpinion:
    identity = {
        "expert_id": "expert-methodology-001",
        "domain": "base",
        "topic": "endpoint selection",
        "statement": EXPERT_TEXT,
        "opinion_type": "methodological_guidance",
        "scope": "Scientific pathway planning",
        "rationale": "Disconnected endpoints cannot support a coherent pathway comparison.",
        "conditions": ("Atom identity and boundary conditions remain consistent.",),
        "evidence_refs": ("ev-expert-guidance",),
        "attribution_refs": (make_attribution().attribution_id,),
        "related_claim_refs": ("claim-connectivity-001",),
        "related_workflow_refs": ("workflow-connectivity-001",),
        "supersedes": None,
    }
    identity.update(updates)
    clean_identity = {key: value for key, value in identity.items() if value is not None}
    if not clean_identity.get("attribution_refs"):
        clean_identity.pop("attribution_refs", None)
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
    }
    identity.update(updates)
    relation_id = f"knowledge-relation-{content_hash(identity)[:24]}"
    payload = {"relation_id": relation_id, **identity}
    return KnowledgeRelation(**payload, content_hash=content_hash(payload))


def make_curation(
    target_type: str,
    target,
    status: CurationStatus,
    *,
    supersedes: str | None = None,
    evidence_refs: tuple[str, ...] = (),
) -> KnowledgeCurationRecord:
    identity_field = {
        "literature_document": "literature_id",
        "expert_opinion": "opinion_id",
        "knowledge_relation": "relation_id",
        "expert_profile": "expert_id",
    }[target_type]
    identity = {
        "target_type": target_type,
        "target_id": getattr(target, identity_field),
        "target_hash": getattr(target, "content_hash", None) or content_hash(target),
        "status": status,
        "curator_id": "curator-k1a1",
        "rationale": f"Curation decision: {status}",
        "evidence_refs": evidence_refs,
        "supersedes_curation_id": supersedes,
    }
    clean_identity = {key: value for key, value in identity.items() if value is not None}
    curation_id = f"knowledge-curation-{content_hash(clean_identity)[:24]}"
    payload = {"curation_id": curation_id, **clean_identity}
    return KnowledgeCurationRecord(**payload, content_hash=content_hash(payload))


def prepare_evidence_store(tmp_path) -> SourceEvidenceStore:
    store = SourceEvidenceStore(tmp_path / ".spc")
    literature_path = tmp_path / "generic-paper.txt"
    literature_path.write_bytes(LITERATURE_SOURCE_TEXT.encode("utf-8"))
    literature_source = store.ingest(
        literature_path,
        "source-generic-paper",
        "v1",
        "Generic methodology source",
        source_role=SourceRole.LITERATURE_AUTHOR,
        source_type=SourceType.LITERATURE_ARTICLE,
    )
    expert_path = tmp_path / "expert-note.txt"
    expert_path.write_bytes(EXPERT_SOURCE_TEXT.encode("utf-8"))
    expert_source = store.ingest(
        expert_path,
        "source-expert-note",
        "v1",
        "Expert methodology meeting note",
        source_role=SourceRole.INTERNAL_RESEARCHER,
        source_type=SourceType.INTERNAL_NOTE,
    )
    for evidence_id, text, source in (
        ("ev-literature-connectivity", LITERATURE_TEXT, literature_source),
        ("ev-expert-guidance", EXPERT_TEXT, expert_source),
    ):
        store.add_evidence(
            EvidenceSpan(
                evidence_id=evidence_id,
                source_id=source.source_id,
                source_version=source.version,
                content_sha256=source.content_sha256,
                start_offset=0,
                end_offset=len(text),
                text=text,
            )
        )
    return store


def make_literature_quote(**updates) -> SourceQuote:
    identity = {
        "evidence_ref": "ev-literature-connectivity",
        "relative_start_offset": 0,
        "relative_end_offset": len(LITERATURE_TEXT),
        "text": LITERATURE_TEXT,
        "source_id": "source-generic-paper",
        "source_version": "v1",
        "source_role": SourceRole.LITERATURE_AUTHOR,
        "source_type": SourceType.LITERATURE_ARTICLE,
    }
    identity.update(updates)
    quote_identity = {
        "evidence_ref": identity["evidence_ref"],
        "relative_start_offset": identity["relative_start_offset"],
        "relative_end_offset": identity["relative_end_offset"],
        "text_hash": content_hash({"text": identity["text"]}),
    }
    return SourceQuote(
        quote_id=f"quote-{content_hash(quote_identity)[:24]}",
        **identity,
    )


def populate_generic_knowledge(
    repositories: KnowledgeRepositories,
    *,
    literature_status: CurationStatus = CurationStatus.ACCEPTED,
    opinion_status: CurationStatus = CurationStatus.ACCEPTED,
    relation_status: CurationStatus = CurationStatus.ACCEPTED,
) -> tuple[LiteratureDocument, ExpertOpinion, tuple[KnowledgeRelation, ...]]:
    literature = make_literature()
    profile = make_expert_profile()
    attribution = make_attribution()
    opinion = make_expert_opinion(
        attribution_refs=(attribution.attribution_id,)
    )
    quote = make_literature_quote()
    claim = SourceClaim(
        claim_id="claim-connectivity-001",
        text=LITERATURE_TEXT,
        claim_type="methodological_claim",
        source_role=SourceRole.LITERATURE_AUTHOR,
        evidence_refs=("ev-literature-connectivity",),
        source_quote_refs=(quote.quote_id,),
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
    repositories.expert_attributions.put(attribution.attribution_id, attribution)
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    repositories.source_quotes.put(quote.quote_id, quote)
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
    curations = (
        make_curation("literature_document", literature, literature_status),
        make_curation("expert_opinion", opinion, opinion_status),
        *(make_curation("knowledge_relation", relation, relation_status) for relation in relations),
    )
    repositories.load_curations(curations)
    return literature, opinion, relations


def make_source_claim(**updates) -> SourceClaim:
    quote = make_literature_quote()
    payload = {
        "claim_id": "claim-connectivity-001",
        "text": LITERATURE_TEXT,
        "claim_type": "methodological_claim",
        "source_role": SourceRole.LITERATURE_AUTHOR,
        "evidence_refs": ("ev-literature-connectivity",),
        "source_quote_refs": (quote.quote_id,),
        "claim_strength": "reported guidance",
        "epistemic_status": EpistemicStatus.SOURCE_REPORTED,
    }
    payload.update(updates)
    return SourceClaim(**payload)


def trust_relation_endpoint(
    repositories: KnowledgeRepositories,
    record_type: str,
    record_id: str,
) -> None:
    literature = make_literature()
    if not (repositories.literature_documents.root / f"{literature.literature_id}.json").exists():
        repositories.literature_documents.put(literature.literature_id, literature)
        curation = make_curation(
            "literature_document", literature, CurationStatus.ACCEPTED
        )
        repositories.curations.put(curation.curation_id, curation)
    relation = make_relation(
        object_type=record_type,
        object_id=record_id,
        evidence_refs=(),
    )
    repositories.relations.put(relation.relation_id, relation)
    curation = make_curation(
        "knowledge_relation", relation, CurationStatus.ACCEPTED
    )
    repositories.curations.put(curation.curation_id, curation)


def test_literature_document_id_and_hash_are_deterministic() -> None:
    assert make_literature() == make_literature()


def test_literature_id_uses_normalized_doi() -> None:
    plain = make_literature(doi="10.0000/example.k1a")
    url_form = make_literature(doi="HTTPS://DOI.ORG/10.0000/EXAMPLE.K1A")
    assert plain.literature_id == url_form.literature_id
    assert plain.content_hash != url_form.content_hash


def test_literature_without_doi_uses_bibliographic_identity() -> None:
    first = make_literature(doi=None)
    second = make_literature(
        doi=None,
        title="  CONNECTIVITY-PRESERVING   SCIENTIFIC WORKFLOW DESIGN ",
        authors=("a. researcher", "b. curator"),
        journal="Changed metadata",
    )
    assert first.literature_id == second.literature_id


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


def test_curation_transition_preserves_target_and_history(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    literature = make_literature()
    repositories.literature_documents.put(literature.literature_id, literature)
    machine = make_curation("literature_document", literature, CurationStatus.MACHINE_EXTRACTED)
    reviewed = make_curation(
        "literature_document",
        literature,
        CurationStatus.HUMAN_REVIEWED,
        supersedes=machine.curation_id,
    )
    accepted = make_curation(
        "literature_document",
        literature,
        CurationStatus.ACCEPTED,
        supersedes=reviewed.curation_id,
    )
    repositories.load_curations((machine, reviewed, accepted))
    assert len({item.curation_id for item in repositories.curations.list()}) == 3
    assert {item.target_id for item in repositories.curations.list()} == {
        literature.literature_id
    }
    current = TrustedKnowledgeValidator(
        repositories, None
    ).resolve_current_curations()
    assert current[("literature_document", literature.literature_id)] == accepted
    assert repositories.curations.get(machine.curation_id) == machine
    with pytest.raises(ValidationError, match="curation_id is not content-bound"):
        machine.model_copy(update={"status": CurationStatus.REJECTED})


def test_opinion_and_relation_identity_ignore_curation_transitions() -> None:
    opinion = make_expert_opinion()
    relation = make_relation()
    opinion_ids = {
        make_curation("expert_opinion", opinion, status).target_id
        for status in CurationStatus
    }
    relation_ids = {
        make_curation("knowledge_relation", relation, status).target_id
        for status in CurationStatus
    }
    assert opinion_ids == {opinion.opinion_id}
    assert relation_ids == {relation.relation_id}


def test_snapshot_binds_valid_accepted_knowledge_and_curations(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    profile = DomainPackLoader().load("base").profile
    before = repositories.create_snapshot(store, profile)
    literature, opinion, relations = populate_generic_knowledge(repositories)
    snapshot = repositories.create_snapshot(store, profile)
    assert snapshot.snapshot_id != before.snapshot_id
    assert snapshot.literature_document_hashes[literature.literature_id] == literature.content_hash
    assert snapshot.expert_opinion_hashes[opinion.opinion_id] == opinion.content_hash
    assert set(snapshot.knowledge_relation_hashes) == {
        relation.relation_id for relation in relations
    }
    assert len(snapshot.curation_record_hashes) == 5


def test_accepted_opinion_with_missing_evidence_is_rejected(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    attribution = make_attribution(evidence_refs=("ev-missing",))
    opinion = make_expert_opinion(
        evidence_refs=("ev-missing",),
        attribution_refs=(attribution.attribution_id,),
        related_claim_refs=(),
        related_workflow_refs=(),
    )
    profile = make_expert_profile()
    repositories.expert_profiles.put(profile.expert_id, profile)
    repositories.expert_attributions.put(attribution.attribution_id, attribution)
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    repositories.curations.put(
        (curation := make_curation("expert_opinion", opinion, CurationStatus.ACCEPTED)).curation_id,
        curation,
    )
    store = prepare_evidence_store(tmp_path)
    with pytest.raises(TrustedKnowledgeError, match="INVALID_KNOWLEDGE_EVIDENCE"):
        repositories.create_snapshot(
            store,
            DomainPackLoader().load("base").profile,
        )


@pytest.mark.parametrize(
    "updates",
    (
        {"expert_id": "expert-that-does-not-exist"},
        {"related_claim_refs": ("claim-that-does-not-exist",)},
        {"related_workflow_refs": ("workflow-that-does-not-exist",)},
        {"supersedes": "opinion-that-does-not-exist"},
    ),
)
def test_accepted_opinion_requires_all_related_records(tmp_path, updates) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    opinion = make_expert_opinion(**updates)
    profile = make_expert_profile()
    attribution = make_attribution()
    quote = make_literature_quote()
    claim = make_source_claim()
    workflow = LiteratureWorkflowPattern(
        pattern_id="workflow-connectivity-001",
        domain="base",
        trigger="A scientific comparison requires consistent endpoints.",
        workflow_capabilities=("comparative_analysis",),
    )
    repositories.expert_profiles.put(profile.expert_id, profile)
    repositories.expert_attributions.put(attribution.attribution_id, attribution)
    repositories.source_quotes.put(quote.quote_id, quote)
    repositories.source_claims.put(claim.claim_id, claim)
    repositories.workflow_patterns.put(workflow.pattern_id, workflow)
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    curation = make_curation("expert_opinion", opinion, CurationStatus.ACCEPTED)
    repositories.curations.put(curation.curation_id, curation)
    with pytest.raises(TrustedKnowledgeError, match="UNKNOWN_KNOWLEDGE_ENDPOINT"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_accepted_curation_evidence_must_exist(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    literature = make_literature()
    repositories.literature_documents.put(literature.literature_id, literature)
    curation = make_curation(
        "literature_document",
        literature,
        CurationStatus.ACCEPTED,
        evidence_refs=("curation-evidence-that-does-not-exist",),
    )
    repositories.curations.put(curation.curation_id, curation)
    with pytest.raises(TrustedKnowledgeError, match="INVALID_KNOWLEDGE_EVIDENCE"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_accepted_literature_with_missing_source_is_rejected(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    literature = make_literature(source_id="missing-source")
    repositories.literature_documents.put(literature.literature_id, literature)
    curation = make_curation("literature_document", literature, CurationStatus.ACCEPTED)
    repositories.curations.put(curation.curation_id, curation)
    with pytest.raises(TrustedKnowledgeError, match="INVALID_LITERATURE_SOURCE"):
        repositories.create_snapshot(
            SourceEvidenceStore(tmp_path / ".spc"),
            DomainPackLoader().load("base").profile,
        )


def test_accepted_relation_with_unknown_endpoint_is_rejected(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    relation = make_relation(
        evidence_refs=(),
        object_id="claim-that-does-not-exist",
    )
    repositories.relations.put(relation.relation_id, relation)
    curation = make_curation("knowledge_relation", relation, CurationStatus.ACCEPTED)
    repositories.curations.put(curation.curation_id, curation)
    with pytest.raises(TrustedKnowledgeError, match="UNKNOWN_KNOWLEDGE_ENDPOINT"):
        repositories.create_snapshot(
            SourceEvidenceStore(tmp_path / ".spc"),
            DomainPackLoader().load("base").profile,
        )


def test_accepted_relation_pointing_to_rejected_knowledge_is_rejected(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    literature = make_literature()
    claim = SourceClaim(
        claim_id="claim-connectivity-001",
        text=LITERATURE_TEXT,
        claim_type="methodological_claim",
        source_role=SourceRole.AUTHOR,
        evidence_refs=("ev-literature-connectivity",),
        source_quote_refs=("quote-connectivity-001",),
        claim_strength="reported guidance",
        epistemic_status=EpistemicStatus.SOURCE_REPORTED,
    )
    relation = make_relation(evidence_refs=())
    repositories.literature_documents.put(literature.literature_id, literature)
    repositories.source_claims.put(claim.claim_id, claim)
    repositories.relations.put(relation.relation_id, relation)
    repositories.load_curations(
        (
            make_curation("literature_document", literature, CurationStatus.REJECTED),
            make_curation("knowledge_relation", relation, CurationStatus.ACCEPTED),
        )
    )
    with pytest.raises(TrustedKnowledgeError, match="UNTRUSTED_RELATION_ENDPOINT"):
        repositories.create_snapshot(
            SourceEvidenceStore(tmp_path / ".spc"),
            DomainPackLoader().load("base").profile,
        )


def test_tampered_evidence_prevents_trusted_snapshot(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    opinion = make_expert_opinion(related_claim_refs=(), related_workflow_refs=())
    profile = make_expert_profile()
    attribution = make_attribution()
    repositories.expert_profiles.put(profile.expert_id, profile)
    repositories.expert_attributions.put(attribution.attribution_id, attribution)
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    curation = make_curation("expert_opinion", opinion, CurationStatus.ACCEPTED)
    repositories.curations.put(curation.curation_id, curation)
    evidence = store.get("ev-expert-guidance")
    dump_json(
        store.evidence_records.root / "ev-expert-guidance.json",
        evidence.model_copy(update={"text": "Tampered evidence text."}),
    )
    with pytest.raises(TrustedKnowledgeError, match="INVALID_KNOWLEDGE_EVIDENCE"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_trusted_and_audit_graph_modes_expose_correct_curation(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    literature, opinion, relations = populate_generic_knowledge(
        repositories,
        literature_status=CurationStatus.MACHINE_EXTRACTED,
        opinion_status=CurationStatus.REJECTED,
        relation_status=CurationStatus.REJECTED,
    )
    trusted = KnowledgeGraphBuilder().build(repositories, store)
    audit = KnowledgeGraphBuilder().build(
        repositories,
        view_mode=KnowledgeViewMode.AUDIT,
    )
    trusted_ids = {node.record_id for node in trusted.nodes}
    audit_by_id = {node.record_id: node for node in audit.nodes}
    assert literature.literature_id not in trusted_ids
    assert opinion.opinion_id not in trusted_ids
    assert audit_by_id[literature.literature_id].curation_status == CurationStatus.MACHINE_EXTRACTED
    assert audit_by_id[opinion.opinion_id].curation_status == CurationStatus.REJECTED
    assert all(edge.curation_status == CurationStatus.REJECTED for edge in audit.edges)
    audit_mermaid = knowledge_graph_to_mermaid(audit)
    assert "status=machine_extracted" in audit_mermaid
    assert "status=rejected" in audit_mermaid
    assert relations


def test_graph_and_mermaid_views_are_deterministic(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    populate_generic_knowledge(repositories)
    builder = KnowledgeGraphBuilder()
    first = builder.build(repositories, store, domain="base")
    second = builder.build(repositories, store, domain="base")
    assert first == second
    assert knowledge_graph_to_mermaid(first) == knowledge_graph_to_mermaid(second)


def test_graph_rejects_relation_with_unknown_record_in_audit_mode(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    relation = make_relation(object_id="claim-that-does-not-exist")
    repositories.relations.put(relation.relation_id, relation)
    with pytest.raises(KnowledgeGraphError, match="unavailable record"):
        KnowledgeGraphBuilder().build(
            repositories,
            view_mode=KnowledgeViewMode.AUDIT,
        )


def test_source_claim_with_missing_quote_cannot_enter_trusted_graph(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    claim = make_source_claim(source_quote_refs=("quote-missing",))
    repositories.source_claims.put(claim.claim_id, claim)
    trust_relation_endpoint(repositories, "source_claim", claim.claim_id)
    with pytest.raises(TrustedKnowledgeError, match="UNKNOWN_SOURCE_QUOTE_REF"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_source_claim_quote_evidence_mismatch_is_rejected(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    quote = make_literature_quote()
    claim = make_source_claim(evidence_refs=("ev-expert-guidance",))
    repositories.source_quotes.put(quote.quote_id, quote)
    repositories.source_claims.put(claim.claim_id, claim)
    trust_relation_endpoint(repositories, "source_claim", claim.claim_id)
    with pytest.raises(TrustedKnowledgeError, match="CLAIM_QUOTE_EVIDENCE_MISMATCH"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_source_claim_source_role_mismatch_is_rejected(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    quote = make_literature_quote()
    claim = make_source_claim(source_role=SourceRole.AUTHOR)
    repositories.source_quotes.put(quote.quote_id, quote)
    repositories.source_claims.put(claim.claim_id, claim)
    trust_relation_endpoint(repositories, "source_claim", claim.claim_id)
    with pytest.raises(TrustedKnowledgeError, match="CLAIM_SOURCE_ROLE_MISMATCH"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_tampered_source_quote_is_rejected(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    quote = make_literature_quote(source_role=SourceRole.INTERNAL_RESEARCHER)
    repositories.source_quotes.put(quote.quote_id, quote)
    trust_relation_endpoint(repositories, "source_quote", quote.quote_id)
    with pytest.raises(TrustedKnowledgeError, match="SOURCE_QUOTE_PROVENANCE_MISMATCH"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


@pytest.mark.parametrize(
    ("record_type", "record"),
    (
        (
            "method_fact",
            MethodFact(
                fact_id="method-invalid-evidence",
                text="A method statement.",
                attributes={},
                evidence_refs=("ev-missing",),
            ),
        ),
        (
            "model_fact",
            ModelFact(
                fact_id="model-invalid-evidence",
                text="A model statement.",
                attributes={},
                evidence_refs=("ev-missing",),
            ),
        ),
        (
            "reported_result",
            ReportedResult(
                result_id="result-invalid-evidence",
                quantity="barrier",
                value=1.2,
                unit="eV",
                system_context={"system": "generic"},
                method_context={"method": "reported"},
                evidence_refs=("ev-missing",),
                result_status="literature_reported",
            ),
        ),
    ),
)
def test_scientific_record_with_invalid_evidence_is_rejected(
    tmp_path, record_type, record
) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    identity_field = "result_id" if record_type == "reported_result" else "fact_id"
    repository_name = {
        "method_fact": "method_facts",
        "model_fact": "model_facts",
        "reported_result": "reported_results",
    }[record_type]
    record_id = getattr(record, identity_field)
    getattr(repositories, repository_name).put(record_id, record)
    trust_relation_endpoint(repositories, record_type, record_id)
    with pytest.raises(TrustedKnowledgeError, match="INVALID_KNOWLEDGE_EVIDENCE"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_snapshot_binds_reachable_quote_claim_result_and_fact_hashes(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    quote = make_literature_quote()
    claim = make_source_claim()
    method = MethodFact(
        fact_id="method-connectivity-001",
        text="The endpoint comparison preserves atom identity.",
        attributes={"method": "pathway comparison"},
        evidence_refs=("ev-literature-connectivity",),
    )
    model = ModelFact(
        fact_id="model-connectivity-001",
        text="The compared endpoints share a connectivity model.",
        attributes={"model": "connectivity preserving"},
        evidence_refs=("ev-literature-connectivity",),
    )
    context_identity = {
        "system_context": {"system": "generic"},
        "method_context": {"method": "pathway comparison"},
        "method_fact_refs": (method.fact_id,),
        "model_fact_refs": (model.fact_id,),
    }
    result_context = ResultContext(
        context_id=f"result-context-{content_hash(context_identity)[:24]}",
        **context_identity,
    )
    result = ReportedResult(
        result_id="result-connectivity-001",
        quantity="connectivity score",
        value=1.0,
        unit="dimensionless",
        system_context=context_identity["system_context"],
        method_context=context_identity["method_context"],
        result_context=result_context,
        evidence_refs=("ev-literature-connectivity",),
        result_status="literature_reported",
    )
    repositories.source_quotes.put(quote.quote_id, quote)
    repositories.source_claims.put(claim.claim_id, claim)
    repositories.method_facts.put(method.fact_id, method)
    repositories.model_facts.put(model.fact_id, model)
    repositories.reported_results.put(result.result_id, result)
    trust_relation_endpoint(repositories, "source_claim", claim.claim_id)
    trust_relation_endpoint(repositories, "reported_result", result.result_id)
    snapshot = repositories.create_snapshot(
        store, DomainPackLoader().load("base").profile
    )
    expected = {
        f"source_quote:{quote.quote_id}": content_hash(quote),
        f"source_claim:{claim.claim_id}": content_hash(claim),
        f"reported_result:{result.result_id}": content_hash(result),
        f"method_fact:{method.fact_id}": content_hash(method),
        f"model_fact:{model.fact_id}": content_hash(model),
    }
    assert expected.items() <= snapshot.trusted_record_hashes.items()


def test_snapshot_changes_when_reachable_source_claim_changes(tmp_path) -> None:
    snapshots = []
    for name, claim_text in (("first", LITERATURE_TEXT), ("second", "Normalized claim text.")):
        root = tmp_path / name
        repositories = KnowledgeRepositories(root / "knowledge")
        store = prepare_evidence_store(root)
        quote = make_literature_quote()
        claim = make_source_claim(text=claim_text)
        repositories.source_quotes.put(quote.quote_id, quote)
        repositories.source_claims.put(claim.claim_id, claim)
        trust_relation_endpoint(repositories, "source_claim", claim.claim_id)
        snapshots.append(
            repositories.create_snapshot(
                store, DomainPackLoader().load("base").profile
            )
        )
    assert snapshots[0].snapshot_id != snapshots[1].snapshot_id


def test_expert_opinion_rejects_attribution_owned_by_another_expert(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    profile = make_expert_profile()
    attribution = make_attribution(expert_id="expert-other")
    opinion = make_expert_opinion(
        attribution_refs=(attribution.attribution_id,),
        related_claim_refs=(),
        related_workflow_refs=(),
    )
    repositories.expert_profiles.put(profile.expert_id, profile)
    repositories.expert_attributions.put(attribution.attribution_id, attribution)
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    curation = make_curation("expert_opinion", opinion, CurationStatus.ACCEPTED)
    repositories.curations.put(curation.curation_id, curation)
    with pytest.raises(TrustedKnowledgeError, match="EXPERT_ATTRIBUTION_OWNER_MISMATCH"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_expert_opinion_requires_explicit_attribution_for_literature_evidence(
    tmp_path,
) -> None:
    profile = make_expert_profile()
    store = prepare_evidence_store(tmp_path)
    invalid_repositories = KnowledgeRepositories(tmp_path / "invalid-knowledge")
    expert_attribution = make_attribution()
    invalid_opinion = make_expert_opinion(
        evidence_refs=("ev-literature-connectivity",),
        attribution_refs=(expert_attribution.attribution_id,),
        related_claim_refs=(),
        related_workflow_refs=(),
    )
    invalid_repositories.expert_profiles.put(profile.expert_id, profile)
    invalid_repositories.expert_attributions.put(
        expert_attribution.attribution_id, expert_attribution
    )
    invalid_repositories.expert_opinions.put(
        invalid_opinion.opinion_id, invalid_opinion
    )
    curation = make_curation(
        "expert_opinion", invalid_opinion, CurationStatus.ACCEPTED
    )
    invalid_repositories.curations.put(curation.curation_id, curation)
    with pytest.raises(
        TrustedKnowledgeError, match="EXPERT_OPINION_EVIDENCE_NOT_ATTRIBUTED"
    ):
        invalid_repositories.create_snapshot(
            store, DomainPackLoader().load("base").profile
        )

    valid_repositories = KnowledgeRepositories(tmp_path / "valid-knowledge")
    literature_attribution = make_attribution(
        source_id="source-generic-paper",
        evidence_refs=("ev-literature-connectivity",),
        attribution_basis="Explicit authorship attribution for the literature source.",
    )
    valid_opinion = make_expert_opinion(
        evidence_refs=("ev-literature-connectivity",),
        attribution_refs=(literature_attribution.attribution_id,),
        related_claim_refs=(),
        related_workflow_refs=(),
    )
    valid_repositories.expert_profiles.put(profile.expert_id, profile)
    valid_repositories.expert_attributions.put(
        literature_attribution.attribution_id, literature_attribution
    )
    valid_repositories.expert_opinions.put(valid_opinion.opinion_id, valid_opinion)
    valid_curation = make_curation(
        "expert_opinion", valid_opinion, CurationStatus.ACCEPTED
    )
    valid_repositories.curations.put(valid_curation.curation_id, valid_curation)
    snapshot = valid_repositories.create_snapshot(
        store, DomainPackLoader().load("base").profile
    )
    assert literature_attribution.attribution_id in snapshot.expert_attribution_hashes


def test_accepted_expert_opinion_requires_attribution_record(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = prepare_evidence_store(tmp_path)
    profile = make_expert_profile()
    opinion = make_expert_opinion(
        attribution_refs=(), related_claim_refs=(), related_workflow_refs=()
    )
    repositories.expert_profiles.put(profile.expert_id, profile)
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    curation = make_curation("expert_opinion", opinion, CurationStatus.ACCEPTED)
    repositories.curations.put(curation.curation_id, curation)
    with pytest.raises(TrustedKnowledgeError, match="MISSING_EXPERT_ATTRIBUTION"):
        repositories.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_generic_non_ft_knowledge_fixture_works(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = SourceEvidenceStore(tmp_path / ".spc")
    fixture = files("spc.knowledge.fixtures").joinpath("generic.yaml")
    with as_file(fixture) as fixture_path:
        load_knowledge_fixture(fixture_path, repositories, store)
    graph = KnowledgeGraphBuilder().build(
        repositories,
        store,
        domain="base",
    )
    assert all("fischer_tropsch" not in node.record_id for node in graph.nodes)
    assert len(graph.edges) == 3
    sources = {item.source_id: item for item in store.source_records.list()}
    assert sources["source-generic-paper"].source_type == SourceType.LITERATURE_ARTICLE
    assert sources["source-expert-note"].source_type == SourceType.INTERNAL_NOTE
    assert repositories.source_claims.get(
        "claim-connectivity-001"
    ).source_role == SourceRole.LITERATURE_AUTHOR
    opinion = repositories.expert_opinions.list()[0]
    attribution = repositories.expert_attributions.get(opinion.attribution_refs[0])
    assert attribution.source_id == "source-expert-note"
