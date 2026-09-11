from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from spc.cli import app
from spc.compiler import ScientificProblemCompiler
from spc.domains import DomainPackLoader
from spc.interpretation import MockInterpretationProvider, ScientificEvidencePacketBuilder
from spc.knowledge.trust import TrustedKnowledgeError
from spc.models import (
    CurationStatus,
    EpistemicStatus,
    EvidenceSpan,
    ExpertAttributionRecord,
    ExpertCase,
    ExpertOpinion,
    ExpertProfile,
    KnowledgeCurationRecord,
    KnowledgePredicate,
    KnowledgeRelation,
    LiteratureDocument,
    MethodFact,
    ReportedResult,
    ResultStatus,
    SourceClaim,
    SourceQuote,
    SourceRole,
    SourceType,
)
from spc.planning import MockPlanningProvider, PlanningContextResolver
from spc.repositories import (
    CompositeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
)
from spc.retrieval import PersistentKnowledgeRetriever, ScientificContextBuilder
from spc.retrieval.query_builder import build_retrieval_query
from spc.serialization import content_hash


LITERATURE_TEXT = "The activation barrier is 1.25 eV using a periodic DFT model."
CONFLICT_TEXT = "The source reports that a competing mechanism remains plausible."
EXPERT_TEXT = "The mechanism is not sufficiently convincing without a distinguishing observation."


def _literature(source_id: str) -> LiteratureDocument:
    identity = {
        "title": "Competing pathways and activation barriers",
        "authors": ("A. Author",),
        "year": 2026,
        "journal": "Journal of Auditable Science",
        "doi": "10.0000/k1h.fixture",
        "url": "https://example.org/k1h",
        "domain": "base",
        "topics": ("competing mechanisms", "activation barrier"),
        "keywords": ("DFT", "mechanism discrimination"),
        "raw_artifact_ref": "raw-fixture",
        "canonical_text_ref": "canonical-fixture",
        "source_id": source_id,
        "source_version": "v1",
        "citation_refs": (),
    }
    stable = {"doi": "10.0000/k1h.fixture"}
    literature_id = f"literature-{content_hash(stable)[:24]}"
    payload = {"literature_id": literature_id, **identity}
    return LiteratureDocument(**payload, content_hash=content_hash(payload))


def _quote(evidence: EvidenceSpan, source_role: SourceRole, source_type: SourceType) -> SourceQuote:
    identity = {
        "evidence_ref": evidence.evidence_id,
        "relative_start_offset": 0,
        "relative_end_offset": len(evidence.text),
        "text_hash": content_hash({"text": evidence.text}),
    }
    return SourceQuote(
        quote_id=f"quote-{content_hash(identity)[:24]}",
        evidence_ref=evidence.evidence_id,
        relative_start_offset=0,
        relative_end_offset=len(evidence.text),
        text=evidence.text,
        source_id=evidence.source_id,
        source_version=evidence.source_version,
        source_role=source_role,
        source_type=source_type,
    )


def _curate(
    repositories: KnowledgeRepositories,
    target_type: str,
    target_id: str,
    target,
    status: CurationStatus,
) -> KnowledgeCurationRecord:
    identity = {
        "target_type": target_type,
        "target_id": target_id,
        "target_hash": getattr(target, "content_hash", None) or content_hash(target),
        "status": status,
        "curator_id": "k1h-reviewer",
        "rationale": f"K1H fixture status: {status.value}",
        "evidence_refs": (),
    }
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    record = KnowledgeCurationRecord(**payload, content_hash=content_hash(payload))
    repositories.curations.put(record.curation_id, record)
    return record


def _relation(
    subject_type: str,
    subject_id: str,
    predicate: KnowledgePredicate,
    object_type: str,
    object_id: str,
    evidence_refs: tuple[str, ...],
) -> KnowledgeRelation:
    identity = {
        "subject_type": subject_type,
        "subject_id": subject_id,
        "predicate": predicate,
        "object_type": object_type,
        "object_id": object_id,
        "domain": "base",
        "evidence_refs": evidence_refs,
        "rationale": "K1H deterministic scientific relationship fixture.",
    }
    relation_id = f"knowledge-relation-{content_hash(identity)[:24]}"
    payload = {"relation_id": relation_id, **identity}
    return KnowledgeRelation(**payload, content_hash=content_hash(payload))


def _prepare(
    tmp_path: Path,
    *,
    claim_status: CurationStatus = CurationStatus.ACCEPTED,
    opinion_status: CurationStatus = CurationStatus.ACCEPTED,
    case_status: CurationStatus = CurationStatus.ACCEPTED,
):
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = repositories.evidence_store
    literature_path = tmp_path / "paper.txt"
    expert_path = tmp_path / "expert.txt"
    literature_path.write_bytes(f"{LITERATURE_TEXT}\n{CONFLICT_TEXT}".encode("utf-8"))
    expert_path.write_bytes(EXPERT_TEXT.encode("utf-8"))
    literature_source = store.ingest(
        literature_path,
        "source-k1h-paper",
        "v1",
        "K1H paper",
        source_role=SourceRole.LITERATURE_AUTHOR,
        source_type=SourceType.LITERATURE_ARTICLE,
    )
    expert_source = store.ingest(
        expert_path,
        "source-k1h-expert",
        "v1",
        "K1H expert note",
        source_role=SourceRole.INTERNAL_RESEARCHER,
        source_type=SourceType.INTERNAL_NOTE,
    )
    evidence = EvidenceSpan(
        evidence_id="ev-k1h-literature",
        source_id=literature_source.source_id,
        source_version=literature_source.version,
        content_sha256=literature_source.content_sha256,
        start_offset=0,
        end_offset=len(LITERATURE_TEXT),
        text=LITERATURE_TEXT,
    )
    conflict_start = len(LITERATURE_TEXT) + 1
    conflict_evidence = EvidenceSpan(
        evidence_id="ev-k1h-conflict",
        source_id=literature_source.source_id,
        source_version=literature_source.version,
        content_sha256=literature_source.content_sha256,
        start_offset=conflict_start,
        end_offset=conflict_start + len(CONFLICT_TEXT),
        text=CONFLICT_TEXT,
    )
    expert_evidence = EvidenceSpan(
        evidence_id="ev-k1h-expert",
        source_id=expert_source.source_id,
        source_version=expert_source.version,
        content_sha256=expert_source.content_sha256,
        start_offset=0,
        end_offset=len(EXPERT_TEXT),
        text=EXPERT_TEXT,
    )
    for item in (evidence, conflict_evidence, expert_evidence):
        store.add_evidence(item)

    document = _literature(literature_source.source_id)
    quote = _quote(evidence, SourceRole.LITERATURE_AUTHOR, SourceType.LITERATURE_ARTICLE)
    conflict_quote = _quote(
        conflict_evidence,
        SourceRole.LITERATURE_AUTHOR,
        SourceType.LITERATURE_ARTICLE,
    )
    claim = SourceClaim(
        claim_id="claim-k1h-barrier",
        text="A periodic DFT calculation reports a 1.25 eV activation barrier.",
        claim_type="reported_result_claim",
        source_role=SourceRole.LITERATURE_AUTHOR,
        evidence_refs=(evidence.evidence_id,),
        source_quote_refs=(quote.quote_id,),
        claim_strength="source reported",
        epistemic_status=EpistemicStatus.SOURCE_REPORTED,
    )
    conflict_claim = SourceClaim(
        claim_id="claim-k1h-conflict",
        text="A competing mechanism remains plausible.",
        claim_type="mechanism_claim",
        source_role=SourceRole.LITERATURE_AUTHOR,
        evidence_refs=(conflict_evidence.evidence_id,),
        source_quote_refs=(conflict_quote.quote_id,),
        claim_strength="source reported",
        epistemic_status=EpistemicStatus.SOURCE_REPORTED,
    )
    method = MethodFact(
        fact_id="method-k1h-dft",
        text="The reported barrier was calculated with DFT.",
        attributes={"method": "DFT"},
        evidence_refs=(evidence.evidence_id,),
    )
    result = ReportedResult(
        result_id="result-k1h-barrier",
        quantity="activation barrier",
        value=1.25,
        unit="eV",
        system_context={"pathway": "reported mechanism"},
        method_context={"method": "DFT"},
        evidence_refs=(evidence.evidence_id,),
        result_status=ResultStatus.COMPUTED_REPORTED,
    )
    profile_identity = {
        "expert_id": "expert-k1h",
        "display_name": "K1H Reviewer",
        "affiliations": ("Methods Group",),
        "expertise_domains": ("base",),
        "expertise_topics": ("mechanism discrimination",),
        "profile_source_refs": ("profile-k1h",),
    }
    profile = ExpertProfile(**profile_identity, content_hash=content_hash(profile_identity))
    attribution_identity = {
        "expert_id": profile.expert_id,
        "source_id": expert_source.source_id,
        "source_version": expert_source.version,
        "evidence_refs": (expert_evidence.evidence_id,),
        "medium": "internal_note",
        "attribution_basis": "Explicitly attributed reviewer note.",
    }
    attribution_id = f"expert-attribution-{content_hash(attribution_identity)[:24]}"
    attribution_payload = {"attribution_id": attribution_id, **attribution_identity}
    attribution = ExpertAttributionRecord(
        **attribution_payload,
        content_hash=content_hash(attribution_payload),
    )
    opinion_identity = {
        "expert_id": profile.expert_id,
        "domain": "base",
        "topic": "mechanism confidence",
        "statement": EXPERT_TEXT,
        "opinion_type": "criticism",
        "scope": "mechanism discrimination",
        "rationale": "Competing mechanisms require a distinguishing observation.",
        "conditions": ("Use matched system and method conditions.",),
        "evidence_refs": (expert_evidence.evidence_id,),
        "attribution_refs": (attribution.attribution_id,),
        "related_claim_refs": (),
        "related_workflow_refs": (),
    }
    opinion_id = f"expert-opinion-{content_hash(opinion_identity)[:24]}"
    opinion_payload = {"opinion_id": opinion_id, **opinion_identity}
    opinion = ExpertOpinion(**opinion_payload, content_hash=content_hash(opinion_payload))
    case_identity = {
        "domain": "base",
        "vague_request": "The mechanism is not convincing. What else should be tested?",
        "translated_questions": ("Which observation distinguishes the competing mechanisms?",),
        "positive": True,
        "rationale": "Translate vague criticism into a discriminating scientific question.",
        "evidence_refs": (expert_evidence.evidence_id,),
        "opinion_refs": (opinion.opinion_id,),
        "original_wording": "The mechanism is not sufficiently convincing.",
        "latent_concern": "Competing mechanisms remain underdetermined.",
        "atomic_questions": ("Which observation differs between the competing mechanisms?",),
        "good_question_formulations": ("What observable discriminates the mechanisms under matched conditions?",),
        "wrong_formulations": ("Run more calculations without a hypothesis.",),
        "answerability_conditions": ("Competing mechanisms are explicitly defined.",),
        "required_evidence_types": ("discriminating_observable",),
        "baseline_guidance": ("Keep system and method conditions matched.",),
        "common_misinterpretations": ("More data alone resolves ambiguity.",),
        "resolution_pattern": "Select a decisive observation and matched baseline.",
        "applicability": ("vague reviewer criticism",),
    }
    case_provenance_hash = content_hash(case_identity)
    case_id = f"expert-case-{case_provenance_hash[:24]}"
    case = ExpertCase(
        case_id=case_id,
        **case_identity,
        provenance_hash=case_provenance_hash,
    )

    repositories.literature_documents.put(document.literature_id, document)
    repositories.source_quotes.put(quote.quote_id, quote)
    repositories.source_quotes.put(conflict_quote.quote_id, conflict_quote)
    repositories.source_claims.put(claim.claim_id, claim)
    repositories.source_claims.put(conflict_claim.claim_id, conflict_claim)
    repositories.method_facts.put(method.fact_id, method)
    repositories.reported_results.put(result.result_id, result)
    repositories.expert_profiles.put(profile.expert_id, profile)
    repositories.expert_attributions.put(attribution.attribution_id, attribution)
    repositories.expert_opinions.put(opinion.opinion_id, opinion)
    repositories.expert_cases.put(case.case_id, case)

    relations = (
        _relation(
            "source_claim",
            claim.claim_id,
            KnowledgePredicate.USES_METHOD,
            "method_fact",
            method.fact_id,
            (evidence.evidence_id,),
        ),
        _relation(
            "source_claim",
            claim.claim_id,
            KnowledgePredicate.REPORTS_RESULT,
            "reported_result",
            result.result_id,
            (evidence.evidence_id,),
        ),
        _relation(
            "source_claim",
            claim.claim_id,
            KnowledgePredicate.CONTRADICTS,
            "source_claim",
            conflict_claim.claim_id,
            (evidence.evidence_id, conflict_evidence.evidence_id),
        ),
    )
    repositories.load_relations(relations)
    for target_type, target_id, target, status in (
        ("literature_document", document.literature_id, document, CurationStatus.ACCEPTED),
        ("source_claim", claim.claim_id, claim, claim_status),
        ("source_claim", conflict_claim.claim_id, conflict_claim, claim_status),
        ("method_fact", method.fact_id, method, claim_status),
        ("reported_result", result.result_id, result, claim_status),
        ("expert_opinion", opinion.opinion_id, opinion, opinion_status),
        ("expert_case", case.case_id, case, case_status),
        *(("knowledge_relation", item.relation_id, item, claim_status) for item in relations),
    ):
        _curate(repositories, target_type, target_id, target, status)
    return repositories, store, claim, method, result, opinion, case, relations


def _retrieve(repositories, store, text: str, **options):
    pack = DomainPackLoader().load("base")
    snapshot = repositories.create_snapshot(store, pack.profile)
    query = build_retrieval_query(text, "base", pack.profile)
    return PersistentKnowledgeRetriever().retrieve(query, repositories, store, pack.profile, snapshot, **options)


def test_trusted_cross_source_retrieval_and_graph_provenance(tmp_path: Path) -> None:
    repositories, store, claim, method, result, opinion, case, relations = _prepare(tmp_path)

    context = _retrieve(
        repositories,
        store,
        "mechanism not convincing activation barrier DFT",
    )

    literature_ids = {item.record_id for item in context.initial_literature_hits}
    expert_ids = {item.record_id for item in context.initial_expert_hits}
    expanded_ids = {item.record_id for item in context.graph_expanded_hits}
    assert {claim.claim_id, method.fact_id, result.result_id} <= literature_ids
    assert {opinion.opinion_id, case.case_id} <= expert_ids
    contradiction = next(item for item in relations if item.predicate == KnowledgePredicate.CONTRADICTS)
    assert contradiction.relation_id in context.conflict_relation_refs
    assert expanded_ids or context.graph_hops == 1
    for hit in (*context.initial_literature_hits, *context.initial_expert_hits):
        assert hit.record_hash
        assert hit.authority_status == "trusted_current"
        if hit.evidence_refs:
            assert hit.source_refs
    repeated = _retrieve(
        repositories,
        store,
        "mechanism not convincing activation barrier DFT",
    )
    assert context.retrieval_context_id == repeated.retrieval_context_id
    assert context.content_hash == repeated.content_hash


@pytest.mark.parametrize(
    "nontrusted_status",
    (CurationStatus.MACHINE_EXTRACTED, CurationStatus.REJECTED),
)
def test_audit_exposes_nontrusted_records_but_trusted_excludes_them(
    tmp_path: Path,
    nontrusted_status: CurationStatus,
) -> None:
    repositories, store, claim, _, _, opinion, case, _ = _prepare(
        tmp_path,
        claim_status=nontrusted_status,
        opinion_status=nontrusted_status,
        case_status=nontrusted_status,
    )

    trusted = _retrieve(repositories, store, "mechanism activation barrier")
    audit = _retrieve(
        repositories,
        store,
        "mechanism activation barrier",
        mode="audit",
    )

    assert claim.claim_id not in {item.record_id for item in trusted.initial_literature_hits}
    assert opinion.opinion_id not in {item.record_id for item in trusted.initial_expert_hits}
    assert case.case_id not in {item.record_id for item in trusted.initial_expert_hits}
    audit_hits = {item.record_id: item for item in (*audit.initial_literature_hits, *audit.initial_expert_hits)}
    assert audit_hits[claim.claim_id].curation_status == nontrusted_status
    assert audit_hits[opinion.opinion_id].authority_status == nontrusted_status.value
    assert audit_hits[case.case_id].source_class == "expert_case"


def test_graph_cycles_terminate_and_caps_are_enforced(tmp_path: Path) -> None:
    repositories, store, claim, method, _, _, _, _ = _prepare(tmp_path)
    reverse = _relation(
        "method_fact",
        method.fact_id,
        KnowledgePredicate.REFINES,
        "source_claim",
        claim.claim_id,
        ("ev-k1h-literature",),
    )
    repositories.relations.put(reverse.relation_id, reverse)
    _curate(
        repositories,
        "knowledge_relation",
        reverse.relation_id,
        reverse,
        CurationStatus.ACCEPTED,
    )

    context = _retrieve(
        repositories,
        store,
        "periodic DFT",
        graph_hops=2,
        max_graph_hits=1,
    )

    assert len(context.graph_expanded_hits) <= 1
    all_ids = [
        item.hit_id
        for item in (
            *context.initial_literature_hits,
            *context.initial_expert_hits,
            *context.graph_expanded_hits,
        )
    ]
    assert len(all_ids) == len(set(all_ids))


def test_corrupt_trusted_evidence_fails_closed(tmp_path: Path) -> None:
    repositories, store, *_ = _prepare(tmp_path)
    source = store.get_source("source-k1h-paper", "v1")
    content = store.state_root / source.stored_path
    content.chmod(0o644)
    content.write_text("tampered", encoding="utf-8")

    with pytest.raises((TrustedKnowledgeError, ValueError), match="INVALID_KNOWLEDGE_EVIDENCE|hash"):
        _retrieve(repositories, store, "activation barrier")


def test_unaccepted_opinion_cannot_support_accepted_expert_case(tmp_path: Path) -> None:
    repositories, store, *_ = _prepare(
        tmp_path,
        opinion_status=CurationStatus.MACHINE_EXTRACTED,
        case_status=CurationStatus.ACCEPTED,
    )

    with pytest.raises(TrustedKnowledgeError, match="UNTRUSTED_SCIENTIFIC_DEPENDENCY"):
        _retrieve(repositories, store, "not sufficiently convincing")


def test_scientific_context_phase2_pipeline_consumes_persistent_knowledge(tmp_path: Path) -> None:
    repositories, store, claim, _, _, _, case, _ = _prepare(tmp_path)
    state_dir = tmp_path / ".spc"
    context = ScientificContextBuilder().build(
        "The mechanism is not convincing; compare the DFT activation barrier.",
        "base",
        state_dir=state_dir,
        knowledge_dir=repositories.root,
    )
    evidence_view = CompositeEvidenceStore(store, ProjectEvidenceStore(state_dir))
    evidence_packet = ScientificEvidencePacketBuilder(MockInterpretationProvider()).build(
        context,
        evidence_view,
    )
    planning_input = PlanningContextResolver().resolve(
        context,
        evidence_packet,
        repositories,
        evidence_view,
    )
    compiled = ScientificProblemCompiler(MockPlanningProvider(), evidence_repository=evidence_view).compile(
        planning_input
    )

    assert claim.claim_id in context.retrieval_manifest.result_ids or claim.claim_id in {
        hit.record_id for hit in context.literature_knowledge_hits
    }
    assert case.case_id in {item.case_id for item in planning_input.expert_cases}
    assert claim.claim_id in evidence_packet.provenance_manifest["retrieved_knowledge_record_ids"]
    assert compiled.candidates
    assert all(not task.runnable for plan in compiled.candidates for task in plan.tasks)


def test_retrieve_knowledge_cli_is_bounded_and_offline(tmp_path: Path) -> None:
    repositories, *_ = _prepare(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "retrieve-knowledge",
            "--query",
            "mechanism activation barrier",
            "--knowledge-dir",
            str(repositories.root),
            "--domain",
            "base",
            "--max-literature-hits",
            "1",
            "--max-expert-hits",
            "1",
            "--max-expert-cases",
            "1",
            "--graph-hops",
            "1",
            "--max-graph-hits",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"retrieval_mode": "trusted"' in result.output
    assert '"knowledge_snapshot_id"' in result.output
