from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from pypdf import PdfReader, PdfWriter
from typer.testing import CliRunner

from spc.cli import app
from spc.domains import DomainPackLoader
from spc.knowledge.ingestion import (
    LiteratureIngestionService,
    LiteratureRepresentationSelector,
    create_evidence_span_from_canonical_text,
    create_supplementary_evidence_span,
    ingest_literature_supplementary_docx,
    ingest_literature_supplementary_pdf,
)
from spc.knowledge.literature_knowledge.validation import curate_knowledge_record
from spc.models import (
    CurationStatus,
    EpistemicStatus,
    KnowledgeCurationRecord,
    MethodFact,
    ReportedResult,
    ResultContext,
    ResultStatus,
    RetrievalSourceType,
)
from spc.repositories import KnowledgeRepositories
from spc.retrieval import PersistentKnowledgeRetriever
from spc.retrieval.knowledge_retriever import KnowledgeRetrievalError
from spc.retrieval.query_builder import build_retrieval_query
from spc.serialization import content_hash


PDF = Path(__file__).parent / "fixtures" / "generic-born-digital.pdf"
PAGE_ONE_QUOTE = "Mechanistic connectivity must be preserved."


def _prepared_literature(tmp_path: Path) -> tuple[KnowledgeRepositories, str, Path]:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    outcome = LiteratureIngestionService().ingest(
        PDF,
        {
            "title": "Generic Born-Digital Scientific Fixture",
            "authors": ["A. Researcher"],
            "year": 2026,
            "doi": "10.0000/supplement-fixture",
            "domain": "base",
        },
        repositories,
        repositories.evidence_store,
    )
    selection = LiteratureRepresentationSelector().select(
        outcome.literature_id,
        outcome.ingestion_id,
        "fixture-curator",
        "Selected the main article, not its supplement.",
        repositories,
        repositories.evidence_store,
    ).selection
    document = repositories.literature_documents.get(outcome.literature_id)
    for target_type, target_id in (
        ("literature_document", document.literature_id),
        ("literature_representation_selection", selection.selection_id),
    ):
        curate_knowledge_record(
            repositories,
            target_type=target_type,
            target_id=target_id,
            status=CurationStatus.ACCEPTED,
            curator_id="fixture-curator",
            rationale="Reviewed main PDF source identity for this offline test.",
        )
    supplement_pdf = tmp_path / "separate-supplement.pdf"
    writer = PdfWriter()
    writer.append_pages_from_reader(PdfReader(PDF))
    writer.add_metadata({"/Title": "Separate supporting information fixture"})
    with supplement_pdf.open("wb") as output:
        writer.write(output)
    return repositories, document.literature_id, supplement_pdf


def test_supplementary_pdf_is_independent_and_snapshot_bound_only_after_curation(
    tmp_path: Path,
) -> None:
    repositories, literature_id, supplement_pdf = _prepared_literature(tmp_path)
    main_selection = repositories.literature_representation_selections.resolve_current(
        literature_id
    )
    supplement = ingest_literature_supplementary_pdf(
        supplement_pdf, literature_id, repositories, repositories.evidence_store
    )
    assert ingest_literature_supplementary_pdf(
        supplement_pdf, literature_id, repositories, repositories.evidence_store
    ) == supplement
    assert (
        repositories.literature_representation_selections.resolve_current(literature_id)
        == main_selection
    )
    assert supplement.source_id != main_selection.source_id
    evidence = create_supplementary_evidence_span(
        supplement.supplement_id,
        1,
        PAGE_ONE_QUOTE,
        "source fixture, page one",
        repositories,
        repositories.evidence_store,
    )
    assert repositories.evidence_store.verify_evidence_integrity(evidence).source_id == (
        supplement.source_id
    )
    before = repositories.create_snapshot(
        repositories.evidence_store, DomainPackLoader().load("base").profile
    )
    assert evidence.evidence_id not in before.evidence_span_hashes
    assert (
        f"{supplement.source_id}@{supplement.source_version}"
        not in before.evidence_source_versions
    )

    curation = CliRunner().invoke(
        app,
        [
            "curate-knowledge",
            "--target-type", "literature_supplementary_source",
            "--target-id", supplement.supplement_id,
            "--status", "accepted",
            "--curator-id", "fixture-curator",
            "--rationale", "Reviewed independent PDF identity and exact cited page.",
            "--knowledge-dir", str(repositories.root),
            "--evidence-ref", evidence.evidence_id,
        ],
    )
    assert curation.exit_code == 0, curation.output
    after = repositories.create_snapshot(
        repositories.evidence_store, DomainPackLoader().load("base").profile
    )
    assert after.snapshot_id != before.snapshot_id
    assert evidence.evidence_id in after.evidence_span_hashes
    assert (
        f"literature_supplementary_source:{supplement.supplement_id}"
        in after.trusted_record_hashes
    )
    assert (
        f"{supplement.source_id}@{supplement.source_version}"
        in after.evidence_source_versions
    )


def test_supplementary_quote_must_be_on_the_stated_page(tmp_path: Path) -> None:
    repositories, literature_id, supplement_pdf = _prepared_literature(tmp_path)
    supplement = ingest_literature_supplementary_pdf(
        supplement_pdf, literature_id, repositories, repositories.evidence_store
    )
    with pytest.raises(ValueError, match="exactly once"):
        create_supplementary_evidence_span(
            supplement.supplement_id,
            2,
            PAGE_ONE_QUOTE,
            "wrong page",
            repositories,
            repositories.evidence_store,
        )


def test_same_pdf_is_not_an_independent_supplement(tmp_path: Path) -> None:
    repositories, literature_id, _ = _prepared_literature(tmp_path)
    with pytest.raises(ValueError, match="must differ"):
        ingest_literature_supplementary_pdf(
            PDF, literature_id, repositories, repositories.evidence_store
        )


def test_supplement_curation_cannot_cite_main_article_evidence(tmp_path: Path) -> None:
    repositories, literature_id, supplement_pdf = _prepared_literature(tmp_path)
    supplement = ingest_literature_supplementary_pdf(
        supplement_pdf, literature_id, repositories, repositories.evidence_store
    )
    selection = repositories.literature_representation_selections.resolve_current(
        literature_id
    )
    main = LiteratureRepresentationSelector.resolve_selection(
        selection, repositories, repositories.evidence_store
    )
    text = repositories.canonical_text_artifacts.read_text(main.canonical_text_id)
    start = text.index(PAGE_ONE_QUOTE)
    main_evidence = create_evidence_span_from_canonical_text(
        main.canonical_text_id,
        start,
        start + len(PAGE_ONE_QUOTE),
        repositories,
        repositories.evidence_store,
    )
    with pytest.raises(ValueError, match="another source"):
        curate_knowledge_record(
            repositories,
            target_type="literature_supplementary_source",
            target_id=supplement.supplement_id,
            status=CurationStatus.ACCEPTED,
            curator_id="fixture-curator",
            rationale="Invalid cross-source reference.",
            evidence_refs=(main_evidence.evidence_id,),
        )
    identity = {
        "target_type": "literature_supplementary_source",
        "target_id": supplement.supplement_id,
        "target_hash": supplement.content_hash,
        "status": CurationStatus.ACCEPTED,
        "curator_id": "fixture-curator",
        "rationale": "Forged direct repository write for trust-boundary test.",
        "evidence_refs": (main_evidence.evidence_id,),
    }
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    forged = KnowledgeCurationRecord(
        **payload, content_hash=content_hash(payload)
    )
    repositories.curations.put(curation_id, forged)
    with pytest.raises(ValueError, match="SUPPLEMENTARY_EVIDENCE_SOURCE_MISMATCH"):
        repositories.create_snapshot(
            repositories.evidence_store, DomainPackLoader().load("base").profile
        )


def test_tampered_supplementary_pdf_breaks_trusted_snapshot(tmp_path: Path) -> None:
    repositories, literature_id, supplement_pdf = _prepared_literature(tmp_path)
    supplement = ingest_literature_supplementary_pdf(
        supplement_pdf, literature_id, repositories, repositories.evidence_store
    )
    curate_knowledge_record(
        repositories,
        target_type="literature_supplementary_source",
        target_id=supplement.supplement_id,
        status=CurationStatus.ACCEPTED,
        curator_id="fixture-curator",
        rationale="Reviewed independent PDF source identity.",
    )
    raw = repositories.raw_literature_artifacts.get(supplement.raw_artifact_id)
    stored_pdf = repositories.root / raw.stored_path
    os.chmod(stored_pdf, 0o666)
    stored_pdf.write_bytes(stored_pdf.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="INVALID_SUPPLEMENTARY_SOURCE"):
        repositories.create_snapshot(
            repositories.evidence_store, DomainPackLoader().load("base").profile
        )


def _supplement_docx(path: Path, *, duplicate_value: bool = False) -> Path:
    second_row = (
        '<w:tr><w:tc><w:p><w:r><w:t>Other barrier (eV)</w:t></w:r></w:p>'
        '</w:tc><w:tc><w:p><w:r><w:t>1.25</w:t></w:r></w:p>'
        '</w:tc></w:tr>'
        if duplicate_value else ""
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body>'
        '<w:p><w:r><w:t>Independent supporting information.</w:t></w:r></w:p>'
        '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Barrier (eV)</w:t></w:r></w:p>'
        '</w:tc><w:tc><w:p><w:r><w:t>1.25</w:t></w:r></w:p>'
        f'</w:tc></w:tr>{second_row}</w:tbl>'
        '</w:body></w:document>'
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml)
    return path


def test_duplicate_docx_value_requires_exact_canonical_block(tmp_path: Path) -> None:
    repositories, literature_id, _ = _prepared_literature(tmp_path)
    docx = _supplement_docx(tmp_path / "duplicate-values.docx", duplicate_value=True)
    supplement = ingest_literature_supplementary_docx(
        docx, literature_id, repositories, repositories.evidence_store
    )
    canonical = repositories.canonical_text_artifacts.get(
        supplement.canonical_text_id
    )
    blocks = {block.section_path: block for block in canonical.blocks}
    with pytest.raises(ValueError, match="identify one block"):
        create_supplementary_evidence_span(
            supplement.supplement_id, 1, "1.25", "table value",
            repositories, repositories.evidence_store,
        )
    evidence = create_supplementary_evidence_span(
        supplement.supplement_id, 1, "1.25", "second row value",
        repositories, repositories.evidence_store,
        block_id=blocks[("table:1", "row:2", "cell:2")].block_id,
    )
    assert "table:1/row:2/cell:2" in evidence.locator
    with pytest.raises(ValueError, match="identify one block"):
        create_supplementary_evidence_span(
            supplement.supplement_id, 1, "1.25", "wrong block",
            repositories, repositories.evidence_store,
            block_id=blocks[("table:1", "row:1", "cell:1")].block_id,
        )


def test_original_docx_supplement_has_verified_source_and_block_locator(
    tmp_path: Path,
) -> None:
    repositories, literature_id, _ = _prepared_literature(tmp_path)
    docx = _supplement_docx(tmp_path / "publisher-original.docx")
    supplement = ingest_literature_supplementary_docx(
        docx, literature_id, repositories, repositories.evidence_store
    )
    assert ingest_literature_supplementary_docx(
        docx, literature_id, repositories, repositories.evidence_store
    ) == supplement
    raw = repositories.raw_literature_artifacts.get(supplement.raw_artifact_id)
    assert raw.media_type.endswith("wordprocessingml.document")
    assert raw.stored_path.endswith("artifact.docx")
    evidence = create_supplementary_evidence_span(
        supplement.supplement_id,
        1,
        "1.25",
        "table value only; unit requires separate header evidence",
        repositories,
        repositories.evidence_store,
    )
    assert "SI DOCX table:1/row:1/cell:2; no physical page" in evidence.locator
    assert repositories.evidence_store.verify_evidence_integrity(evidence).source_id == (
        supplement.source_id
    )
    before = repositories.create_snapshot(
        repositories.evidence_store, DomainPackLoader().load("base").profile
    )
    assert evidence.evidence_id not in before.evidence_span_hashes
    curate_knowledge_record(
        repositories,
        target_type="literature_supplementary_source",
        target_id=supplement.supplement_id,
        status=CurationStatus.ACCEPTED,
        curator_id="fixture-curator",
        rationale="Reviewed original DOCX identity, not its scientific conclusions.",
        evidence_refs=(evidence.evidence_id,),
    )
    after = repositories.create_snapshot(
        repositories.evidence_store, DomainPackLoader().load("base").profile
    )
    assert after.snapshot_id != before.snapshot_id
    assert evidence.evidence_id in after.evidence_span_hashes
    assert (
        f"literature_supplementary_source:{supplement.supplement_id}"
        in after.trusted_record_hashes
    )


def test_curated_supplement_method_fact_keeps_literature_domain(tmp_path: Path) -> None:
    repositories, literature_id, _ = _prepared_literature(tmp_path)
    supplement = ingest_literature_supplementary_docx(
        _supplement_docx(tmp_path / "method.docx"),
        literature_id,
        repositories,
        repositories.evidence_store,
    )
    evidence = create_supplementary_evidence_span(
        supplement.supplement_id,
        1,
        "Independent supporting information.",
        "reviewed SI method text",
        repositories,
        repositories.evidence_store,
    )
    curate_knowledge_record(
        repositories,
        target_type="literature_supplementary_source",
        target_id=supplement.supplement_id,
        status=CurationStatus.ACCEPTED,
        curator_id="fixture-curator",
        rationale="Reviewed SI source identity, not scientific conclusions.",
        evidence_refs=(evidence.evidence_id,),
    )
    identity = {
        "text": "The SI reports a reviewed method condition.",
        "attributes": {"scope": "SI method"},
        "evidence_refs": (evidence.evidence_id,),
        "epistemic_status": EpistemicStatus.METHOD_STATEMENT,
    }
    fact = MethodFact(fact_id=f"method-fact-{content_hash(identity)[:24]}", **identity)
    repositories.method_facts.put(fact.fact_id, fact)
    curate_knowledge_record(
        repositories,
        target_type="method_fact",
        target_id=fact.fact_id,
        status=CurationStatus.ACCEPTED,
        curator_id="fixture-curator",
        rationale="Reviewed only the cited SI method text.",
        evidence_refs=(evidence.evidence_id,),
    )

    retriever = PersistentKnowledgeRetriever()
    metadata = retriever._literature_metadata(repositories)
    assert (literature_id, "Generic Born-Digital Scientific Fixture", "base") in metadata[fact.fact_id]
    assert retriever._domain_matches(fact, ("method_fact", fact.fact_id), "base", metadata)
    assert not retriever._domain_matches(
        fact, ("method_fact", fact.fact_id), "fischer_tropsch", metadata
    )
    for domain, should_find in (("base", True), ("fischer_tropsch", False)):
        profile = DomainPackLoader().load(domain).profile
        snapshot = repositories.create_snapshot(repositories.evidence_store, profile)
        query = build_retrieval_query(
            "reviewed method condition",
            domain,
            profile,
            evidence_types=(RetrievalSourceType.METHOD_FACT,),
        )
        context = retriever.retrieve(
            query,
            repositories,
            repositories.evidence_store,
            profile,
            snapshot,
            graph_hops=0,
        )
        assert (fact.fact_id in {hit.record_id for hit in context.initial_literature_hits}) is should_find


def test_result_citing_same_literature_main_and_si_keeps_domain(tmp_path: Path) -> None:
    repositories, literature_id, _ = _prepared_literature(tmp_path)
    supplement = ingest_literature_supplementary_docx(
        _supplement_docx(tmp_path / "mixed-source.docx"),
        literature_id,
        repositories,
        repositories.evidence_store,
    )
    si_evidence = create_supplementary_evidence_span(
        supplement.supplement_id,
        1,
        "1.25",
        "SI numeric fixture",
        repositories,
        repositories.evidence_store,
    )
    selection = repositories.literature_representation_selections.resolve_current(
        literature_id
    )
    main = LiteratureRepresentationSelector.resolve_selection(
        selection, repositories, repositories.evidence_store
    )
    text = repositories.canonical_text_artifacts.read_text(main.canonical_text_id)
    start = text.index(PAGE_ONE_QUOTE)
    main_evidence = create_evidence_span_from_canonical_text(
        main.canonical_text_id,
        start,
        start + len(PAGE_ONE_QUOTE),
        repositories,
        repositories.evidence_store,
    )
    context_identity = {
        "system_context": {"system": "same-paper fixture"},
        "method_context": {"method": "main plus SI"},
        "method_fact_refs": (),
        "model_fact_refs": (),
    }
    context = ResultContext(
        context_id=f"result-context-{content_hash(context_identity)[:24]}",
        **context_identity,
    )
    identity = {
        "quantity": "fixture quantity",
        "value": 1.25,
        "unit": "eV",
        "system_context": context.system_context,
        "method_context": context.method_context,
        "result_context": context,
        "evidence_refs": tuple(sorted((si_evidence.evidence_id, main_evidence.evidence_id))),
        "result_status": ResultStatus.UNKNOWN_ORIGIN,
    }
    result = ReportedResult(
        result_id=f"result-{content_hash(identity)[:24]}", **identity
    )
    repositories.reported_results.put(result.result_id, result)

    metadata = PersistentKnowledgeRetriever._literature_metadata(repositories)
    assert metadata[result.result_id] == (
        (literature_id, "Generic Born-Digital Scientific Fixture", "base"),
    )
    curate_knowledge_record(
        repositories,
        target_type="literature_supplementary_source",
        target_id=supplement.supplement_id,
        status=CurationStatus.ACCEPTED,
        curator_id="fixture-curator",
        rationale="Accepted this synthetic SI source identity only.",
        evidence_refs=(si_evidence.evidence_id,),
    )
    curate_knowledge_record(
        repositories,
        target_type="reported_result",
        target_id=result.result_id,
        status=CurationStatus.ACCEPTED,
        curator_id="fixture-curator",
        rationale="Synthetic result accepted only inside this isolated test fixture.",
        evidence_refs=result.evidence_refs,
    )
    profile = DomainPackLoader().load("base").profile
    snapshot = repositories.create_snapshot(repositories.evidence_store, profile)
    query = build_retrieval_query(
        "fixture quantity",
        "base",
        profile,
        evidence_types=(RetrievalSourceType.REPORTED_RESULT,),
    )
    context = PersistentKnowledgeRetriever().retrieve(
        query,
        repositories,
        repositories.evidence_store,
        profile,
        snapshot,
        graph_hops=0,
    )
    assert result.result_id in {hit.record_id for hit in context.initial_literature_hits}

    other = LiteratureIngestionService().ingest(
        PDF,
        {
            "title": "Unrelated Scientific Fixture",
            "authors": ["B. Researcher"],
            "year": 2026,
            "doi": "10.0000/unrelated-source-fixture",
            "domain": "fischer_tropsch",
        },
        repositories,
        repositories.evidence_store,
    )
    other_selection = LiteratureRepresentationSelector().select(
        other.literature_id,
        other.ingestion_id,
        "fixture-curator",
        "Separate paper for cross-source rejection test.",
        repositories,
        repositories.evidence_store,
    ).selection
    other_main = LiteratureRepresentationSelector.resolve_selection(
        other_selection, repositories, repositories.evidence_store
    )
    other_text = repositories.canonical_text_artifacts.read_text(
        other_main.canonical_text_id
    )
    other_start = other_text.index(PAGE_ONE_QUOTE)
    other_evidence = create_evidence_span_from_canonical_text(
        other_main.canonical_text_id,
        other_start,
        other_start + len(PAGE_ONE_QUOTE),
        repositories,
        repositories.evidence_store,
    )
    cross_identity = {
        **identity,
        "evidence_refs": tuple(sorted((si_evidence.evidence_id, other_evidence.evidence_id))),
    }
    cross_result = ReportedResult(
        result_id=f"result-{content_hash(cross_identity)[:24]}",
        **cross_identity,
    )
    repositories.reported_results.put(cross_result.result_id, cross_result)
    with pytest.raises(KnowledgeRetrievalError, match="ambiguous literature identity"):
        PersistentKnowledgeRetriever._literature_metadata(repositories)


def test_docx_cannot_replace_main_pdf_or_enter_pdf_only_entrance(tmp_path: Path) -> None:
    repositories, literature_id, _ = _prepared_literature(tmp_path)
    docx = _supplement_docx(tmp_path / "supplement.docx")
    raw_count = len(repositories.raw_literature_artifacts.list())
    with pytest.raises(ValueError, match="not a PDF"):
        ingest_literature_supplementary_pdf(
            docx, literature_id, repositories, repositories.evidence_store
        )
    with pytest.raises(ValueError, match="retain its original DOCX format"):
        ingest_literature_supplementary_docx(
            PDF, literature_id, repositories, repositories.evidence_store
        )
    assert len(repositories.raw_literature_artifacts.list()) == raw_count
    assert repositories.literature_representation_selections.resolve_current(
        literature_id
    ).literature_id == literature_id


def test_tampered_docx_supplement_breaks_trusted_snapshot(tmp_path: Path) -> None:
    repositories, literature_id, _ = _prepared_literature(tmp_path)
    supplement = ingest_literature_supplementary_docx(
        _supplement_docx(tmp_path / "supplement.docx"),
        literature_id,
        repositories,
        repositories.evidence_store,
    )
    curate_knowledge_record(
        repositories,
        target_type="literature_supplementary_source",
        target_id=supplement.supplement_id,
        status=CurationStatus.ACCEPTED,
        curator_id="fixture-curator",
        rationale="Reviewed original DOCX source identity only.",
    )
    raw = repositories.raw_literature_artifacts.get(supplement.raw_artifact_id)
    stored = repositories.root / raw.stored_path
    os.chmod(stored, 0o666)
    stored.write_bytes(stored.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="INVALID_SUPPLEMENTARY_SOURCE"):
        repositories.create_snapshot(
            repositories.evidence_store, DomainPackLoader().load("base").profile
        )
