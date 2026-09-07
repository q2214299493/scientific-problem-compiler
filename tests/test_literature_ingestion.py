from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from typer.testing import CliRunner

from spc.cli import app
from spc.domains import DomainPackLoader
from spc.knowledge.ingestion import (
    LiteratureExtractionError,
    LiteratureIngestionService,
    LiteratureRepresentationSelector,
    PypdfLiteratureTextExtractor,
    create_evidence_span_from_canonical_text,
)
from spc.models import (
    CurationStatus,
    HistoricalLiteratureEvidenceAuthorization,
    KnowledgeCurationRecord,
    LiteratureRepresentationSelection,
)
from spc.repositories import KnowledgeRepositories, SourceEvidenceStore
from spc.serialization import content_hash, dump_yaml


FIXTURES = Path(__file__).parent / "fixtures"
BORN_DIGITAL_PDF = FIXTURES / "generic-born-digital.pdf"
TEXTLESS_PDF = FIXTURES / "generic-textless.pdf"


def metadata() -> dict[str, object]:
    return {
        "title": "Generic Born-Digital Scientific Fixture",
        "authors": ["A. Researcher", "B. Curator"],
        "year": 2026,
        "journal": "Journal of Offline Fixtures",
        "doi": "10.0000/example.k1b",
        "url": "https://example.org/k1b",
        "domain": "base",
        "topics": ["pathway evidence"],
        "keywords": ["canonical text", "provenance"],
        "citation_refs": [],
    }


def ingest_fixture(
    tmp_path: Path,
    *,
    pdf_path: Path = BORN_DIGITAL_PDF,
    parser_version: str = "1.0.0",
    select: bool = True,
) -> tuple[KnowledgeRepositories, SourceEvidenceStore, object]:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    evidence_store = SourceEvidenceStore(tmp_path / ".spc")
    outcome = LiteratureIngestionService(
        PypdfLiteratureTextExtractor(parser_version=parser_version)
    ).ingest(pdf_path, metadata(), repositories, evidence_store)
    if select and outcome.canonical_text_id is not None:
        selection_outcome = LiteratureRepresentationSelector().select(
            outcome.literature_id,
            outcome.ingestion_id,
            "test-selector",
            "Selected explicitly for the ingestion test fixture.",
            repositories,
            evidence_store,
        )
        outcome = outcome.model_copy(
            update={"selection_id": selection_outcome.selection.selection_id}
        )
    return repositories, evidence_store, outcome


def accept_literature(repositories: KnowledgeRepositories, literature_id: str) -> None:
    document = repositories.literature_documents.get(literature_id)
    identity = {
        "target_type": "literature_document",
        "target_id": literature_id,
        "target_hash": document.content_hash,
        "status": CurationStatus.ACCEPTED,
        "curator_id": "curator-k1b",
        "rationale": "Accepted after deterministic artifact ingestion.",
        "evidence_refs": (),
    }
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    curation = KnowledgeCurationRecord(
        **payload,
        content_hash=content_hash(payload),
    )
    repositories.curations.put(curation.curation_id, curation)
    selection = repositories.literature_representation_selections.resolve_current(
        literature_id
    )
    selection_identity = {
        "target_type": "literature_representation_selection",
        "target_id": selection.selection_id,
        "target_hash": selection.content_hash,
        "status": CurationStatus.ACCEPTED,
        "curator_id": "curator-k1b",
        "rationale": "Accepted after explicit representation review.",
        "evidence_refs": (),
    }
    selection_curation_id = (
        f"knowledge-curation-{content_hash(selection_identity)[:24]}"
    )
    selection_payload = {
        "curation_id": selection_curation_id,
        **selection_identity,
    }
    selection_curation = KnowledgeCurationRecord(
        **selection_payload,
        content_hash=content_hash(selection_payload),
    )
    repositories.curations.put(
        selection_curation.curation_id, selection_curation
    )


def make_selection(
    ingestion,
    *,
    supersedes_selection_id: str | None = None,
    **overrides,
) -> LiteratureRepresentationSelection:
    identity = {
        "literature_id": ingestion.literature_id,
        "ingestion_id": ingestion.ingestion_id,
        "ingestion_hash": ingestion.content_hash,
        "canonical_text_id": ingestion.canonical_text_id,
        "canonical_text_hash": ingestion.canonical_text_hash,
        "source_id": ingestion.source_id,
        "source_version": ingestion.source_version,
        "selected_by": "test-curator",
        "rationale": "Test representation selection.",
        "supersedes_selection_id": supersedes_selection_id,
    }
    identity.update(overrides)
    identity = {key: value for key, value in identity.items() if value is not None}
    selection_id = f"literature-selection-{content_hash(identity)[:24]}"
    payload = {"selection_id": selection_id, **identity}
    return LiteratureRepresentationSelection(
        **payload,
        content_hash=content_hash(payload),
    )


def put_curation(
    repositories: KnowledgeRepositories,
    *,
    target_type: str,
    target_id: str,
    target_hash: str,
    evidence_refs: tuple[str, ...] = (),
    supersedes: str | None = None,
) -> KnowledgeCurationRecord:
    identity = {
        "target_type": target_type,
        "target_id": target_id,
        "target_hash": target_hash,
        "status": CurationStatus.ACCEPTED,
        "curator_id": "curator-k1b2",
        "rationale": "Accepted under the K1B.2 promotion policy.",
        "evidence_refs": evidence_refs,
        "supersedes_curation_id": supersedes,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    curation = KnowledgeCurationRecord(
        **payload,
        content_hash=content_hash(payload),
    )
    repositories.curations.put(curation.curation_id, curation)
    return curation


def test_raw_pdf_bytes_are_stored_unchanged_and_hash_is_deterministic(tmp_path) -> None:
    repositories, _, outcome = ingest_fixture(tmp_path)
    artifact = repositories.raw_literature_artifacts.get(outcome.artifact_id)
    stored = repositories.root / artifact.stored_path
    assert stored.read_bytes() == BORN_DIGITAL_PDF.read_bytes()
    assert artifact.sha256 == hashlib.sha256(BORN_DIGITAL_PDF.read_bytes()).hexdigest()
    assert repositories.raw_literature_artifacts.put(
        BORN_DIGITAL_PDF, outcome.literature_id
    ) == artifact


def test_same_pdf_bytes_under_different_filenames_reuse_raw_artifact(tmp_path) -> None:
    repositories, _, outcome = ingest_fixture(tmp_path)
    renamed = tmp_path / "renamed-local-copy.pdf"
    renamed.write_bytes(BORN_DIGITAL_PDF.read_bytes())
    reused = repositories.raw_literature_artifacts.put(
        renamed, outcome.literature_id
    )
    assert reused.artifact_id == outcome.artifact_id
    assert reused == repositories.raw_literature_artifacts.get(outcome.artifact_id)
    assert len(repositories.raw_literature_artifacts.list()) == 1


def test_raw_pdf_cannot_be_overwritten_with_different_bytes(tmp_path) -> None:
    repositories, _, outcome = ingest_fixture(tmp_path)
    artifact = repositories.raw_literature_artifacts.get(outcome.artifact_id)
    stored = repositories.root / artifact.stored_path
    os.chmod(stored, 0o666)
    stored.write_bytes(b"%PDF-1.3\nmodified")
    with pytest.raises(ValueError, match="byte size changed|hash changed"):
        repositories.raw_literature_artifacts.put(
            BORN_DIGITAL_PDF, outcome.literature_id
        )


def test_artifact_store_rejects_path_traversal_and_symlink(tmp_path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    with pytest.raises(ValueError, match="safe path component"):
        repositories.raw_literature_artifacts.put(BORN_DIGITAL_PDF, "../paper")
    link = tmp_path / "paper-link.pdf"
    try:
        link.symlink_to(BORN_DIGITAL_PDF)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(ValueError, match="non-symlink"):
        repositories.raw_literature_artifacts.put(link, "literature-safe")


def test_pdf_bytes_are_not_used_as_evidence_source(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path)
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    source = evidence_store.source_records.get(
        f"{ingestion.source_id}--{ingestion.source_version}"
    )
    stored_source = evidence_store.state_root / source.stored_path
    assert not stored_source.read_bytes().startswith(b"%PDF-")
    assert stored_source.read_text(encoding="utf-8") == (
        repositories.canonical_text_artifacts.read_text(outcome.canonical_text_id)
    )


def test_canonical_text_page_order_utf8_and_block_offsets_are_deterministic(
    tmp_path,
) -> None:
    repositories, _, outcome = ingest_fixture(tmp_path)
    canonical = repositories.canonical_text_artifacts.get(outcome.canonical_text_id)
    text = repositories.canonical_text_artifacts.read_text(outcome.canonical_text_id)
    assert text.encode("utf-8").decode("utf-8") == text
    assert text.index("page one") < text.index("page two")
    assert canonical.page_count == 2
    assert [block.page_number for block in canonical.blocks] == [1, 2]
    for block in canonical.blocks:
        recovered = text[block.start_offset : block.end_offset]
        assert hashlib.sha256(recovered.encode("utf-8")).hexdigest() == block.text_hash


def test_tampered_raw_artifact_invalidates_trusted_snapshot(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path)
    accept_literature(repositories, outcome.literature_id)
    artifact = repositories.raw_literature_artifacts.get(outcome.artifact_id)
    path = repositories.root / artifact.stored_path
    os.chmod(path, 0o666)
    path.write_bytes(b"%PDF-1.3\ntampered")
    with pytest.raises(ValueError, match="byte size changed|hash changed"):
        repositories.create_snapshot(
            evidence_store, DomainPackLoader().load("base").profile
        )


def test_trusted_snapshot_binds_complete_literature_ingestion_chain(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path)
    accept_literature(repositories, outcome.literature_id)
    snapshot = repositories.create_snapshot(
        evidence_store, DomainPackLoader().load("base").profile
    )
    assert outcome.literature_id in snapshot.literature_document_hashes
    assert outcome.artifact_id in snapshot.raw_literature_artifact_hashes
    assert outcome.canonical_text_id in snapshot.canonical_text_artifact_hashes
    assert outcome.ingestion_id in snapshot.literature_ingestion_hashes
    assert (
        outcome.selection_id
        in snapshot.literature_representation_selection_hashes
    )
    for prefix in (
        "literature_document:",
        "raw_literature_artifact:",
        "canonical_text_artifact:",
        "literature_ingestion:",
        "literature_representation_selection:",
        "source_document:",
    ):
        assert any(key.startswith(prefix) for key in snapshot.trusted_record_hashes)


def test_tampered_canonical_text_invalidates_trusted_snapshot(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path)
    accept_literature(repositories, outcome.literature_id)
    canonical = repositories.canonical_text_artifacts.get(outcome.canonical_text_id)
    path = repositories.root / canonical.stored_path
    os.chmod(path, 0o666)
    path.write_text("tampered canonical text", encoding="utf-8")
    with pytest.raises(ValueError, match="character count changed|text hash changed"):
        repositories.create_snapshot(
            evidence_store, DomainPackLoader().load("base").profile
        )


def test_evidence_span_offsets_recover_exact_canonical_text(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path)
    text = repositories.canonical_text_artifacts.read_text(outcome.canonical_text_id)
    start = text.index("Mechanistic connectivity")
    end = start + len("Mechanistic connectivity must be preserved.")
    evidence = create_evidence_span_from_canonical_text(
        outcome.canonical_text_id,
        start,
        end,
        repositories,
        evidence_store,
        locator="page=1;block=2",
    )
    assert evidence.text == text[start:end]
    assert evidence_store.verify_evidence_integrity(evidence)


def test_current_representation_selection_is_deterministic(tmp_path) -> None:
    repositories, evidence_store, first = ingest_fixture(tmp_path)
    repeated = LiteratureIngestionService().ingest(
        BORN_DIGITAL_PDF, metadata(), repositories, evidence_store
    )
    current = repositories.literature_representation_selections.resolve_current(
        first.literature_id
    )
    assert repeated.ingestion_id == first.ingestion_id
    assert repeated.selection_id is None
    assert first.selection_id == current.selection_id
    assert len(repositories.literature_representation_selections.list()) == 1


def test_new_parser_version_preserves_literature_identity_and_old_ingestion(
    tmp_path,
) -> None:
    repositories, evidence_store, first = ingest_fixture(tmp_path)
    first_record = repositories.literature_ingestions.get(first.ingestion_id)
    first_document = repositories.literature_documents.get(first.literature_id)
    second = LiteratureIngestionService(
        PypdfLiteratureTextExtractor(parser_version="1.1.0")
    ).ingest(BORN_DIGITAL_PDF, metadata(), repositories, evidence_store)
    assert second.literature_id == first.literature_id
    assert second.canonical_text_id != first.canonical_text_id
    assert second.ingestion_id != first.ingestion_id
    assert repositories.literature_ingestions.get(first.ingestion_id) == first_record
    assert repositories.literature_documents.get(first.literature_id) == first_document
    assert len(repositories.literature_ingestions.list()) == 2
    first_selection = repositories.literature_representation_selections.get(
        first.selection_id
    )
    current = repositories.literature_representation_selections.resolve_current(
        first.literature_id
    )
    assert second.selection_id is None
    assert current.selection_id == first.selection_id
    promoted = LiteratureRepresentationSelector().select(
        second.literature_id,
        second.ingestion_id,
        "test-selector",
        "Promote the reprocessed representation.",
        repositories,
        evidence_store,
    )
    current = repositories.literature_representation_selections.resolve_current(
        first.literature_id
    )
    assert current.selection_id == promoted.selection.selection_id
    assert current.supersedes_selection_id == first.selection_id
    assert repositories.literature_representation_selections.get(
        first.selection_id
    ) == first_selection


def test_ambiguous_current_representation_is_rejected(tmp_path) -> None:
    repositories, _, outcome = ingest_fixture(tmp_path)
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    first = repositories.literature_representation_selections.get(
        outcome.selection_id
    )
    for selected_by in ("curator-a", "curator-b"):
        selection = make_selection(
            ingestion,
            supersedes_selection_id=first.selection_id,
            selected_by=selected_by,
        )
        repositories.literature_representation_selections.put(
            selection.selection_id, selection
        )
    with pytest.raises(ValueError, match="multiple successors"):
        repositories.literature_representation_selections.resolve_current(
            outcome.literature_id
        )


def test_selection_cannot_point_to_ingestion_from_another_literature(tmp_path) -> None:
    repositories, evidence_store, first = ingest_fixture(tmp_path)
    other_metadata = {**metadata(), "doi": "10.0000/example.k1b.other"}
    other = LiteratureIngestionService().ingest(
        BORN_DIGITAL_PDF, other_metadata, repositories, evidence_store
    )
    other_ingestion = repositories.literature_ingestions.get(other.ingestion_id)
    bad = make_selection(
        other_ingestion,
        literature_id=first.literature_id,
        supersedes_selection_id=first.selection_id,
    )
    repositories.literature_representation_selections.put(bad.selection_id, bad)
    accept_literature(repositories, first.literature_id)
    with pytest.raises(ValueError, match="selection binding is invalid"):
        repositories.create_snapshot(
            evidence_store, DomainPackLoader().load("base").profile
        )


def test_selection_ingestion_hash_mismatch_is_rejected(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path)
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    bad = make_selection(
        ingestion,
        supersedes_selection_id=outcome.selection_id,
        ingestion_hash="0" * 64,
    )
    repositories.literature_representation_selections.put(bad.selection_id, bad)
    accept_literature(repositories, outcome.literature_id)
    with pytest.raises(ValueError, match="selection binding is invalid"):
        repositories.create_snapshot(
            evidence_store, DomainPackLoader().load("base").profile
        )


def test_evidence_span_requires_selected_representation_by_default(tmp_path) -> None:
    repositories, evidence_store, first = ingest_fixture(tmp_path)
    first_text = repositories.canonical_text_artifacts.read_text(
        first.canonical_text_id
    )
    second = LiteratureIngestionService(
        PypdfLiteratureTextExtractor(parser_version="1.1.0")
    ).ingest(BORN_DIGITAL_PDF, metadata(), repositories, evidence_store)
    LiteratureRepresentationSelector().select(
        second.literature_id,
        second.ingestion_id,
        "test-selector",
        "Promote the second representation.",
        repositories,
        evidence_store,
    )
    start = first_text.index("Mechanistic connectivity")
    end = start + len("Mechanistic connectivity must be preserved.")
    with pytest.raises(ValueError, match="not the current selected"):
        create_evidence_span_from_canonical_text(
            first.canonical_text_id,
            start,
            end,
            repositories,
            evidence_store,
        )
    historical = create_evidence_span_from_canonical_text(
        first.canonical_text_id,
        start,
        end,
        repositories,
        evidence_store,
        historical_ingestion=True,
    )
    assert historical.text == first_text[start:end]


def test_evidence_span_rejects_broken_raw_canonical_source_chain(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path)
    artifact = repositories.raw_literature_artifacts.get(outcome.artifact_id)
    path = repositories.root / artifact.stored_path
    os.chmod(path, 0o666)
    path.write_bytes(b"%PDF-1.3\ntampered")
    with pytest.raises(ValueError, match="byte size changed|hash changed"):
        create_evidence_span_from_canonical_text(
            outcome.canonical_text_id,
            0,
            1,
            repositories,
            evidence_store,
        )


def test_evidence_span_rejects_tampered_source_binding(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path)
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    source = evidence_store.source_records.get(
        f"{ingestion.source_id}--{ingestion.source_version}"
    )
    source_path = evidence_store.state_root / source.stored_path
    os.chmod(source_path, 0o666)
    source_path.write_text("tampered source text", encoding="utf-8")
    with pytest.raises(ValueError, match="stored source content hash"):
        create_evidence_span_from_canonical_text(
            outcome.canonical_text_id,
            0,
            1,
            repositories,
            evidence_store,
        )


def test_expected_parser_failure_persists_failed_ingestion(tmp_path) -> None:
    class FailingExtractor:
        parser_id = "failing-parser"
        parser_version = "1.0.0"
        parser_config_hash = "f" * 64

        def extract(self, artifact, artifact_path):
            raise LiteratureExtractionError("fixture parser failure")

    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    evidence_store = SourceEvidenceStore(tmp_path / ".spc")
    outcome = LiteratureIngestionService(FailingExtractor()).ingest(
        BORN_DIGITAL_PDF, metadata(), repositories, evidence_store
    )
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    assert ingestion.ingestion_status == "failed"
    assert ingestion.warnings == ("PDF extraction failed",)
    assert ingestion.total_pages == 0
    assert repositories.raw_literature_artifacts.get(outcome.artifact_id)
    assert repositories.literature_representation_selections.list() == ()
    assert repositories.literature_documents.list() == ()


def test_partially_text_bearing_pdf_records_extraction_coverage(tmp_path) -> None:
    partial_pdf = tmp_path / "partial.pdf"
    reader = PdfReader(BORN_DIGITAL_PDF)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.add_blank_page(width=612, height=792)
    with partial_pdf.open("wb") as stream:
        writer.write(stream)
    repositories, _, outcome = ingest_fixture(
        tmp_path, pdf_path=partial_pdf, select=False
    )
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    assert ingestion.ingestion_status == "accepted"
    assert ingestion.total_pages == 2
    assert ingestion.pages_with_text == 1
    assert ingestion.pages_without_text == 1
    assert ingestion.text_coverage_ratio == 0.5
    assert ingestion.warnings == ("page 2 has no extractable text",)
    assert repositories.literature_representation_selections.list() == ()


def test_textless_pdf_requires_ocr_without_attempting_it(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(
        tmp_path, pdf_path=TEXTLESS_PDF
    )
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    assert outcome.requires_ocr is True
    assert outcome.canonical_text_id is None
    assert outcome.source_id is None
    assert ingestion.ingestion_status == "requires_ocr"
    assert ingestion.total_pages == 1
    assert ingestion.pages_with_text == 0
    assert ingestion.pages_without_text == 1
    assert ingestion.text_coverage_ratio == 0.0
    assert "OCR is required" in ingestion.warnings[-1]
    assert repositories.canonical_text_artifacts.list() == ()
    assert evidence_store.source_records.list() == ()
    assert repositories.literature_documents.list() == ()


def test_generic_non_ft_pdf_fixture_and_cli_work_offline(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.yaml"
    dump_yaml(metadata_path, metadata())
    result = CliRunner().invoke(
        app,
        [
            "ingest-literature",
            str(BORN_DIGITAL_PDF),
            "--metadata",
            str(metadata_path),
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--state-dir",
            str(tmp_path / ".spc"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"requires_ocr": false' in result.output
    assert "fischer_tropsch" not in result.output


def test_trusted_snapshot_changes_when_accepted_ingestion_changes(tmp_path) -> None:
    repositories, evidence_store, first = ingest_fixture(tmp_path)
    accept_literature(repositories, first.literature_id)
    first_snapshot = repositories.create_snapshot(
        evidence_store, DomainPackLoader().load("base").profile
    )
    second = LiteratureIngestionService(
        PypdfLiteratureTextExtractor(parser_version="1.1.0")
    ).ingest(BORN_DIGITAL_PDF, metadata(), repositories, evidence_store)
    promoted = LiteratureRepresentationSelector().select(
        second.literature_id,
        second.ingestion_id,
        "test-selector",
        "Promote the second parser output.",
        repositories,
        evidence_store,
    )
    accept_literature(repositories, second.literature_id)
    second_snapshot = repositories.create_snapshot(
        evidence_store, DomainPackLoader().load("base").profile
    )
    assert second_snapshot.snapshot_id != first_snapshot.snapshot_id
    assert second.ingestion_id in second_snapshot.literature_ingestion_hashes
    assert (
        first.selection_id
        in first_snapshot.literature_representation_selection_hashes
    )
    assert (
        promoted.selection.selection_id
        in second_snapshot.literature_representation_selection_hashes
    )
    assert (
        first.selection_id
        not in second_snapshot.literature_representation_selection_hashes
    )


def test_snapshot_changes_when_only_current_representation_selection_changes(
    tmp_path,
) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path)
    accept_literature(repositories, outcome.literature_id)
    first_snapshot = repositories.create_snapshot(
        evidence_store, DomainPackLoader().load("base").profile
    )
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    replacement = make_selection(
        ingestion,
        supersedes_selection_id=outcome.selection_id,
        selected_by="human-curator",
        rationale="Prefer the reviewed representation.",
    )
    repositories.literature_representation_selections.put(
        replacement.selection_id, replacement
    )
    accept_literature(repositories, outcome.literature_id)
    second_snapshot = repositories.create_snapshot(
        evidence_store, DomainPackLoader().load("base").profile
    )
    assert second_snapshot.snapshot_id != first_snapshot.snapshot_id
    assert first_snapshot.literature_ingestion_hashes == (
        second_snapshot.literature_ingestion_hashes
    )
    assert replacement.selection_id in (
        second_snapshot.literature_representation_selection_hashes
    )


def test_successful_ingestion_does_not_create_initial_selection(tmp_path) -> None:
    repositories, _, outcome = ingest_fixture(tmp_path, select=False)
    assert outcome.selection_id is None
    assert repositories.literature_ingestions.get(outcome.ingestion_id)
    assert repositories.literature_representation_selections.list() == ()


def test_unselected_successful_ingestion_is_excluded_from_trusted_snapshot(
    tmp_path,
) -> None:
    repositories, evidence_store, selected = ingest_fixture(tmp_path)
    accept_literature(repositories, selected.literature_id)
    unselected = LiteratureIngestionService(
        PypdfLiteratureTextExtractor(parser_version="2.0.0")
    ).ingest(BORN_DIGITAL_PDF, metadata(), repositories, evidence_store)
    snapshot = repositories.create_snapshot(
        evidence_store, DomainPackLoader().load("base").profile
    )
    assert selected.ingestion_id in snapshot.literature_ingestion_hashes
    assert unselected.ingestion_id not in snapshot.literature_ingestion_hashes
    assert selected.canonical_text_id in snapshot.canonical_text_artifact_hashes
    assert unselected.canonical_text_id not in snapshot.canonical_text_artifact_hashes
    assert repositories.literature_ingestions.get(unselected.ingestion_id)


def test_tampered_unselected_canonical_does_not_break_current_snapshot(
    tmp_path,
) -> None:
    repositories, evidence_store, selected = ingest_fixture(tmp_path)
    accept_literature(repositories, selected.literature_id)
    unselected = LiteratureIngestionService(
        PypdfLiteratureTextExtractor(parser_version="2.0.0")
    ).ingest(BORN_DIGITAL_PDF, metadata(), repositories, evidence_store)
    canonical = repositories.canonical_text_artifacts.get(
        unselected.canonical_text_id
    )
    path = repositories.root / canonical.stored_path
    os.chmod(path, 0o666)
    path.write_text("tampered unused historical representation", encoding="utf-8")
    snapshot = repositories.create_snapshot(
        evidence_store, DomainPackLoader().load("base").profile
    )
    assert selected.ingestion_id in snapshot.literature_ingestion_hashes
    assert unselected.ingestion_id not in snapshot.literature_ingestion_hashes


def test_uncurated_current_selection_cannot_enter_trusted_snapshot(tmp_path) -> None:
    repositories, evidence_store, outcome = ingest_fixture(tmp_path, select=False)
    selection = LiteratureRepresentationSelector().select(
        outcome.literature_id,
        outcome.ingestion_id,
        "human-selector",
        "Candidate representation requires independent curation.",
        repositories,
        evidence_store,
    ).selection
    document = repositories.literature_documents.get(outcome.literature_id)
    put_curation(
        repositories,
        target_type="literature_document",
        target_id=document.literature_id,
        target_hash=document.content_hash,
    )
    with pytest.raises(ValueError, match="selection is not accepted"):
        repositories.create_snapshot(
            evidence_store, DomainPackLoader().load("base").profile
        )
    assert repositories.literature_representation_selections.get(
        selection.selection_id
    ) == selection


def test_selector_cli_promotes_without_curating_or_extracting_science(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.yaml"
    dump_yaml(metadata_path, metadata())
    knowledge_dir = tmp_path / "knowledge"
    state_dir = tmp_path / ".spc"
    ingest_result = CliRunner().invoke(
        app,
        [
            "ingest-literature",
            str(BORN_DIGITAL_PDF),
            "--metadata",
            str(metadata_path),
            "--knowledge-dir",
            str(knowledge_dir),
            "--state-dir",
            str(state_dir),
        ],
    )
    assert ingest_result.exit_code == 0, ingest_result.output
    ingested = json.loads(ingest_result.output)
    assert ingested["selection_id"] is None
    select_result = CliRunner().invoke(
        app,
        [
            "select-literature-representation",
            "--literature-id",
            ingested["literature_id"],
            "--ingestion-id",
            ingested["ingestion_id"],
            "--selected-by",
            "cli-curator",
            "--rationale",
            "Explicit promotion after reviewing extraction metrics.",
            "--knowledge-dir",
            str(knowledge_dir),
            "--state-dir",
            str(state_dir),
        ],
    )
    assert select_result.exit_code == 0, select_result.output
    promoted = json.loads(select_result.output)
    assert promoted["ingestion_status"] == "accepted"
    assert promoted["text_coverage_ratio"] == 1.0
    assert promoted["warnings"] == []
    repositories = KnowledgeRepositories(knowledge_dir)
    assert repositories.literature_representation_selections.get(
        promoted["selection"]["selection_id"]
    )
    assert repositories.curations.list() == ()


def test_historical_evidence_requires_explicit_authorization(tmp_path) -> None:
    repositories, evidence_store, first = ingest_fixture(tmp_path)
    accept_literature(repositories, first.literature_id)
    second = LiteratureIngestionService(
        PypdfLiteratureTextExtractor(parser_version="2.0.0")
    ).ingest(BORN_DIGITAL_PDF, metadata(), repositories, evidence_store)
    current = LiteratureRepresentationSelector().select(
        second.literature_id,
        second.ingestion_id,
        "human-selector",
        "Promote the reviewed second representation.",
        repositories,
        evidence_store,
    ).selection
    accept_literature(repositories, second.literature_id)

    first_text = repositories.canonical_text_artifacts.read_text(
        first.canonical_text_id
    )
    start = first_text.index("Mechanistic connectivity")
    historical_evidence = create_evidence_span_from_canonical_text(
        first.canonical_text_id,
        start,
        start + len("Mechanistic connectivity must be preserved."),
        repositories,
        evidence_store,
        historical_ingestion=True,
    )
    current_curations = [
        item
        for item in repositories.curations.list()
        if item.target_type == "literature_representation_selection"
        and item.target_id == current.selection_id
    ]
    assert len(current_curations) == 1
    put_curation(
        repositories,
        target_type="literature_representation_selection",
        target_id=current.selection_id,
        target_hash=current.content_hash,
        evidence_refs=(historical_evidence.evidence_id,),
        supersedes=current_curations[0].curation_id,
    )
    with pytest.raises(ValueError, match="historical literature evidence"):
        repositories.create_snapshot(
            evidence_store, DomainPackLoader().load("base").profile
        )

    first_ingestion = repositories.literature_ingestions.get(first.ingestion_id)
    authorization_identity = {
        "literature_id": first_ingestion.literature_id,
        "ingestion_id": first_ingestion.ingestion_id,
        "ingestion_hash": first_ingestion.content_hash,
        "source_id": first_ingestion.source_id,
        "source_version": first_ingestion.source_version,
        "evidence_refs": (historical_evidence.evidence_id,),
        "authorized_by": "historical-evidence-curator",
        "rationale": "Permit this exact historical quotation for comparison.",
    }
    authorization_id = (
        "historical-evidence-authorization-"
        f"{content_hash(authorization_identity)[:24]}"
    )
    authorization_payload = {
        "authorization_id": authorization_id,
        **authorization_identity,
    }
    authorization = HistoricalLiteratureEvidenceAuthorization(
        **authorization_payload,
        content_hash=content_hash(authorization_payload),
    )
    repositories.historical_evidence_authorizations.put(
        authorization.authorization_id, authorization
    )
    snapshot = repositories.create_snapshot(
        evidence_store, DomainPackLoader().load("base").profile
    )
    assert historical_evidence.evidence_id in snapshot.evidence_span_hashes
    assert authorization.authorization_id in (
        snapshot.historical_evidence_authorization_hashes
    )
