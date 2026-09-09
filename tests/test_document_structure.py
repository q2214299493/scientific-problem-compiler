from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spc.cli import app
from spc.domains import DomainPackLoader
from spc.knowledge.acquisition import (
    HTTPResponse,
    LiteratureAcquisitionService,
    SafeHTTPFetcher,
)
from spc.knowledge.ingestion import (
    LiteratureIngestionService,
    LiteratureRepresentationSelector,
    PypdfLiteratureTextExtractor,
)
from spc.knowledge.structure import (
    DocumentStructureService,
    create_evidence_from_block,
    create_evidence_from_figure_caption,
    create_evidence_from_table_cell,
    format_structured_evidence_locator,
    validate_document_structure,
    verify_structured_evidence_locator,
)
from spc.models import (
    CurationStatus,
    DocumentBlockType,
    KnowledgeCurationRecord,
)
from spc.repositories import KnowledgeEvidenceStore, KnowledgeRepositories
from spc.serialization import content_hash


FIXTURES = Path(__file__).parent / "fixtures"
PDF = FIXTURES / "generic-born-digital.pdf"
ARTICLE_URL = "https://journal.example/structured"


class StaticHTMLTransport:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def request(
        self,
        url: str,
        *,
        validated_ips: tuple[str, ...],
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse:
        assert validated_ips and timeout > 0 and max_bytes > len(self.body)
        return HTTPResponse(
            url=url,
            status=200,
            headers={"content-type": "text/html"},
            body=self.body,
            connected_ip="93.184.216.34",
        )


def pdf_metadata(suffix: str = "current") -> dict[str, object]:
    return {
        "title": "Representation-Bound PDF Structure",
        "authors": ["A. Curator"],
        "year": 2026,
        "doi": f"10.0000/k1e.{suffix}",
        "domain": "base",
    }


def structured_html() -> bytes:
    return b"""<html><head>
    <meta name="citation_title" content="Representation-Bound HTML Structure">
    <meta name="citation_author" content="A. Curator">
    <meta name="citation_publication_date" content="2026">
    </head><body><article>
    <h1>Results</h1>
    <nav><h2>Navigation Heading</h2><p>Navigation text.</p></nav>
    <h2>CO dissociation</h2>
    <p>Exact paragraph evidence.</p>
    <table><caption>Table 2 Energies</caption>
      <tr><th>State</th><th>Ea</th></tr>
      <tr><td>TS1</td><td>1.2 eV</td></tr>
    </table>
    <figure><figcaption>Figure 4 pathway comparison.</figcaption></figure>
    </article></body></html>"""


def pdf_structure(tmp_path: Path, *, parser_version: str = "1.0.0"):
    root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(root)
    store = KnowledgeEvidenceStore(root)
    outcome = LiteratureIngestionService(PypdfLiteratureTextExtractor(parser_version=parser_version)).ingest(
        PDF, pdf_metadata(), repositories, store
    )
    representation = LiteratureRepresentationSelector.resolve_reference(outcome.ingestion_id, repositories, store)
    LiteratureRepresentationSelector().select(
        outcome.literature_id,
        representation.representation_id,
        "structure-test",
        "Select PDF representation for exact-offset evidence.",
        repositories,
        store,
    )
    result = DocumentStructureService().extract(
        outcome.literature_id,
        representation.representation_id,
        repositories,
        store,
    )
    return repositories, store, outcome, representation, result


def html_structure(tmp_path: Path, html: bytes | None = None):
    root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(root)
    store = KnowledgeEvidenceStore(root)
    fetcher = SafeHTTPFetcher(
        StaticHTMLTransport(html or structured_html()),
        dns_resolver=lambda _host: ("93.184.216.34",),
    )
    outcome = LiteratureAcquisitionService(fetcher).add(ARTICLE_URL, "base", repositories, store)
    LiteratureRepresentationSelector().select(
        outcome.literature_id or "",
        outcome.representation_id or "",
        "structure-test",
        "Select HTML representation for exact-offset evidence.",
        repositories,
        store,
    )
    result = DocumentStructureService().extract(
        outcome.literature_id or "",
        outcome.representation_id or "",
        repositories,
        store,
    )
    return repositories, store, outcome, result


def accept_record(
    repositories: KnowledgeRepositories,
    target_type: str,
    target_id: str,
    target_hash: str,
    *,
    evidence_refs: tuple[str, ...] = (),
    supersedes: str | None = None,
) -> KnowledgeCurationRecord:
    identity = {
        "target_type": target_type,
        "target_id": target_id,
        "target_hash": target_hash,
        "status": CurationStatus.ACCEPTED,
        "curator_id": "structure-test-curator",
        "rationale": "Accepted after representation and structure review.",
        "evidence_refs": evidence_refs,
        "supersedes_curation_id": supersedes,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    record = KnowledgeCurationRecord(**payload, content_hash=content_hash(payload))
    repositories.curations.put(record.curation_id, record)
    return record


def test_pdf_structure_binds_pages_and_exact_offsets(tmp_path: Path) -> None:
    repositories, store, _, representation, result = pdf_structure(tmp_path)
    canonical = repositories.canonical_text_artifacts.get(representation.canonical_text_id)
    text = repositories.canonical_text_artifacts.read_text(representation.canonical_text_id)

    assert result.artifact.representation_id == representation.representation_id
    assert result.artifact.representation_hash == representation.content_hash
    assert result.artifact.page_count == canonical.page_count == 2
    page_blocks = [block for block in result.blocks if block.block_type == DocumentBlockType.PAGE]
    assert [block.page_number for block in page_blocks] == [1, 2]
    assert result.table_cells == ()
    for block in result.blocks:
        recovered = text[block.start_offset : block.end_offset]
        assert block.text_hash == hashlib.sha256(recovered.encode("utf-8")).hexdigest()
    assert validate_document_structure(result.artifact, repositories, store)


def test_pdf_canonical_tamper_breaks_structure_validation(tmp_path: Path) -> None:
    repositories, store, _, representation, result = pdf_structure(tmp_path)
    canonical = repositories.canonical_text_artifacts.get(representation.canonical_text_id)
    path = repositories.root / canonical.stored_path
    os.chmod(path, 0o666)
    path.write_text("tampered canonical structure source", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical text"):
        validate_document_structure(result.artifact, repositories, store)


def test_html_hierarchy_table_cells_figure_and_structured_evidence(
    tmp_path: Path,
) -> None:
    repositories, store, outcome, result = html_structure(tmp_path)
    text = repositories.canonical_html_text_artifacts.read_text(outcome.canonical_text_id or "")
    headings = [block for block in result.blocks if block.block_type == DocumentBlockType.HEADING]
    paragraph = next(
        block for block in result.blocks if text[block.start_offset : block.end_offset] == "Exact paragraph evidence."
    )

    assert [block.section_path for block in headings] == [
        ("Results",),
        ("Results", "CO dissociation"),
    ]
    assert paragraph.section_path == ("Results", "CO dissociation")
    assert len(result.tables) == 1
    assert len(result.table_cells) == 4
    assert {(cell.row_index, cell.column_index) for cell in result.table_cells} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert len(result.figures) == 1
    for cell in result.table_cells:
        assert text[cell.start_offset : cell.end_offset]

    paragraph_evidence, paragraph_locator = create_evidence_from_block(
        result.artifact.structure_id,
        paragraph.block_id,
        repositories,
        store,
    )
    cell_evidence, cell_locator = create_evidence_from_table_cell(
        result.artifact.structure_id,
        result.tables[0].table_id,
        result.table_cells[-1].cell_id,
        repositories,
        store,
    )
    caption_evidence, caption_locator = create_evidence_from_figure_caption(
        result.artifact.structure_id,
        result.figures[0].figure_id,
        repositories,
        store,
    )

    assert paragraph_evidence.text == "Exact paragraph evidence."
    assert cell_evidence.text == "1.2 eV"
    assert caption_evidence.text == "Figure 4 pathway comparison."
    assert cell_locator.row_index == 1 and cell_locator.column_index == 1
    assert caption_locator.figure_id == result.figures[0].figure_id
    for evidence, locator in (
        (paragraph_evidence, paragraph_locator),
        (cell_evidence, cell_locator),
        (caption_evidence, caption_locator),
    ):
        assert locator.canonical_start_offset == evidence.start_offset
        assert locator.canonical_end_offset == evidence.end_offset
        assert verify_structured_evidence_locator(locator, repositories, store)
        assert format_structured_evidence_locator(locator, repositories)


def test_locator_and_evidence_tampering_fail_closed(tmp_path: Path) -> None:
    repositories, store, _, result = html_structure(tmp_path)
    paragraph = next(
        block
        for block in result.blocks
        if block.block_type == DocumentBlockType.PARAGRAPH and block.section_path == ("Results", "CO dissociation")
    )
    evidence, locator = create_evidence_from_block(
        result.artifact.structure_id, paragraph.block_id, repositories, store
    )
    locator_path = repositories.structured_evidence_locators.root / f"{locator.locator_id}.json"
    payload = json.loads(locator_path.read_text(encoding="utf-8"))
    payload["canonical_start_offset"] += 1
    locator_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_structured_evidence_locator(locator.locator_id, repositories, store)

    locator_path.write_text(locator.model_dump_json(), encoding="utf-8")
    evidence_path = store.evidence_records.root / f"{evidence.evidence_id}.json"
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_payload["text"] = "tampered"
    evidence_path.write_text(json.dumps(evidence_payload), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_structured_evidence_locator(locator, repositories, store)


def test_structure_record_tampering_fails_closed(tmp_path: Path) -> None:
    repositories, store, _, _, result = pdf_structure(tmp_path)
    block = result.blocks[1]
    block_path = repositories.document_structure_blocks.root / f"{block.block_id}.json"
    payload = json.loads(block_path.read_text(encoding="utf-8"))
    payload["text_hash"] = "0" * 64
    block_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_document_structure(result.artifact, repositories, store)


def test_malformed_html_returns_partial_structure_warning(tmp_path: Path) -> None:
    html = structured_html().replace(b"<td>TS1</td>", b'<td rowspan="invalid">TS1</td>')
    _, _, _, result = html_structure(tmp_path, html)

    assert result.artifact.extraction_status.value == "partial"
    assert any("rowspan" in warning for warning in result.artifact.warnings)


def test_historical_structure_cannot_bypass_current_representation(
    tmp_path: Path,
) -> None:
    repositories, store, first, _, first_structure = pdf_structure(tmp_path)
    second = LiteratureIngestionService(PypdfLiteratureTextExtractor(parser_version="2.0.0")).ingest(
        PDF, pdf_metadata(), repositories, store
    )
    second_representation = LiteratureRepresentationSelector.resolve_reference(second.ingestion_id, repositories, store)
    LiteratureRepresentationSelector().select(
        second.literature_id,
        second_representation.representation_id,
        "structure-test",
        "Promote the second representation.",
        repositories,
        store,
    )
    second_structure = DocumentStructureService().extract(
        second.literature_id,
        second_representation.representation_id,
        repositories,
        store,
    )
    paragraph = next(block for block in first_structure.blocks if block.block_type == DocumentBlockType.PARAGRAPH)

    with pytest.raises(ValueError, match="current selected representation"):
        create_evidence_from_block(
            first_structure.artifact.structure_id,
            paragraph.block_id,
            repositories,
            store,
        )
    evidence, locator = create_evidence_from_block(
        first_structure.artifact.structure_id,
        paragraph.block_id,
        repositories,
        store,
        historical_representation=True,
    )
    assert evidence and locator.representation_id != second_representation.representation_id
    assert first.literature_id == second.literature_id
    assert first_structure.artifact.structure_id != second_structure.artifact.structure_id


def test_trusted_snapshot_binds_current_structure_and_locator(tmp_path: Path) -> None:
    repositories, store, outcome, result = html_structure(tmp_path)
    document = repositories.literature_documents.get(outcome.literature_id or "")
    selection = repositories.literature_representation_selections.resolve_current(document.literature_id)
    document_curation = accept_record(
        repositories,
        "literature_document",
        document.literature_id,
        document.content_hash,
    )
    del document_curation
    selection_curation = accept_record(
        repositories,
        "literature_representation_selection",
        selection.selection_id,
        selection.content_hash,
    )
    paragraph = next(
        block
        for block in result.blocks
        if block.block_type == DocumentBlockType.PARAGRAPH and block.section_path == ("Results", "CO dissociation")
    )
    evidence, locator = create_evidence_from_block(
        result.artifact.structure_id, paragraph.block_id, repositories, store
    )
    accept_record(
        repositories,
        "literature_representation_selection",
        selection.selection_id,
        selection.content_hash,
        evidence_refs=(evidence.evidence_id,),
        supersedes=selection_curation.curation_id,
    )

    snapshot = repositories.create_snapshot(store, DomainPackLoader().load("base").profile)

    assert snapshot.document_structure_hashes == {result.artifact.structure_id: result.artifact.content_hash}
    assert snapshot.structured_evidence_locator_hashes == {locator.locator_id: locator.content_hash}


def test_structure_cli_and_shared_repository_are_project_independent(
    tmp_path: Path,
) -> None:
    repositories, _, outcome, representation, result = pdf_structure(tmp_path)
    command = CliRunner().invoke(
        app,
        [
            "structure-literature",
            "--literature-id",
            outcome.literature_id,
            "--representation-id",
            representation.representation_id,
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
        ],
    )
    assert command.exit_code == 0, command.output
    assert json.loads(command.output)["structure_id"] == result.artifact.structure_id
    reopened = KnowledgeRepositories(tmp_path / "knowledge")
    assert reopened.document_structure_artifacts.get(result.artifact.structure_id) == result.artifact
    assert not (tmp_path / "project-a" / ".spc" / "document_structure_artifacts").exists()

    inspection = CliRunner().invoke(
        app,
        [
            "inspect-literature-structure",
            "--structure-id",
            result.artifact.structure_id,
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--page",
            "1",
        ],
    )
    assert inspection.exit_code == 0, inspection.output
    assert all(block["page_number"] == 1 for block in json.loads(inspection.output)["blocks"])
