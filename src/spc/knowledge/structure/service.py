"""Persist and inspect document structure."""

from __future__ import annotations

from pathlib import PurePosixPath

from ...models import (
    DocumentStructureArtifact,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
)
from ...repositories import EvidenceStore, KnowledgeRepositories
from ...serialization import content_hash
from ..representation import validate_literature_representation
from .html import HTMLDocumentStructureExtractor
from .pdf import PDFDocumentStructureExtractor
from .records import DocumentStructureExtractionResult, DocumentStructureExtractor, DocumentStructureInput
from .validation import _validate_structure_records, validate_document_structure


def _structure_id(
    representation: LiteratureRepresentationReference,
    extractor: DocumentStructureExtractor,
) -> str:
    identity = {
        "literature_id": representation.literature_id,
        "representation_id": representation.representation_id,
        "representation_hash": representation.content_hash,
        "representation_kind": representation.representation_kind,
        "canonical_text_id": representation.canonical_text_id,
        "canonical_text_hash": representation.canonical_text_hash,
        "source_id": representation.source_id,
        "source_version": representation.source_version,
        "extractor_id": extractor.extractor_id,
        "extractor_version": extractor.extractor_version,
        "extractor_config_hash": extractor.extractor_config_hash,
    }
    return f"document-structure-{content_hash(identity)[:24]}"


class DocumentStructureService:
    def extract(
        self,
        literature_id: str,
        representation_id: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
        extractor: DocumentStructureExtractor | None = None,
    ) -> DocumentStructureExtractionResult:
        store = evidence_store or repositories.evidence_store
        custom_extractor = extractor is not None
        representation = repositories.literature_representation_refs.get(representation_id)
        validate_literature_representation(representation, repositories, store)
        if representation.literature_id != literature_id:
            raise ValueError("representation belongs to another literature work")
        if representation.representation_kind == LiteratureRepresentationKind.PDF:
            canonical = repositories.canonical_text_artifacts.get(representation.canonical_text_id)
            text = repositories.canonical_text_artifacts.read_text(representation.canonical_text_id)
            raw_content = None
            selected_extractor = extractor or PDFDocumentStructureExtractor()
        else:
            canonical = repositories.canonical_html_text_artifacts.get(representation.canonical_text_id)
            text = repositories.canonical_html_text_artifacts.read_text(representation.canonical_text_id)
            raw = repositories.raw_html_literature_artifacts.get(representation.raw_artifact_id)
            raw_path = repositories.root.joinpath(*PurePosixPath(raw.stored_path).parts)
            raw_content = raw_path.read_bytes()
            selected_extractor = extractor or HTMLDocumentStructureExtractor()
        if selected_extractor.representation_kind != representation.representation_kind:
            raise ValueError("structure extractor is incompatible with representation kind")
        structure_id = _structure_id(representation, selected_extractor)
        extracted = selected_extractor.extract(
            structure_id,
            DocumentStructureInput(
                representation=representation,
                canonical=canonical,
                canonical_text=text,
                raw_content=raw_content,
            ),
        )
        identity = {
            "literature_id": representation.literature_id,
            "representation_id": representation.representation_id,
            "representation_hash": representation.content_hash,
            "representation_kind": representation.representation_kind,
            "canonical_text_id": representation.canonical_text_id,
            "canonical_text_hash": representation.canonical_text_hash,
            "source_id": representation.source_id,
            "source_version": representation.source_version,
            "extractor_id": selected_extractor.extractor_id,
            "extractor_version": selected_extractor.extractor_version,
            "extractor_config_hash": selected_extractor.extractor_config_hash,
            "block_hashes": {block.block_id: block.content_hash for block in extracted.blocks},
            "table_hashes": {table.table_id: table.content_hash for table in extracted.tables},
            "figure_hashes": {figure.figure_id: figure.content_hash for figure in extracted.figures},
            "page_count": extracted.page_count,
            "extraction_status": extracted.status,
            "warnings": extracted.warnings,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        artifact_payload = {"structure_id": structure_id, **identity}
        artifact = DocumentStructureArtifact(
            **artifact_payload,
            content_hash=content_hash(artifact_payload),
        )
        _validate_structure_records(
            artifact,
            extracted.blocks,
            extracted.tables,
            extracted.table_cells,
            extracted.figures,
            representation,
            canonical,
            text,
        )
        for block in extracted.blocks:
            repositories.document_structure_blocks.put(block.block_id, block)
        for cell in extracted.table_cells:
            repositories.table_cell_structures.put(cell.cell_id, cell)
        for table in extracted.tables:
            repositories.table_structures.put(table.table_id, table)
        for figure in extracted.figures:
            repositories.figure_structures.put(figure.figure_id, figure)
        repositories.document_structure_artifacts.put(artifact.structure_id, artifact)
        validate_document_structure(artifact, repositories, store)
        from ..structure_selection import DocumentStructureSelector

        selection = None
        if not custom_extractor:
            selection = DocumentStructureSelector().consider_auto_promotion(
                literature_id,
                representation_id,
                artifact.structure_id,
                repositories,
                store,
            )
        return DocumentStructureExtractionResult(
            artifact=artifact,
            selection=selection,
            blocks=extracted.blocks,
            tables=extracted.tables,
            table_cells=extracted.table_cells,
            figures=extracted.figures,
        )


def inspect_document_structure(
    structure_id: str,
    repositories: KnowledgeRepositories,
    *,
    page: int | None = None,
    section_path: tuple[str, ...] | None = None,
    table_label: str | None = None,
    figure_label: str | None = None,
) -> dict[str, object]:
    artifact = validate_document_structure(structure_id, repositories)
    blocks = tuple(
        block
        for block in repositories.document_structure_blocks.list()
        if block.structure_id == structure_id
        and (page is None or block.page_number == page)
        and (section_path is None or block.section_path == section_path)
    )
    tables = tuple(
        table
        for table in repositories.table_structures.list()
        if table.structure_id == structure_id
        and (page is None or table.page_number == page)
        and (section_path is None or table.section_path == section_path)
        and (table_label is None or table.label == table_label)
    )
    figures = tuple(
        figure
        for figure in repositories.figure_structures.list()
        if figure.structure_id == structure_id
        and (page is None or figure.page_number == page)
        and (section_path is None or figure.section_path == section_path)
        and (figure_label is None or figure.label == figure_label)
    )
    table_ids = {table.table_id for table in tables}
    cells = tuple(cell for cell in repositories.table_cell_structures.list() if cell.table_id in table_ids)
    return {
        "artifact": artifact,
        "blocks": blocks,
        "tables": tables,
        "table_cells": cells,
        "figures": figures,
    }
