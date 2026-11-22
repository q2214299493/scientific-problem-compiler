"""Structure extractor contracts and deterministic record construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import hashlib

from ...models import (
    CanonicalHTMLTextArtifact,
    CanonicalTextArtifact,
    DocumentBlockType,
    DocumentContentRegion,
    DocumentStructureArtifact,
    DocumentStructureBlock,
    DocumentStructureSelection,
    FigureStructure,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    StructureExtractionStatus,
    TableCellStructure,
    TableStructure,
)
from ...serialization import content_hash


CanonicalArtifact = CanonicalTextArtifact | CanonicalHTMLTextArtifact


@dataclass(frozen=True)
class DocumentStructureInput:
    representation: LiteratureRepresentationReference
    canonical: CanonicalArtifact
    canonical_text: str
    raw_content: bytes | None = None


@dataclass(frozen=True)
class ExtractedDocumentStructure:
    blocks: tuple[DocumentStructureBlock, ...]
    tables: tuple[TableStructure, ...]
    table_cells: tuple[TableCellStructure, ...]
    figures: tuple[FigureStructure, ...]
    page_count: int | None
    status: StructureExtractionStatus
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentStructureExtractionResult:
    artifact: DocumentStructureArtifact
    selection: DocumentStructureSelection | None
    blocks: tuple[DocumentStructureBlock, ...]
    tables: tuple[TableStructure, ...]
    table_cells: tuple[TableCellStructure, ...]
    figures: tuple[FigureStructure, ...]


class DocumentStructureExtractor(Protocol):
    extractor_id: str
    extractor_version: str
    extractor_config_hash: str
    representation_kind: LiteratureRepresentationKind

    def extract(self, structure_id: str, source: DocumentStructureInput) -> ExtractedDocumentStructure: ...


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_block(
    *,
    structure_id: str,
    block_type: DocumentBlockType,
    ordinal: int,
    start_offset: int,
    end_offset: int,
    text: str,
    parent_block_id: str | None = None,
    page_number: int | None = None,
    heading_level: int | None = None,
    section_path: tuple[str, ...] = (),
    content_region: DocumentContentRegion = DocumentContentRegion.UNKNOWN,
    label: str | None = None,
) -> DocumentStructureBlock:
    identity = {
        "structure_id": structure_id,
        "block_type": block_type,
        "ordinal": ordinal,
        "parent_block_id": parent_block_id,
        "page_number": page_number,
        "heading_level": heading_level,
        "section_path": section_path,
        "content_region": content_region,
        "label": label,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "text_hash": _text_hash(text),
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    if content_region == DocumentContentRegion.UNKNOWN:
        identity.pop("content_region", None)
    block_id = f"document-block-{content_hash(identity)[:24]}"
    payload = {"block_id": block_id, **identity}
    return DocumentStructureBlock(
        **payload,
        content_hash=content_hash(payload),
    )


def _make_table(
    *,
    structure_id: str,
    ordinal: int,
    label: str,
    caption_block_ref: str | None,
    page_number: int | None,
    section_path: tuple[str, ...],
    cell_refs: tuple[str, ...],
    status: StructureExtractionStatus,
) -> TableStructure:
    identity = {
        "structure_id": structure_id,
        "ordinal": ordinal,
        "label": label,
        "caption_block_ref": caption_block_ref,
        "page_number": page_number,
        "section_path": section_path,
        "extraction_status": status,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    table_id = f"table-structure-{content_hash(identity)[:24]}"
    payload = {"table_id": table_id, **identity, "cell_refs": cell_refs}
    return TableStructure(**payload, content_hash=content_hash(payload))


def _make_cell(
    *,
    table_id: str,
    row_index: int,
    column_index: int,
    row_span: int,
    column_span: int,
    is_header: bool,
    canonical_block_refs: tuple[str, ...],
    start_offset: int | None,
    end_offset: int | None,
    text: str | None,
) -> TableCellStructure:
    identity = {
        "table_id": table_id,
        "row_index": row_index,
        "column_index": column_index,
        "row_span": row_span,
        "column_span": column_span,
        "is_header": is_header,
        "canonical_block_refs": canonical_block_refs,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "text_hash": _text_hash(text) if text is not None else None,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    if not canonical_block_refs and start_offset is not None:
        identity.pop("canonical_block_refs", None)
    cell_id = f"table-cell-{content_hash(identity)[:24]}"
    payload = {"cell_id": cell_id, **identity}
    return TableCellStructure(**payload, content_hash=content_hash(payload))


def _make_figure(
    *,
    structure_id: str,
    ordinal: int,
    label: str,
    page_number: int | None,
    section_path: tuple[str, ...],
    caption_block_ref: str,
    status: StructureExtractionStatus = StructureExtractionStatus.COMPLETE,
) -> FigureStructure:
    identity = {
        "structure_id": structure_id,
        "ordinal": ordinal,
        "label": label,
        "page_number": page_number,
        "section_path": section_path,
        "caption_block_ref": caption_block_ref,
        "extraction_status": status,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    figure_id = f"figure-structure-{content_hash(identity)[:24]}"
    payload = {"figure_id": figure_id, **identity}
    return FigureStructure(**payload, content_hash=content_hash(payload))
