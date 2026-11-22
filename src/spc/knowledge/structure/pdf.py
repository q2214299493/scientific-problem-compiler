"""Extract conservative document structure from canonical PDF pages."""

from __future__ import annotations

import re

from ...models import (
    CanonicalTextArtifact,
    DocumentBlockType,
    DocumentStructureBlock,
    FigureStructure,
    LiteratureRepresentationKind,
    StructureExtractionStatus,
    TableStructure,
)
from ...serialization import content_hash
from .records import (
    DocumentStructureInput,
    ExtractedDocumentStructure,
    _make_block,
    _make_figure,
    _make_table,
)


_PDF_CAPTION_RE = re.compile(
    r"^(?P<label>(?:Extended Data\s+)?(?:Table|Fig(?:ure)?\.?)\s+\d+[A-Za-z]?)"
    r"(?P<separator>\s*\|\s*|\s*:\s*|\.\s+)"
    r"(?P<title>\S.*)$",
    re.IGNORECASE,
)


_PDF_BARE_TABLE_RE = re.compile(r"^(?P<label>Table\s+\d+[A-Za-z]?)$", re.IGNORECASE)


def _pdf_caption(text: str) -> tuple[DocumentBlockType, str] | None:
    match = _PDF_CAPTION_RE.match(text)
    if match is None:
        bare_table = _PDF_BARE_TABLE_RE.match(text)
        if bare_table is None:
            return None
        return DocumentBlockType.TABLE_CAPTION, " ".join(bare_table.group("label").split())
    label = " ".join(match.group("label").split())
    if match.group("separator").lstrip().startswith(".") and re.search(r"\d+[A-Za-z]$", label):
        return None  # A panel reference such as "Fig. 7h. Note ..." is not a caption.
    block_type = (
        DocumentBlockType.TABLE_CAPTION
        if re.search(r"(?i)\btable\b", label)
        else DocumentBlockType.FIGURE_CAPTION
    )
    return block_type, label


def _heading_level(text: str) -> int | None:
    stripped = text.strip()
    numbered = re.match(r"^(\d+(?:\.\d+)+)[.)]?\s+\S", stripped)
    if numbered:
        return min(6, numbered.group(1).count(".") + 1)
    return None


class PDFDocumentStructureExtractor:
    extractor_id = "pdf-canonical-structure"
    extractor_version = "3.0.2"
    representation_kind = LiteratureRepresentationKind.PDF
    extractor_config_hash = content_hash(
        {
            "source": "canonical_page_blocks",
            "heading_policy": "hierarchical_numbered_heading_only_v3",
            "paragraph_policy": "unresolved_without_layout",
            "caption_policy": "title_delimiter_and_panel_guard_v3",
            "figure_caption_status": "partial_first_line_only",
            "table_cells": "never_infer",
            "warning_policy": "deduplicate_by_first_occurrence",
        }
    )

    def extract(self, structure_id: str, source: DocumentStructureInput) -> ExtractedDocumentStructure:
        if not isinstance(source.canonical, CanonicalTextArtifact):
            raise ValueError("PDF structure extractor requires PDF canonical text")
        blocks: list[DocumentStructureBlock] = []
        tables: list[TableStructure] = []
        figures: list[FigureStructure] = []
        warnings: list[str] = ["PDF paragraph boundaries are unresolved without layout evidence"]
        heading_blocks: dict[int, DocumentStructureBlock] = {}
        heading_titles: dict[int, str] = {}
        ordinal = 0
        for page in source.canonical.blocks:
            page_text = source.canonical_text[page.start_offset : page.end_offset]
            page_block = _make_block(
                structure_id=structure_id,
                block_type=DocumentBlockType.PAGE,
                ordinal=ordinal,
                page_number=page.page_number,
                start_offset=page.start_offset,
                end_offset=page.end_offset,
                text=page_text,
                label=f"Page {page.page_number}",
            )
            blocks.append(page_block)
            ordinal += 1
            cursor = page.start_offset
            for line in page_text.splitlines(keepends=True):
                raw_line = line.rstrip("\n")
                text = raw_line.strip()
                leading = len(raw_line) - len(raw_line.lstrip())
                start = cursor + leading
                end = start + len(text)
                cursor += len(line)
                if not text:
                    continue
                lower = text.casefold()
                block_type = DocumentBlockType.OTHER_TEXT
                heading_level = _heading_level(text)
                label: str | None = None
                caption = _pdf_caption(text)
                if re.match(r"^(?:[-*•])\s+", text):
                    block_type = DocumentBlockType.LIST_ITEM
                    heading_level = None
                elif caption is not None:
                    block_type, label = caption
                    heading_level = None
                elif heading_level is not None:
                    block_type = DocumentBlockType.HEADING
                if block_type == DocumentBlockType.HEADING:
                    if heading_level is None:
                        raise ValueError("heading classification lost its level")
                    for level in tuple(heading_titles):
                        if level >= heading_level:
                            heading_titles.pop(level, None)
                            heading_blocks.pop(level, None)
                    heading_titles[heading_level] = text
                    section_path = tuple(heading_titles[level] for level in sorted(heading_titles))
                    parent = heading_blocks[max(heading_blocks)] if heading_blocks else page_block
                else:
                    section_path = tuple(heading_titles[level] for level in sorted(heading_titles))
                    parent = heading_blocks[max(heading_blocks)] if heading_blocks else page_block
                block = _make_block(
                    structure_id=structure_id,
                    block_type=block_type,
                    ordinal=ordinal,
                    parent_block_id=parent.block_id,
                    page_number=page.page_number,
                    heading_level=heading_level,
                    section_path=section_path,
                    label=label,
                    start_offset=start,
                    end_offset=end,
                    text=source.canonical_text[start:end],
                )
                blocks.append(block)
                ordinal += 1
                if block_type == DocumentBlockType.HEADING:
                    heading_blocks[heading_level or 1] = block
                elif block_type == DocumentBlockType.TABLE_CAPTION:
                    tables.append(
                        _make_table(
                            structure_id=structure_id,
                            ordinal=len(tables),
                            label=label or f"Table {len(tables) + 1}",
                            caption_block_ref=block.block_id,
                            page_number=page.page_number,
                            section_path=section_path,
                            cell_refs=(),
                            status=StructureExtractionStatus.PARTIAL,
                        )
                    )
                    warnings.append(f"{label or lower}: PDF table cells remain unresolved")
                elif block_type == DocumentBlockType.FIGURE_CAPTION:
                    figures.append(
                        _make_figure(
                            structure_id=structure_id,
                            ordinal=len(figures),
                            label=label or f"Figure {len(figures) + 1}",
                            page_number=page.page_number,
                            section_path=section_path,
                            caption_block_ref=block.block_id,
                            status=StructureExtractionStatus.PARTIAL,
                        )
                    )
        return ExtractedDocumentStructure(
            blocks=tuple(blocks),
            tables=tuple(tables),
            table_cells=(),
            figures=tuple(figures),
            page_count=source.canonical.page_count,
            status=StructureExtractionStatus.PARTIAL,
            warnings=tuple(dict.fromkeys(warnings)),
        )
