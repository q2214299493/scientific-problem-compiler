from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
from pathlib import PurePosixPath
import re
from typing import Protocol

from ..models import (
    CanonicalHTMLTextArtifact,
    CanonicalTextArtifact,
    DocumentBlockType,
    DocumentStructureArtifact,
    DocumentStructureBlock,
    EvidenceSpan,
    FigureStructure,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    StructureExtractionStatus,
    StructuredEvidenceLocator,
    TableCellStructure,
    TableStructure,
)
from ..repositories import EvidenceStore, KnowledgeRepositories
from ..serialization import content_hash
from .acquisition import validate_literature_representation
from .ingestion import create_evidence_span_from_canonical_text


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
        "label": label,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "text_hash": _text_hash(text),
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    block_id = f"document-block-{content_hash(identity)[:24]}"
    payload = {"block_id": block_id, **identity}
    return DocumentStructureBlock(
        **payload,
        content_hash=content_hash(payload),
    )


def _label(text: str, prefix: str, fallback_number: int) -> str:
    match = re.match(rf"(?i)^\s*({re.escape(prefix)}(?:ure)?\.?\s*[A-Za-z0-9.-]+)", text)
    return " ".join(match.group(1).split()) if match else f"{prefix.title()} {fallback_number}"


def _heading_level(text: str) -> int | None:
    stripped = text.strip()
    numbered = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+\S", stripped)
    if numbered:
        return min(6, numbered.group(1).count(".") + 1)
    words = stripped.split()
    if not words or len(words) > 12 or stripped.endswith((".", ";", ":", "?", "!")):
        return None
    if stripped.isupper() and any(character.isalpha() for character in stripped):
        return 1
    alpha_words = [word for word in words if any(character.isalpha() for character in word)]
    if alpha_words and all(word[0].isupper() for word in alpha_words):
        return 1
    return None


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
    start_offset: int,
    end_offset: int,
    text: str,
) -> TableCellStructure:
    identity = {
        "table_id": table_id,
        "row_index": row_index,
        "column_index": column_index,
        "row_span": row_span,
        "column_span": column_span,
        "is_header": is_header,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "text_hash": _text_hash(text),
    }
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
) -> FigureStructure:
    identity = {
        "structure_id": structure_id,
        "ordinal": ordinal,
        "label": label,
        "page_number": page_number,
        "section_path": section_path,
        "caption_block_ref": caption_block_ref,
        "extraction_status": StructureExtractionStatus.COMPLETE,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    figure_id = f"figure-structure-{content_hash(identity)[:24]}"
    payload = {"figure_id": figure_id, **identity}
    return FigureStructure(**payload, content_hash=content_hash(payload))


class PDFDocumentStructureExtractor:
    extractor_id = "pdf-canonical-structure"
    extractor_version = "1.0.0"
    representation_kind = LiteratureRepresentationKind.PDF
    extractor_config_hash = content_hash(
        {
            "source": "canonical_page_blocks",
            "heading_policy": "conservative_line_shape_v1",
            "caption_policy": "line_prefix_table_figure_v1",
            "table_cells": "never_infer",
        }
    )

    def extract(self, structure_id: str, source: DocumentStructureInput) -> ExtractedDocumentStructure:
        if not isinstance(source.canonical, CanonicalTextArtifact):
            raise ValueError("PDF structure extractor requires PDF canonical text")
        blocks: list[DocumentStructureBlock] = []
        tables: list[TableStructure] = []
        figures: list[FigureStructure] = []
        warnings: list[str] = []
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
                block_type = DocumentBlockType.PARAGRAPH
                heading_level = _heading_level(text)
                label: str | None = None
                if re.match(r"^(?:[-*•]|\d+[.)])\s+", text):
                    block_type = DocumentBlockType.LIST_ITEM
                    heading_level = None
                elif re.match(r"(?i)^table\s*[A-Za-z0-9.-]+", text):
                    block_type = DocumentBlockType.TABLE_CAPTION
                    heading_level = None
                    label = _label(text, "table", len(tables) + 1)
                elif re.match(r"(?i)^(?:figure|fig\.)\s*[A-Za-z0-9.-]+", text):
                    block_type = DocumentBlockType.FIGURE_CAPTION
                    heading_level = None
                    label = _label(text, "fig", len(figures) + 1)
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
                        )
                    )
        return ExtractedDocumentStructure(
            blocks=tuple(blocks),
            tables=tuple(tables),
            table_cells=(),
            figures=tuple(figures),
            page_count=source.canonical.page_count,
            status=(StructureExtractionStatus.PARTIAL if warnings else StructureExtractionStatus.COMPLETE),
            warnings=tuple(warnings),
        )


@dataclass
class _HTMLElement:
    tag: str
    text: str
    negative_context: bool
    table_index: int | None
    row_index: int | None
    column_index: int | None
    row_span: int
    column_span: int
    figure_index: int | None


class _HTMLStructureParser(HTMLParser):
    CAPTURE_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "caption", "th", "td", "figcaption"})
    NEGATIVE_TOKENS = frozenset(
        {
            "references",
            "reference",
            "bibliography",
            "citations",
            "cited-by",
            "footer",
            "navigation",
            "nav",
            "related-articles",
            "recommended",
            "references-list",
        }
    )
    VOID_TAGS = frozenset(
        {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.active: list[tuple[str, list[str], dict[str, object]]] = []
        self.elements: list[_HTMLElement] = []
        self.ignored_depth = 0
        self.table_stack: list[dict[str, int]] = []
        self.figure_stack: list[int] = []
        self.table_count = 0
        self.figure_count = 0
        self.attribute_warnings: list[str] = []

    def _span(self, value: str | None, field: str) -> int:
        try:
            parsed = int(value or 1)
        except ValueError:
            parsed = 1
        if parsed < 1:
            parsed = 1
        if value is not None and str(parsed) != value.strip():
            self.attribute_warnings.append(f"invalid HTML {field} was conservatively treated as 1")
        return parsed

    @classmethod
    def _negative(cls, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in {"nav", "footer", "aside"}:
            return True
        values = " ".join(value or "" for key, value in attrs if key in {"id", "class", "role"})
        tokens = {token for token in re.split(r"[^a-z0-9-]+", values.casefold()) if token}
        return bool(tokens & cls.NEGATIVE_TOKENS)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        negative = self._negative(normalized, attrs) or any(item[1] for item in self.stack)
        if normalized not in self.VOID_TAGS:
            self.stack.append((normalized, negative))
        if self.ignored_depth:
            return
        if normalized == "table":
            self.table_count += 1
            self.table_stack.append({"index": self.table_count - 1, "row": -1, "column": -1})
        elif normalized == "tr" and self.table_stack:
            self.table_stack[-1]["row"] += 1
            self.table_stack[-1]["column"] = -1
        elif normalized == "figure":
            self.figure_count += 1
            self.figure_stack.append(self.figure_count - 1)
        if normalized not in self.CAPTURE_TAGS:
            return
        table = self.table_stack[-1] if self.table_stack else None
        if normalized in {"th", "td"} and table is not None:
            table["column"] += 1
        attr_map = dict(attrs)
        metadata: dict[str, object] = {
            "negative": negative,
            "table_index": table["index"] if table is not None else None,
            "row_index": table["row"] if table is not None else None,
            "column_index": table["column"] if table is not None else None,
            "row_span": self._span(attr_map.get("rowspan"), "rowspan"),
            "column_span": self._span(attr_map.get("colspan"), "colspan"),
            "figure_index": self.figure_stack[-1] if self.figure_stack else None,
        }
        self.active.append((normalized, [], metadata))

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in self.CAPTURE_TAGS and not self.ignored_depth:
            for index in range(len(self.active) - 1, -1, -1):
                active_tag, content, metadata = self.active[index]
                if active_tag != normalized:
                    continue
                text = " ".join("".join(content).split())
                if text:
                    self.elements.append(
                        _HTMLElement(
                            tag=normalized,
                            text=text,
                            negative_context=bool(metadata["negative"]),
                            table_index=metadata["table_index"],
                            row_index=metadata["row_index"],
                            column_index=metadata["column_index"],
                            row_span=int(metadata["row_span"]),
                            column_span=int(metadata["column_span"]),
                            figure_index=metadata["figure_index"],
                        )
                    )
                del self.active[index]
                break
        if normalized in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
        if normalized == "table" and self.table_stack:
            self.table_stack.pop()
        elif normalized == "figure" and self.figure_stack:
            self.figure_stack.pop()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == normalized:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth and self.active:
            self.active[-1][1].append(data)


class HTMLDocumentStructureExtractor:
    extractor_id = "html-canonical-structure"
    extractor_version = "1.0.0"
    representation_kind = LiteratureRepresentationKind.HTML
    extractor_config_hash = content_hash(
        {
            "tags": tuple(sorted(_HTMLStructureParser.CAPTURE_TAGS)),
            "negative_context": tuple(sorted(_HTMLStructureParser.NEGATIVE_TOKENS)),
            "binding": "canonical_block_tag_and_text_hash",
        }
    )

    def extract(self, structure_id: str, source: DocumentStructureInput) -> ExtractedDocumentStructure:
        if not isinstance(source.canonical, CanonicalHTMLTextArtifact) or source.raw_content is None:
            raise ValueError("HTML structure extractor requires raw and canonical HTML")
        try:
            raw_text = source.raw_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("raw HTML is not UTF-8") from error
        parser = _HTMLStructureParser()
        parser.feed(raw_text)
        parser.close()
        warnings: list[str] = []
        if parser.active or parser.stack:
            warnings.append("malformed HTML left unclosed structural elements")
        warnings.extend(parser.attribute_warnings)
        queues: dict[tuple[str, str], deque] = defaultdict(deque)
        for canonical_block in source.canonical.blocks:
            tag = canonical_block.block_type.removeprefix("html_")
            queues[(tag, canonical_block.text_hash)].append(canonical_block)
        matched_ids: set[str] = set()
        heading_blocks: dict[int, DocumentStructureBlock] = {}
        heading_titles: dict[int, str] = {}
        blocks: list[DocumentStructureBlock] = []
        event_blocks: dict[int, DocumentStructureBlock] = {}
        ordinal = 0
        for event_index, event in enumerate(parser.elements):
            queue = queues[(event.tag, _text_hash(event.text))]
            if not queue:
                warnings.append(f"HTML {event.tag} text is absent from canonical representation")
                continue
            canonical_block = queue.popleft()
            matched_ids.add(canonical_block.block_id)
            block_type = {
                "li": DocumentBlockType.LIST_ITEM,
                "caption": DocumentBlockType.TABLE_CAPTION,
                "th": DocumentBlockType.TABLE_CELL,
                "td": DocumentBlockType.TABLE_CELL,
                "figcaption": DocumentBlockType.FIGURE_CAPTION,
            }.get(event.tag, DocumentBlockType.PARAGRAPH)
            level: int | None = None
            if event.tag.startswith("h") and len(event.tag) == 2 and event.tag[1].isdigit():
                if event.negative_context:
                    block_type = DocumentBlockType.OTHER_TEXT
                else:
                    block_type = DocumentBlockType.HEADING
                    level = int(event.tag[1])
                    for existing in tuple(heading_titles):
                        if existing >= level:
                            heading_titles.pop(existing, None)
                            heading_blocks.pop(existing, None)
                    heading_titles[level] = event.text
            section_path = tuple(heading_titles[item] for item in sorted(heading_titles))
            parent = heading_blocks[max(heading_blocks)].block_id if heading_blocks else None
            label: str | None = None
            if block_type == DocumentBlockType.TABLE_CAPTION:
                label = _label(event.text, "table", (event.table_index or 0) + 1)
            elif block_type == DocumentBlockType.FIGURE_CAPTION:
                label = _label(event.text, "fig", (event.figure_index or 0) + 1)
            recovered = source.canonical_text[canonical_block.start_offset : canonical_block.end_offset]
            block = _make_block(
                structure_id=structure_id,
                block_type=block_type,
                ordinal=ordinal,
                parent_block_id=parent,
                heading_level=level,
                section_path=section_path,
                label=label,
                start_offset=canonical_block.start_offset,
                end_offset=canonical_block.end_offset,
                text=recovered,
            )
            blocks.append(block)
            event_blocks[event_index] = block
            ordinal += 1
            if block_type == DocumentBlockType.HEADING:
                heading_blocks[level or 1] = block
        for canonical_block in source.canonical.blocks:
            if canonical_block.block_id in matched_ids:
                continue
            recovered = source.canonical_text[canonical_block.start_offset : canonical_block.end_offset]
            section_path = tuple(heading_titles[item] for item in sorted(heading_titles))
            parent = heading_blocks[max(heading_blocks)].block_id if heading_blocks else None
            blocks.append(
                _make_block(
                    structure_id=structure_id,
                    block_type=DocumentBlockType.OTHER_TEXT,
                    ordinal=ordinal,
                    parent_block_id=parent,
                    section_path=section_path,
                    start_offset=canonical_block.start_offset,
                    end_offset=canonical_block.end_offset,
                    text=recovered,
                )
            )
            ordinal += 1
            warnings.append(f"canonical block has no recoverable HTML structural context: {canonical_block.block_id}")
        tables: list[TableStructure] = []
        cells: list[TableCellStructure] = []
        for table_index in range(parser.table_count):
            related = [
                (index, event, event_blocks.get(index))
                for index, event in enumerate(parser.elements)
                if event.table_index == table_index
            ]
            caption = next(
                (block for _, event, block in related if event.tag == "caption" and block),
                None,
            )
            cell_events = [
                (event, block) for _, event, block in related if event.tag in {"th", "td"} and block is not None
            ]
            section_path = (
                caption.section_path if caption is not None else (cell_events[0][1].section_path if cell_events else ())
            )
            table_label = caption.label if caption is not None and caption.label else f"Table {table_index + 1}"
            base_table = _make_table(
                structure_id=structure_id,
                ordinal=table_index,
                label=table_label,
                caption_block_ref=caption.block_id if caption else None,
                page_number=None,
                section_path=section_path,
                cell_refs=(),
                status=(StructureExtractionStatus.COMPLETE if cell_events else StructureExtractionStatus.PARTIAL),
            )
            table_cells = tuple(
                _make_cell(
                    table_id=base_table.table_id,
                    row_index=event.row_index or 0,
                    column_index=event.column_index or 0,
                    row_span=event.row_span,
                    column_span=event.column_span,
                    is_header=event.tag == "th",
                    start_offset=block.start_offset,
                    end_offset=block.end_offset,
                    text=source.canonical_text[block.start_offset : block.end_offset],
                )
                for event, block in cell_events
            )
            cells.extend(table_cells)
            tables.append(
                _make_table(
                    structure_id=structure_id,
                    ordinal=table_index,
                    label=table_label,
                    caption_block_ref=caption.block_id if caption else None,
                    page_number=None,
                    section_path=section_path,
                    cell_refs=tuple(cell.cell_id for cell in table_cells),
                    status=(StructureExtractionStatus.COMPLETE if cell_events else StructureExtractionStatus.PARTIAL),
                )
            )
            if not cell_events:
                warnings.append(f"{table_label}: table cell structure is unresolved")
        figures: list[FigureStructure] = []
        for figure_index in range(parser.figure_count):
            caption = next(
                (
                    event_blocks.get(index)
                    for index, event in enumerate(parser.elements)
                    if event.figure_index == figure_index and event.tag == "figcaption"
                ),
                None,
            )
            if caption is None:
                warnings.append(f"Figure {figure_index + 1}: textual caption is unresolved")
                continue
            figures.append(
                _make_figure(
                    structure_id=structure_id,
                    ordinal=figure_index,
                    label=caption.label or f"Figure {figure_index + 1}",
                    page_number=None,
                    section_path=caption.section_path,
                    caption_block_ref=caption.block_id,
                )
            )
        return ExtractedDocumentStructure(
            blocks=tuple(sorted(blocks, key=lambda item: item.ordinal)),
            tables=tuple(tables),
            table_cells=tuple(cells),
            figures=tuple(figures),
            page_count=None,
            status=(StructureExtractionStatus.PARTIAL if warnings else StructureExtractionStatus.COMPLETE),
            warnings=tuple(dict.fromkeys(warnings)),
        )


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
        return DocumentStructureExtractionResult(
            artifact=artifact,
            blocks=extracted.blocks,
            tables=extracted.tables,
            table_cells=extracted.table_cells,
            figures=extracted.figures,
        )


def _validate_structure_records(
    artifact: DocumentStructureArtifact,
    blocks: tuple[DocumentStructureBlock, ...],
    tables: tuple[TableStructure, ...],
    cells: tuple[TableCellStructure, ...],
    figures: tuple[FigureStructure, ...],
    representation: LiteratureRepresentationReference,
    canonical: CanonicalArtifact,
    text: str,
) -> None:
    if (
        artifact.representation_id != representation.representation_id
        or artifact.representation_hash != representation.content_hash
        or artifact.literature_id != representation.literature_id
        or artifact.representation_kind != representation.representation_kind
        or artifact.canonical_text_id != representation.canonical_text_id
        or artifact.canonical_text_hash != representation.canonical_text_hash
        or artifact.source_id != representation.source_id
        or artifact.source_version != representation.source_version
    ):
        raise ValueError("document structure representation binding is invalid")
    if canonical.canonical_text_id != artifact.canonical_text_id:
        raise ValueError("document structure canonical text binding is invalid")
    block_map = {block.block_id: block for block in blocks}
    table_map = {table.table_id: table for table in tables}
    cell_map = {cell.cell_id: cell for cell in cells}
    figure_map = {figure.figure_id: figure for figure in figures}
    if (
        len(block_map) != len(blocks)
        or len(table_map) != len(tables)
        or len(cell_map) != len(cells)
        or len(figure_map) != len(figures)
    ):
        raise ValueError("document structure contains duplicate IDs")
    if artifact.block_hashes != {key: value.content_hash for key, value in block_map.items()}:
        raise ValueError("document structure block hashes are invalid")
    if artifact.table_hashes != {key: value.content_hash for key, value in table_map.items()}:
        raise ValueError("document structure table hashes are invalid")
    if artifact.figure_hashes != {key: value.content_hash for key, value in figure_map.items()}:
        raise ValueError("document structure figure hashes are invalid")
    if sorted(block.ordinal for block in blocks) != list(range(len(blocks))):
        raise ValueError("document structure block ordinals must be contiguous")
    for block in blocks:
        if block.structure_id != artifact.structure_id:
            raise ValueError("cross-representation structure block reference")
        if block.end_offset > len(text) or _text_hash(text[block.start_offset : block.end_offset]) != block.text_hash:
            raise ValueError(f"document block does not recover exact canonical text: {block.block_id}")
        if block.parent_block_id is not None and block.parent_block_id not in block_map:
            raise ValueError("document structure parent block is missing")
        if (
            block.parent_block_id is not None
            and block_map[block.parent_block_id].ordinal >= block.ordinal
        ):
            raise ValueError("document structure parent must precede its child")
    heading_titles: dict[int, str] = {}
    for block in sorted(blocks, key=lambda item: item.ordinal):
        if block.block_type == DocumentBlockType.PAGE:
            continue
        if block.block_type == DocumentBlockType.HEADING:
            level = block.heading_level or 1
            for existing in tuple(heading_titles):
                if existing >= level:
                    heading_titles.pop(existing, None)
            heading_titles[level] = text[block.start_offset : block.end_offset]
        expected_path = tuple(heading_titles[level] for level in sorted(heading_titles))
        if block.section_path != expected_path:
            raise ValueError("document structure section hierarchy is invalid")
    for block in blocks:
        seen: set[str] = set()
        cursor = block
        while cursor.parent_block_id is not None:
            if cursor.block_id in seen:
                raise ValueError("document structure parent cycle detected")
            seen.add(cursor.block_id)
            cursor = block_map[cursor.parent_block_id]
    if isinstance(canonical, CanonicalTextArtifact):
        if artifact.page_count != canonical.page_count:
            raise ValueError("document structure page count is invalid")
        pages = {page.page_number: page for page in canonical.blocks}
        for block in blocks:
            if block.page_number not in pages:
                raise ValueError("PDF structure block has invalid page binding")
            page = pages[block.page_number]
            if block.start_offset < page.start_offset or block.end_offset > page.end_offset:
                raise ValueError("PDF structure block escapes canonical page range")
    elif artifact.page_count is not None or any(block.page_number is not None for block in blocks):
        raise ValueError("HTML structure must not invent PDF page numbers")
    for table in tables:
        if table.structure_id != artifact.structure_id:
            raise ValueError("cross-representation table reference")
        if table.caption_block_ref is not None:
            caption = block_map.get(table.caption_block_ref)
            if (
                caption is None
                or caption.block_type != DocumentBlockType.TABLE_CAPTION
                or caption.page_number != table.page_number
                or caption.section_path != table.section_path
            ):
                raise ValueError("table caption block binding is invalid")
        if set(table.cell_refs) != {cell.cell_id for cell in cells if cell.table_id == table.table_id}:
            raise ValueError("table cell bindings are invalid")
        coordinates: set[tuple[int, int]] = set()
        for cell_id in table.cell_refs:
            cell = cell_map[cell_id]
            coordinate = (cell.row_index, cell.column_index)
            if coordinate in coordinates:
                raise ValueError("table contains duplicate cell coordinates")
            coordinates.add(coordinate)
            if cell.end_offset > len(text) or _text_hash(text[cell.start_offset : cell.end_offset]) != cell.text_hash:
                raise ValueError("table cell does not recover exact canonical text")
            matching_blocks = [
                block
                for block in blocks
                if block.block_type == DocumentBlockType.TABLE_CELL
                and block.start_offset == cell.start_offset
                and block.end_offset == cell.end_offset
                and block.section_path == table.section_path
            ]
            if len(matching_blocks) != 1:
                raise ValueError("table cell has no exact structure block")
    for figure in figures:
        if figure.structure_id != artifact.structure_id:
            raise ValueError("cross-representation figure reference")
        caption = block_map.get(figure.caption_block_ref)
        if (
            caption is None
            or caption.block_type != DocumentBlockType.FIGURE_CAPTION
            or caption.page_number != figure.page_number
            or caption.section_path != figure.section_path
        ):
            raise ValueError("figure caption block binding is invalid")


def validate_document_structure(
    structure: str | DocumentStructureArtifact,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
) -> DocumentStructureArtifact:
    store = evidence_store or repositories.evidence_store
    artifact = repositories.document_structure_artifacts.get(structure) if isinstance(structure, str) else structure
    stored = repositories.document_structure_artifacts.get(artifact.structure_id)
    if stored != artifact:
        raise ValueError("document structure differs from repository record")
    representation = repositories.literature_representation_refs.get(artifact.representation_id)
    validate_literature_representation(representation, repositories, store)
    if artifact.representation_kind == LiteratureRepresentationKind.PDF:
        canonical: CanonicalArtifact = repositories.canonical_text_artifacts.get(artifact.canonical_text_id)
        text = repositories.canonical_text_artifacts.read_text(artifact.canonical_text_id)
    else:
        canonical = repositories.canonical_html_text_artifacts.get(artifact.canonical_text_id)
        text = repositories.canonical_html_text_artifacts.read_text(artifact.canonical_text_id)
    blocks = tuple(repositories.document_structure_blocks.get(block_id) for block_id in artifact.block_hashes)
    tables = tuple(repositories.table_structures.get(table_id) for table_id in artifact.table_hashes)
    figures = tuple(repositories.figure_structures.get(figure_id) for figure_id in artifact.figure_hashes)
    table_ids = {table.table_id for table in tables}
    cells = tuple(cell for cell in repositories.table_cell_structures.list() if cell.table_id in table_ids)
    actual_blocks = {
        item.block_id
        for item in repositories.document_structure_blocks.list()
        if item.structure_id == artifact.structure_id
    }
    actual_tables = {
        item.table_id for item in repositories.table_structures.list() if item.structure_id == artifact.structure_id
    }
    actual_figures = {
        item.figure_id for item in repositories.figure_structures.list() if item.structure_id == artifact.structure_id
    }
    if (
        actual_blocks != set(artifact.block_hashes)
        or actual_tables != set(artifact.table_hashes)
        or actual_figures != set(artifact.figure_hashes)
    ):
        raise ValueError("document structure repository inventory is inconsistent")
    _validate_structure_records(artifact, blocks, tables, cells, figures, representation, canonical, text)
    return artifact


def format_structured_evidence_locator(
    locator: StructuredEvidenceLocator,
    repositories: KnowledgeRepositories | None = None,
) -> str:
    prefix = "PDF" if locator.page_number is not None else "HTML"
    parts = [f"{prefix} p.{locator.page_number}" if locator.page_number else prefix]
    parts.extend(locator.section_path)
    if locator.table_id is not None:
        label = locator.table_id
        if repositories is not None:
            label = repositories.table_structures.get(locator.table_id).label or label
        parts.append(label)
        parts.append(f"row {(locator.row_index or 0) + 1}")
        parts.append(f"column {(locator.column_index or 0) + 1}")
    elif locator.figure_id is not None:
        label = locator.figure_id
        if repositories is not None:
            label = repositories.figure_structures.get(locator.figure_id).label or label
        parts.append(f"{label} caption")
    return " > ".join(parts)


def verify_structured_evidence_locator(
    locator: str | StructuredEvidenceLocator,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
) -> StructuredEvidenceLocator:
    store = evidence_store or repositories.evidence_store
    record = repositories.structured_evidence_locators.get(locator) if isinstance(locator, str) else locator
    stored = repositories.structured_evidence_locators.get(record.locator_id)
    if stored != record:
        raise ValueError("structured evidence locator differs from repository record")
    artifact = validate_document_structure(record.structure_id, repositories, store)
    evidence = store.get_evidence(record.evidence_id)
    store.verify_evidence_integrity(evidence)
    block = repositories.document_structure_blocks.get(record.block_id)
    if (
        record.source_id != evidence.source_id
        or record.source_version != evidence.source_version
        or record.source_id != artifact.source_id
        or record.source_version != artifact.source_version
        or record.literature_id != artifact.literature_id
        or record.representation_id != artifact.representation_id
        or block.structure_id != artifact.structure_id
        or record.canonical_start_offset != evidence.start_offset
        or record.canonical_end_offset != evidence.end_offset
        or evidence.start_offset < block.start_offset
        or evidence.end_offset > block.end_offset
        or record.page_number != block.page_number
        or record.section_path != block.section_path
    ):
        raise ValueError("structured evidence locator provenance or offsets are invalid")
    if record.table_cell_id is not None:
        table = repositories.table_structures.get(record.table_id or "")
        cell = repositories.table_cell_structures.get(record.table_cell_id)
        if (
            table.structure_id != artifact.structure_id
            or cell.table_id != table.table_id
            or cell.cell_id not in table.cell_refs
            or record.row_index != cell.row_index
            or record.column_index != cell.column_index
            or evidence.start_offset != cell.start_offset
            or evidence.end_offset != cell.end_offset
        ):
            raise ValueError("structured table-cell locator is invalid")
    if record.figure_id is not None:
        figure = repositories.figure_structures.get(record.figure_id)
        if figure.structure_id != artifact.structure_id or figure.caption_block_ref != block.block_id:
            raise ValueError("structured figure locator is invalid")
    return record


def _create_locator(
    *,
    evidence: EvidenceSpan,
    artifact: DocumentStructureArtifact,
    block: DocumentStructureBlock,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
    table: TableStructure | None = None,
    cell: TableCellStructure | None = None,
    figure: FigureStructure | None = None,
) -> tuple[EvidenceSpan, StructuredEvidenceLocator]:
    identity = {
        "evidence_id": evidence.evidence_id,
        "source_id": evidence.source_id,
        "source_version": evidence.source_version,
        "literature_id": artifact.literature_id,
        "representation_id": artifact.representation_id,
        "structure_id": artifact.structure_id,
        "block_id": block.block_id,
        "page_number": block.page_number,
        "section_path": block.section_path,
        "table_id": table.table_id if table else None,
        "table_cell_id": cell.cell_id if cell else None,
        "row_index": cell.row_index if cell else None,
        "column_index": cell.column_index if cell else None,
        "figure_id": figure.figure_id if figure else None,
        "canonical_start_offset": evidence.start_offset,
        "canonical_end_offset": evidence.end_offset,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    locator_id = f"structured-locator-{content_hash(identity)[:24]}"
    payload = {"locator_id": locator_id, **identity}
    locator = StructuredEvidenceLocator(**payload, content_hash=content_hash(payload))
    repositories.structured_evidence_locators.put(locator.locator_id, locator)
    verify_structured_evidence_locator(locator, repositories, evidence_store)
    return evidence, locator


def _display_for_block(
    block: DocumentStructureBlock,
    *,
    table: TableStructure | None = None,
    cell: TableCellStructure | None = None,
    figure: FigureStructure | None = None,
) -> str:
    parts = [f"PDF p.{block.page_number}" if block.page_number else "HTML"]
    parts.extend(block.section_path)
    if table is not None and cell is not None:
        parts.extend(
            (
                table.label or table.table_id,
                f"row {cell.row_index + 1}",
                f"column {cell.column_index + 1}",
            )
        )
    elif figure is not None:
        parts.append(f"{figure.label or figure.figure_id} caption")
    elif block.label is not None:
        parts.append(block.label)
    return " > ".join(parts)


def create_evidence_from_block(
    structure_id: str,
    block_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
    *,
    historical_representation: bool = False,
) -> tuple[EvidenceSpan, StructuredEvidenceLocator]:
    store = evidence_store or repositories.evidence_store
    artifact = validate_document_structure(structure_id, repositories, store)
    block = repositories.document_structure_blocks.get(block_id)
    if block.structure_id != artifact.structure_id:
        raise ValueError("block belongs to another document structure")
    evidence = create_evidence_span_from_canonical_text(
        artifact.canonical_text_id,
        block.start_offset,
        block.end_offset,
        repositories,
        store,
        locator=_display_for_block(block),
        historical_ingestion=historical_representation,
    )
    return _create_locator(
        evidence=evidence,
        artifact=artifact,
        block=block,
        repositories=repositories,
        evidence_store=store,
    )


def create_evidence_from_table_cell(
    structure_id: str,
    table_id: str,
    cell_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
    *,
    historical_representation: bool = False,
) -> tuple[EvidenceSpan, StructuredEvidenceLocator]:
    store = evidence_store or repositories.evidence_store
    artifact = validate_document_structure(structure_id, repositories, store)
    table = repositories.table_structures.get(table_id)
    cell = repositories.table_cell_structures.get(cell_id)
    if (
        table.structure_id != artifact.structure_id
        or cell.table_id != table.table_id
        or cell.cell_id not in table.cell_refs
    ):
        raise ValueError("table cell belongs to another document structure")
    blocks = [
        block
        for block in repositories.document_structure_blocks.list()
        if block.structure_id == artifact.structure_id
        and block.block_type == DocumentBlockType.TABLE_CELL
        and block.start_offset == cell.start_offset
        and block.end_offset == cell.end_offset
    ]
    if len(blocks) != 1:
        raise ValueError("table cell has no unique exact structure block")
    block = blocks[0]
    evidence = create_evidence_span_from_canonical_text(
        artifact.canonical_text_id,
        cell.start_offset,
        cell.end_offset,
        repositories,
        store,
        locator=_display_for_block(block, table=table, cell=cell),
        historical_ingestion=historical_representation,
    )
    return _create_locator(
        evidence=evidence,
        artifact=artifact,
        block=block,
        table=table,
        cell=cell,
        repositories=repositories,
        evidence_store=store,
    )


def create_evidence_from_figure_caption(
    structure_id: str,
    figure_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
    *,
    historical_representation: bool = False,
) -> tuple[EvidenceSpan, StructuredEvidenceLocator]:
    store = evidence_store or repositories.evidence_store
    artifact = validate_document_structure(structure_id, repositories, store)
    figure = repositories.figure_structures.get(figure_id)
    block = repositories.document_structure_blocks.get(figure.caption_block_ref)
    if figure.structure_id != artifact.structure_id or block.structure_id != artifact.structure_id:
        raise ValueError("figure belongs to another document structure")
    evidence = create_evidence_span_from_canonical_text(
        artifact.canonical_text_id,
        block.start_offset,
        block.end_offset,
        repositories,
        store,
        locator=_display_for_block(block, figure=figure),
        historical_ingestion=historical_representation,
    )
    return _create_locator(
        evidence=evidence,
        artifact=artifact,
        block=block,
        figure=figure,
        repositories=repositories,
        evidence_store=store,
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
