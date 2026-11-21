"""Extract HTML sections, tables and figures with exact canonical offsets."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
import re

from ...models import (
    CanonicalHTMLTextArtifact,
    CanonicalTextBlock,
    DocumentBlockType,
    DocumentContentRegion,
    DocumentStructureBlock,
    FigureStructure,
    LiteratureRepresentationKind,
    StructureExtractionStatus,
    TableCellStructure,
    TableStructure,
)
from ...serialization import content_hash
from .records import (
    DocumentStructureInput,
    ExtractedDocumentStructure,
    _make_block,
    _make_cell,
    _make_figure,
    _make_table,
    _text_hash,
)


def _label(text: str, prefix: str, fallback_number: int) -> str:
    match = re.match(rf"(?i)^\s*({re.escape(prefix)}(?:ure)?\.?\s*[A-Za-z0-9.-]+)", text)
    return " ".join(match.group(1).split()) if match else f"{prefix.title()} {fallback_number}"


@dataclass
class _HTMLElement:
    tag: str
    text: str
    content_region: DocumentContentRegion
    table_index: int | None
    row_index: int | None
    row_span: int
    column_span: int
    figure_index: int | None
    cell_index: int | None


@dataclass
class _HTMLTableCell:
    table_index: int
    row_index: int
    row_span: int
    column_span: int
    is_header: bool
    invalid_span: bool
    event_indices: list[int] = field(default_factory=list)
    has_text: bool = False


class _HTMLStructureParser(HTMLParser):
    CAPTURE_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "caption", "th", "td", "figcaption"})
    REFERENCE_TOKENS = frozenset(
        {
            "references",
            "reference",
            "bibliography",
            "citations",
            "cited-by",
            "references-list",
        }
    )
    NAVIGATION_TOKENS = frozenset({"navigation", "nav", "menu"})
    FOOTER_TOKENS = frozenset({"footer", "site-footer"})
    RELATED_TOKENS = frozenset({"related", "related-articles", "recommended", "recommended-articles"})
    SUPPLEMENTARY_TOKENS = frozenset({"supplement", "supplementary", "supplementary-material"})
    MAIN_TOKENS = frozenset({"article", "article-content", "main", "main-content"})
    VOID_TAGS = frozenset(
        {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, DocumentContentRegion]] = []
        self.active: list[tuple[str, list[str], dict[str, object]]] = []
        self.elements: list[_HTMLElement] = []
        self.cells: list[_HTMLTableCell] = []
        self.ignored_depth = 0
        self.table_stack: list[dict[str, int]] = []
        self.cell_stack: list[int] = []
        self.figure_stack: list[int] = []
        self.table_count = 0
        self.figure_count = 0
        self.attribute_warnings: list[str] = []
        self.structural_errors: list[str] = []

    def _span(self, value: str | None, field: str) -> tuple[int, bool]:
        try:
            parsed = int(value or 1)
        except ValueError:
            parsed = 1
        invalid = parsed < 1 or parsed > 1000
        if invalid:
            parsed = 1
        if value is not None and (invalid or str(parsed) != value.strip()):
            self.attribute_warnings.append(f"invalid HTML {field} was conservatively treated as 1")
            invalid = True
        return parsed, invalid

    @classmethod
    def _region(
        cls,
        tag: str,
        attrs: list[tuple[str, str | None]],
        parent: DocumentContentRegion,
    ) -> DocumentContentRegion:
        values = " ".join(value or "" for key, value in attrs if key in {"id", "class", "role"})
        tokens = {token for token in re.split(r"[^a-z0-9-]+", values.casefold()) if token}
        if tokens & cls.REFERENCE_TOKENS:
            return DocumentContentRegion.REFERENCES
        if tag == "nav" or tokens & cls.NAVIGATION_TOKENS:
            return DocumentContentRegion.NAVIGATION
        if tag == "footer" or tokens & cls.FOOTER_TOKENS:
            return DocumentContentRegion.FOOTER
        if tokens & cls.RELATED_TOKENS:
            return DocumentContentRegion.RELATED_CONTENT
        if tag == "aside" or tokens & cls.SUPPLEMENTARY_TOKENS:
            return DocumentContentRegion.SUPPLEMENTARY_CONTEXT
        if parent not in {
            DocumentContentRegion.UNKNOWN,
            DocumentContentRegion.MAIN_CONTENT,
        }:
            return parent
        if tag in {"article", "main"} or tokens & cls.MAIN_TOKENS:
            return DocumentContentRegion.MAIN_CONTENT
        return parent

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        parent_region = self.stack[-1][1] if self.stack else DocumentContentRegion.UNKNOWN
        region = self._region(normalized, attrs, parent_region)
        if normalized not in self.VOID_TAGS:
            self.stack.append((normalized, region))
        if self.ignored_depth:
            return
        if normalized == "table":
            self.table_count += 1
            self.table_stack.append({"index": self.table_count - 1, "row": -1})
        elif normalized == "tr" and self.table_stack:
            self.table_stack[-1]["row"] += 1
        elif normalized == "figure":
            self.figure_count += 1
            self.figure_stack.append(self.figure_count - 1)
        attr_map = dict(attrs)
        if normalized in {"th", "td"}:
            if not self.table_stack or self.table_stack[-1]["row"] < 0:
                self.structural_errors.append("HTML table cell occurs outside a table row")
            else:
                row_span, invalid_row_span = self._span(attr_map.get("rowspan"), "rowspan")
                column_span, invalid_column_span = self._span(attr_map.get("colspan"), "colspan")
                table = self.table_stack[-1]
                self.cells.append(
                    _HTMLTableCell(
                        table_index=table["index"],
                        row_index=table["row"],
                        row_span=row_span,
                        column_span=column_span,
                        is_header=normalized == "th",
                        invalid_span=invalid_row_span or invalid_column_span,
                    )
                )
                self.cell_stack.append(len(self.cells) - 1)
        if normalized not in self.CAPTURE_TAGS:
            return
        table = self.table_stack[-1] if self.table_stack else None
        cell_index = self.cell_stack[-1] if self.cell_stack else None
        if normalized in {"th", "td"} and cell_index is None:
            row_span, _ = self._span(attr_map.get("rowspan"), "rowspan")
            column_span, _ = self._span(attr_map.get("colspan"), "colspan")
        elif cell_index is not None:
            owner = self.cells[cell_index]
            row_span = owner.row_span
            column_span = owner.column_span
        else:
            row_span = 1
            column_span = 1
        metadata: dict[str, object] = {
            "content_region": region,
            "table_index": table["index"] if table is not None else None,
            "row_index": table["row"] if table is not None else None,
            "row_span": row_span,
            "column_span": column_span,
            "figure_index": self.figure_stack[-1] if self.figure_stack else None,
            "cell_index": cell_index,
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
                    event_index = len(self.elements)
                    self.elements.append(
                        _HTMLElement(
                            tag=normalized,
                            text=text,
                            content_region=DocumentContentRegion(metadata["content_region"]),
                            table_index=metadata["table_index"],
                            row_index=metadata["row_index"],
                            row_span=int(metadata["row_span"]),
                            column_span=int(metadata["column_span"]),
                            figure_index=metadata["figure_index"],
                            cell_index=metadata["cell_index"],
                        )
                    )
                    cell_index = metadata["cell_index"]
                    if cell_index is not None:
                        self.cells[int(cell_index)].event_indices.append(event_index)
                del self.active[index]
                break
        if normalized in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
        if normalized == "table" and self.table_stack:
            self.table_stack.pop()
        elif normalized == "figure" and self.figure_stack:
            self.figure_stack.pop()
        elif normalized in {"th", "td"} and self.cell_stack:
            self.cell_stack.pop()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == normalized:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if self.cell_stack and data.strip():
            self.cells[self.cell_stack[-1]].has_text = True
        if self.active:
            self.active[-1][1].append(data)


def _table_occupancy_layout(
    cells: tuple[_HTMLTableCell, ...],
) -> tuple[tuple[_HTMLTableCell, int, int], ...]:
    occupied: set[tuple[int, int]] = set()
    positioned: list[tuple[_HTMLTableCell, int, int]] = []
    previous_row = -1
    for cell in cells:
        if cell.row_index < previous_row:
            raise ValueError("HTML table cell rows are not in deterministic order")
        previous_row = cell.row_index
        column = 0
        while True:
            candidate = {
                (row, col)
                for row in range(cell.row_index, cell.row_index + cell.row_span)
                for col in range(column, column + cell.column_span)
            }
            if not candidate & occupied:
                break
            column += 1
        if candidate & occupied:
            raise ValueError("HTML table cell spans overlap")
        occupied.update(candidate)
        positioned.append((cell, cell.row_index, column))
    return tuple(positioned)


class HTMLDocumentStructureExtractor:
    extractor_id = "html-canonical-structure"
    extractor_version = "3.0.0"
    representation_kind = LiteratureRepresentationKind.HTML
    extractor_config_hash = content_hash(
        {
            "tags": tuple(sorted(_HTMLStructureParser.CAPTURE_TAGS)),
            "content_regions": {
                "references": tuple(sorted(_HTMLStructureParser.REFERENCE_TOKENS)),
                "navigation": tuple(sorted(_HTMLStructureParser.NAVIGATION_TOKENS)),
                "footer": tuple(sorted(_HTMLStructureParser.FOOTER_TOKENS)),
                "related": tuple(sorted(_HTMLStructureParser.RELATED_TOKENS)),
                "supplementary": tuple(sorted(_HTMLStructureParser.SUPPLEMENTARY_TOKENS)),
            },
            "binding": "dom_owner_to_exact_canonical_blocks_v2",
            "table_coordinates": "rowspan_colspan_occupancy_grid_v1",
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
        if parser.structural_errors:
            raise ValueError("; ".join(dict.fromkeys(parser.structural_errors)))
        if parser.active or parser.stack or parser.cell_stack:
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
        event_canonical_blocks: dict[int, CanonicalTextBlock] = {}
        ordinal = 0
        for event_index, event in enumerate(parser.elements):
            queue = queues[(event.tag, _text_hash(event.text))]
            if not queue:
                warnings.append(f"HTML {event.tag} text is absent from canonical representation")
                continue
            canonical_block = queue.popleft()
            matched_ids.add(canonical_block.block_id)
            block_type = (
                DocumentBlockType.TABLE_CELL
                if event.cell_index is not None
                else {
                    "li": DocumentBlockType.LIST_ITEM,
                    "caption": DocumentBlockType.TABLE_CAPTION,
                    "figcaption": DocumentBlockType.FIGURE_CAPTION,
                }.get(event.tag, DocumentBlockType.PARAGRAPH)
            )
            level: int | None = None
            if event.tag.startswith("h") and len(event.tag) == 2 and event.tag[1].isdigit():
                if event.content_region not in {
                    DocumentContentRegion.MAIN_CONTENT,
                    DocumentContentRegion.UNKNOWN,
                }:
                    block_type = DocumentBlockType.OTHER_TEXT
                else:
                    block_type = DocumentBlockType.HEADING
                    level = int(event.tag[1])
                    for existing in tuple(heading_titles):
                        if existing >= level:
                            heading_titles.pop(existing, None)
                            heading_blocks.pop(existing, None)
                    heading_titles[level] = event.text
            if event.content_region in {
                DocumentContentRegion.MAIN_CONTENT,
                DocumentContentRegion.UNKNOWN,
            }:
                section_path = tuple(heading_titles[item] for item in sorted(heading_titles))
                parent = heading_blocks[max(heading_blocks)].block_id if heading_blocks else None
            else:
                section_path = (event.content_region.value,)
                parent = None
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
                content_region=event.content_region,
                label=label,
                start_offset=canonical_block.start_offset,
                end_offset=canonical_block.end_offset,
                text=recovered,
            )
            blocks.append(block)
            event_blocks[event_index] = block
            event_canonical_blocks[event_index] = canonical_block
            ordinal += 1
            if block_type == DocumentBlockType.HEADING:
                heading_blocks[level or 1] = block
        for canonical_block in source.canonical.blocks:
            if canonical_block.block_id in matched_ids:
                continue
            recovered = source.canonical_text[canonical_block.start_offset : canonical_block.end_offset]
            section_path = ()
            parent = None
            blocks.append(
                _make_block(
                    structure_id=structure_id,
                    block_type=DocumentBlockType.OTHER_TEXT,
                    ordinal=ordinal,
                    parent_block_id=parent,
                    section_path=section_path,
                    content_region=DocumentContentRegion.UNKNOWN,
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
            table_owners = tuple(owner for owner in parser.cells if owner.table_index == table_index)
            positioned = _table_occupancy_layout(table_owners)
            mapped_events = {
                event_index
                for owner in table_owners
                for event_index in owner.event_indices
                if event_index in event_blocks and event_index in event_canonical_blocks
            }
            table_complete = bool(table_owners) and all(
                not owner.invalid_span
                and (
                    not owner.has_text
                    or (
                        bool(owner.event_indices)
                        and all(event_index in mapped_events for event_index in owner.event_indices)
                    )
                )
                for owner in table_owners
            )
            first_cell_block = next(
                (
                    event_blocks[event_index]
                    for owner in table_owners
                    for event_index in owner.event_indices
                    if event_index in event_blocks
                ),
                None,
            )
            section_path = (
                caption.section_path
                if caption is not None
                else (first_cell_block.section_path if first_cell_block else ())
            )
            table_label = caption.label if caption is not None and caption.label else f"Table {table_index + 1}"
            table_status = StructureExtractionStatus.COMPLETE if table_complete else StructureExtractionStatus.PARTIAL
            base_table = _make_table(
                structure_id=structure_id,
                ordinal=table_index,
                label=table_label,
                caption_block_ref=caption.block_id if caption else None,
                page_number=None,
                section_path=section_path,
                cell_refs=(),
                status=table_status,
            )
            table_cells: list[TableCellStructure] = []
            for owner, row_index, column_index in positioned:
                canonical_regions = tuple(
                    event_canonical_blocks[event_index]
                    for event_index in owner.event_indices
                    if event_index in event_canonical_blocks
                )
                start_offset = min(region.start_offset for region in canonical_regions) if canonical_regions else None
                end_offset = max(region.end_offset for region in canonical_regions) if canonical_regions else None
                table_cells.append(
                    _make_cell(
                        table_id=base_table.table_id,
                        row_index=row_index,
                        column_index=column_index,
                        row_span=owner.row_span,
                        column_span=owner.column_span,
                        is_header=owner.is_header,
                        canonical_block_refs=tuple(region.block_id for region in canonical_regions),
                        start_offset=start_offset,
                        end_offset=end_offset,
                        text=(
                            source.canonical_text[start_offset:end_offset]
                            if start_offset is not None and end_offset is not None
                            else None
                        ),
                    )
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
                    status=table_status,
                )
            )
            if not table_owners:
                warnings.append(f"{table_label}: table cell structure is unresolved")
            elif not table_complete:
                warnings.append(f"{table_label}: table topology or canonical cell mapping is partial")
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
