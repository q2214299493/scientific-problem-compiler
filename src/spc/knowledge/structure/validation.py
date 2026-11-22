"""Verify persisted structure, canonical text, table topology and ownership."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from ...models import (
    CanonicalTextArtifact,
    DocumentBlockType,
    DocumentContentRegion,
    DocumentStructureArtifact,
    DocumentStructureBlock,
    FigureStructure,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    StructureExtractionStatus,
    TableCellStructure,
    TableStructure,
)
from ...repositories import (
    CanonicalHTMLTextArtifactRepository,
    CanonicalTextArtifactRepository,
    EvidenceStore,
    KnowledgeRepositories,
)
from ..representation import validate_literature_representation
from .records import CanonicalArtifact, _text_hash


# Function identities only: custom readers must retain their original path.
_CANONICAL_READERS = {
    repository_type: (repository_type.get, repository_type.read_text)
    for repository_type in (CanonicalTextArtifactRepository, CanonicalHTMLTextArtifactRepository)
}


def validate_table_cell_topology(
    cells: tuple[TableCellStructure, ...],
) -> None:
    occupied: set[tuple[int, int]] = set()
    logical_order: list[tuple[int, int]] = []
    for cell in cells:
        logical_order.append((cell.row_index, cell.column_index))
        occupied_by_cell = {
            (row, column)
            for row in range(cell.row_index, cell.row_index + cell.row_span)
            for column in range(cell.column_index, cell.column_index + cell.column_span)
        }
        if occupied & occupied_by_cell:
            raise ValueError("table cell span topology overlaps")
        occupied.update(occupied_by_cell)
    if logical_order != sorted(logical_order):
        raise ValueError("table cells are not in deterministic coordinate order")


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
    canonical_block_map = {block.block_id: block for block in canonical.blocks}
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
        if block.parent_block_id is not None and block_map[block.parent_block_id].ordinal >= block.ordinal:
            raise ValueError("document structure parent must precede its child")
    heading_titles: dict[int, str] = {}
    for block in sorted(blocks, key=lambda item: item.ordinal):
        if block.block_type == DocumentBlockType.PAGE:
            continue
        if block.content_region not in {
            DocumentContentRegion.MAIN_CONTENT,
            DocumentContentRegion.UNKNOWN,
        }:
            if block.section_path != (block.content_region.value,):
                raise ValueError("specialized content region has misleading section path")
            continue
        if block.content_region == DocumentContentRegion.UNKNOWN and not block.section_path:
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
        if table.extraction_status == StructureExtractionStatus.COMPLETE and not table.cell_refs:
            raise ValueError("complete table must contain mapped cells")
        table_cells = tuple(cell_map[cell_id] for cell_id in table.cell_refs)
        validate_table_cell_topology(table_cells)
        for cell in table_cells:
            if cell.table_id != table.table_id:
                raise ValueError("table cell points to another table")
            if cell.start_offset is None:
                if cell.canonical_block_refs or cell.end_offset is not None or cell.text_hash is not None:
                    raise ValueError("empty table cell has an invalid text binding")
                continue
            if (
                cell.end_offset is None
                or cell.text_hash is None
                or cell.end_offset > len(text)
                or _text_hash(text[cell.start_offset : cell.end_offset]) != cell.text_hash
            ):
                raise ValueError("table cell does not recover exact canonical text")
            if cell.canonical_block_refs:
                try:
                    canonical_regions = tuple(canonical_block_map[block_id] for block_id in cell.canonical_block_refs)
                except KeyError as error:
                    raise ValueError("table cell references an unknown canonical block") from error
                if (
                    tuple(
                        sorted(
                            canonical_regions,
                            key=lambda item: (item.start_offset, item.end_offset),
                        )
                    )
                    != canonical_regions
                ):
                    raise ValueError("table cell canonical regions are out of order")
                if (
                    canonical_regions[0].start_offset != cell.start_offset
                    or canonical_regions[-1].end_offset != cell.end_offset
                    or any(
                        left.end_offset > right.start_offset
                        for left, right in zip(canonical_regions, canonical_regions[1:])
                    )
                ):
                    raise ValueError("table cell canonical region topology is invalid")
                for region in canonical_regions:
                    matching_blocks = [
                        block
                        for block in blocks
                        if block.block_type == DocumentBlockType.TABLE_CELL
                        and block.start_offset == region.start_offset
                        and block.end_offset == region.end_offset
                        and block.text_hash == region.text_hash
                        and block.section_path == table.section_path
                    ]
                    if len(matching_blocks) != 1:
                        raise ValueError("table cell canonical region has no exact structure block")
            else:
                matching_blocks = [
                    block
                    for block in blocks
                    if block.block_type == DocumentBlockType.TABLE_CELL
                    and block.start_offset == cell.start_offset
                    and block.end_offset == cell.end_offset
                    and block.section_path == table.section_path
                ]
                if len(matching_blocks) != 1:
                    raise ValueError("legacy table cell has no exact structure block")
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
    *,
    _file_hashes: dict[Path, str] | None = None,
) -> DocumentStructureArtifact:
    store = evidence_store or repositories.evidence_store

    def read(repository, identity):
        if _file_hashes is None:
            return repository.get(identity)
        return repository.get(identity, _file_hashes=_file_hashes)

    artifact = read(repositories.document_structure_artifacts, structure) if isinstance(structure, str) else structure
    stored = read(repositories.document_structure_artifacts, artifact.structure_id)
    if stored != artifact:
        raise ValueError("document structure differs from repository record")
    representation = repositories.literature_representation_refs.get(artifact.representation_id)
    validate_literature_representation(representation, repositories, store)
    canonical_repository = (
        repositories.canonical_text_artifacts
        if artifact.representation_kind == LiteratureRepresentationKind.PDF
        else repositories.canonical_html_text_artifacts
    )
    standard_readers = _CANONICAL_READERS.get(type(canonical_repository))
    if (
        _file_hashes is not None
        and standard_readers is not None
        and getattr(canonical_repository.get, "__func__", None) is standard_readers[0]
        and getattr(canonical_repository.read_text, "__func__", None) is standard_readers[1]
    ):
        # Retain the text from the same read that passed every canonical check.
        # This value lives only in this validation; the trusted read still
        # revalidates the representation/source dependencies before returning.
        canonical, text = canonical_repository._read_validated_text(artifact.canonical_text_id)
    else:
        canonical = canonical_repository.get(artifact.canonical_text_id)
        text = canonical_repository.read_text(artifact.canonical_text_id)
    # Large papers contain many independently stored immutable blocks. Preserve
    # manifest order while bounding parallel file reads; each get still validates
    # the repository key and record identity before the full structure check.
    with ThreadPoolExecutor(max_workers=32) as executor:
        blocks = tuple(executor.map(partial(read, repositories.document_structure_blocks), artifact.block_hashes))
    tables = tuple(read(repositories.table_structures, table_id) for table_id in artifact.table_hashes)
    figures = tuple(read(repositories.figure_structures, figure_id) for figure_id in artifact.figure_hashes)
    cells = tuple(read(repositories.table_cell_structures, cell_id) for table in tables for cell_id in table.cell_refs)
    _validate_structure_records(artifact, blocks, tables, cells, figures, representation, canonical, text)
    return artifact
