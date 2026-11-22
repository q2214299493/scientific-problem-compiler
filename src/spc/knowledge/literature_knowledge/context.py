from __future__ import annotations

from collections.abc import Iterable

from ...immutable import FrozenDict
from ...models import (
    CurationStatus,
    DocumentBlockType,
    DocumentContentRegion,
    DocumentStructureArtifact,
    FigureStructure,
    LiteratureRepresentationKind,
    TableCellStructure,
    TableStructure,
)
from ...repositories import (
    EvidenceStore,
    KnowledgeRepositories,
    ModelRepository,
    TableCellStructureRepository,
    TableStructureRepository,
)
from ...serialization import build_content_bound_record as _content_bound
from ..ingestion import LiteratureRepresentationSelector
from ..structure import validate_document_structure
from ..structure_selection import resolve_current_structure_selection
from ..trust import TrustedKnowledgeValidator
from .contracts import LiteratureKnowledgeChunk, LiteratureKnowledgeCompilationInput
from .repositories import (
    LiteratureKnowledgeChunkRepository,
    LiteratureKnowledgeCompilationInputRepository,
)


_STANDARD_TABLE_GET = TableStructureRepository.get
_STANDARD_TABLE_ID_CHECK = TableStructureRepository._record_id
_STANDARD_MODEL_GET = ModelRepository.get


COMPILER_POLICY_ID = "spc-literature-knowledge-current-authority"
COMPILER_POLICY_VERSION = "1.0.0"
CHUNK_POLICY_ID = "spc-structure-block-bounded-chunks"
CHUNK_POLICY_VERSION = "1.0.0"
MAX_CHUNK_CHARACTERS = 8_000


def resolve_literature_knowledge_input(
    literature_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
    *,
    include_supplementary: bool = False,
) -> LiteratureKnowledgeCompilationInput:
    store = evidence_store or repositories.evidence_store
    document = repositories.literature_documents.get(literature_id)
    current_curations = TrustedKnowledgeValidator(repositories, store).resolve_current_curations()
    document_curation = current_curations.get(("literature_document", literature_id))
    if document_curation is None or document_curation.status != CurationStatus.ACCEPTED:
        raise ValueError("literature document is not accepted")
    representation_selection = repositories.literature_representation_selections.resolve_current(literature_id)
    selection_curation = current_curations.get(
        ("literature_representation_selection", representation_selection.selection_id)
    )
    if selection_curation is None or selection_curation.status != CurationStatus.ACCEPTED:
        raise ValueError("current literature representation selection is not accepted")
    representation = LiteratureRepresentationSelector.resolve_selection(
        representation_selection,
        repositories,
        store,
    )
    structure_selection = resolve_current_structure_selection(
        representation.representation_id,
        repositories,
        store,
    )
    structure = validate_document_structure(structure_selection.structure_id, repositories, store)
    included = {DocumentContentRegion.MAIN_CONTENT}
    if include_supplementary:
        included.add(DocumentContentRegion.SUPPLEMENTARY_CONTEXT)
    if representation.representation_kind == LiteratureRepresentationKind.PDF:
        included.add(DocumentContentRegion.UNKNOWN)
    identity = {
        "literature_id": document.literature_id,
        "literature_hash": document.content_hash,
        "representation_selection_id": representation_selection.selection_id,
        "representation_selection_hash": representation_selection.content_hash,
        "representation_id": representation.representation_id,
        "representation_hash": representation.content_hash,
        "representation_kind": representation.representation_kind,
        "canonical_text_id": representation.canonical_text_id,
        "canonical_text_hash": representation.canonical_text_hash,
        "source_id": representation.source_id,
        "source_version": representation.source_version,
        "structure_selection_id": structure_selection.selection_id,
        "structure_selection_hash": structure_selection.content_hash,
        "structure_id": structure.structure_id,
        "structure_hash": structure.content_hash,
        "compiler_policy_id": COMPILER_POLICY_ID,
        "compiler_policy_version": COMPILER_POLICY_VERSION,
        "included_content_regions": tuple(sorted(included)),
        "chunk_policy_id": CHUNK_POLICY_ID,
        "chunk_policy_version": CHUNK_POLICY_VERSION,
    }
    record = _content_bound(
        LiteratureKnowledgeCompilationInput,
        "literature-knowledge-input",
        "compilation_input_id",
        identity,
    )
    LiteratureKnowledgeCompilationInputRepository(repositories.root).put(
        record.compilation_input_id,
        record,
    )
    return record


def validate_literature_knowledge_input(
    record: LiteratureKnowledgeCompilationInput,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
) -> LiteratureKnowledgeCompilationInput:
    stored = LiteratureKnowledgeCompilationInputRepository(repositories.root).get(
        record.compilation_input_id
    )
    if stored != record:
        raise ValueError("literature knowledge input differs from repository record")
    current = resolve_literature_knowledge_input(
        record.literature_id,
        repositories,
        evidence_store,
        include_supplementary=(
            DocumentContentRegion.SUPPLEMENTARY_CONTEXT in record.included_content_regions
        ),
    )
    if current != record:
        raise ValueError("literature knowledge input is not bound to current authority")
    return record


def _bounded_ranges(start: int, end: int, text: str) -> Iterable[tuple[int, int]]:
    cursor = start
    while cursor < end:
        boundary = min(cursor + MAX_CHUNK_CHARACTERS, end)
        if boundary < end:
            relative = text[cursor:boundary]
            split = max(relative.rfind("\n"), relative.rfind(" "))
            if split > MAX_CHUNK_CHARACTERS // 2:
                boundary = cursor + split + 1
        while cursor < boundary and text[cursor].isspace():
            cursor += 1
        while boundary > cursor and text[boundary - 1].isspace():
            boundary -= 1
        if boundary > cursor:
            yield cursor, boundary
        cursor = max(boundary, cursor + 1)


def _canonical_block_map(
    structure: DocumentStructureArtifact,
    repositories: KnowledgeRepositories,
) -> dict[str, object]:
    canonical = (
        repositories.canonical_text_artifacts.get(structure.canonical_text_id)
        if structure.representation_kind == LiteratureRepresentationKind.PDF
        else repositories.canonical_html_text_artifacts.get(structure.canonical_text_id)
    )
    return {block.block_id: block for block in canonical.blocks}


def resolve_table_ownership(
    block,
    structure: DocumentStructureArtifact,
    repositories: KnowledgeRepositories,
    *,
    _table_records: dict[str, tuple[TableStructure, tuple[TableCellStructure, ...]]] | None = None,
) -> tuple[TableStructure, TableCellStructure | None] | None:
    if block.block_type not in {DocumentBlockType.TABLE_CAPTION, DocumentBlockType.TABLE_CELL}:
        return None
    if _table_records is not None and not (
        type(repositories.table_structures) is TableStructureRepository
        and type(repositories.table_cell_structures) is TableCellStructureRepository
        and repositories.table_structures.model_type is TableStructure
        and repositories.table_cell_structures.model_type is TableCellStructure
        and repositories.table_structures.identity_field == "table_id"
        and repositories.table_cell_structures.identity_field == "cell_id"
        and ModelRepository.get is _STANDARD_MODEL_GET
        and all(
            getattr(repository.get, "__func__", None) is _STANDARD_TABLE_GET
            and getattr(repository._record_id, "__func__", None) is _STANDARD_TABLE_ID_CHECK
            for repository in (repositories.table_structures, repositories.table_cell_structures)
        )
    ):
        # Preserve custom repository/model/identity checks on every lookup.
        _table_records = None
    canonical_blocks = _canonical_block_map(structure, repositories)
    matches: list[tuple[TableStructure, TableCellStructure | None]] = []
    for table_id, expected_hash in structure.table_hashes.items():
        cached = _table_records.get(table_id) if _table_records is not None else None
        table = repositories.table_structures.get(table_id) if cached is None else cached[0]
        if table.structure_id != structure.structure_id or table.content_hash != expected_hash:
            raise ValueError("table is not bound to the selected structure")
        if table.caption_block_ref == block.block_id:
            matches.append((table, None))
        cells = cached[1] if cached is not None else (
            repositories.table_cell_structures.get(cell_id) for cell_id in table.cell_refs
        )
        collected = [] if _table_records is not None and cached is None else None
        for cell in cells:
            if collected is not None:
                collected.append(cell)
            if cell.table_id != table.table_id:
                raise ValueError("table cell belongs to another table")
            if cell.canonical_block_refs:
                regions = []
                for canonical_block_id in cell.canonical_block_refs:
                    region = canonical_blocks.get(canonical_block_id)
                    if region is None:
                        raise ValueError("table cell references an unknown canonical block")
                    regions.append(region)
                owns_block = any(
                    region.start_offset == block.start_offset
                    and region.end_offset == block.end_offset
                    and region.text_hash == block.text_hash
                    for region in regions
                )
            else:
                owns_block = (
                    cell.start_offset == block.start_offset
                    and cell.end_offset == block.end_offset
                    and cell.text_hash == block.text_hash
                )
            if owns_block:
                matches.append((table, cell))
        if collected is not None:
            _table_records[table_id] = (table, tuple(collected))
    if len(matches) > 1:
        raise ValueError("document block has ambiguous table ownership")
    return matches[0] if matches else None


def _table_context(
    block,
    structure: DocumentStructureArtifact,
    repositories: KnowledgeRepositories,
) -> FrozenDict | None:
    ownership = resolve_table_ownership(block, structure, repositories)
    if ownership is None:
        return None
    table, cell = ownership
    values = {
        "table_id": table.table_id,
        "table_hash": table.content_hash,
        "label": table.label,
        "caption_block_ref": table.caption_block_ref,
        "cell_id": cell.cell_id if cell else None,
        "cell_hash": cell.content_hash if cell else None,
        "row_index": cell.row_index if cell else None,
        "column_index": cell.column_index if cell else None,
        "row_span": cell.row_span if cell else None,
        "column_span": cell.column_span if cell else None,
        "is_header": cell.is_header if cell else None,
        "canonical_block_refs": cell.canonical_block_refs if cell else (),
    }
    return FrozenDict({key: value for key, value in values.items() if value is not None})


def resolve_figure_ownership(
    block,
    structure: DocumentStructureArtifact,
    repositories: KnowledgeRepositories,
) -> FigureStructure | None:
    if block.block_type != DocumentBlockType.FIGURE_CAPTION:
        return None
    matches = []
    for figure_id, expected_hash in structure.figure_hashes.items():
        figure = repositories.figure_structures.get(figure_id)
        if figure.structure_id != structure.structure_id or figure.content_hash != expected_hash:
            raise ValueError("figure is not bound to the selected structure")
        if figure.caption_block_ref == block.block_id:
            matches.append(figure)
    if len(matches) > 1:
        raise ValueError("document block has ambiguous figure ownership")
    return matches[0] if matches else None


def _figure_context(
    block,
    structure: DocumentStructureArtifact,
    repositories: KnowledgeRepositories,
) -> FrozenDict | None:
    figure = resolve_figure_ownership(block, structure, repositories)
    if figure is None:
        return None
    return FrozenDict(
        {
            "figure_id": figure.figure_id,
            "figure_hash": figure.content_hash,
            "label": figure.label,
            "caption_only": True,
        }
    )


def validate_literature_knowledge_chunk(
    chunk: LiteratureKnowledgeChunk,
    compilation_input: LiteratureKnowledgeCompilationInput,
    repositories: KnowledgeRepositories,
) -> LiteratureKnowledgeChunk:
    stored = LiteratureKnowledgeChunkRepository(repositories.root).get(chunk.chunk_id)
    if stored != chunk:
        raise ValueError("literature knowledge chunk differs from repository record")
    if (
        chunk.literature_id != compilation_input.literature_id
        or chunk.representation_id != compilation_input.representation_id
        or chunk.structure_id != compilation_input.structure_id
    ):
        raise ValueError("literature knowledge chunk authority is invalid")
    if chunk.content_region not in compilation_input.included_content_regions:
        raise ValueError("literature knowledge chunk is outside content policy")
    if compilation_input.representation_kind == LiteratureRepresentationKind.PDF:
        canonical_text = repositories.canonical_text_artifacts.read_text(
            compilation_input.canonical_text_id
        )
    else:
        canonical_text = repositories.canonical_html_text_artifacts.read_text(
            compilation_input.canonical_text_id
        )
    if canonical_text[chunk.canonical_start_offset : chunk.canonical_end_offset] != chunk.text:
        raise ValueError("literature knowledge chunk text binding is invalid")
    structure = repositories.document_structure_artifacts.get(
        compilation_input.structure_id
    )
    for block_id in chunk.block_refs:
        block = repositories.document_structure_blocks.get(block_id)
        if (
            block.structure_id != compilation_input.structure_id
            or block.content_hash != chunk.block_hashes[block_id]
            or chunk.canonical_start_offset < block.start_offset
            or chunk.canonical_end_offset > block.end_offset
            or block.content_region != chunk.content_region
        ):
            raise ValueError("literature knowledge chunk block binding is invalid")
        if chunk.table_context != _table_context(block, structure, repositories):
            raise ValueError("literature knowledge chunk table context is invalid")
        if chunk.figure_context != _figure_context(block, structure, repositories):
            raise ValueError("literature knowledge chunk figure context is invalid")
    return chunk


def build_literature_knowledge_chunks(
    compilation_input: LiteratureKnowledgeCompilationInput,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
) -> tuple[LiteratureKnowledgeChunk, ...]:
    validate_literature_knowledge_input(compilation_input, repositories, evidence_store)
    if compilation_input.representation_kind == LiteratureRepresentationKind.PDF:
        canonical_text = repositories.canonical_text_artifacts.read_text(
            compilation_input.canonical_text_id
        )
    else:
        canonical_text = repositories.canonical_html_text_artifacts.read_text(
            compilation_input.canonical_text_id
        )
    structure = repositories.document_structure_artifacts.get(compilation_input.structure_id)
    chunks: list[LiteratureKnowledgeChunk] = []
    for block_id in structure.block_hashes:
        block = repositories.document_structure_blocks.get(block_id)
        if (
            block.block_type == DocumentBlockType.PAGE
            or block.content_region not in compilation_input.included_content_regions
        ):
            continue
        for start, end in _bounded_ranges(block.start_offset, block.end_offset, canonical_text):
            identity = {
                "literature_id": compilation_input.literature_id,
                "representation_id": compilation_input.representation_id,
                "structure_id": compilation_input.structure_id,
                "block_refs": (block.block_id,),
                "block_hashes": {block.block_id: block.content_hash},
                "page_number": block.page_number,
                "section_path": block.section_path,
                "content_region": block.content_region,
                "region_uncertain": block.content_region == DocumentContentRegion.UNKNOWN,
                "canonical_start_offset": start,
                "canonical_end_offset": end,
                "text": canonical_text[start:end],
                "table_context": _table_context(block, structure, repositories),
                "figure_context": _figure_context(block, structure, repositories),
            }
            identity = {key: value for key, value in identity.items() if value is not None}
            chunk = _content_bound(
                LiteratureKnowledgeChunk,
                "literature-knowledge-chunk",
                "chunk_id",
                identity,
            )
            LiteratureKnowledgeChunkRepository(repositories.root).put(chunk.chunk_id, chunk)
            chunks.append(chunk)
    return tuple(sorted(chunks, key=lambda item: (item.canonical_start_offset, item.chunk_id)))
