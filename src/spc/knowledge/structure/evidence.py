"""Create and verify structured evidence spans and locators."""

from __future__ import annotations

from ...models import (
    DocumentBlockType,
    DocumentContentRegion,
    DocumentStructureArtifact,
    DocumentStructureBlock,
    EvidenceSpan,
    FigureStructure,
    LiteratureRepresentationKind,
    StructuredEvidenceLocator,
    TableCellStructure,
    TableStructure,
)
from ...repositories import EvidenceStore, KnowledgeRepositories
from ...serialization import content_hash
from ..ingestion.evidence import create_evidence_span_from_canonical_text
from .validation import validate_document_structure


def format_structured_evidence_locator(
    locator: StructuredEvidenceLocator,
    repositories: KnowledgeRepositories | None = None,
) -> str:
    prefix = "PDF" if locator.page_number is not None else "HTML"
    parts = [f"{prefix} p.{locator.page_number}" if locator.page_number else prefix]
    if locator.content_region != DocumentContentRegion.UNKNOWN:
        parts.append(locator.content_region.value)
    parts.extend(item for item in locator.section_path if item != locator.content_region.value)
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
    artifact = validate_document_structure(record.structure_id, repositories, store)
    return _verify_locator_against_validated_structure(record, artifact, repositories, store)


def _verify_locator_against_validated_structure(
    record: StructuredEvidenceLocator,
    artifact: DocumentStructureArtifact,
    repositories: KnowledgeRepositories,
    store: EvidenceStore,
    *,
    verified_evidence: EvidenceSpan | None = None,
) -> StructuredEvidenceLocator:
    """Check one locator after its complete structure was validated in this read."""
    stored = repositories.structured_evidence_locators.get(record.locator_id)
    if stored != record:
        raise ValueError("structured evidence locator differs from repository record")
    if verified_evidence is None:
        evidence = store.get_evidence(record.evidence_id)
        store.verify_evidence_integrity(evidence)
    else:
        evidence = verified_evidence
        if evidence.evidence_id != record.evidence_id:
            raise ValueError("locator evidence differs from validated evidence")
    block = repositories.document_structure_blocks.get(record.block_id)
    if (
        record.structure_id != artifact.structure_id
        or
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
        or record.content_region != block.content_region
    ):
        raise ValueError("structured evidence locator provenance or offsets are invalid")
    if record.table_cell_id is not None:
        table = repositories.table_structures.get(record.table_id or "")
        cell = repositories.table_cell_structures.get(record.table_cell_id)
        if cell.canonical_block_refs:
            canonical = (
                repositories.canonical_text_artifacts.get(artifact.canonical_text_id)
                if artifact.representation_kind == LiteratureRepresentationKind.PDF
                else repositories.canonical_html_text_artifacts.get(artifact.canonical_text_id)
            )
            regions = {item.block_id: item for item in canonical.blocks if item.block_id in cell.canonical_block_refs}
            region_matches = any(
                region.start_offset == evidence.start_offset and region.end_offset == evidence.end_offset
                for region in regions.values()
            )
        else:
            region_matches = evidence.start_offset == cell.start_offset and evidence.end_offset == cell.end_offset
        if (
            table.structure_id != artifact.structure_id
            or cell.table_id != table.table_id
            or cell.cell_id not in table.cell_refs
            or record.row_index != cell.row_index
            or record.column_index != cell.column_index
            or not region_matches
            or block.start_offset != evidence.start_offset
            or block.end_offset != evidence.end_offset
        ):
            raise ValueError("structured table-cell locator is invalid")
    if record.figure_id is not None:
        figure = repositories.figure_structures.get(record.figure_id)
        if figure.structure_id != artifact.structure_id or figure.caption_block_ref != block.block_id:
            raise ValueError("structured figure locator is invalid")
    return record


def create_structured_evidence_locator(
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
        "content_region": block.content_region,
        "table_id": table.table_id if table else None,
        "table_cell_id": cell.cell_id if cell else None,
        "row_index": cell.row_index if cell else None,
        "column_index": cell.column_index if cell else None,
        "figure_id": figure.figure_id if figure else None,
        "canonical_start_offset": evidence.start_offset,
        "canonical_end_offset": evidence.end_offset,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    if block.content_region == DocumentContentRegion.UNKNOWN:
        identity.pop("content_region", None)
    locator_id = f"structured-locator-{content_hash(identity)[:24]}"
    payload = {"locator_id": locator_id, **identity}
    locator = StructuredEvidenceLocator(**payload, content_hash=content_hash(payload))
    repositories.structured_evidence_locators.put(locator.locator_id, locator)
    verify_structured_evidence_locator(locator, repositories, evidence_store)
    return evidence, locator


_create_locator = create_structured_evidence_locator


def _display_for_block(
    block: DocumentStructureBlock,
    *,
    table: TableStructure | None = None,
    cell: TableCellStructure | None = None,
    figure: FigureStructure | None = None,
) -> str:
    parts = [f"PDF p.{block.page_number}" if block.page_number else "HTML"]
    if block.content_region != DocumentContentRegion.UNKNOWN:
        parts.append(block.content_region.value)
    parts.extend(item for item in block.section_path if item != block.content_region.value)
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
    historical_structure: bool = False,
) -> tuple[EvidenceSpan, StructuredEvidenceLocator]:
    store = evidence_store or repositories.evidence_store
    artifact = validate_document_structure(structure_id, repositories, store)
    if not historical_structure:
        from ..structure_selection import require_current_structure

        require_current_structure(artifact, repositories, store)
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
    return create_structured_evidence_locator(
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
    historical_structure: bool = False,
    region_index: int | None = None,
) -> tuple[EvidenceSpan, StructuredEvidenceLocator]:
    store = evidence_store or repositories.evidence_store
    artifact = validate_document_structure(structure_id, repositories, store)
    if not historical_structure:
        from ..structure_selection import require_current_structure

        require_current_structure(artifact, repositories, store)
    table = repositories.table_structures.get(table_id)
    cell = repositories.table_cell_structures.get(cell_id)
    if (
        table.structure_id != artifact.structure_id
        or cell.table_id != table.table_id
        or cell.cell_id not in table.cell_refs
    ):
        raise ValueError("table cell belongs to another document structure")
    if cell.canonical_block_refs:
        canonical = (
            repositories.canonical_text_artifacts.get(artifact.canonical_text_id)
            if artifact.representation_kind == LiteratureRepresentationKind.PDF
            else repositories.canonical_html_text_artifacts.get(artifact.canonical_text_id)
        )
        canonical_map = {item.block_id: item for item in canonical.blocks}
        regions = tuple(canonical_map[item] for item in cell.canonical_block_refs)
        if len(regions) > 1 and region_index is None:
            raise ValueError("table cell has multiple exact regions; select region_index explicitly")
        selected_index = 0 if region_index is None else region_index
        if selected_index < 0 or selected_index >= len(regions):
            raise ValueError("table cell region_index is outside its exact regions")
        region = regions[selected_index]
        start_offset = region.start_offset
        end_offset = region.end_offset
    else:
        if cell.start_offset is None or cell.end_offset is None:
            raise ValueError("empty table cell has no evidence-capable text region")
        if region_index not in {None, 0}:
            raise ValueError("legacy table cell has only one exact region")
        start_offset = cell.start_offset
        end_offset = cell.end_offset
    structure_blocks = tuple(repositories.document_structure_blocks.get(block_id) for block_id in artifact.block_hashes)
    blocks = [
        block
        for block in structure_blocks
        if block.structure_id == artifact.structure_id
        and block.block_type == DocumentBlockType.TABLE_CELL
        and block.start_offset == start_offset
        and block.end_offset == end_offset
    ]
    if len(blocks) != 1:
        raise ValueError("table cell has no unique exact structure block")
    block = blocks[0]
    evidence = create_evidence_span_from_canonical_text(
        artifact.canonical_text_id,
        start_offset,
        end_offset,
        repositories,
        store,
        locator=_display_for_block(block, table=table, cell=cell),
        historical_ingestion=historical_representation,
    )
    return create_structured_evidence_locator(
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
    historical_structure: bool = False,
) -> tuple[EvidenceSpan, StructuredEvidenceLocator]:
    store = evidence_store or repositories.evidence_store
    artifact = validate_document_structure(structure_id, repositories, store)
    if not historical_structure:
        from ..structure_selection import require_current_structure

        require_current_structure(artifact, repositories, store)
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
    return create_structured_evidence_locator(
        evidence=evidence,
        artifact=artifact,
        block=block,
        figure=figure,
        repositories=repositories,
        evidence_store=store,
    )
