from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import unicodedata

from .contracts import (
    BackendCapability,
    BackendInputBinding,
    BackendRunStatus,
    DocumentParsingBackend,
    ExternalBindingStatus,
    ExternalDocumentElement,
    ExternalDocumentElementKind,
    ExternalDocumentParseInput,
    ExternalDocumentParseProposal,
    ExternalProposalStatus,
    ExternalStructurePromotionRecord,
    ExternalStructureRebindingResult,
    ReboundDocumentElement,
)
from .provenance import (
    BackendInvocationError,
    BackendRunRecord,
    BackendRunRepository,
    create_backend_run_record,
    probe_backend_runtime,
)
from ..knowledge.acquisition import validate_literature_representation
from ..knowledge.structure import (
    DocumentStructureExtractionResult,
    DocumentStructureInput,
    DocumentStructureService,
    ExtractedDocumentStructure,
    _make_block,
    _make_cell,
    _make_figure,
    _make_table,
)
from ..knowledge.structure_selection import DocumentStructureSelector
from ..models import (
    CanonicalHTMLTextArtifact,
    CanonicalTextArtifact,
    DocumentBlockType,
    DocumentContentRegion,
    DocumentStructureSelection,
    LiteratureRepresentationKind,
    StructureExtractionStatus,
)
from ..repositories import EvidenceStore, IdentityBoundRepository, KnowledgeRepositories
from ..serialization import content_hash, file_sha256


class BackendUnavailableError(RuntimeError):
    pass


class ExternalDocumentProposalRepository(IdentityBoundRepository[ExternalDocumentParseProposal]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "backend_document_proposals",
            ExternalDocumentParseProposal,
            "proposal_id",
        )


class ExternalStructureRebindingRepository(IdentityBoundRepository[ExternalStructureRebindingResult]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "backend_structure_rebindings",
            ExternalStructureRebindingResult,
            "rebinding_id",
        )


class ExternalStructurePromotionRepository(IdentityBoundRepository[ExternalStructurePromotionRecord]):
    def __init__(self, knowledge_root: Path) -> None:
        self.knowledge_root = knowledge_root
        super().__init__(
            knowledge_root / "backend_structure_promotions",
            ExternalStructurePromotionRecord,
            "promotion_id",
        )

    def get(self, key: str) -> ExternalStructurePromotionRecord:
        record = super().get(key)
        run = BackendRunRepository(self.knowledge_root).get(record.backend_run_id)
        proposal = ExternalDocumentProposalRepository(self.knowledge_root).get(record.proposal_id)
        rebinding = ExternalStructureRebindingRepository(self.knowledge_root).get(record.rebinding_id)
        repositories = KnowledgeRepositories(self.knowledge_root)
        structure = repositories.document_structure_artifacts.get(record.structure_id)
        selection = repositories.document_structure_selections.get(record.selection_id)
        actual = (
            run.content_hash,
            proposal.content_hash,
            rebinding.content_hash,
            structure.content_hash,
            selection.content_hash,
        )
        expected = (
            record.backend_run_hash,
            record.proposal_hash,
            record.rebinding_hash,
            record.structure_hash,
            record.selection_hash,
        )
        if actual != expected:
            raise ValueError("external structure promotion provenance is invalid")
        if (
            run.output_hash != proposal.content_hash
            or rebinding.proposal_id != proposal.proposal_id
            or rebinding.proposal_hash != proposal.content_hash
            or selection.structure_id != structure.structure_id
            or selection.structure_hash != structure.content_hash
            or proposal.backend_descriptor_hash != run.backend_descriptor_hash
            or proposal.runtime_identity_hash != run.runtime_identity_hash
        ):
            raise ValueError("external structure promotion chain is inconsistent")
        return record


def _normalized_with_offsets(value: str) -> tuple[str, tuple[int, ...]]:
    characters: list[str] = []
    offsets: list[int] = []
    pending_space: int | None = None
    for index, original in enumerate(value):
        normalized = unicodedata.normalize("NFKC", original)
        for character in normalized:
            if character.isspace():
                if characters and pending_space is None:
                    pending_space = index
                continue
            if pending_space is not None:
                characters.append(" ")
                offsets.append(pending_space)
                pending_space = None
            characters.append(character)
            offsets.append(index)
    return "".join(characters), tuple(offsets)


def exact_normalized_matches(candidate: str, canonical_text: str) -> tuple[tuple[int, int], ...]:
    normalized_candidate, _ = _normalized_with_offsets(candidate)
    normalized_text, offsets = _normalized_with_offsets(canonical_text)
    if not normalized_candidate:
        return ()
    matches: list[tuple[int, int]] = []
    cursor = 0
    while True:
        index = normalized_text.find(normalized_candidate, cursor)
        if index < 0:
            break
        start = offsets[index]
        end = offsets[index + len(normalized_candidate) - 1] + 1
        matches.append((start, end))
        cursor = index + 1
    return tuple(matches)


def _page_for_offsets(
    canonical: CanonicalTextArtifact | CanonicalHTMLTextArtifact,
    start: int,
    end: int,
) -> int | None:
    if isinstance(canonical, CanonicalHTMLTextArtifact):
        return None
    pages = tuple(
        block.page_number for block in canonical.blocks if start >= block.start_offset and end <= block.end_offset
    )
    return pages[0] if len(pages) == 1 else None


def _rebound_element(
    source: ExternalDocumentElement,
    *,
    status: ExternalBindingStatus,
    start_offset: int | None = None,
    end_offset: int | None = None,
    text_hash: str | None = None,
    page_number: int | None = None,
    page_hint_consistent: bool | None = None,
    reason: str | None = None,
) -> ReboundDocumentElement:
    identity = {
        "source_element_id": source.element_id,
        "kind": source.kind,
        "status": status,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "text_hash": text_hash,
        "page_number": page_number,
        "page_hint_consistent": page_hint_consistent,
        "heading_level": source.heading_level,
        "proposed_content_region": source.proposed_content_region,
        "content_region": DocumentContentRegion.UNKNOWN,
        "table_ref": source.table_ref,
        "row_index": source.row_index,
        "column_index": source.column_index,
        "row_span": source.row_span,
        "column_span": source.column_span,
        "is_header": source.is_header,
        "reason": reason,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    binding_id = f"rebound-document-element-{content_hash(identity)[:24]}"
    payload = {"binding_id": binding_id, **identity}
    return ReboundDocumentElement(
        **payload,
        content_hash=content_hash(payload),
    )


class ExternalStructureRebinder:
    def rebind(
        self,
        proposal: ExternalDocumentParseProposal,
        canonical: CanonicalTextArtifact | CanonicalHTMLTextArtifact,
        canonical_text: str,
    ) -> ExternalStructureRebindingResult:
        if hashlib.sha256(canonical_text.encode("utf-8")).hexdigest() != canonical.text_sha256:
            raise ValueError("canonical text bytes do not match canonical artifact")
        bindings: list[ReboundDocumentElement] = []
        for element in proposal.elements:
            matches = exact_normalized_matches(element.text, canonical_text)
            if len(matches) == 1:
                start, end = matches[0]
                page = _page_for_offsets(canonical, start, end)
                bindings.append(
                    _rebound_element(
                        element,
                        status=ExternalBindingStatus.RESOLVED,
                        start_offset=start,
                        end_offset=end,
                        text_hash=hashlib.sha256(canonical_text[start:end].encode("utf-8")).hexdigest(),
                        page_number=page,
                        page_hint_consistent=(None if element.page_hint is None else page == element.page_hint),
                    )
                )
            else:
                status = ExternalBindingStatus.MISSING if not matches else ExternalBindingStatus.AMBIGUOUS
                bindings.append(
                    _rebound_element(
                        element,
                        status=status,
                        reason=(
                            "candidate text is absent from canonical text"
                            if not matches
                            else "candidate text has multiple canonical matches"
                        ),
                    )
                )
        resolved_count = sum(item.status == ExternalBindingStatus.RESOLVED for item in bindings)
        identity = {
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.content_hash,
            "canonical_text_id": canonical.canonical_text_id,
            "canonical_text_hash": canonical.content_hash,
            "bindings": tuple(bindings),
            "resolved_count": resolved_count,
            "unresolved_count": len(bindings) - resolved_count,
        }
        rebinding_id = f"external-structure-rebinding-{content_hash(identity)[:24]}"
        payload = {"rebinding_id": rebinding_id, **identity}
        return ExternalStructureRebindingResult(
            **payload,
            content_hash=content_hash(payload),
        )


_BLOCK_TYPES = {
    ExternalDocumentElementKind.PAGE: DocumentBlockType.PAGE,
    ExternalDocumentElementKind.HEADING: DocumentBlockType.HEADING,
    ExternalDocumentElementKind.PARAGRAPH: DocumentBlockType.PARAGRAPH,
    ExternalDocumentElementKind.LIST_ITEM: DocumentBlockType.LIST_ITEM,
    ExternalDocumentElementKind.TABLE_CAPTION: DocumentBlockType.TABLE_CAPTION,
    ExternalDocumentElementKind.TABLE_CELL: DocumentBlockType.TABLE_CELL,
    ExternalDocumentElementKind.FIGURE_CAPTION: DocumentBlockType.FIGURE_CAPTION,
    ExternalDocumentElementKind.OTHER_TEXT: DocumentBlockType.OTHER_TEXT,
}


class ReboundDocumentStructureExtractor:
    extractor_id = "external-proposal-rebinder"
    extractor_version = "1.0.0"

    def __init__(
        self,
        proposal: ExternalDocumentParseProposal,
        rebinding: ExternalStructureRebindingResult,
        representation_kind: LiteratureRepresentationKind,
    ) -> None:
        self.proposal = proposal
        self.rebinding = rebinding
        self.representation_kind = representation_kind
        self.extractor_config_hash = content_hash(
            {
                "proposal_hash": proposal.content_hash,
                "rebinding_hash": rebinding.content_hash,
            }
        )

    def extract(self, structure_id: str, source: DocumentStructureInput) -> ExtractedDocumentStructure:
        if source.canonical.canonical_text_id != self.rebinding.canonical_text_id:
            raise ValueError("rebinding targets another canonical representation")
        resolved = tuple(item for item in self.rebinding.bindings if item.status == ExternalBindingStatus.RESOLVED)
        if not resolved:
            raise ValueError("external proposal has no evidence-capable exact bindings")
        blocks = []
        block_by_binding: dict[str, object] = {}
        heading_titles: dict[int, str] = {}
        heading_blocks: dict[int, object] = {}
        for ordinal, binding in enumerate(resolved):
            start = binding.start_offset
            end = binding.end_offset
            if start is None or end is None:
                raise ValueError("resolved binding lost exact offsets")
            block_type = _BLOCK_TYPES[binding.kind]
            level = binding.heading_level
            if block_type == DocumentBlockType.HEADING:
                for existing in tuple(heading_titles):
                    if existing >= (level or 1):
                        heading_titles.pop(existing, None)
                        heading_blocks.pop(existing, None)
                heading_titles[level or 1] = source.canonical_text[start:end]
            section_path = tuple(heading_titles[key] for key in sorted(heading_titles))
            parent = heading_blocks[max(heading_blocks)] if heading_blocks else None
            block = _make_block(
                structure_id=structure_id,
                block_type=block_type,
                ordinal=ordinal,
                parent_block_id=(parent.block_id if parent is not None else None),
                page_number=binding.page_number,
                heading_level=level,
                section_path=section_path,
                content_region=DocumentContentRegion.UNKNOWN,
                start_offset=start,
                end_offset=end,
                text=source.canonical_text[start:end],
            )
            blocks.append(block)
            block_by_binding[binding.binding_id] = block
            if block_type == DocumentBlockType.HEADING:
                heading_blocks[level or 1] = block

        tables = []
        cells = []
        table_refs = sorted(
            {
                item.table_ref
                for item in resolved
                if item.kind == ExternalDocumentElementKind.TABLE_CELL and item.table_ref is not None
            }
        )
        for ordinal, table_ref in enumerate(table_refs):
            table_bindings = tuple(
                item
                for item in resolved
                if item.kind == ExternalDocumentElementKind.TABLE_CELL and item.table_ref == table_ref
            )
            caption_binding = next(
                (
                    item
                    for item in resolved
                    if item.kind == ExternalDocumentElementKind.TABLE_CAPTION and item.table_ref == table_ref
                ),
                None,
            )
            representative = caption_binding or table_bindings[0]
            representative_block = block_by_binding[representative.binding_id]
            table_status = (
                StructureExtractionStatus.COMPLETE
                if self.proposal.status == ExternalProposalStatus.COMPLETE
                and len(table_bindings)
                == sum(
                    item.kind == ExternalDocumentElementKind.TABLE_CELL and item.table_ref == table_ref
                    for item in self.proposal.elements
                )
                else StructureExtractionStatus.PARTIAL
            )
            base = _make_table(
                structure_id=structure_id,
                ordinal=ordinal,
                label=table_ref,
                caption_block_ref=(
                    block_by_binding[caption_binding.binding_id].block_id if caption_binding is not None else None
                ),
                page_number=representative.page_number,
                section_path=representative_block.section_path,
                cell_refs=(),
                status=table_status,
            )
            table_cells = []
            for item in table_bindings:
                start = item.start_offset
                end = item.end_offset
                if start is None or end is None:
                    raise ValueError("resolved table cell lost exact offsets")
                table_cells.append(
                    _make_cell(
                        table_id=base.table_id,
                        row_index=item.row_index or 0,
                        column_index=item.column_index or 0,
                        row_span=item.row_span,
                        column_span=item.column_span,
                        is_header=item.is_header,
                        canonical_block_refs=(),
                        start_offset=start,
                        end_offset=end,
                        text=source.canonical_text[start:end],
                    )
                )
            cells.extend(table_cells)
            tables.append(
                _make_table(
                    structure_id=structure_id,
                    ordinal=ordinal,
                    label=table_ref,
                    caption_block_ref=base.caption_block_ref,
                    page_number=base.page_number,
                    section_path=base.section_path,
                    cell_refs=tuple(item.cell_id for item in table_cells),
                    status=table_status,
                )
            )

        figures = []
        figure_bindings = tuple(item for item in resolved if item.kind == ExternalDocumentElementKind.FIGURE_CAPTION)
        for ordinal, item in enumerate(figure_bindings):
            block = block_by_binding[item.binding_id]
            figures.append(
                _make_figure(
                    structure_id=structure_id,
                    ordinal=ordinal,
                    label=f"Figure {ordinal + 1}",
                    page_number=item.page_number,
                    section_path=block.section_path,
                    caption_block_ref=block.block_id,
                )
            )
        warnings = tuple(
            f"unresolved external element: {item.source_element_id} ({item.status.value})"
            for item in self.rebinding.bindings
            if item.status != ExternalBindingStatus.RESOLVED
        )
        status = (
            StructureExtractionStatus.COMPLETE
            if not warnings and self.proposal.status == ExternalProposalStatus.COMPLETE
            else StructureExtractionStatus.PARTIAL
        )
        return ExtractedDocumentStructure(
            blocks=tuple(blocks),
            tables=tuple(tables),
            table_cells=tuple(cells),
            figures=tuple(figures),
            page_count=(source.canonical.page_count if isinstance(source.canonical, CanonicalTextArtifact) else None),
            status=status,
            warnings=warnings,
        )


@dataclass(frozen=True)
class ExternalDocumentStructureOutcome:
    proposal: ExternalDocumentParseProposal
    rebinding: ExternalStructureRebindingResult
    structure: DocumentStructureExtractionResult
    run_record: BackendRunRecord
    selection: DocumentStructureSelection | None
    promotion: ExternalStructurePromotionRecord | None


class ExternalDocumentStructureService:
    def structure(
        self,
        literature_id: str,
        representation_id: str,
        backend: DocumentParsingBackend,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
        *,
        config: dict[str, str | int | float | bool] | None = None,
        promote: bool = False,
    ) -> ExternalDocumentStructureOutcome:
        store = evidence_store or repositories.evidence_store
        descriptor = backend.descriptor
        if BackendCapability.DOCUMENT_PARSING not in descriptor.capability_types:
            raise ValueError("backend does not provide document_parsing")
        representation = repositories.literature_representation_refs.get(representation_id)
        validate_literature_representation(representation, repositories, store)
        if representation.literature_id != literature_id:
            raise ValueError("representation belongs to another literature work")
        if representation.representation_kind == LiteratureRepresentationKind.PDF:
            raw = repositories.raw_literature_artifacts.get(representation.raw_artifact_id)
            raw_path = repositories.root.joinpath(*PurePosixPath(raw.stored_path).parts)
            canonical = repositories.canonical_text_artifacts.get(representation.canonical_text_id)
            canonical_text = repositories.canonical_text_artifacts.read_text(representation.canonical_text_id)
            media_type = "application/pdf"
            artifact_id = raw.artifact_id
            artifact_hash = raw.sha256
        else:
            raw = repositories.raw_html_literature_artifacts.get(representation.raw_artifact_id)
            raw_path = repositories.root.joinpath(*PurePosixPath(raw.stored_path).parts)
            canonical = repositories.canonical_html_text_artifacts.get(representation.canonical_text_id)
            canonical_text = repositories.canonical_html_text_artifacts.read_text(representation.canonical_text_id)
            media_type = raw.media_type
            artifact_id = raw.artifact_id
            artifact_hash = raw.sha256
        if file_sha256(raw_path) != artifact_hash:
            raise ValueError("raw artifact integrity check failed before backend invocation")
        config_hash = content_hash(config or {})
        parse_input = ExternalDocumentParseInput(
            artifact_id=artifact_id,
            artifact_path=str(raw_path.resolve()),
            artifact_sha256=artifact_hash,
            media_type=media_type,
            parsing_config=dict(config or {}),
            config_hash=config_hash,
        )
        run_repository = BackendRunRepository(repositories.root)
        input_bindings = (
            BackendInputBinding(
                input_id=artifact_id,
                input_hash=artifact_hash,
            ),
        )
        runtime_probe = probe_backend_runtime(backend, repositories.root)
        runtime = runtime_probe.runtime_identity
        if runtime_probe.failure_status is not None:
            run = create_backend_run_record(
                descriptor=descriptor,
                runtime_identity=runtime,
                capability=BackendCapability.DOCUMENT_PARSING,
                input_bindings=input_bindings,
                config_hash=config_hash,
                output_hash=None,
                status=runtime_probe.failure_status,
                warnings=(runtime_probe.failure_category or "runtime_probe_failed",),
            )
            run_repository.put(run.run_id, run)
            error_type = (
                BackendUnavailableError
                if runtime_probe.failure_status == BackendRunStatus.UNAVAILABLE
                else BackendInvocationError
            )
            raise error_type(f"backend {descriptor.backend_id} runtime probe failed; run_id={run.run_id}")
        proposal: ExternalDocumentParseProposal | None = None
        try:
            proposal = ExternalDocumentParseProposal.model_validate(backend.parse(parse_input))
            if (
                proposal.backend_id != descriptor.backend_id
                or proposal.backend_descriptor_hash != descriptor.content_hash
                or proposal.runtime_identity_hash != runtime.content_hash
                or proposal.artifact_id != artifact_id
                or proposal.artifact_sha256 != artifact_hash
            ):
                raise ValueError("external document proposal binding is invalid")
        except Exception as error:
            run = create_backend_run_record(
                descriptor=descriptor,
                runtime_identity=runtime,
                capability=BackendCapability.DOCUMENT_PARSING,
                input_bindings=input_bindings,
                config_hash=config_hash,
                output_hash=None,
                status=BackendRunStatus.FAILED,
                warnings=("backend invocation or output validation failed",),
            )
            run_repository.put(run.run_id, run)
            raise BackendInvocationError(f"external document backend failed; run_id={run.run_id}") from error
        ExternalDocumentProposalRepository(repositories.root).put(proposal.proposal_id, proposal)
        try:
            rebinding = ExternalStructureRebinder().rebind(proposal, canonical, canonical_text)
            ExternalStructureRebindingRepository(repositories.root).put(rebinding.rebinding_id, rebinding)
            extractor = ReboundDocumentStructureExtractor(proposal, rebinding, representation.representation_kind)
            structure = DocumentStructureService().extract(
                literature_id,
                representation_id,
                repositories,
                store,
                extractor=extractor,
            )
        except Exception as error:
            run = create_backend_run_record(
                descriptor=descriptor,
                runtime_identity=runtime,
                capability=BackendCapability.DOCUMENT_PARSING,
                input_bindings=input_bindings,
                config_hash=config_hash,
                output_hash=proposal.content_hash,
                status=BackendRunStatus.FAILED,
                warnings=("spc_exact_rebinding_or_structure_validation_failed",),
                output_count=len(proposal.elements),
            )
            run_repository.put(run.run_id, run)
            raise BackendInvocationError(f"external document normalization failed; run_id={run.run_id}") from error
        run_status = (
            BackendRunStatus.SUCCEEDED
            if rebinding.unresolved_count == 0 and proposal.status == ExternalProposalStatus.COMPLETE
            else BackendRunStatus.PARTIAL
        )
        run = create_backend_run_record(
            descriptor=descriptor,
            runtime_identity=runtime,
            capability=BackendCapability.DOCUMENT_PARSING,
            input_bindings=input_bindings,
            config_hash=config_hash,
            output_hash=proposal.content_hash,
            status=run_status,
            warnings=proposal.warnings,
            output_count=len(proposal.elements),
            resolved_count=rebinding.resolved_count,
            unresolved_count=rebinding.unresolved_count,
        )
        run_repository.put(run.run_id, run)
        selection = None
        promotion = None
        if promote:
            selected = DocumentStructureSelector().consider_auto_promotion(
                literature_id,
                representation_id,
                structure.artifact.structure_id,
                repositories,
                store,
                rationale="external_exact_rebinding_promotion",
            )
            if (
                selected is not None
                and selected.structure_id == structure.artifact.structure_id
                and selected.structure_hash == structure.artifact.content_hash
            ):
                selection = selected
                promotion_identity = {
                    "backend_run_id": run.run_id,
                    "backend_run_hash": run.content_hash,
                    "proposal_id": proposal.proposal_id,
                    "proposal_hash": proposal.content_hash,
                    "rebinding_id": rebinding.rebinding_id,
                    "rebinding_hash": rebinding.content_hash,
                    "structure_id": structure.artifact.structure_id,
                    "structure_hash": structure.artifact.content_hash,
                    "selection_id": selection.selection_id,
                    "selection_hash": selection.content_hash,
                    "promotion_policy": "external_exact_rebinding_promotion",
                }
                promotion_id = f"external-structure-promotion-{content_hash(promotion_identity)[:24]}"
                promotion_payload = {
                    "promotion_id": promotion_id,
                    **promotion_identity,
                }
                promotion = ExternalStructurePromotionRecord(
                    **promotion_payload,
                    content_hash=content_hash(promotion_payload),
                )
                ExternalStructurePromotionRepository(repositories.root).put(promotion.promotion_id, promotion)
        return ExternalDocumentStructureOutcome(
            proposal=proposal,
            rebinding=rebinding,
            structure=structure,
            run_record=run,
            selection=selection,
            promotion=promotion,
        )
