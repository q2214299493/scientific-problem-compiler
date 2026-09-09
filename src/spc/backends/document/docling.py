from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
from typing import Any, Iterable

from ..builders import (
    build_external_document_element,
    build_external_document_parse_proposal,
)
from ..contracts import (
    BackendCapability,
    BackendIntegrationMode,
    BackendLicenseStatus,
    BackendRuntimeAvailability,
    BackendRuntimeIdentity,
    ExternalBackendDescriptor,
    ExternalDocumentElement,
    ExternalDocumentElementKind,
    ExternalDocumentParseInput,
    ExternalDocumentParseProposal,
    ExternalProposalStatus,
)
from ..provenance import build_backend_runtime_identity
from ...models import DocumentContentRegion
from ...serialization import content_hash, file_sha256


SUPPORTED_DOCLING_MAJOR = 2


def _descriptor() -> ExternalBackendDescriptor:
    payload = {
        "backend_id": "docling",
        "backend_name": "Docling",
        "backend_version": "runtime-resolved",
        "adapter_version": "1.1.0",
        "capability_types": (BackendCapability.DOCUMENT_PARSING,),
        "integration_mode": BackendIntegrationMode.PYTHON_LIBRARY,
        "package_name": "docling",
        "source_project": "https://github.com/docling-project/docling",
        "license_id": "MIT",
        "license_status": BackendLicenseStatus.CONFIRMED,
        "deterministic": False,
        "requires_network": False,
        "optional_dependency": True,
        "vendored": False,
        "redistributable_confirmed": False,
    }
    return ExternalBackendDescriptor(**payload, content_hash=content_hash(payload))


def _version_is_supported(version: str) -> bool:
    try:
        return int(version.split(".", 1)[0]) == SUPPORTED_DOCLING_MAJOR
    except ValueError:
        return False


def _value(value: object) -> str:
    native = getattr(value, "value", value)
    return str(native)


def _native_ref(item: object) -> str | None:
    reference = getattr(item, "self_ref", None)
    if reference is None:
        return None
    value = getattr(reference, "cref", reference)
    rendered = str(value).strip()
    return rendered or None


def _page_hint(item: object) -> int | None:
    provenance = getattr(item, "prov", None)
    if isinstance(provenance, Iterable):
        for entry in provenance:
            value = getattr(entry, "page_no", None)
            if isinstance(value, int) and value >= 1:
                return value
    return None


def _kind(label: str) -> ExternalDocumentElementKind:
    normalized = label.casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"title", "section_header", "heading"}:
        return ExternalDocumentElementKind.HEADING
    if normalized in {"list_item", "listitem"}:
        return ExternalDocumentElementKind.LIST_ITEM
    if normalized in {"paragraph", "text"}:
        return ExternalDocumentElementKind.PARAGRAPH
    return ExternalDocumentElementKind.OTHER_TEXT


def _proposed_content_region(item: object) -> DocumentContentRegion:
    raw = getattr(item, "content_region", None)
    if raw is None:
        return DocumentContentRegion.UNKNOWN
    try:
        return DocumentContentRegion(_value(raw))
    except ValueError:
        return DocumentContentRegion.UNKNOWN


def _resolve_caption(document: object, reference: object) -> object | None:
    if isinstance(reference, str):
        return reference
    if isinstance(getattr(reference, "text", None), str):
        return reference
    resolver = getattr(document, "resolve_item", None)
    if callable(resolver):
        try:
            return resolver(reference)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _caption_items(document: object, item: object) -> tuple[object, ...]:
    candidates: list[object] = []
    plural = getattr(item, "captions", None)
    if isinstance(plural, Iterable) and not isinstance(plural, (str, bytes)):
        candidates.extend(plural)
    singular = getattr(item, "caption", None)
    if singular is not None:
        candidates.append(singular)
    resolved = tuple(_resolve_caption(document, candidate) for candidate in candidates)
    return tuple(item for item in resolved if item is not None)


def _caption_ref_ids(items: tuple[tuple[object, int], ...]) -> frozenset[str]:
    refs: set[str] = set()
    for item, _level in items:
        for candidate in getattr(item, "captions", ()) or ():
            value = getattr(candidate, "cref", candidate)
            if isinstance(value, str) and value.strip():
                refs.add(value.strip())
    return frozenset(refs)


def normalize_docling_document(
    document: object,
    *,
    descriptor: ExternalBackendDescriptor,
    runtime_identity: BackendRuntimeIdentity,
    request: ExternalDocumentParseInput,
) -> ExternalDocumentParseProposal:
    iterator = getattr(document, "iterate_items", None)
    if not callable(iterator):
        raise ValueError("Docling document does not expose iterate_items()")
    raw_items = tuple(iterator())
    items: tuple[tuple[object, int], ...] = tuple(
        (entry[0], entry[1] if isinstance(entry[1], int) else 1)
        for entry in raw_items
        if isinstance(entry, tuple) and len(entry) == 2
    )
    if len(items) != len(raw_items):
        raise ValueError("Docling iterate_items() returned an unsupported item shape")
    owned_caption_refs = _caption_ref_ids(items)
    elements: list[ExternalDocumentElement] = []
    warnings: list[str] = []

    def add_element(**values: Any) -> None:
        elements.append(
            build_external_document_element(
                reading_order=len(elements),
                **values,
            )
        )

    for table_ordinal, (item, traversal_level) in enumerate(items):
        native_ref = _native_ref(item)
        label = _value(getattr(item, "label", item.__class__.__name__))
        data = getattr(item, "data", None)
        table_cells = getattr(data, "table_cells", None)
        if table_cells is not None:
            table_ref = native_ref or f"docling-table-{table_ordinal}"
            for caption in _caption_items(document, item):
                caption_text = caption if isinstance(caption, str) else getattr(caption, "text", None)
                if isinstance(caption_text, str) and caption_text.strip():
                    add_element(
                        kind=ExternalDocumentElementKind.TABLE_CAPTION,
                        text=caption_text.strip(),
                        page_hint=_page_hint(item),
                        proposed_content_region=_proposed_content_region(caption),
                        table_ref=table_ref,
                        backend_native_ref=_native_ref(caption),
                        backend_metadata={"native_label": "table_caption"},
                    )
            for cell_ordinal, cell in enumerate(table_cells):
                text = getattr(cell, "text", None)
                row_index = getattr(cell, "start_row_offset_idx", None)
                column_index = getattr(cell, "start_col_offset_idx", None)
                row_span = getattr(cell, "row_span", 1)
                column_span = getattr(cell, "col_span", 1)
                if not isinstance(text, str) or not text.strip():
                    warnings.append(f"table {table_ref} omitted blank cell {cell_ordinal}")
                    continue
                if not all(
                    isinstance(value, int) and value >= minimum
                    for value, minimum in (
                        (row_index, 0),
                        (column_index, 0),
                        (row_span, 1),
                        (column_span, 1),
                    )
                ):
                    warnings.append(f"table {table_ref} omitted invalid cell {cell_ordinal}")
                    continue
                add_element(
                    kind=ExternalDocumentElementKind.TABLE_CELL,
                    text=text.strip(),
                    page_hint=_page_hint(item),
                    proposed_content_region=_proposed_content_region(item),
                    table_ref=table_ref,
                    row_index=row_index,
                    column_index=column_index,
                    row_span=row_span,
                    column_span=column_span,
                    is_header=bool(
                        getattr(cell, "column_header", False)
                        or getattr(cell, "row_header", False)
                        or getattr(cell, "row_section", False)
                    ),
                    backend_native_ref=f"{table_ref}/cell/{cell_ordinal}",
                    backend_metadata={"native_label": "table_cell"},
                )
            continue

        captions = _caption_items(document, item)
        if captions or label.casefold() in {"picture", "figure"}:
            for caption in captions:
                caption_text = caption if isinstance(caption, str) else getattr(caption, "text", None)
                if isinstance(caption_text, str) and caption_text.strip():
                    add_element(
                        kind=ExternalDocumentElementKind.FIGURE_CAPTION,
                        text=caption_text.strip(),
                        page_hint=_page_hint(item),
                        proposed_content_region=_proposed_content_region(caption),
                        backend_native_ref=_native_ref(caption),
                        backend_metadata={"native_label": "figure_caption"},
                    )
            continue

        if native_ref in owned_caption_refs:
            continue
        text = getattr(item, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        kind = _kind(label)
        heading_level = None
        if kind == ExternalDocumentElementKind.HEADING:
            candidate_level = getattr(item, "level", traversal_level)
            heading_level = min(
                6,
                max(1, candidate_level if isinstance(candidate_level, int) else 1),
            )
        content_layer = getattr(item, "content_layer", None)
        add_element(
            kind=kind,
            text=text.strip(),
            page_hint=_page_hint(item),
            heading_level=heading_level,
            proposed_content_region=_proposed_content_region(item),
            backend_native_ref=native_ref,
            backend_metadata={
                "native_label": label,
                "native_content_layer": (_value(content_layer) if content_layer is not None else None),
            },
        )
    status = (
        ExternalProposalStatus.COMPLETE
        if elements and not warnings
        else ExternalProposalStatus.PARTIAL
        if elements
        else ExternalProposalStatus.UNRESOLVED
    )
    if not elements:
        warnings.append("Docling returned no supported typed document elements")
    return build_external_document_parse_proposal(
        descriptor=descriptor,
        runtime_identity=runtime_identity,
        request=request,
        elements=tuple(elements),
        status=status,
        warnings=tuple(warnings),
    )


class DoclingDocumentParsingBackend:
    descriptor = _descriptor()

    def __init__(self, artifacts_path: Path | None = None) -> None:
        self.artifacts_path = artifacts_path

    def _installed_version(self) -> str | None:
        try:
            if util.find_spec("docling") is None:
                return None
            return metadata.version("docling")
        except (
            ImportError,
            metadata.PackageNotFoundError,
            ModuleNotFoundError,
            ValueError,
        ):
            return None

    def inspect_availability(self) -> BackendRuntimeAvailability:
        version = self._installed_version()
        if version is None:
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                reason="optional package 'docling' is not installed",
            )
        if not _version_is_supported(version):
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                detected_version=version,
                reason=(f"Docling major version {version} is outside the tested 2.x API range"),
            )
        if self.artifacts_path is None:
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                detected_version=version,
                reason="an explicit local Docling artifacts path is required for offline mode",
            )
        if (
            not self.artifacts_path.is_absolute()
            or not self.artifacts_path.is_dir()
            or self.artifacts_path.is_symlink()
        ):
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                detected_version=version,
                reason=("the configured Docling artifacts path is not a regular absolute directory"),
            )
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=True,
            detected_version=version,
        )

    def resolve_runtime_identity(self) -> BackendRuntimeIdentity:
        availability = self.inspect_availability()
        if not availability.available or availability.detected_version is None:
            raise RuntimeError(availability.reason or "Docling runtime is unavailable")
        return build_backend_runtime_identity(
            self.descriptor,
            resolved_backend_version=availability.detected_version,
            runtime_provider="python-package:docling",
            runtime_config_hash=content_hash({"offline_artifacts_path": str(self.artifacts_path.resolve())}),
        )

    def _convert_document(self, path: Path) -> object:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions(
            artifacts_path=self.artifacts_path,
            enable_remote_services=False,
            allow_external_plugins=False,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        converted = converter.convert(str(path))
        return getattr(converted, "document", converted)

    def parse(self, request: ExternalDocumentParseInput) -> ExternalDocumentParseProposal:
        if request.parsing_config:
            raise ValueError("non-empty Docling parsing_config is not supported")
        runtime = self.resolve_runtime_identity()
        path = Path(request.artifact_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError("Docling input must be a regular local artifact file")
        if file_sha256(path) != request.artifact_sha256:
            raise ValueError("Docling input artifact hash mismatch")
        document = self._convert_document(path)
        return normalize_docling_document(
            document,
            descriptor=self.descriptor,
            runtime_identity=runtime,
            request=request,
        )
