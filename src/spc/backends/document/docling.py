from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..builders import build_external_document_element
from ..contracts import (
    BackendCapability,
    BackendIntegrationMode,
    BackendLicenseStatus,
    BackendRuntimeAvailability,
    ExternalBackendDescriptor,
    ExternalDocumentElement,
    ExternalDocumentElementKind,
    ExternalDocumentParseInput,
    ExternalDocumentParseProposal,
    ExternalProposalStatus,
)
from ...serialization import content_hash, file_sha256


def _descriptor() -> ExternalBackendDescriptor:
    payload = {
        "backend_id": "docling",
        "backend_name": "Docling",
        "backend_version": "runtime-resolved",
        "adapter_version": "1.0.0",
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


def _element(
    *,
    kind: ExternalDocumentElementKind,
    text: str,
    reading_order: int,
    page_hint: int | None = None,
    heading_level: int | None = None,
    backend_native_ref: str | None = None,
    backend_metadata: Mapping[str, str | int | float | bool | None] | None = None,
    table_ref: str | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
) -> ExternalDocumentElement:
    return build_external_document_element(
        kind=kind,
        text=text,
        reading_order=reading_order,
        page_hint=page_hint,
        heading_level=heading_level,
        backend_native_ref=backend_native_ref,
        backend_metadata=dict(backend_metadata or {}),
        table_ref=table_ref,
        row_index=row_index,
        column_index=column_index,
    )


def _page_hint(node: Mapping[str, Any]) -> int | None:
    provenance = node.get("prov") or node.get("provenance")
    if isinstance(provenance, list) and provenance:
        first = provenance[0]
        if isinstance(first, Mapping):
            value = first.get("page_no", first.get("page"))
            if isinstance(value, int) and value >= 1:
                return value
    value = node.get("page_no", node.get("page"))
    return value if isinstance(value, int) and value >= 1 else None


def _kind(label: str) -> ExternalDocumentElementKind:
    normalized = label.casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"title", "section_header", "heading"}:
        return ExternalDocumentElementKind.HEADING
    if normalized in {"list_item", "listitem"}:
        return ExternalDocumentElementKind.LIST_ITEM
    if normalized in {"table_caption", "tablecaption"}:
        return ExternalDocumentElementKind.TABLE_CAPTION
    if normalized in {"table_cell", "tablecell"}:
        return ExternalDocumentElementKind.TABLE_CELL
    if normalized in {"figure_caption", "picture_caption", "caption"}:
        return ExternalDocumentElementKind.FIGURE_CAPTION
    if normalized in {"paragraph", "text"}:
        return ExternalDocumentElementKind.PARAGRAPH
    return ExternalDocumentElementKind.OTHER_TEXT


def _walk_nodes(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        text = value.get("text")
        label = value.get("label", value.get("type"))
        if isinstance(text, str) and text.strip() and isinstance(label, str):
            yield value
        for child in value.values():
            yield from _walk_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_nodes(child)


class DoclingDocumentParsingBackend:
    descriptor = _descriptor()

    def inspect_availability(self) -> BackendRuntimeAvailability:
        try:
            installed = util.find_spec("docling") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                reason="optional package 'docling' is not installed",
            )
        try:
            version = metadata.version("docling")
        except metadata.PackageNotFoundError:
            version = "installed-version-unresolved"
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=True,
            detected_version=version,
        )

    @staticmethod
    def _converter_type():
        from docling.document_converter import DocumentConverter

        return DocumentConverter

    def parse(self, request: ExternalDocumentParseInput) -> ExternalDocumentParseProposal:
        availability = self.inspect_availability()
        if not availability.available:
            raise RuntimeError(availability.reason)
        path = Path(request.artifact_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError("Docling input must be a regular local artifact file")
        if file_sha256(path) != request.artifact_sha256:
            raise ValueError("Docling input artifact hash mismatch")
        converter = self._converter_type()()
        converted = converter.convert(str(path))
        document = getattr(converted, "document", converted)
        export = getattr(document, "export_to_dict", None)
        if not callable(export):
            raise ValueError("Docling output does not expose export_to_dict()")
        native = export()
        if not isinstance(native, Mapping):
            raise ValueError("Docling output must normalize from a mapping")
        elements: list[ExternalDocumentElement] = []
        for order, node in enumerate(_walk_nodes(native)):
            label = str(node.get("label", node.get("type", "text")))
            kind = _kind(label)
            native_ref = str(node["self_ref"]) if isinstance(node.get("self_ref"), str) else None
            table_ref_value = node.get("table_ref", node.get("parent_ref"))
            table_ref = str(table_ref_value) if isinstance(table_ref_value, (str, int)) else None
            row_index = node.get("row_index")
            column_index = node.get("column_index")
            if kind == ExternalDocumentElementKind.TABLE_CAPTION and table_ref is None:
                table_ref = native_ref
            if kind == ExternalDocumentElementKind.TABLE_CELL and not (
                table_ref is not None and isinstance(row_index, int) and isinstance(column_index, int)
            ):
                kind = ExternalDocumentElementKind.OTHER_TEXT
                table_ref = None
                row_index = None
                column_index = None
            if kind == ExternalDocumentElementKind.TABLE_CAPTION and table_ref is None:
                kind = ExternalDocumentElementKind.OTHER_TEXT
            heading_level = None
            if kind == ExternalDocumentElementKind.HEADING:
                raw_level = node.get("level", 1)
                heading_level = raw_level if isinstance(raw_level, int) else 1
                heading_level = min(6, max(1, heading_level))
            elements.append(
                _element(
                    kind=kind,
                    text=str(node["text"]).strip(),
                    reading_order=order,
                    page_hint=_page_hint(node),
                    heading_level=heading_level,
                    backend_native_ref=native_ref,
                    backend_metadata={"native_label": label},
                    table_ref=table_ref,
                    row_index=(row_index if isinstance(row_index, int) else None),
                    column_index=(column_index if isinstance(column_index, int) else None),
                )
            )
        status = ExternalProposalStatus.COMPLETE if elements else ExternalProposalStatus.UNRESOLVED
        identity = {
            "backend_id": self.descriptor.backend_id,
            "backend_descriptor_hash": self.descriptor.content_hash,
            "artifact_id": request.artifact_id,
            "artifact_sha256": request.artifact_sha256,
            "media_type": request.media_type,
            "elements": tuple(elements),
            "status": status,
            "warnings": (() if elements else ("Docling returned no text elements",)),
            "trust_class": "external_proposal",
        }
        proposal_id = f"external-document-proposal-{content_hash(identity)[:24]}"
        payload = {"proposal_id": proposal_id, **identity}
        return ExternalDocumentParseProposal(
            **payload,
            content_hash=content_hash(payload),
        )
