from __future__ import annotations

from .contracts import (
    ExternalBackendDescriptor,
    BackendRuntimeIdentity,
    ExternalDocumentElement,
    ExternalDocumentElementKind,
    ExternalDocumentParseInput,
    ExternalDocumentParseProposal,
    ExternalLiteratureRetrievalHit,
    ExternalProposalStatus,
)
from ..models import DocumentContentRegion
from ..serialization import content_hash


def build_external_document_element(
    *,
    kind: ExternalDocumentElementKind,
    text: str,
    reading_order: int,
    page_hint: int | None = None,
    claimed_start_offset: int | None = None,
    claimed_end_offset: int | None = None,
    heading_level: int | None = None,
    proposed_content_region: DocumentContentRegion = DocumentContentRegion.UNKNOWN,
    table_ref: str | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
    row_span: int = 1,
    column_span: int = 1,
    is_header: bool = False,
    backend_native_ref: str | None = None,
    confidence: float | None = None,
    backend_metadata: dict[str, str | int | float | bool | None] | None = None,
) -> ExternalDocumentElement:
    identity = {
        "kind": kind,
        "text": text,
        "reading_order": reading_order,
        "page_hint": page_hint,
        "claimed_start_offset": claimed_start_offset,
        "claimed_end_offset": claimed_end_offset,
        "heading_level": heading_level,
        "proposed_content_region": proposed_content_region,
        "table_ref": table_ref,
        "row_index": row_index,
        "column_index": column_index,
        "row_span": row_span,
        "column_span": column_span,
        "is_header": is_header,
        "backend_native_ref": backend_native_ref,
        "confidence": confidence,
        "backend_metadata": dict(backend_metadata or {}),
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    element_id = f"external-document-element-{content_hash(identity)[:24]}"
    payload = {"element_id": element_id, **identity}
    return ExternalDocumentElement(
        **payload,
        content_hash=content_hash(payload),
    )


def build_external_document_parse_proposal(
    *,
    descriptor: ExternalBackendDescriptor,
    runtime_identity: BackendRuntimeIdentity,
    request: ExternalDocumentParseInput,
    elements: tuple[ExternalDocumentElement, ...],
    status: ExternalProposalStatus = ExternalProposalStatus.COMPLETE,
    warnings: tuple[str, ...] = (),
) -> ExternalDocumentParseProposal:
    identity = {
        "backend_id": descriptor.backend_id,
        "backend_descriptor_hash": descriptor.content_hash,
        "runtime_identity_hash": runtime_identity.content_hash,
        "artifact_id": request.artifact_id,
        "artifact_sha256": request.artifact_sha256,
        "media_type": request.media_type,
        "elements": elements,
        "status": status,
        "warnings": warnings,
        "trust_class": "external_proposal",
    }
    proposal_id = f"external-document-proposal-{content_hash(identity)[:24]}"
    payload = {"proposal_id": proposal_id, **identity}
    return ExternalDocumentParseProposal(
        **payload,
        content_hash=content_hash(payload),
    )


def build_external_retrieval_hit(
    *,
    external_document_id: str | None = None,
    doi: str | None = None,
    title: str | None = None,
    source_url: str | None = None,
    page_hint: int | None = None,
    text_snippet: str | None = None,
    score: float | None = None,
    backend_metadata: dict[str, str | int | float | bool | None] | None = None,
) -> ExternalLiteratureRetrievalHit:
    identity = {
        "external_document_id": external_document_id,
        "doi": doi,
        "title": title,
        "source_url": source_url,
        "page_hint": page_hint,
        "text_snippet": text_snippet,
        "score": score,
        "backend_metadata": dict(backend_metadata or {}),
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    hit_id = f"external-retrieval-hit-{content_hash(identity)[:24]}"
    payload = {"hit_id": hit_id, **identity}
    return ExternalLiteratureRetrievalHit(
        **payload,
        content_hash=content_hash(payload),
    )
