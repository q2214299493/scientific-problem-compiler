from __future__ import annotations

from pathlib import Path

from ..contracts import (
    ExternalLiteratureRetrievalHit,
    ExternalLiteratureRetrievalResult,
    ExternalRetrievalAuthorityBinding,
    ExternalRetrievalResolution,
    ExternalRetrievalResolutionBatch,
)
from ..rebinding import exact_normalized_matches
from ...knowledge.acquisition import normalize_doi
from ...knowledge.ingestion import (
    LiteratureRepresentationSelector,
    create_evidence_span_from_canonical_text,
)
from ...knowledge.structure import _create_locator, validate_document_structure
from ...knowledge.structure_selection import resolve_current_structure_selection
from ...knowledge.trust import TrustedKnowledgeValidator
from ...models import CurationStatus, DocumentBlockType
from ...repositories import EvidenceStore, IdentityBoundRepository, KnowledgeRepositories
from ...serialization import content_hash


class ExternalRetrievalResolutionBatchRepository(IdentityBoundRepository[ExternalRetrievalResolutionBatch]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "backend_retrieval_resolution_batches",
            ExternalRetrievalResolutionBatch,
            "batch_id",
        )


def _resolution(
    hit: ExternalLiteratureRetrievalHit,
    *,
    resolved: bool,
    literature_id: str | None = None,
    representation_id: str | None = None,
    structure_id: str | None = None,
    evidence_id: str | None = None,
    locator_id: str | None = None,
    reason: str | None = None,
) -> ExternalRetrievalResolution:
    identity = {
        "hit_id": hit.hit_id,
        "resolved": resolved,
        "literature_id": literature_id,
        "representation_id": representation_id,
        "structure_id": structure_id,
        "evidence_id": evidence_id,
        "locator_id": locator_id,
        "reason": reason,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    resolution_id = f"external-retrieval-resolution-{content_hash(identity)[:24]}"
    payload = {"resolution_id": resolution_id, **identity}
    return ExternalRetrievalResolution(
        **payload,
        content_hash=content_hash(payload),
    )


def _unresolved(hit: ExternalLiteratureRetrievalHit, reason: str) -> ExternalRetrievalResolution:
    return _resolution(hit, resolved=False, reason=reason)


class ExternalRetrievalEvidenceResolver:
    resolver_id = "spc-external-retrieval-evidence-resolver"
    resolver_version = "1.1.0"

    def resolve(
        self,
        hit: ExternalLiteratureRetrievalHit,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
    ) -> ExternalRetrievalResolution:
        store = evidence_store or repositories.evidence_store
        documents = tuple(repositories.literature_documents.list())
        candidates = set(range(len(documents)))
        identity_used = False
        if hit.external_document_id is not None:
            identity_used = True
            candidates &= {
                index for index, document in enumerate(documents) if document.literature_id == hit.external_document_id
            }
        if hit.doi is not None:
            identity_used = True
            normalized = normalize_doi(hit.doi)
            candidates &= {
                index
                for index, document in enumerate(documents)
                if document.doi is not None and normalize_doi(document.doi) == normalized
            }
        if not identity_used and hit.title is not None:
            identity_used = True
            title = " ".join(hit.title.casefold().split())
            candidates &= {
                index
                for index, document in enumerate(documents)
                if " ".join(document.title.casefold().split()) == title
            }
        if not identity_used or len(candidates) != 1:
            return _unresolved(hit, "external document identity is missing or ambiguous")
        document = documents[next(iter(candidates))]
        try:
            representation_selection = repositories.literature_representation_selections.resolve_current(
                document.literature_id
            )
            current_curations = TrustedKnowledgeValidator(repositories, store).resolve_current_curations()
        except (FileNotFoundError, ValueError) as error:
            return _unresolved(hit, f"trusted representation is unavailable: {error}")
        document_curation = current_curations.get(("literature_document", document.literature_id))
        representation_curation = current_curations.get(
            (
                "literature_representation_selection",
                representation_selection.selection_id,
            )
        )
        if (
            document_curation is None
            or document_curation.status != CurationStatus.ACCEPTED
            or representation_curation is None
            or representation_curation.status != CurationStatus.ACCEPTED
        ):
            return _unresolved(hit, "literature representation is not accepted")
        try:
            representation = LiteratureRepresentationSelector.resolve_selection(
                representation_selection, repositories, store
            )
            structure_selection = resolve_current_structure_selection(
                representation.representation_id, repositories, store
            )
            structure = validate_document_structure(structure_selection.structure_id, repositories, store)
        except (FileNotFoundError, ValueError) as error:
            return _unresolved(hit, f"current structure is unavailable: {error}")
        if hit.text_snippet is None:
            return _unresolved(hit, "external page hint without exact text is insufficient")
        if representation.representation_kind.value == "pdf":
            canonical_text = repositories.canonical_text_artifacts.read_text(representation.canonical_text_id)
        else:
            canonical_text = repositories.canonical_html_text_artifacts.read_text(representation.canonical_text_id)
        matches = exact_normalized_matches(hit.text_snippet, canonical_text)
        if len(matches) != 1:
            return _unresolved(
                hit,
                "external snippet is absent or ambiguous in current canonical text",
            )
        start, end = matches[0]
        containing = tuple(
            block
            for block_id in structure.block_hashes
            if (
                (block := repositories.document_structure_blocks.get(block_id))
                and block.block_type != DocumentBlockType.PAGE
                and start >= block.start_offset
                and end <= block.end_offset
            )
        )
        if not containing:
            return _unresolved(hit, "exact snippet is outside current structure blocks")
        smallest_size = min(block.end_offset - block.start_offset for block in containing)
        smallest = tuple(block for block in containing if block.end_offset - block.start_offset == smallest_size)
        if len(smallest) != 1:
            return _unresolved(hit, "exact snippet has ambiguous structure ownership")
        block = smallest[0]
        try:
            evidence = create_evidence_span_from_canonical_text(
                representation.canonical_text_id,
                start,
                end,
                repositories,
                store,
                locator=f"external retrieval rebound via {block.block_id}",
            )
            evidence, locator = _create_locator(
                evidence=evidence,
                artifact=structure,
                block=block,
                repositories=repositories,
                evidence_store=store,
            )
        except (FileNotFoundError, ValueError) as error:
            return _unresolved(hit, f"SPC evidence binding failed: {error}")
        return _resolution(
            hit,
            resolved=True,
            literature_id=document.literature_id,
            representation_id=representation.representation_id,
            structure_id=structure.structure_id,
            evidence_id=evidence.evidence_id,
            locator_id=locator.locator_id,
        )

    def resolve_batch(
        self,
        result: ExternalLiteratureRetrievalResult,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
    ) -> ExternalRetrievalResolutionBatch:
        from .service import ExternalRetrievalResultRepository

        stored_result = ExternalRetrievalResultRepository(repositories.root).get(result.result_id)
        if stored_result != result:
            raise ValueError("external retrieval result differs from stored record")
        store = evidence_store or repositories.evidence_store
        resolutions = tuple(self.resolve(hit, repositories, store) for hit in result.hits)
        bindings: dict[str, ExternalRetrievalAuthorityBinding] = {}
        for resolution in resolutions:
            if not resolution.resolved or resolution.literature_id is None:
                continue
            document = repositories.literature_documents.get(resolution.literature_id)
            representation_selection = repositories.literature_representation_selections.resolve_current(
                document.literature_id
            )
            representation = repositories.literature_representation_refs.get(resolution.representation_id or "")
            structure_selection = resolve_current_structure_selection(
                representation.representation_id, repositories, store
            )
            structure = repositories.document_structure_artifacts.get(structure_selection.structure_id)
            bindings[document.literature_id] = ExternalRetrievalAuthorityBinding(
                literature_id=document.literature_id,
                literature_hash=document.content_hash,
                representation_selection_id=representation_selection.selection_id,
                representation_selection_hash=representation_selection.content_hash,
                representation_id=representation.representation_id,
                representation_hash=representation.content_hash,
                structure_selection_id=structure_selection.selection_id,
                structure_selection_hash=structure_selection.content_hash,
                structure_id=structure.structure_id,
                structure_hash=structure.content_hash,
            )
        resolved_count = sum(item.resolved for item in resolutions)
        identity = {
            "external_result_id": result.result_id,
            "external_result_hash": result.content_hash,
            "authority_bindings": [bindings[key].model_dump(mode="json") for key in sorted(bindings)],
            "resolutions": [resolution.model_dump(mode="json") for resolution in resolutions],
            "resolved_count": resolved_count,
            "unresolved_count": len(resolutions) - resolved_count,
            "resolver_id": self.resolver_id,
            "resolver_version": self.resolver_version,
        }
        batch_id = f"external-retrieval-resolution-batch-{content_hash(identity)[:24]}"
        payload = {"batch_id": batch_id, **identity}
        batch = ExternalRetrievalResolutionBatch(
            **payload,
            content_hash=content_hash(payload),
        )
        ExternalRetrievalResolutionBatchRepository(repositories.root).put(batch.batch_id, batch)
        return batch


def validate_resolution_batch_current(
    batch: ExternalRetrievalResolutionBatch,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
) -> ExternalRetrievalResolutionBatch:
    from .service import ExternalRetrievalResultRepository

    stored = ExternalRetrievalResolutionBatchRepository(repositories.root).get(batch.batch_id)
    if stored != batch:
        raise ValueError("retrieval resolution batch differs from stored record")
    result = ExternalRetrievalResultRepository(repositories.root).get(batch.external_result_id)
    if result.content_hash != batch.external_result_hash:
        raise ValueError("retrieval resolution batch result binding is stale")
    if tuple(item.hit_id for item in batch.resolutions) != tuple(item.hit_id for item in result.hits):
        raise ValueError("retrieval resolution batch does not match the external result hits")
    store = evidence_store or repositories.evidence_store
    current_curations = TrustedKnowledgeValidator(repositories, store).resolve_current_curations()
    authority_by_literature = {item.literature_id: item for item in batch.authority_bindings}
    resolved_literature_ids: set[str] = set()
    for resolution in batch.resolutions:
        if not resolution.resolved or resolution.literature_id is None:
            continue
        authority = authority_by_literature.get(resolution.literature_id)
        if (
            authority is None
            or resolution.representation_id != authority.representation_id
            or resolution.structure_id != authority.structure_id
        ):
            raise ValueError("retrieval resolution batch authority coverage is invalid")
        resolved_literature_ids.add(resolution.literature_id)
    if resolved_literature_ids != set(authority_by_literature):
        raise ValueError("retrieval resolution batch has unused authority bindings")
    for authority in batch.authority_bindings:
        document = repositories.literature_documents.get(authority.literature_id)
        representation_selection = repositories.literature_representation_selections.resolve_current(
            authority.literature_id
        )
        document_curation = current_curations.get(("literature_document", authority.literature_id))
        selection_curation = current_curations.get(
            (
                "literature_representation_selection",
                representation_selection.selection_id,
            )
        )
        if (
            document_curation is None
            or document_curation.status != CurationStatus.ACCEPTED
            or selection_curation is None
            or selection_curation.status != CurationStatus.ACCEPTED
        ):
            raise ValueError("retrieval resolution batch representation is no longer accepted")
        representation = repositories.literature_representation_refs.get(authority.representation_id)
        structure_selection = resolve_current_structure_selection(authority.representation_id, repositories, store)
        structure = repositories.document_structure_artifacts.get(structure_selection.structure_id)
        current = ExternalRetrievalAuthorityBinding(
            literature_id=document.literature_id,
            literature_hash=document.content_hash,
            representation_selection_id=representation_selection.selection_id,
            representation_selection_hash=representation_selection.content_hash,
            representation_id=representation.representation_id,
            representation_hash=representation.content_hash,
            structure_selection_id=structure_selection.selection_id,
            structure_selection_hash=structure_selection.content_hash,
            structure_id=structure.structure_id,
            structure_hash=structure.content_hash,
        )
        if current != authority:
            raise ValueError("retrieval resolution batch authority binding is stale")
    return batch
