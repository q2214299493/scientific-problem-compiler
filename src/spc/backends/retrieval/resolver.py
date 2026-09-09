from __future__ import annotations

from ..contracts import ExternalLiteratureRetrievalHit, ExternalRetrievalResolution
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
from ...repositories import EvidenceStore, KnowledgeRepositories


def _unresolved(hit: ExternalLiteratureRetrievalHit, reason: str) -> ExternalRetrievalResolution:
    return ExternalRetrievalResolution(
        hit_id=hit.hit_id,
        resolved=False,
        reason=reason,
    )


class ExternalRetrievalEvidenceResolver:
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
        return ExternalRetrievalResolution(
            hit_id=hit.hit_id,
            resolved=True,
            literature_id=document.literature_id,
            representation_id=representation.representation_id,
            structure_id=structure.structure_id,
            evidence_id=evidence.evidence_id,
            locator_id=locator.locator_id,
        )
