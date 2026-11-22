"""Create source-bound evidence spans from canonical text."""

from __future__ import annotations

from ...models import EvidenceSpan, LiteratureRepresentationKind, SourceDocument
from ...repositories import EvidenceStore, KnowledgeEvidenceStore, KnowledgeRepositories
from ...serialization import content_hash
from ..representation import LiteratureRepresentationSelector, validate_literature_representation
from .supplementary import validate_literature_supplementary_source


def _store_canonical_span(
    source: SourceDocument, canonical_text_id: str, start_offset: int, end_offset: int,
    text: str, locator: str | None, evidence_store: EvidenceStore,
) -> EvidenceSpan:
    evidence_identity = {
        "canonical_text_id": canonical_text_id,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "text_hash": content_hash({"text": text}),
    }
    evidence = EvidenceSpan(
        evidence_id=f"evidence-{content_hash(evidence_identity)[:24]}",
        source_id=source.source_id,
        source_version=source.version,
        content_sha256=source.content_sha256,
        start_offset=start_offset,
        end_offset=end_offset,
        text=text,
        locator=locator,
    )
    evidence_store.add_evidence(evidence)
    return evidence


def create_evidence_span_from_canonical_text(
    canonical_text_id: str,
    start_offset: int,
    end_offset: int,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
    *,
    locator: str | None = None,
    historical_ingestion: bool = False,
) -> EvidenceSpan:
    pdf_canonical_path = repositories.canonical_text_artifacts.root / canonical_text_id
    if pdf_canonical_path.exists() or pdf_canonical_path.is_symlink():
        canonical = repositories.canonical_text_artifacts.get(canonical_text_id)
        text = repositories.canonical_text_artifacts.read_text(canonical_text_id)
        representation_kind = LiteratureRepresentationKind.PDF
    else:
        canonical = repositories.canonical_html_text_artifacts.get(
            canonical_text_id
        )
        text = repositories.canonical_html_text_artifacts.read_text(
            canonical_text_id
        )
        representation_kind = LiteratureRepresentationKind.HTML
    if start_offset < 0 or end_offset <= start_offset or end_offset > len(text):
        raise ValueError("EvidenceSpan offsets are outside canonical text")
    matching = tuple(
        representation
        for representation in repositories.literature_representation_refs.list()
        if representation.representation_kind == representation_kind
        and representation.canonical_text_id == canonical_text_id
    )
    if len(matching) != 1:
        raise ValueError("canonical text must have exactly one representation binding")
    representation = matching[0]
    if not historical_ingestion:
        selection = repositories.literature_representation_selections.resolve_current(
            canonical.literature_id
        )
        selected_representation = LiteratureRepresentationSelector.resolve_selection(
            selection, repositories, evidence_store
        )
        if selected_representation.representation_id != representation.representation_id:
            raise ValueError("canonical text is not the current selected representation")
    validate_literature_representation(
        representation, repositories, evidence_store
    )
    source = evidence_store.get_source(
        representation.source_id, representation.source_version
    )
    evidence_store.verify_source_integrity(source)
    recovered = text[start_offset:end_offset]
    return _store_canonical_span(source, canonical_text_id, start_offset, end_offset, recovered, locator, evidence_store)


def create_supplementary_evidence_span(
    supplement_id: str,
    page_number: int,
    exact_text: str,
    locator: str,
    repositories: KnowledgeRepositories,
    evidence_store: KnowledgeEvidenceStore,
    *,
    block_id: str | None = None,
) -> EvidenceSpan:
    """Bind a unique verbatim SI passage to a verified source segment."""
    if not exact_text or not locator.strip():
        raise ValueError("supplementary evidence requires text and a locator")
    supplement = repositories.literature_supplementary_sources.get(supplement_id)
    raw, canonical, source = validate_literature_supplementary_source(
        supplement, repositories, evidence_store
    )
    text = repositories.canonical_text_artifacts.read_text(canonical.canonical_text_id)
    if raw.media_type == "application/pdf":
        pages = tuple(
            block for block in canonical.blocks if block.page_number == page_number
        )
        if len(pages) != 1:
            raise ValueError("supplementary page has no unique canonical block")
        block = pages[0]
        if block_id is not None and block.block_id != block_id:
            raise ValueError("supplementary block ID does not match the stated page")
        source_locator = f"SI PDF page {page_number}"
    else:
        if page_number != 1:
            raise ValueError("DOCX has no physical page numbers; use logical segment 1")
        matches = tuple(
            block
            for block in canonical.blocks
            if block_id is None or block.block_id == block_id
            if exact_text in text[block.start_offset : block.end_offset]
        )
        if len(matches) != 1:
            raise ValueError("supplementary DOCX quote must identify one block")
        block = matches[0]
        source_locator = f"SI DOCX {'/'.join(block.section_path)}; no physical page"
    block_text = text[block.start_offset : block.end_offset]
    if block_text.count(exact_text) != 1:
        raise ValueError("supplementary quote must occur exactly once in its block")
    start_offset = block.start_offset + block_text.index(exact_text)
    end_offset = start_offset + len(exact_text)
    return _store_canonical_span(
        source, canonical.canonical_text_id, start_offset, end_offset,
        exact_text, f"{source_locator}; {locator}", evidence_store,
    )
