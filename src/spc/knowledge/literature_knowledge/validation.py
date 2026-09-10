from __future__ import annotations

from ...interpretation.validators import source_quote_record_issues
from ...models import CurationStatus, KnowledgeCurationRecord
from ...repositories import EvidenceStore, KnowledgeRepositories
from ...serialization import content_hash
from ..ingestion import LiteratureRepresentationSelector
from ..structure import validate_document_structure, verify_structured_evidence_locator
from ..structure_selection import validate_structure_selection
from .contracts import LiteratureKnowledgeGroundingRecord
from .repositories import LiteratureKnowledgeGroundingRepository


CURATION_TRANSITIONS = {
    CurationStatus.MACHINE_EXTRACTED: frozenset(
        {
            CurationStatus.HUMAN_REVIEWED,
            CurationStatus.ACCEPTED,
            CurationStatus.REJECTED,
        }
    ),
    CurationStatus.HUMAN_REVIEWED: frozenset(
        {CurationStatus.ACCEPTED, CurationStatus.REJECTED}
    ),
    CurationStatus.ACCEPTED: frozenset(),
    CurationStatus.REJECTED: frozenset(),
}

def curate_knowledge_record(
    repositories: KnowledgeRepositories,
    *,
    target_type: str,
    target_id: str,
    status: CurationStatus,
    curator_id: str,
    rationale: str,
) -> KnowledgeCurationRecord:
    from ..trust import TrustedKnowledgeValidator

    records = TrustedKnowledgeValidator.record_index(repositories)
    target = records.get((target_type, target_id))
    if target is None:
        raise ValueError(f"unknown knowledge curation target: {target_type}:{target_id}")
    target_hash = getattr(target, "content_hash", None) or content_hash(target)
    current = TrustedKnowledgeValidator(repositories, repositories.evidence_store).resolve_current_curations().get(
        (target_type, target_id)
    )
    if current is not None:
        if current.status == status:
            return current
        if status not in CURATION_TRANSITIONS[current.status]:
            raise ValueError(f"invalid curation transition: {current.status.value} -> {status.value}")
    identity = {
        "target_type": target_type,
        "target_id": target_id,
        "target_hash": target_hash,
        "status": status,
        "curator_id": curator_id,
        "rationale": rationale,
        "evidence_refs": (),
        "supersedes_curation_id": current.curation_id if current is not None else None,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    record = KnowledgeCurationRecord(**payload, content_hash=content_hash(payload))
    repositories.curations.put(record.curation_id, record)
    return record


def ensure_machine_curation(
    repositories: KnowledgeRepositories,
    target_type: str,
    target_id: str,
) -> KnowledgeCurationRecord:
    from ..trust import TrustedKnowledgeValidator

    current = TrustedKnowledgeValidator(repositories, repositories.evidence_store).resolve_current_curations().get(
        (target_type, target_id)
    )
    if current is not None:
        return current
    return curate_knowledge_record(
        repositories,
        target_type=target_type,
        target_id=target_id,
        status=CurationStatus.MACHINE_EXTRACTED,
        curator_id="spc-literature-knowledge-compiler",
        rationale="Machine-extracted K1F proposal; requires explicit scientific review.",
    )


def validate_grounding_record(
    record: LiteratureKnowledgeGroundingRecord,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
    *,
    require_current: bool,
) -> LiteratureKnowledgeGroundingRecord:
    store = evidence_store or repositories.evidence_store
    stored = LiteratureKnowledgeGroundingRepository(repositories.root).get(record.grounding_id)
    if stored != record:
        raise ValueError("literature knowledge grounding differs from repository record")
    document = repositories.literature_documents.get(record.literature_id)
    representation_selection = repositories.literature_representation_selections.get(
        record.representation_selection_id
    )
    representation = LiteratureRepresentationSelector.resolve_selection(
        representation_selection,
        repositories,
        store,
    )
    structure_selection = validate_structure_selection(
        record.structure_selection_id,
        repositories,
        store,
    )
    structure = validate_document_structure(record.structure_id, repositories, store)
    block = repositories.document_structure_blocks.get(record.block_id)
    evidence = store.get_evidence(record.evidence_id)
    store.verify_evidence_integrity(evidence)
    locator = verify_structured_evidence_locator(record.locator_id, repositories, store)
    from .context import resolve_figure_ownership, resolve_table_ownership

    table_ownership = resolve_table_ownership(block, structure, repositories)
    table, cell = table_ownership if table_ownership is not None else (None, None)
    figure = resolve_figure_ownership(block, structure, repositories)
    quote = repositories.source_quotes.get(record.quote_id)
    issues, _source = source_quote_record_issues(
        quote,
        store,
        path=f"source_quote:{quote.quote_id}",
    )
    if issues:
        raise ValueError(f"invalid grounded SourceQuote: {issues[0].code}")
    actual = (
        document.content_hash,
        representation_selection.content_hash,
        representation.content_hash,
        structure_selection.content_hash,
        structure.content_hash,
        block.content_hash,
        content_hash(evidence),
        locator.content_hash,
    )
    expected = (
        record.literature_hash,
        record.representation_selection_hash,
        record.representation_hash,
        record.structure_selection_hash,
        record.structure_hash,
        record.block_hash,
        record.evidence_hash,
        record.locator_hash,
    )
    if actual != expected:
        raise ValueError("literature knowledge grounding hash binding is invalid")
    if (
        representation.literature_id != document.literature_id
        or structure_selection.structure_id != structure.structure_id
        or structure.representation_id != representation.representation_id
        or block.structure_id != structure.structure_id
        or locator.evidence_id != evidence.evidence_id
        or locator.block_id != block.block_id
        or quote.evidence_ref != evidence.evidence_id
        or record.content_region != block.content_region
        or locator.table_id != (table.table_id if table is not None else None)
        or locator.table_cell_id != (cell.cell_id if cell is not None else None)
        or locator.figure_id != (figure.figure_id if figure is not None else None)
    ):
        raise ValueError("literature knowledge grounding provenance is invalid")
    if require_current:
        current_representation = repositories.literature_representation_selections.resolve_current(
            document.literature_id
        )
        current_structure = repositories.document_structure_selections.resolve_current(
            representation.representation_id
        )
        if (
            current_representation.selection_id != representation_selection.selection_id
            or current_structure.selection_id != structure_selection.selection_id
        ):
            raise ValueError("literature knowledge grounding is not current")
    return record
