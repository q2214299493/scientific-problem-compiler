from __future__ import annotations

from ..models import DocumentStructureArtifact, DocumentStructureSelection
from ..repositories import EvidenceStore, KnowledgeRepositories
from ..serialization import content_hash
from .acquisition import validate_literature_representation


class DocumentStructureSelector:
    policy_id = "deterministic-structure-selection"
    selector_version = "1.0.0"

    def select(
        self,
        literature_id: str,
        representation_id: str,
        structure_id: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
        *,
        rationale: str = "Select the successful deterministic structure extraction.",
    ) -> DocumentStructureSelection:
        store = evidence_store or repositories.evidence_store
        from .structure import validate_document_structure

        structure = validate_document_structure(structure_id, repositories, store)
        representation = repositories.literature_representation_refs.get(representation_id)
        validate_literature_representation(representation, repositories, store)
        if (
            structure.literature_id != literature_id
            or structure.representation_id != representation.representation_id
            or structure.representation_hash != representation.content_hash
        ):
            raise ValueError("structure selection binding is invalid")
        repository = repositories.document_structure_selections
        try:
            current = repository.resolve_current(representation_id)
        except FileNotFoundError:
            current = None
        if (
            current is not None
            and current.structure_id == structure.structure_id
            and current.structure_hash == structure.content_hash
        ):
            return validate_structure_selection(current, repositories, store)
        identity = {
            "literature_id": literature_id,
            "representation_id": representation.representation_id,
            "representation_hash": representation.content_hash,
            "structure_id": structure.structure_id,
            "structure_hash": structure.content_hash,
            "selected_by_policy": self.policy_id,
            "selector_version": self.selector_version,
            "rationale": rationale,
            "supersedes_selection_id": (current.selection_id if current is not None else None),
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        selection_id = f"document-structure-selection-{content_hash(identity)[:24]}"
        payload = {"selection_id": selection_id, **identity}
        selection = DocumentStructureSelection(
            **payload,
            content_hash=content_hash(payload),
        )
        repository.put(selection.selection_id, selection)
        return validate_structure_selection(selection, repositories, store)


def validate_structure_selection(
    selection: str | DocumentStructureSelection,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
) -> DocumentStructureSelection:
    store = evidence_store or repositories.evidence_store
    record = repositories.document_structure_selections.get(selection) if isinstance(selection, str) else selection
    stored = repositories.document_structure_selections.get(record.selection_id)
    if stored != record:
        raise ValueError("document structure selection differs from repository record")
    representation = repositories.literature_representation_refs.get(record.representation_id)
    validate_literature_representation(representation, repositories, store)
    from .structure import validate_document_structure

    structure = validate_document_structure(record.structure_id, repositories, store)
    if (
        record.literature_id != representation.literature_id
        or record.representation_hash != representation.content_hash
        or record.structure_hash != structure.content_hash
        or structure.literature_id != record.literature_id
        or structure.representation_id != record.representation_id
        or structure.representation_hash != record.representation_hash
    ):
        raise ValueError("document structure selection authority binding is invalid")
    return record


def resolve_current_structure_selection(
    representation_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
) -> DocumentStructureSelection:
    selection = repositories.document_structure_selections.resolve_current(representation_id)
    return validate_structure_selection(selection, repositories, evidence_store)


def require_current_structure(
    structure: DocumentStructureArtifact,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None = None,
) -> DocumentStructureSelection:
    selection = resolve_current_structure_selection(structure.representation_id, repositories, evidence_store)
    if selection.structure_id != structure.structure_id or selection.structure_hash != structure.content_hash:
        raise ValueError("document structure is not the current selected structure")
    return selection
