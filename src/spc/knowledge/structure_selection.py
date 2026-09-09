from __future__ import annotations

from ..models import (
    DocumentStructureArtifact,
    DocumentStructureSelection,
    LiteratureRepresentationReference,
    StructureExtractionStatus,
)
from ..repositories import EvidenceStore, KnowledgeRepositories
from ..serialization import content_hash
from .acquisition import validate_literature_representation


class DocumentStructureSelector:
    policy_id = "deterministic-structure-selection"
    selector_version = "2.0.0"
    known_policy_versions = frozenset({"1.0.0", selector_version})

    _admissible_auto_successors = {
        StructureExtractionStatus.UNSUPPORTED: frozenset(
            {
                StructureExtractionStatus.PARTIAL,
                StructureExtractionStatus.COMPLETE,
            }
        ),
        StructureExtractionStatus.UNRESOLVED: frozenset(
            {
                StructureExtractionStatus.PARTIAL,
                StructureExtractionStatus.COMPLETE,
            }
        ),
        StructureExtractionStatus.PARTIAL: frozenset(
            {
                StructureExtractionStatus.PARTIAL,
                StructureExtractionStatus.COMPLETE,
            }
        ),
        StructureExtractionStatus.COMPLETE: frozenset({StructureExtractionStatus.COMPLETE}),
    }

    def consider_auto_promotion(
        self,
        literature_id: str,
        representation_id: str,
        structure_id: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
    ) -> DocumentStructureSelection | None:
        """Promote an official extraction only when authority cannot regress."""
        store = evidence_store or repositories.evidence_store
        structure, representation = self._validate_target(
            literature_id,
            representation_id,
            structure_id,
            repositories,
            store,
        )
        current = self._current_or_none(representation_id, repositories, store)
        if current is not None and self._same_structure(current, structure):
            return current
        if structure.extraction_status in {
            StructureExtractionStatus.UNSUPPORTED,
            StructureExtractionStatus.UNRESOLVED,
        }:
            return current
        if current is not None:
            current_structure = repositories.document_structure_artifacts.get(current.structure_id)
            if structure.extraction_status not in self._admissible_auto_successors[current_structure.extraction_status]:
                return current
            if structure.structure_id in self._history_structure_ids(current, repositories):
                return current
        return self._create_selection(
            literature_id,
            representation.representation_id,
            representation.content_hash,
            structure,
            current,
            repositories,
            store,
            rationale="Automatically promote a non-downgrading official structure extraction.",
        )

    def select(
        self,
        literature_id: str,
        representation_id: str,
        structure_id: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
        *,
        rationale: str = "Explicitly select the reviewed document structure.",
        allow_rollback: bool = False,
    ) -> DocumentStructureSelection:
        """Explicitly promote one stored artifact, optionally allowing rollback."""
        store = evidence_store or repositories.evidence_store
        structure, representation = self._validate_target(
            literature_id,
            representation_id,
            structure_id,
            repositories,
            store,
        )
        current = self._current_or_none(representation_id, repositories, store)
        if current is not None and self._same_structure(current, structure):
            return current
        if (
            current is not None
            and structure.structure_id in self._history_structure_ids(current, repositories)
            and not allow_rollback
        ):
            raise ValueError("document structure rollback requires explicit allow_rollback")
        return self._create_selection(
            literature_id,
            representation.representation_id,
            representation.content_hash,
            structure,
            current,
            repositories,
            store,
            rationale=rationale,
        )

    @staticmethod
    def _same_structure(
        selection: DocumentStructureSelection,
        structure: DocumentStructureArtifact,
    ) -> bool:
        return selection.structure_id == structure.structure_id and selection.structure_hash == structure.content_hash

    @staticmethod
    def _history_structure_ids(
        current: DocumentStructureSelection,
        repositories: KnowledgeRepositories,
    ) -> frozenset[str]:
        records = {
            item.selection_id: item
            for item in repositories.document_structure_selections.list()
            if item.representation_id == current.representation_id
        }
        structure_ids: set[str] = set()
        seen: set[str] = set()
        cursor: DocumentStructureSelection | None = current
        while cursor is not None:
            if cursor.selection_id in seen:
                raise ValueError("cyclic document structure selection history")
            seen.add(cursor.selection_id)
            structure_ids.add(cursor.structure_id)
            predecessor_id = cursor.supersedes_selection_id
            cursor = records.get(predecessor_id) if predecessor_id else None
            if predecessor_id and cursor is None:
                raise ValueError(f"missing superseded structure selection: {predecessor_id}")
        return frozenset(structure_ids)

    @staticmethod
    def _current_or_none(
        representation_id: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> DocumentStructureSelection | None:
        try:
            current = repositories.document_structure_selections.resolve_current(representation_id)
        except FileNotFoundError:
            return None
        return validate_structure_selection(current, repositories, evidence_store)

    @staticmethod
    def _validate_target(
        literature_id: str,
        representation_id: str,
        structure_id: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> tuple[DocumentStructureArtifact, LiteratureRepresentationReference]:
        from .structure import validate_document_structure

        structure = validate_document_structure(structure_id, repositories, evidence_store)
        representation = repositories.literature_representation_refs.get(representation_id)
        validate_literature_representation(representation, repositories, evidence_store)
        if (
            structure.literature_id != literature_id
            or structure.representation_id != representation.representation_id
            or structure.representation_hash != representation.content_hash
        ):
            raise ValueError("structure selection binding is invalid")
        return structure, representation

    def _create_selection(
        self,
        literature_id: str,
        representation_id: str,
        representation_hash: str,
        structure: DocumentStructureArtifact,
        current: DocumentStructureSelection | None,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        *,
        rationale: str,
    ) -> DocumentStructureSelection:
        identity = {
            "literature_id": literature_id,
            "representation_id": representation_id,
            "representation_hash": representation_hash,
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
        repositories.document_structure_selections.put(selection.selection_id, selection)
        return validate_structure_selection(selection, repositories, evidence_store)


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
    if (
        record.selected_by_policy != DocumentStructureSelector.policy_id
        or record.selector_version not in DocumentStructureSelector.known_policy_versions
    ):
        raise ValueError("unknown document structure selection policy")
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
