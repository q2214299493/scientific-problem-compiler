from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

from pydantic import BaseModel

from ..models import (
    CurationStatus,
    DocumentStructureSelection,
    EvidenceSpan,
    KnowledgeCurationRecord,
    SourceDocument,
    TableCellStructure,
    TableStructure,
)

if TYPE_CHECKING:
    from ..repositories import EvidenceStore, KnowledgeRepositories


K1F_RECORD_FIELDS = {
    "source_claim": ("source_claim_ids", "source_claims", "claim_id"),
    "method_fact": ("method_fact_ids", "method_facts", "fact_id"),
    "model_fact": ("model_fact_ids", "model_facts", "fact_id"),
    "reported_result": ("reported_result_ids", "reported_results", "result_id"),
    "knowledge_relation": (
        "knowledge_relation_ids",
        "relations",
        "relation_id",
    ),
}


class K1FScientificAuthorityStatus(StrEnum):
    NOT_K1F = "not_k1f"
    CURRENT = "current"
    STALE = "stale"


def _require_accepted_curation(
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord],
    record_type: str,
    record_id: str,
) -> None:
    curation = current_curations.get((record_type, record_id))
    if curation is None or curation.status != CurationStatus.ACCEPTED:
        raise ValueError(f"current K1F authority is not accepted: {record_type}:{record_id}")


def _validate_compilation_input(
    compilation,
    repositories: KnowledgeRepositories,
    store: EvidenceStore,
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord],
    structure_selection_validator: Callable[..., DocumentStructureSelection] | None = None,
):
    from .ingestion import LiteratureRepresentationSelector
    from .structure_selection import (
        validate_structure_selection,
    )

    record = repositories.literature_knowledge_inputs.get(
        compilation.compilation_input_id
    )
    if (
        record.content_hash != compilation.compilation_input_hash
        or record.representation_selection_id != compilation.representation_selection_id
        or record.representation_selection_hash
        != compilation.representation_selection_hash
        or record.structure_selection_id != compilation.structure_selection_id
        or record.structure_selection_hash != compilation.structure_selection_hash
    ):
        raise ValueError("K1F compilation input binding is invalid")
    proposal = repositories.literature_knowledge_proposals.get(
        compilation.proposal_set_id
    )
    if (
        proposal.content_hash != compilation.proposal_set_hash
        or proposal.compilation_input_id != record.compilation_input_id
        or proposal.compilation_input_hash != record.content_hash
        or proposal.provider_id != compilation.provider_id
        or proposal.provider_version != compilation.provider_version
        or proposal.provider_config_hash != compilation.provider_config_hash
    ):
        raise ValueError("K1F compilation proposal binding is invalid")
    document = repositories.literature_documents.get(record.literature_id)
    selection = repositories.literature_representation_selections.get(
        record.representation_selection_id
    )
    representation = LiteratureRepresentationSelector.resolve_selection(
        selection, repositories, store
    )
    structure_selection = (structure_selection_validator or validate_structure_selection)(
        record.structure_selection_id, repositories, store
    )
    # validate_structure_selection already performed the complete structure
    # integrity check; read the same bound artifact without repeating it.
    if structure_selection.structure_id != record.structure_id:
        raise ValueError("K1F compilation structure selection is inconsistent")
    structure = repositories.document_structure_artifacts.get(record.structure_id)
    actual = (
        document.content_hash,
        selection.content_hash,
        representation.content_hash,
        representation.representation_kind,
        representation.canonical_text_id,
        representation.canonical_text_hash,
        representation.source_id,
        representation.source_version,
        structure_selection.content_hash,
        structure.content_hash,
    )
    expected = (
        record.literature_hash,
        record.representation_selection_hash,
        record.representation_hash,
        record.representation_kind,
        record.canonical_text_id,
        record.canonical_text_hash,
        record.source_id,
        record.source_version,
        record.structure_selection_hash,
        record.structure_hash,
    )
    if actual != expected:
        raise ValueError("K1F compilation authority hash binding is invalid")
    current_representation = (
        repositories.literature_representation_selections.resolve_current(
            record.literature_id
        )
    )
    if current_representation.selection_id != selection.selection_id:
        return False, record
    current_structure = repositories.document_structure_selections.resolve_current(
        representation.representation_id
    )
    if current_structure.selection_id != structure_selection.selection_id:
        return False, record
    _require_accepted_curation(
        current_curations, "literature_document", document.literature_id
    )
    _require_accepted_curation(
        current_curations,
        "literature_representation_selection",
        selection.selection_id,
    )
    return True, record


def _record_grounding_requirements(
    record_type: str,
    record: BaseModel,
) -> tuple[set[str], set[str]]:
    if record_type == "source_claim":
        return set(record.source_quote_refs), set(record.evidence_refs)
    return set(), set(record.evidence_refs)


def validate_k1f_compilation_authority(
    compilation,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord],
    *,
    structure_selection_validator: Callable[..., DocumentStructureSelection] | None = None,
    evidence_validator: Callable[[str], tuple[EvidenceSpan, SourceDocument]] | None = None,
    _table_records: dict[str, tuple[TableStructure, tuple[TableCellStructure, ...]]] | None = None,
) -> K1FScientificAuthorityStatus:
    from .literature_knowledge.validation import validate_grounding_inventory

    is_current, compilation_input = _validate_compilation_input(
        compilation, repositories, evidence_store, current_curations,
        structure_selection_validator,
    )
    groundings = tuple(
        repositories.literature_knowledge_groundings.get(grounding_id)
        for grounding_id in compilation.grounding_hashes
    )
    if {item.grounding_id: item.content_hash for item in groundings} != dict(
        compilation.grounding_hashes
    ):
        raise ValueError("K1F compilation grounding inventory is invalid")
    validate_grounding_inventory(
        groundings, repositories, evidence_store, require_current=is_current,
        structure_selection_validator=structure_selection_validator,
        # Historical compilations still require integrity, but do not acquire
        # current scientific eligibility through the trusted record cache.
        evidence_validator=evidence_validator if is_current else None,
        _table_records=_table_records,
    )
    for grounding in groundings:
        if (
            grounding.literature_id != compilation_input.literature_id
            or grounding.representation_selection_id
            != compilation_input.representation_selection_id
            or grounding.structure_selection_id
            != compilation_input.structure_selection_id
        ):
            raise ValueError("K1F grounding belongs to another compilation authority")
    quote_ids = {item.quote_id for item in groundings}
    evidence_ids = {item.evidence_id for item in groundings}
    if quote_ids != set(compilation.source_quote_ids):
        raise ValueError("K1F compilation SourceQuote grounding coverage is incomplete")
    for record_type, (field_name, repository_name, identity_field) in K1F_RECORD_FIELDS.items():
        for record_id in getattr(compilation, field_name):
            record = getattr(repositories, repository_name).get(record_id)
            if getattr(record, identity_field) != record_id:
                raise ValueError("K1F scientific record identity binding is invalid")
            required_quotes, required_evidence = _record_grounding_requirements(
                record_type, record
            )
            if not required_quotes.issubset(quote_ids) or not required_evidence.issubset(
                evidence_ids
            ):
                raise ValueError(
                    f"K1F grounding coverage is incomplete for {record_type}:{record_id}"
                )
    return (
        K1FScientificAuthorityStatus.CURRENT
        if is_current
        else K1FScientificAuthorityStatus.STALE
    )


def classify_k1f_scientific_authority(
    record_type: str,
    record_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord],
) -> K1FScientificAuthorityStatus:
    specification = K1F_RECORD_FIELDS.get(record_type)
    if specification is None:
        return K1FScientificAuthorityStatus.NOT_K1F
    field_name = specification[0]
    compilations = tuple(
        compilation
        for compilation in repositories.literature_knowledge_compilations.list()
        if record_id in getattr(compilation, field_name)
    )
    if not compilations:
        return K1FScientificAuthorityStatus.NOT_K1F
    statuses = {
        validate_k1f_compilation_authority(
            compilation,
            repositories,
            evidence_store,
            current_curations,
        )
        for compilation in compilations
    }
    if K1FScientificAuthorityStatus.CURRENT in statuses:
        return K1FScientificAuthorityStatus.CURRENT
    return K1FScientificAuthorityStatus.STALE


class K1FCompilationAuthorityIndex:
    """Reuse complete compilation checks only within one trusted read."""

    def __init__(
        self,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord],
        *,
        evidence_validator: Callable[[str], tuple[EvidenceSpan, SourceDocument]] | None = None,
        structure_fingerprinter: Callable[
            [Mapping[str, DocumentStructureSelection]], Mapping[Path, str]
        ] | None = None,
    ) -> None:
        self.repositories = repositories
        self.evidence_store = evidence_store
        self.current_curations = current_curations
        self._evidence_validator = evidence_validator
        self._structure_fingerprinter = structure_fingerprinter
        self._structure_file_hashes: dict[Path, str] = {}
        self._compilations: dict[tuple[str, str], list] = {}
        self._validated: dict[str, K1FScientificAuthorityStatus] = {}
        self._validated_structures: dict[str, DocumentStructureSelection] = {}
        for compilation in repositories.literature_knowledge_compilations.list():
            for record_type, (field_name, _, _) in K1F_RECORD_FIELDS.items():
                for record_id in getattr(compilation, field_name):
                    self._compilations.setdefault((record_type, record_id), []).append(
                        compilation
                    )

    def classify(
        self, record_type: str, record_id: str
    ) -> K1FScientificAuthorityStatus:
        if record_type not in K1F_RECORD_FIELDS:
            return K1FScientificAuthorityStatus.NOT_K1F
        compilations = self._compilations.get((record_type, record_id), ())
        if not compilations:
            return K1FScientificAuthorityStatus.NOT_K1F
        statuses = set()
        for compilation in compilations:
            status = self._validated.get(compilation.compilation_id)
            if status is None:
                # Parsed table/cell records live only during this compilation
                # check. Reuse requires the trusted read's closing fingerprints.
                read_options = {"_table_records": {}} if self._structure_fingerprinter is not None else {}
                status = validate_k1f_compilation_authority(
                    compilation,
                    self.repositories,
                    self.evidence_store,
                    self.current_curations,
                    structure_selection_validator=self._validate_structure_once,
                    evidence_validator=self._evidence_validator,
                    **read_options,
                )
                self._validated[compilation.compilation_id] = status
            statuses.add(status)
        if K1FScientificAuthorityStatus.CURRENT in statuses:
            return K1FScientificAuthorityStatus.CURRENT
        return K1FScientificAuthorityStatus.STALE

    def _validate_structure_once(
        self,
        selection: str | DocumentStructureSelection,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> DocumentStructureSelection:
        from .structure_selection import validate_structure_selection

        selection_id = selection if isinstance(selection, str) else selection.selection_id
        cached = self._validated_structures.get(selection_id)
        if cached is None:
            fingerprints = {} if self._structure_fingerprinter is not None else None
            read_options = {"_file_hashes": fingerprints} if fingerprints is not None else {}
            cached = validate_structure_selection(selection, repositories, evidence_store, **read_options)
            if fingerprints is not None:
                for path, digest in fingerprints.items():
                    previous = self._structure_file_hashes.get(path)
                    if previous is not None and previous != digest:
                        raise ValueError("document structure changed during trusted read")
                self._structure_file_hashes.update(fingerprints)
            self._validated_structures[selection_id] = cached
            return cached
        if repositories.document_structure_selections.get(selection_id) != cached:
            raise ValueError("document structure selection changed during trusted read")
        structure = repositories.document_structure_artifacts.get(cached.structure_id)
        if structure.content_hash != cached.structure_hash:
            raise ValueError("document structure changed during trusted read")
        return cached

    def revalidate_structures(self) -> None:
        """Close the read by checking every bound structure file and source."""
        from .structure_selection import validate_structure_selection

        if self._structure_fingerprinter is not None:
            # Identical file bytes preserve the complete structure/selection
            # validation. Representation dependencies are outside that manifest,
            # so their raw/canonical/ingestion/source chains are checked afresh.
            from .representation import validate_literature_representation

            checked_representations: set[str] = set()
            for cached in self._validated_structures.values():
                representation = self.repositories.literature_representation_refs.get(
                    cached.representation_id
                )
                if representation.content_hash != cached.representation_hash:
                    raise ValueError("document structure representation changed during trusted read")
                if representation.representation_id not in checked_representations:
                    validate_literature_representation(
                        representation, self.repositories, self.evidence_store
                    )
                    checked_representations.add(representation.representation_id)
            # Close the structure binding last, including changes that occur
            # while the representation/source chains are being revalidated.
            current = self._structure_fingerprinter(self._validated_structures)
            if dict(current) != self._structure_file_hashes:
                raise ValueError("document structure changed during trusted read")
            return

        for selection_id, cached in sorted(self._validated_structures.items()):
            if validate_structure_selection(
                selection_id, self.repositories, self.evidence_store
            ) != cached:
                raise ValueError("document structure changed during trusted read")

    def validated_structure_selections(self) -> Mapping[str, DocumentStructureSelection]:
        """Return structures fully checked during this trusted read."""
        return dict(self._validated_structures)

    def validated_structure_file_hashes(self) -> Mapping[Path, str]:
        """Return hashes of the bytes actually parsed, never a recaptured baseline."""
        return dict(self._structure_file_hashes)
