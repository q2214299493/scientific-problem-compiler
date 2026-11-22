from __future__ import annotations

import hashlib
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import BaseModel

from ..interpretation.validators import (
    _source_quote_binding_issues as _source_quote_binding_issues,
    result_context_record_issues as result_context_record_issues,
    source_claim_binding_issues as source_claim_binding_issues,
)
from ..models import (
    CanonicalHTMLTextArtifact,
    CanonicalTextArtifact,
    CurationStatus,
    DocumentStructureSelection,
    EvidenceSpan,
    ExpertAttributionRecord,
    ExpertCase,
    ExpertOpinion,
    ExpertProfile,
    HistoricalLiteratureEvidenceAuthorization,
    HTMLLiteratureIngestionRecord,
    KnowledgeCurationRecord,
    KnowledgeRelation,
    LiteratureDocument,
    LiteratureIngestionRecord,
    LiteratureIngestionStatus,
    LiteratureRepresentationKind as LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    LiteratureRepresentationSelection,
    LiteratureSupplementarySource,
    LiteratureWorkflowPattern,
    MethodFact,
    ModelFact,
    ReportedResult,
    RawHTMLLiteratureArtifact,
    RawLiteratureArtifact,
    ScientificCapability,
    SourceClaim,
    SourceDocument,
    SourceQuote,
    SourceRole as SourceRole,
    SourceType,
)
from ..serialization import content_hash
from .k1f_authority import (
    K1FCompilationAuthorityIndex,
    K1FScientificAuthorityStatus,
    classify_k1f_scientific_authority,
)

from .trust_rules import (
    CURATION_REQUIRED_TYPES, REPRESENTATION_COMPONENT_TYPES as REPRESENTATION_COMPONENT_TYPES,
    TrustedKnowledgeError, expert, interpretation, literature,
)

if TYPE_CHECKING:
    from ..repositories import EvidenceStore, KnowledgeRepositories


REPOSITORY_NODE_SPECS = (
    ("expert_source", "expert_sources", "expert_source_id"),
    ("literature_document", "literature_documents", "literature_id"),
    ("raw_literature_artifact", "raw_literature_artifacts", "artifact_id"),
    ("canonical_text_artifact", "canonical_text_artifacts", "canonical_text_id"),
    ("literature_ingestion", "literature_ingestions", "ingestion_id"),
    (
        "raw_html_literature_artifact",
        "raw_html_literature_artifacts",
        "artifact_id",
    ),
    (
        "canonical_html_text_artifact",
        "canonical_html_text_artifacts",
        "canonical_text_id",
    ),
    (
        "html_literature_ingestion",
        "html_literature_ingestions",
        "ingestion_id",
    ),
    (
        "literature_representation_ref",
        "literature_representation_refs",
        "representation_id",
    ),
    (
        "literature_representation_selection",
        "literature_representation_selections",
        "selection_id",
    ),
    (
        "literature_supplementary_source",
        "literature_supplementary_sources",
        "supplement_id",
    ),
    (
        "historical_evidence_authorization",
        "historical_evidence_authorizations",
        "authorization_id",
    ),
    ("expert_profile", "expert_profiles", "expert_id"),
    ("expert_attribution", "expert_attributions", "attribution_id"),
    ("expert_opinion", "expert_opinions", "opinion_id"),
    ("expert_case", "expert_cases", "case_id"),
    ("workflow_pattern", "workflow_patterns", "pattern_id"),
    ("scientific_capability", "capabilities", "capability_id"),
    ("source_quote", "source_quotes", "quote_id"),
    ("source_claim", "source_claims", "claim_id"),
    ("reported_result", "reported_results", "result_id"),
    ("method_fact", "method_facts", "fact_id"),
    ("model_fact", "model_facts", "fact_id"),
    ("knowledge_relation", "relations", "relation_id"),
)


@dataclass(frozen=True)
class TrustedKnowledgeSelection:
    literature_documents: tuple[LiteratureDocument, ...]
    expert_profiles: tuple[ExpertProfile, ...]
    expert_attributions: tuple[ExpertAttributionRecord, ...]
    expert_opinions: tuple[ExpertOpinion, ...]
    relations: tuple[KnowledgeRelation, ...]
    curations: tuple[KnowledgeCurationRecord, ...]
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord]
    trusted_records: Mapping[tuple[str, str], BaseModel]


@dataclass(frozen=True)
class _ValidatedTrustedRead:
    repositories: KnowledgeRepositories
    evidence_store: EvidenceStore
    records: Mapping[tuple[str, str], BaseModel]
    selection: TrustedKnowledgeSelection
    structure_selections: Mapping[str, DocumentStructureSelection]
    structure_file_hashes: Mapping[Path, str]

    @classmethod
    def prepare(
        cls, repositories: KnowledgeRepositories, evidence_store: EvidenceStore
    ) -> _ValidatedTrustedRead:
        validator = TrustedKnowledgeValidator(repositories, evidence_store)
        selection = validator.validate()
        structure_selections = (
            validator._k1f_index.validated_structure_selections()
            if validator._k1f_index is not None else {}
        )
        return cls(
            repositories,
            evidence_store,
            MappingProxyType(dict(validator.records)),
            selection,
            MappingProxyType(dict(structure_selections)),
            MappingProxyType(
                dict(validator._k1f_index.validated_structure_file_hashes())
                if validator._k1f_index is not None else {}
            ),
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        try:
            regular_file = stat.S_ISREG(path.lstat().st_mode)
        except OSError as error:
            raise TrustedKnowledgeError(
                "TRUSTED_READ_STRUCTURE_CHANGED", f"structure file is unavailable: {path}"
            ) from error
        if not regular_file:
            raise TrustedKnowledgeError(
                "TRUSTED_READ_STRUCTURE_CHANGED", f"structure file is unavailable: {path}"
            )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def _structure_file_hashes(
        cls,
        repositories: KnowledgeRepositories,
        selections: Mapping[str, DocumentStructureSelection],
    ) -> dict[Path, str]:
        paths: set[Path] = set()
        for selection in selections.values():
            structure = repositories.document_structure_artifacts.get(selection.structure_id)
            paths.add(repositories.document_structure_selections.root / f"{selection.selection_id}.json")
            paths.add(repositories.document_structure_artifacts.root / f"{structure.structure_id}.json")
            paths.update(
                repositories.document_structure_blocks.root / f"{block_id}.json"
                for block_id in structure.block_hashes
            )
            for table_id in structure.table_hashes:
                table = repositories.table_structures.get(table_id)
                paths.add(repositories.table_structures.root / f"{table_id}.json")
                paths.update(
                    repositories.table_cell_structures.root / f"{cell_id}.json"
                    for cell_id in table.cell_refs
                )
            paths.update(
                repositories.figure_structures.root / f"{figure_id}.json"
                for figure_id in structure.figure_hashes
            )
        ordered_paths = sorted(paths)
        with ThreadPoolExecutor(max_workers=8) as executor:
            digests = tuple(executor.map(cls._file_hash, ordered_paths))
        return dict(zip(ordered_paths, digests, strict=True))

    def require_unchanged_structures(self) -> None:
        paths = tuple(self.structure_file_hashes)
        with ThreadPoolExecutor(max_workers=8) as executor:
            digests = tuple(executor.map(self._file_hash, paths))
        for path, digest in zip(paths, digests, strict=True):
            if digest != self.structure_file_hashes[path]:
                raise TrustedKnowledgeError(
                    "TRUSTED_READ_STRUCTURE_CHANGED",
                    f"structure file changed after validation: {path}",
                )

    def require_same_sources(
        self, repositories: KnowledgeRepositories, evidence_store: EvidenceStore
    ) -> None:
        if self.repositories is not repositories or self.evidence_store is not evidence_store:
            raise TrustedKnowledgeError(
                "TRUSTED_READ_SOURCE_MISMATCH",
                "validated trusted read belongs to another repository or evidence store",
            )


class TrustedKnowledgeValidator:
    def __init__(
        self,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None,
    ) -> None:
        self.repositories = repositories
        self.evidence_store = evidence_store
        self.records = self.record_index(repositories)
        self._reachable: dict[tuple[str, str], BaseModel] = {}
        self._all_literature_ingestion_sources: set[tuple[str, str]] = set()
        self._trusted_current_literature_sources: set[tuple[str, str]] = set()
        self._current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord] = {}
        self._k1f_index: K1FCompilationAuthorityIndex | None = None
        self._verified_evidence: dict[str, tuple[EvidenceSpan, SourceDocument]] = {}

    @staticmethod
    def record_index(
        repositories: KnowledgeRepositories,
    ) -> dict[tuple[str, str], BaseModel]:
        records: dict[tuple[str, str], BaseModel] = {}
        for record_type, repository_name, identity_field in REPOSITORY_NODE_SPECS:
            repository: Any = getattr(repositories, repository_name)
            listing = (
                repository.list_metadata()
                if repository_name
                in {
                    "raw_literature_artifacts",
                    "canonical_text_artifacts",
                    "raw_html_literature_artifacts",
                    "canonical_html_text_artifacts",
                }
                else repository.list()
            )
            for record in listing:
                key = (record_type, getattr(record, identity_field))
                if key in records:
                    raise TrustedKnowledgeError(
                        "DUPLICATE_KNOWLEDGE_RECORD",
                        f"duplicate record {key[0]}:{key[1]}",
                    )
                records[key] = record
        return records

    def resolve_current_curations(
        self,
    ) -> Mapping[tuple[str, str], KnowledgeCurationRecord]:
        curations = self.repositories.curations.list()
        by_id = {item.curation_id: item for item in curations}
        groups: dict[tuple[str, str], list[KnowledgeCurationRecord]] = {}
        successors: dict[str, str] = {}
        for curation in curations:
            key = (curation.target_type, curation.target_id)
            target = self.records.get(key)
            if target is None:
                raise TrustedKnowledgeError(
                    "UNKNOWN_CURATION_TARGET",
                    f"curation target does not exist: {key[0]}:{key[1]}",
                )
            if curation.target_hash != self._record_hash(target):
                raise TrustedKnowledgeError(
                    "CURATION_TARGET_HASH_MISMATCH",
                    f"curation target hash is stale: {key[0]}:{key[1]}",
                )
            groups.setdefault(key, []).append(curation)
            predecessor_id = curation.supersedes_curation_id
            if predecessor_id is None:
                continue
            predecessor = by_id.get(predecessor_id)
            if predecessor is None:
                raise TrustedKnowledgeError(
                    "MISSING_CURATION_PREDECESSOR",
                    f"missing superseded curation {predecessor_id}",
                )
            if (predecessor.target_type, predecessor.target_id) != key:
                raise TrustedKnowledgeError(
                    "CURATION_TARGET_MISMATCH",
                    "a curation transition must retain the same target",
                )
            if predecessor_id in successors:
                raise TrustedKnowledgeError(
                    "AMBIGUOUS_CURATION_CHAIN",
                    f"curation {predecessor_id} has multiple successors",
                )
            successors[predecessor_id] = curation.curation_id

        current: dict[tuple[str, str], KnowledgeCurationRecord] = {}
        for key, group in groups.items():
            heads = [item for item in group if item.curation_id not in successors]
            if len(heads) != 1:
                raise TrustedKnowledgeError(
                    "AMBIGUOUS_CURATION_CHAIN",
                    f"expected one current curation for {key[0]}:{key[1]}",
                )
            self._assert_acyclic(heads[0], by_id)
            current[key] = heads[0]
        return MappingProxyType(current)

    def validate(self) -> TrustedKnowledgeSelection:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires an EvidenceStore",
            )
        self._reachable = {}
        self._k1f_index = None
        self._verified_evidence = {}
        current = self.resolve_current_curations()
        self._current_curations = current
        self._k1f_index = K1FCompilationAuthorityIndex(
            self.repositories, self.evidence_store, current,
            evidence_validator=self._verify_evidence,
            structure_fingerprinter=lambda selections: _ValidatedTrustedRead._structure_file_hashes(
                self.repositories, selections
            ),
        )
        accepted = {
            key: curation
            for key, curation in current.items()
            if curation.status == CurationStatus.ACCEPTED
        }
        accepted_ingestions = tuple(
            item
            for item in self.repositories.literature_ingestions.list()
            if item.ingestion_status == LiteratureIngestionStatus.ACCEPTED
        )
        self._all_literature_ingestion_sources = {
            (item.source_id or "", item.source_version or "")
            for item in accepted_ingestions
        }
        self._all_literature_ingestion_sources.update(
            (item.source_id, item.source_version)
            for item in self.repositories.html_literature_ingestions.list()
            if item.ingestion_status == LiteratureIngestionStatus.ACCEPTED
        )
        selection_literature_ids = {
            item.literature_id
            for item in self.repositories.literature_representation_selections.list()
        }
        current_selection_ids: set[str] = set()
        self._trusted_current_literature_sources = set()
        for literature_id in sorted(selection_literature_ids):
            try:
                selection = (
                    self.repositories.literature_representation_selections.resolve_current(
                        literature_id
                    )
                )
            except (FileNotFoundError, ValueError) as error:
                raise TrustedKnowledgeError(
                    "INVALID_LITERATURE_REPRESENTATION_SELECTION",
                    f"cannot resolve current representation for {literature_id}",
                ) from error
            current_selection_ids.add(selection.selection_id)
            curation = current.get(
                ("literature_representation_selection", selection.selection_id)
            )
            if curation is not None and curation.status == CurationStatus.ACCEPTED:
                from .ingestion import LiteratureRepresentationSelector

                try:
                    representation = LiteratureRepresentationSelector.resolve_selection(
                        selection, self.repositories, self.evidence_store
                    )
                except (FileNotFoundError, ValueError) as error:
                    raise TrustedKnowledgeError(
                        "INVALID_LITERATURE_REPRESENTATION_SELECTION",
                        str(error),
                    ) from error
                self._trusted_current_literature_sources.add(
                    (
                        representation.source_id or "",
                        representation.source_version or "",
                    )
                )
        active_accepted: dict[tuple[str, str], KnowledgeCurationRecord] = {}
        for key, curation in accepted.items():
            if (
                key[0] == "literature_representation_selection"
                and key[1] not in current_selection_ids
            ):
                continue
            authority = self._k1f_authority(key, current)
            if authority == K1FScientificAuthorityStatus.STALE:
                continue
            active_accepted[key] = curation
        for curation in active_accepted.values():
            self._verify_evidence_refs(curation.evidence_refs)
            self._reachable[("knowledge_curation", curation.curation_id)] = curation

        roots = set(active_accepted)
        roots.update(
            ("expert_case", item.case_id)
            for item in self.repositories.expert_cases.list()
            if not item.opinion_refs
        )
        roots.update(
            ("workflow_pattern", item.pattern_id)
            for item in self.repositories.workflow_patterns.list()
        )
        roots.update(
            ("scientific_capability", item.capability_id)
            for item in self.repositories.capabilities.list()
        )
        for key in sorted(roots):
            self._validate_record(key, current, set())

        # The K1F index shares fully validated structures only inside this read.
        # Reopen each one before returning trusted records so a changed block
        # cannot be hidden by the temporary cache.
        self._k1f_index.revalidate_structures()
        self._revalidate_cached_evidence()

        documents = self._records_of_type("literature_document", LiteratureDocument)
        profiles = self._records_of_type("expert_profile", ExpertProfile)
        attributions = self._records_of_type(
            "expert_attribution", ExpertAttributionRecord
        )
        opinions = self._records_of_type("expert_opinion", ExpertOpinion)
        relations = self._records_of_type("knowledge_relation", KnowledgeRelation)
        accepted_curations = tuple(
            sorted(active_accepted.values(), key=lambda item: item.curation_id)
        )
        return TrustedKnowledgeSelection(
            literature_documents=documents,
            expert_profiles=profiles,
            expert_attributions=attributions,
            expert_opinions=opinions,
            relations=relations,
            curations=accepted_curations,
            current_curations=current,
            trusted_records=MappingProxyType(dict(sorted(self._reachable.items()))),
        )

    def _revalidate_cached_evidence(self) -> None:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires an EvidenceStore",
            )
        for evidence_id, (evidence, source) in sorted(self._verified_evidence.items()):
            current_evidence = self.evidence_store.get_evidence(evidence_id)
            if current_evidence != evidence:
                raise TrustedKnowledgeError(
                    "INVALID_KNOWLEDGE_EVIDENCE",
                    f"evidence changed during trusted read: {evidence_id}",
                )
            current_source = self.evidence_store.verify_evidence_integrity(current_evidence)
            if current_source != source:
                raise TrustedKnowledgeError(
                    "INVALID_KNOWLEDGE_EVIDENCE",
                    f"source changed during trusted read: {evidence_id}",
                )

    def _validate_record(
        self,
        key: tuple[str, str],
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> BaseModel:
        if key in self._reachable:
            return self._reachable[key]
        if key in visiting:
            return self._require_record(*key)
        record = self._require_record(*key)
        if self._requires_curation(key, record):
            curation = current.get(key)
            if curation is None or curation.status != CurationStatus.ACCEPTED:
                raise TrustedKnowledgeError(
                    "UNTRUSTED_SCIENTIFIC_DEPENDENCY",
                    f"trusted record depends on unaccepted {key[0]}:{key[1]}",
                )
            if self._k1f_authority(key, current) == K1FScientificAuthorityStatus.STALE:
                raise TrustedKnowledgeError(
                    "STALE_K1F_SCIENTIFIC_AUTHORITY",
                    f"trusted record depends on stale K1F authority {key[0]}:{key[1]}",
                )
        def visit(dependency: tuple[str, str]) -> BaseModel:
            return self._validate_record(dependency, current, visiting)

        visiting.add(key)
        try:
            if key[0] == "expert_source":
                expert.validate_expert_source(
                    record,
                    repositories=self.repositories,
                    visit=visit,
                )
            elif isinstance(record, LiteratureDocument):
                literature.validate_literature(
                    record,
                    repositories=self.repositories,
                    reachable=self._reachable,
                    visit=visit,
                    verify_source=self._verify_source,
                    current=current,
                )
            elif isinstance(record, RawLiteratureArtifact):
                self.repositories.raw_literature_artifacts.get(record.artifact_id)
            elif isinstance(record, CanonicalTextArtifact):
                literature.validate_canonical_text(
                    record,
                    repositories=self.repositories,
                    visit=visit,
                )
            elif isinstance(record, LiteratureIngestionRecord):
                literature.validate_ingestion(
                    record,
                    reachable=self._reachable,
                    visit=visit,
                    require_record=self._require_record,
                    verify_source=self._verify_source,
                )
            elif isinstance(record, RawHTMLLiteratureArtifact):
                self.repositories.raw_html_literature_artifacts.get(
                    record.artifact_id
                )
            elif isinstance(record, CanonicalHTMLTextArtifact):
                literature.validate_html_canonical(
                    record,
                    repositories=self.repositories,
                    visit=visit,
                )
            elif isinstance(record, HTMLLiteratureIngestionRecord):
                literature.validate_html_ingestion(
                    record,
                    reachable=self._reachable,
                    visit=visit,
                    require_record=self._require_record,
                    verify_source=self._verify_source,
                )
            elif isinstance(record, LiteratureRepresentationReference):
                literature.validate_representation(
                    record,
                    repositories=self.repositories,
                    evidence_store=self.evidence_store,
                    visit=visit,
                    current=current,
                )
            elif isinstance(record, LiteratureRepresentationSelection):
                literature.validate_representation_selection(
                    record,
                    repositories=self.repositories,
                    evidence_store=self.evidence_store,
                    records=self.records,
                    visit=visit,
                    current=current,
                )
            elif isinstance(record, LiteratureSupplementarySource):
                literature.validate_supplementary_source(
                    record,
                    repositories=self.repositories,
                    evidence_store=self.evidence_store,
                    reachable=self._reachable,
                    visit=visit,
                    current=current,
                )
            elif isinstance(record, HistoricalLiteratureEvidenceAuthorization):
                self._validate_historical_authorization(record)
            elif isinstance(record, ExpertProfile):
                pass
            elif isinstance(record, ExpertAttributionRecord):
                expert.validate_attribution(
                    record,
                    reachable=self._reachable,
                    verify_evidence=self._verify_evidence,
                    verify_source=self._verify_source,
                    remember_evidence=self._remember_evidence,
                )
            elif isinstance(record, ExpertOpinion):
                expert.validate_opinion(
                    record,
                    visit=visit,
                    verify_evidence_refs=self._verify_evidence_refs,
                    verify_evidence=self._verify_evidence,
                )
            elif isinstance(record, KnowledgeRelation):
                interpretation.validate_relation(
                    record,
                    visit=visit,
                    require_record=self._require_record,
                    verify_evidence_refs=self._verify_evidence_refs,
                    current=current,
                )
            elif isinstance(record, SourceQuote):
                interpretation.validate_source_quote(
                    record,
                    evidence_store=self.evidence_store,
                    verify_evidence=self._verify_evidence,
                    remember_evidence=self._remember_evidence,
                )
            elif isinstance(record, SourceClaim):
                interpretation.validate_source_claim(
                    record,
                    records=self.records,
                    visit=visit,
                    verify_evidence_refs=self._verify_evidence_refs,
                )
            elif isinstance(record, MethodFact | ModelFact):
                self._verify_evidence_refs(record.evidence_refs)
            elif isinstance(record, ReportedResult):
                interpretation.validate_reported_result(
                    record,
                    records=self.records,
                    visit=visit,
                    verify_evidence_refs=self._verify_evidence_refs,
                )
            elif isinstance(record, ExpertCase):
                expert.validate_expert_case(
                    record,
                    visit=visit,
                    verify_evidence_refs=self._verify_evidence_refs,
                )
            elif isinstance(record, LiteratureWorkflowPattern):
                self._verify_evidence_refs(record.evidence_refs)
            elif isinstance(record, ScientificCapability):
                pass
            else:
                raise TrustedKnowledgeError(
                    "UNSUPPORTED_TRUSTED_RECORD",
                    f"no provenance validator for {key[0]}:{key[1]}",
                )
            self._reachable[key] = record
            return record
        finally:
            visiting.remove(key)

    def _requires_curation(
        self,
        key: tuple[str, str],
        record: BaseModel,
    ) -> bool:
        if key[0] in CURATION_REQUIRED_TYPES:
            return True
        if isinstance(record, ExpertCase):
            return bool(record.opinion_refs)
        if isinstance(record, ExpertAttributionRecord):
            return any(
                item.attribution_id == record.attribution_id
                for item in self.repositories.expert_sources.list()
            )
        return False

    def _k1f_authority(
        self,
        key: tuple[str, str],
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
    ) -> K1FScientificAuthorityStatus:
        if self.evidence_store is None:
            return K1FScientificAuthorityStatus.NOT_K1F
        try:
            if self._k1f_index is not None and current is self._current_curations:
                return self._k1f_index.classify(key[0], key[1])
            return classify_k1f_scientific_authority(
                key[0],
                key[1],
                self.repositories,
                self.evidence_store,
                current,
            )
        except (FileNotFoundError, ValueError) as error:
            if isinstance(error, TrustedKnowledgeError) and error.code != "INVALID_KNOWLEDGE_EVIDENCE":
                raise
            raise TrustedKnowledgeError(
                "INVALID_K1F_SCIENTIFIC_AUTHORITY",
                f"K1F authority is missing or corrupted for {key[0]}:{key[1]}",
            ) from error

    def _verify_evidence_refs(self, evidence_refs: tuple[str, ...]) -> None:
        for evidence_id in evidence_refs:
            evidence, source = self._verify_evidence(evidence_id)
            self._remember_evidence(evidence, source)

    def _verify_evidence(self, evidence_id: str) -> tuple[EvidenceSpan, SourceDocument]:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires an EvidenceStore",
            )
        cached = self._verified_evidence.get(evidence_id)
        if cached is not None:
            return cached
        try:
            evidence = self.evidence_store.get_evidence(evidence_id)
            source = self.evidence_store.verify_evidence_integrity(evidence)
        except Exception as error:
            raise TrustedKnowledgeError(
                "INVALID_KNOWLEDGE_EVIDENCE",
                f"evidence is unavailable or invalid: {evidence_id}",
            ) from error
        source_key = (source.source_id, source.version)
        shared_supplement = False
        if source.source_type == SourceType.SUPPORTING_INFORMATION:
            try:
                self.repositories.evidence_store.get_source(*source_key)
                shared_supplement = True
            except FileNotFoundError:
                pass
        if shared_supplement:
            matches = tuple(
                item
                for item in self.repositories.literature_supplementary_sources.list()
                if (item.source_id, item.source_version) == source_key
                and (
                    curation := self._current_curations.get(
                        ("literature_supplementary_source", item.supplement_id)
                    )
                ) is not None
                and curation.status == CurationStatus.ACCEPTED
            )
            if not matches:
                raise TrustedKnowledgeError(
                    "UNBOUND_SUPPLEMENTARY_EVIDENCE",
                    f"supporting-information evidence lacks an accepted source binding: {evidence_id}",
                )
            for supplement in matches:
                self._validate_record(
                    ("literature_supplementary_source", supplement.supplement_id),
                    self._current_curations,
                    set(),
                )
        if (
            source_key in self._all_literature_ingestion_sources
            and source_key not in self._trusted_current_literature_sources
        ):
            self._require_historical_authorization(evidence, source)
        self._verified_evidence[evidence_id] = (evidence, source)
        return evidence, source

    def _require_historical_authorization(
        self, evidence: EvidenceSpan, source: SourceDocument
    ) -> None:
        matches = tuple(
            authorization
            for authorization in self.repositories.historical_evidence_authorizations.list()
            if evidence.evidence_id in authorization.evidence_refs
            and (authorization.source_id, authorization.source_version)
            == (source.source_id, source.version)
        )
        if len(matches) != 1:
            raise TrustedKnowledgeError(
                "HISTORICAL_LITERATURE_EVIDENCE_NOT_AUTHORIZED",
                f"historical literature evidence is not explicitly authorized: {evidence.evidence_id}",
            )
        self._validate_historical_authorization(matches[0])

    def _validate_historical_authorization(
        self, authorization: HistoricalLiteratureEvidenceAuthorization
    ) -> None:
        literature.validate_historical_authorization(
            authorization,
            repositories=self.repositories,
            evidence_store=self.evidence_store,
            reachable=self._reachable,
            verify_source=self._verify_source,
            remember_evidence=self._remember_evidence,
        )

    def _verify_source(self, source_id: str, source_version: str) -> SourceDocument:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires an EvidenceStore",
            )
        try:
            source = self.evidence_store.get_source(source_id, source_version)
            if (source.source_id, source.version) != (source_id, source_version):
                raise ValueError("source identity mismatch")
            self.evidence_store.verify_source_integrity(source)
        except Exception as error:
            raise TrustedKnowledgeError(
                "INVALID_LITERATURE_SOURCE",
                f"source is unavailable or invalid: {source_id}@{source_version}",
            ) from error
        return source

    def _remember_evidence(
        self, evidence: EvidenceSpan, source: SourceDocument
    ) -> None:
        self._reachable[("evidence_span", evidence.evidence_id)] = evidence
        self._reachable[("source_document", f"{source.source_id}@{source.version}")] = source

    def _require_record(self, record_type: str, record_id: str) -> BaseModel:
        record = self.records.get((record_type, record_id))
        if record is None:
            raise TrustedKnowledgeError(
                "UNKNOWN_KNOWLEDGE_ENDPOINT",
                f"record does not exist: {record_type}:{record_id}",
            )
        return record

    def _records_of_type(self, record_type: str, model_type: type[Any]) -> tuple[Any, ...]:
        return tuple(
            record
            for (kind, _), record in sorted(self._reachable.items())
            if kind == record_type and isinstance(record, model_type)
        )

    @staticmethod
    def _record_hash(record: BaseModel) -> str:
        return getattr(record, "content_hash", None) or content_hash(record)

    @staticmethod
    def _assert_acyclic(
        head: KnowledgeCurationRecord,
        by_id: Mapping[str, KnowledgeCurationRecord],
    ) -> None:
        seen: set[str] = set()
        current: KnowledgeCurationRecord | None = head
        while current is not None:
            if current.curation_id in seen:
                raise TrustedKnowledgeError(
                    "CURATION_CHAIN_CYCLE",
                    f"curation chain contains a cycle at {current.curation_id}",
                )
            seen.add(current.curation_id)
            predecessor_id = current.supersedes_curation_id
            current = by_id.get(predecessor_id) if predecessor_id is not None else None
