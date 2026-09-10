from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import BaseModel

from ..interpretation.validators import (
    result_context_record_issues,
    source_claim_binding_issues,
    source_quote_record_issues,
)
from ..models import (
    CanonicalHTMLTextArtifact,
    CanonicalTextArtifact,
    CurationStatus,
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
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    LiteratureRepresentationSelection,
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
    SourceRole,
    SourceType,
)
from ..serialization import content_hash
from .k1f_authority import (
    K1FScientificAuthorityStatus,
    classify_k1f_scientific_authority,
)

if TYPE_CHECKING:
    from ..repositories import EvidenceStore, KnowledgeRepositories


REPOSITORY_NODE_SPECS = (
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

CURATION_REQUIRED_TYPES = frozenset(
    {
        "literature_document",
        "literature_representation_selection",
        "expert_opinion",
        "source_claim",
        "method_fact",
        "model_fact",
        "reported_result",
        "knowledge_relation",
    }
)

REPRESENTATION_COMPONENT_TYPES = frozenset(
    {
        "raw_literature_artifact",
        "canonical_text_artifact",
        "literature_ingestion",
        "raw_html_literature_artifact",
        "canonical_html_text_artifact",
        "html_literature_ingestion",
    }
)


class TrustedKnowledgeError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


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
        current = self.resolve_current_curations()
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
        if key[0] in CURATION_REQUIRED_TYPES:
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
        visiting.add(key)
        try:
            if isinstance(record, LiteratureDocument):
                self._validate_literature(record, current, visiting)
            elif isinstance(record, RawLiteratureArtifact):
                self.repositories.raw_literature_artifacts.get(record.artifact_id)
            elif isinstance(record, CanonicalTextArtifact):
                self._validate_canonical_text(record, current, visiting)
            elif isinstance(record, LiteratureIngestionRecord):
                self._validate_ingestion(record, current, visiting)
            elif isinstance(record, RawHTMLLiteratureArtifact):
                self.repositories.raw_html_literature_artifacts.get(
                    record.artifact_id
                )
            elif isinstance(record, CanonicalHTMLTextArtifact):
                self._validate_html_canonical(record, current, visiting)
            elif isinstance(record, HTMLLiteratureIngestionRecord):
                self._validate_html_ingestion(record, current, visiting)
            elif isinstance(record, LiteratureRepresentationReference):
                self._validate_representation(record, current, visiting)
            elif isinstance(record, LiteratureRepresentationSelection):
                self._validate_representation_selection(record, current, visiting)
            elif isinstance(record, HistoricalLiteratureEvidenceAuthorization):
                self._validate_historical_authorization(record)
            elif isinstance(record, ExpertProfile):
                pass
            elif isinstance(record, ExpertAttributionRecord):
                self._validate_attribution(record)
            elif isinstance(record, ExpertOpinion):
                self._validate_opinion(record, current, visiting)
            elif isinstance(record, KnowledgeRelation):
                self._validate_relation(record, current, visiting)
            elif isinstance(record, SourceQuote):
                self._validate_source_quote(record)
            elif isinstance(record, SourceClaim):
                self._validate_source_claim(record, current, visiting)
            elif isinstance(record, MethodFact | ModelFact):
                self._verify_evidence_refs(record.evidence_refs)
            elif isinstance(record, ReportedResult):
                self._validate_reported_result(record, current, visiting)
            elif isinstance(record, ExpertCase | LiteratureWorkflowPattern):
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

    def _k1f_authority(
        self,
        key: tuple[str, str],
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
    ) -> K1FScientificAuthorityStatus:
        if self.evidence_store is None:
            return K1FScientificAuthorityStatus.NOT_K1F
        try:
            return classify_k1f_scientific_authority(
                key[0],
                key[1],
                self.repositories,
                self.evidence_store,
                current,
            )
        except (FileNotFoundError, ValueError) as error:
            if isinstance(error, TrustedKnowledgeError):
                raise
            raise TrustedKnowledgeError(
                "INVALID_K1F_SCIENTIFIC_AUTHORITY",
                f"K1F authority is missing or corrupted for {key[0]}:{key[1]}",
            ) from error

    def _validate_literature(
        self,
        document: LiteratureDocument,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        ingestions = tuple(
            ingestion
            for ingestion in self.repositories.literature_ingestions.list()
            if ingestion.literature_id == document.literature_id
        )
        html_ingestions = tuple(
            ingestion
            for ingestion in self.repositories.html_literature_ingestions.list()
            if ingestion.literature_id == document.literature_id
        )
        if ingestions or html_ingestions:
            try:
                selection = (
                    self.repositories.literature_representation_selections.resolve_current(
                        document.literature_id
                    )
                )
            except (FileNotFoundError, ValueError) as error:
                raise TrustedKnowledgeError(
                    "INVALID_LITERATURE_REPRESENTATION_SELECTION",
                    str(error),
                ) from error
            selection_curation = current.get(
                ("literature_representation_selection", selection.selection_id)
            )
            if (
                selection_curation is None
                or selection_curation.status != CurationStatus.ACCEPTED
            ):
                raise TrustedKnowledgeError(
                    "UNTRUSTED_LITERATURE_REPRESENTATION_SELECTION",
                    f"current representation selection is not accepted: {selection.selection_id}",
                )
            self._validate_record(
                ("literature_representation_selection", selection.selection_id),
                current,
                visiting,
            )
            return
        source = self._verify_source(document.source_id, document.source_version)
        self._reachable[("source_document", f"{source.source_id}@{source.version}")] = source

    def _validate_representation_selection(
        self,
        selection: LiteratureRepresentationSelection,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        try:
            current_selection = (
                self.repositories.literature_representation_selections.resolve_current(
                    selection.literature_id
                )
            )
        except (FileNotFoundError, ValueError) as error:
            raise TrustedKnowledgeError(
                "INVALID_LITERATURE_REPRESENTATION_SELECTION", str(error)
            ) from error
        selection_curation = current.get(
            ("literature_representation_selection", selection.selection_id)
        )
        if current_selection.selection_id != selection.selection_id:
            raise TrustedKnowledgeError(
                "HISTORICAL_REPRESENTATION_NOT_TRUSTED",
                f"selection is not current: {selection.selection_id}",
            )
        if (
            selection_curation is None
            or selection_curation.status != CurationStatus.ACCEPTED
        ):
            raise TrustedKnowledgeError(
                "UNTRUSTED_LITERATURE_REPRESENTATION_SELECTION",
                f"selection is not explicitly accepted: {selection.selection_id}",
            )
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
        self.records[
            ("literature_representation_ref", representation.representation_id)
        ] = representation
        self._validate_record(
            ("literature_representation_ref", representation.representation_id),
            current,
            visiting,
        )

    def _validate_canonical_text(
        self,
        canonical: CanonicalTextArtifact,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        self.repositories.canonical_text_artifacts.get(canonical.canonical_text_id)
        raw = self._validate_record(
            ("raw_literature_artifact", canonical.raw_artifact_id), current, visiting
        )
        if (
            not isinstance(raw, RawLiteratureArtifact)
            or canonical.raw_artifact_hash != raw.content_hash
            or canonical.literature_id != raw.literature_id
        ):
            raise TrustedKnowledgeError(
                "CANONICAL_RAW_ARTIFACT_MISMATCH",
                f"canonical text does not bind its raw artifact: {canonical.canonical_text_id}",
            )

    def _validate_html_canonical(
        self,
        canonical: CanonicalHTMLTextArtifact,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        self.repositories.canonical_html_text_artifacts.get(
            canonical.canonical_text_id
        )
        raw = self._validate_record(
            ("raw_html_literature_artifact", canonical.raw_artifact_id),
            current,
            visiting,
        )
        if (
            not isinstance(raw, RawHTMLLiteratureArtifact)
            or canonical.raw_artifact_hash != raw.content_hash
            or canonical.literature_id != raw.literature_id
            or canonical.source_url != raw.source_url
        ):
            raise TrustedKnowledgeError(
                "HTML_CANONICAL_RAW_ARTIFACT_MISMATCH",
                f"HTML canonical text does not bind its raw artifact: {canonical.canonical_text_id}",
            )

    def _validate_representation(
        self,
        representation: LiteratureRepresentationReference,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        from .acquisition import validate_literature_representation
        from .ingestion import LiteratureRepresentationSelector

        try:
            selection = (
                self.repositories.literature_representation_selections.resolve_current(
                    representation.literature_id
                )
            )
            selection_curation = current.get(
                ("literature_representation_selection", selection.selection_id)
            )
            selected = LiteratureRepresentationSelector.resolve_selection(
                selection, self.repositories, self.evidence_store
            )
            if (
                selection_curation is None
                or selection_curation.status != CurationStatus.ACCEPTED
                or selected.representation_id != representation.representation_id
            ):
                raise ValueError("representation is not the accepted current selection")
            validate_literature_representation(
                representation, self.repositories, self.evidence_store
            )
        except (FileNotFoundError, ValueError) as error:
            raise TrustedKnowledgeError(
                "INVALID_LITERATURE_REPRESENTATION", str(error)
            ) from error
        ingestion_type = (
            "literature_ingestion"
            if representation.representation_kind == LiteratureRepresentationKind.PDF
            else "html_literature_ingestion"
        )
        ingestion = self._validate_record(
            (ingestion_type, representation.ingestion_id), current, visiting
        )
        if (
            representation.literature_id != getattr(ingestion, "literature_id", None)
            or representation.ingestion_hash
            != getattr(ingestion, "content_hash", None)
        ):
            raise TrustedKnowledgeError(
                "INVALID_LITERATURE_REPRESENTATION",
                f"representation ingestion binding is invalid: {representation.representation_id}",
            )

    def _validate_html_ingestion(
        self,
        ingestion: HTMLLiteratureIngestionRecord,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        if ingestion.ingestion_status != LiteratureIngestionStatus.ACCEPTED:
            raise TrustedKnowledgeError(
                "UNTRUSTED_HTML_LITERATURE_INGESTION",
                f"only accepted HTML ingestion records are trusted: {ingestion.ingestion_id}",
            )
        document = self._require_record("literature_document", ingestion.literature_id)
        if not isinstance(document, LiteratureDocument):
            raise TrustedKnowledgeError(
                "INVALID_HTML_LITERATURE_INGESTION_BINDING",
                f"HTML ingestion has no LiteratureDocument: {ingestion.ingestion_id}",
            )
        raw = self._validate_record(
            ("raw_html_literature_artifact", ingestion.raw_artifact_id),
            current,
            visiting,
        )
        canonical = self._validate_record(
            ("canonical_html_text_artifact", ingestion.canonical_text_id),
            current,
            visiting,
        )
        source = self._verify_source(ingestion.source_id, ingestion.source_version)
        if (
            not isinstance(raw, RawHTMLLiteratureArtifact)
            or not isinstance(canonical, CanonicalHTMLTextArtifact)
            or ingestion.raw_artifact_hash != raw.content_hash
            or ingestion.canonical_text_hash != canonical.content_hash
            or ingestion.literature_id != raw.literature_id
            or ingestion.literature_id != canonical.literature_id
            or ingestion.extractor_id != canonical.extractor_id
            or ingestion.extractor_version != canonical.extractor_version
            or ingestion.extractor_config_hash != canonical.extractor_config_hash
            or source.content_sha256 != canonical.text_sha256
            or source.source_role != SourceRole.LITERATURE_AUTHOR
            or source.source_type != SourceType.LITERATURE_ARTICLE
        ):
            raise TrustedKnowledgeError(
                "HTML_LITERATURE_INGESTION_HASH_MISMATCH",
                f"HTML ingestion binding is invalid: {ingestion.ingestion_id}",
            )
        self._reachable[
            ("source_document", f"{source.source_id}@{source.version}")
        ] = source

    def _validate_ingestion(
        self,
        ingestion: LiteratureIngestionRecord,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        if ingestion.ingestion_status != LiteratureIngestionStatus.ACCEPTED:
            raise TrustedKnowledgeError(
                "UNTRUSTED_LITERATURE_INGESTION",
                f"only accepted ingestion records are trusted: {ingestion.ingestion_id}",
            )
        document = self._require_record("literature_document", ingestion.literature_id)
        if not isinstance(document, LiteratureDocument):
            raise TrustedKnowledgeError(
                "INVALID_LITERATURE_INGESTION_BINDING",
                f"ingestion has no LiteratureDocument: {ingestion.ingestion_id}",
            )
        raw = self._validate_record(
            ("raw_literature_artifact", ingestion.raw_artifact_id), current, visiting
        )
        canonical = self._validate_record(
            ("canonical_text_artifact", ingestion.canonical_text_id or ""),
            current,
            visiting,
        )
        if (
            not isinstance(raw, RawLiteratureArtifact)
            or not isinstance(canonical, CanonicalTextArtifact)
            or ingestion.raw_artifact_hash != raw.content_hash
            or ingestion.canonical_text_hash != canonical.content_hash
            or ingestion.literature_id != raw.literature_id
            or ingestion.literature_id != canonical.literature_id
            or ingestion.parser_id != canonical.parser_id
            or ingestion.parser_version != canonical.parser_version
            or ingestion.parser_config_hash != canonical.parser_config_hash
        ):
            raise TrustedKnowledgeError(
                "LITERATURE_INGESTION_HASH_MISMATCH",
                f"ingestion artifact/parser binding is invalid: {ingestion.ingestion_id}",
            )
        source = self._verify_source(
            ingestion.source_id or "", ingestion.source_version or ""
        )
        if (
            source.content_sha256 != canonical.text_sha256
            or source.source_role != SourceRole.LITERATURE_AUTHOR
            or source.source_type != SourceType.LITERATURE_ARTICLE
        ):
            raise TrustedKnowledgeError(
                "CANONICAL_SOURCE_HASH_MISMATCH",
                f"SourceDocument is not the canonical literature source: {ingestion.ingestion_id}",
            )
        self._reachable[("source_document", f"{source.source_id}@{source.version}")] = source

    def _validate_attribution(self, attribution: ExpertAttributionRecord) -> None:
        source = self._verify_source(attribution.source_id, attribution.source_version)
        for evidence_id in attribution.evidence_refs:
            evidence, evidence_source = self._verify_evidence(evidence_id)
            if (evidence.source_id, evidence.source_version) != (
                attribution.source_id,
                attribution.source_version,
            ):
                raise TrustedKnowledgeError(
                    "EXPERT_ATTRIBUTION_SOURCE_MISMATCH",
                    f"attribution evidence comes from another source: {evidence_id}",
                )
            self._remember_evidence(evidence, evidence_source)
        self._reachable[("source_document", f"{source.source_id}@{source.version}")] = source

    def _validate_opinion(
        self,
        opinion: ExpertOpinion,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        self._validate_record(("expert_profile", opinion.expert_id), current, visiting)
        if not opinion.attribution_refs:
            raise TrustedKnowledgeError(
                "MISSING_EXPERT_ATTRIBUTION",
                f"accepted ExpertOpinion has no attribution: {opinion.opinion_id}",
            )
        attributed_evidence: set[str] = set()
        for attribution_id in opinion.attribution_refs:
            attribution = self._validate_record(
                ("expert_attribution", attribution_id), current, visiting
            )
            if not isinstance(attribution, ExpertAttributionRecord):
                raise TrustedKnowledgeError(
                    "INVALID_EXPERT_ATTRIBUTION",
                    f"record is not an ExpertAttributionRecord: {attribution_id}",
                )
            if attribution.expert_id != opinion.expert_id:
                raise TrustedKnowledgeError(
                    "EXPERT_ATTRIBUTION_OWNER_MISMATCH",
                    f"attribution belongs to another expert: {attribution_id}",
                )
            attributed_evidence.update(attribution.evidence_refs)
        if not set(opinion.evidence_refs).issubset(attributed_evidence):
            raise TrustedKnowledgeError(
                "EXPERT_OPINION_EVIDENCE_NOT_ATTRIBUTED",
                f"opinion evidence is not covered by its attributions: {opinion.opinion_id}",
            )
        self._verify_evidence_refs(opinion.evidence_refs)
        for claim_id in opinion.related_claim_refs:
            self._validate_record(("source_claim", claim_id), current, visiting)
        for workflow_id in opinion.related_workflow_refs:
            self._validate_record(("workflow_pattern", workflow_id), current, visiting)
        if opinion.supersedes is not None:
            self._validate_record(("expert_opinion", opinion.supersedes), current, visiting)

    def _validate_relation(
        self,
        relation: KnowledgeRelation,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        self._verify_evidence_refs(relation.evidence_refs)
        for key in (
            (relation.subject_type, relation.subject_id),
            (relation.object_type, relation.object_id),
        ):
            if key[0] in REPRESENTATION_COMPONENT_TYPES:
                raise TrustedKnowledgeError(
                    "UNTRUSTED_RELATION_ENDPOINT",
                    "relations must reference the common literature representation, "
                    f"not an internal component: {key[0]}:{key[1]}",
                )
            self._require_record(*key)
            curation = current.get(key)
            if (key[0] in CURATION_REQUIRED_TYPES or curation is not None) and (
                curation is None or curation.status != CurationStatus.ACCEPTED
            ):
                raise TrustedKnowledgeError(
                    "UNTRUSTED_RELATION_ENDPOINT",
                    f"accepted relation points to untrusted target {key[0]}:{key[1]}",
                )
            self._validate_record(key, current, visiting)

    def _validate_source_quote(self, quote: SourceQuote) -> None:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires an EvidenceStore",
            )
        issues, source = source_quote_record_issues(
            quote,
            self.evidence_store,
            path=f"source_quote:{quote.quote_id}",
        )
        self._raise_issues(issues)
        evidence, evidence_source = self._verify_evidence(quote.evidence_ref)
        self._remember_evidence(evidence, evidence_source)
        if source is not None:
            self._reachable[("source_document", f"{source.source_id}@{source.version}")] = source

    def _validate_source_claim(
        self,
        claim: SourceClaim,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        quotes_by_id = {
            quote_id: quote
            for quote_id in claim.source_quote_refs
            if isinstance(
                (quote := self.records.get(("source_quote", quote_id))), SourceQuote
            )
        }
        self._raise_issues(
            source_claim_binding_issues(
                claim,
                quotes_by_id,
                path=f"source_claim:{claim.claim_id}",
            )
        )
        self._verify_evidence_refs(claim.evidence_refs)
        for quote_id in claim.source_quote_refs:
            self._validate_record(("source_quote", quote_id), current, visiting)

    def _validate_reported_result(
        self,
        result: ReportedResult,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        self._verify_evidence_refs(result.evidence_refs)
        result_context = result.result_context
        if result_context is None:
            return
        method_facts = {
            fact_id: fact
            for fact_id in result_context.method_fact_refs
            if isinstance((fact := self.records.get(("method_fact", fact_id))), MethodFact)
        }
        model_facts = {
            fact_id: fact
            for fact_id in result_context.model_fact_refs
            if isinstance((fact := self.records.get(("model_fact", fact_id))), ModelFact)
        }
        self._raise_issues(
            result_context_record_issues(
                result,
                method_facts,
                model_facts,
                path=f"reported_result:{result.result_id}.result_context",
                require_context=False,
            )
        )
        for fact_id in result_context.method_fact_refs:
            self._validate_record(("method_fact", fact_id), current, visiting)
        for fact_id in result_context.model_fact_refs:
            self._validate_record(("model_fact", fact_id), current, visiting)

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
        try:
            evidence = self.evidence_store.get_evidence(evidence_id)
            source = self.evidence_store.verify_evidence_integrity(evidence)
        except Exception as error:
            raise TrustedKnowledgeError(
                "INVALID_KNOWLEDGE_EVIDENCE",
                f"evidence is unavailable or invalid: {evidence_id}",
            ) from error
        source_key = (source.source_id, source.version)
        if (
            source_key in self._all_literature_ingestion_sources
            and source_key not in self._trusted_current_literature_sources
        ):
            self._require_historical_authorization(evidence, source)
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
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "historical evidence authorization requires an EvidenceStore",
            )
        from .acquisition import validate_literature_representation
        from .ingestion import LiteratureRepresentationSelector

        try:
            if authorization.representation_id is not None:
                representation = LiteratureRepresentationSelector.resolve_reference(
                    authorization.representation_id,
                    self.repositories,
                    self.evidence_store,
                )
                if authorization.representation_hash != representation.content_hash:
                    raise ValueError("historical representation hash mismatch")
            else:
                representation = LiteratureRepresentationSelector.resolve_reference(
                    authorization.ingestion_id or "",
                    self.repositories,
                    self.evidence_store,
                )
                if authorization.ingestion_hash != representation.ingestion_hash:
                    raise ValueError("historical ingestion hash mismatch")
            validate_literature_representation(
                representation, self.repositories, self.evidence_store
            )
            if (
                authorization.literature_id != representation.literature_id
                or authorization.source_id != representation.source_id
                or authorization.source_version != representation.source_version
            ):
                raise ValueError("historical authorization binding mismatch")
            for evidence_id in authorization.evidence_refs:
                evidence = self.evidence_store.get_evidence(evidence_id)
                evidence_source = self.evidence_store.verify_evidence_integrity(evidence)
                if (evidence_source.source_id, evidence_source.version) != (
                    authorization.source_id,
                    authorization.source_version,
                ):
                    raise ValueError("historical evidence comes from another source")
                self._remember_evidence(evidence, evidence_source)
        except Exception as error:
            if isinstance(error, TrustedKnowledgeError):
                raise
            raise TrustedKnowledgeError(
                "INVALID_HISTORICAL_EVIDENCE_AUTHORIZATION",
                f"historical evidence authorization is invalid: {authorization.authorization_id}",
            ) from error
        self._reachable[
            ("historical_evidence_authorization", authorization.authorization_id)
        ] = authorization
        self._reachable[
            ("literature_representation_ref", representation.representation_id)
        ] = representation
        if representation.representation_kind == LiteratureRepresentationKind.PDF:
            ingestion = self.repositories.literature_ingestions.get(
                representation.ingestion_id
            )
            canonical = self.repositories.canonical_text_artifacts.get(
                representation.canonical_text_id or ""
            )
            raw = self.repositories.raw_literature_artifacts.get(
                representation.raw_artifact_id
            )
            record_types = (
                ("literature_ingestion", ingestion.ingestion_id, ingestion),
                ("canonical_text_artifact", canonical.canonical_text_id, canonical),
                ("raw_literature_artifact", raw.artifact_id, raw),
            )
        else:
            ingestion = self.repositories.html_literature_ingestions.get(
                representation.ingestion_id
            )
            canonical = self.repositories.canonical_html_text_artifacts.get(
                representation.canonical_text_id or ""
            )
            raw = self.repositories.raw_html_literature_artifacts.get(
                representation.raw_artifact_id
            )
            record_types = (
                ("html_literature_ingestion", ingestion.ingestion_id, ingestion),
                (
                    "canonical_html_text_artifact",
                    canonical.canonical_text_id,
                    canonical,
                ),
                ("raw_html_literature_artifact", raw.artifact_id, raw),
            )
        for record_type, record_id, record in record_types:
            self._reachable[(record_type, record_id)] = record
        source = self._verify_source(
            representation.source_id or "", representation.source_version or ""
        )
        self._reachable[
            ("source_document", f"{source.source_id}@{source.version}")
        ] = source

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
    def _raise_issues(issues: list[Any]) -> None:
        if issues:
            issue = issues[0]
            raise TrustedKnowledgeError(issue.code, issue.message)

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
