from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..models import (
    DomainProfile,
    ExpertAttributionRecord,
    ExpertCase,
    ExpertOpinion,
    ExpertProfile,
    HistoricalLiteratureEvidenceAuthorization,
    KnowledgeSnapshot,
    KnowledgeCurationRecord,
    KnowledgeRelation,
    LiteratureAcquisitionRecord,
    LiteratureAcquisitionRequest,
    LiteratureDocument,
    LiteratureIngestionRecord,
    LiteratureRepresentationSelection,
    LiteratureWorkflowPattern,
    ResolvedLiteratureResource,
    ScientificCapability,
)

from .base import IdentityBoundRepository
from .collection import CollectionRepositories
from .evidence import EvidenceStore, KnowledgeEvidenceStore
from .expert import ExpertRepositories
from .literature import LiteratureRepositories
from .scientific_records import ScientificRecordRepositories

from .snapshot import create_snapshot

if TYPE_CHECKING:
    from ..knowledge.trust import _ValidatedTrustedRead


class KnowledgeRelationRepository(IdentityBoundRepository[KnowledgeRelation]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "relations",
            KnowledgeRelation,
            "relation_id",
        )

    def put(self, key: str, model: KnowledgeRelation) -> Path:
        if self._record_path(key).exists():
            raise FileExistsError(f"duplicate knowledge relation ID: {key}")
        return super().put(key, model)


class KnowledgeCurationRepository(
    IdentityBoundRepository[KnowledgeCurationRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "curations",
            KnowledgeCurationRecord,
            "curation_id",
        )


class KnowledgeRepositories:
    def __init__(self, root: Path, *, initialize: bool = True) -> None:
        self.root = root
        self.evidence_store = KnowledgeEvidenceStore(root, initialize=initialize)
        self.literature = LiteratureRepositories(root, evidence_store=self.evidence_store)
        self.expert = ExpertRepositories(root, evidence_store=self.evidence_store)
        self.collection = CollectionRepositories(self.literature)
        self.scientific_records = ScientificRecordRepositories(root)
        self.expert_sources = self.expert.expert_sources
        self.expert_cases = self.expert.expert_cases
        self.workflow_patterns = self.literature.workflow_patterns
        self.capabilities = self.scientific_records.capabilities
        self.literature_documents = self.literature.literature_documents
        self.literature_artifacts = self.literature.literature_artifacts
        self.raw_literature_artifacts = self.literature.raw_literature_artifacts
        self.canonical_text_artifacts = self.literature.canonical_text_artifacts
        self.raw_html_literature_artifacts = self.literature.raw_html_literature_artifacts
        self.canonical_html_text_artifacts = self.literature.canonical_html_text_artifacts
        self.literature_ingestions = self.literature.literature_ingestions
        self.html_literature_ingestions = self.literature.html_literature_ingestions
        self.literature_representation_selections = self.literature.literature_representation_selections
        self.literature_supplementary_sources = self.literature.literature_supplementary_sources
        self.historical_evidence_authorizations = self.literature.historical_evidence_authorizations
        self.acquisition_requests = self.literature.acquisition_requests
        self.resolved_literature_resources = self.literature.resolved_literature_resources
        self.literature_acquisitions = self.literature.literature_acquisitions
        self.acquisition_attempts = self.literature.acquisition_attempts
        self.metadata_retrievals = self.literature.metadata_retrievals
        self.metadata_merge_manifests = self.literature.metadata_merge_manifests
        self.literature_representation_refs = self.literature.literature_representation_refs
        self.document_structure_artifacts = self.literature.document_structure_artifacts
        self.document_structure_selections = self.literature.document_structure_selections
        self.document_structure_blocks = self.literature.document_structure_blocks
        self.table_structures = self.literature.table_structures
        self.table_cell_structures = self.literature.table_cell_structures
        self.figure_structures = self.literature.figure_structures
        self.structured_evidence_locators = self.literature.structured_evidence_locators
        self.collection_definitions = self.collection.collection_definitions
        self.collection_pages = self.collection.collection_pages
        self.collection_page_artifacts = self.collection.collection_page_artifacts
        self.collection_resources = self.collection.collection_resources
        self.collection_occurrences = self.collection.collection_occurrences
        self.collection_snapshots = self.collection.collection_snapshots
        self.collection_acquisition_links = self.collection.collection_acquisition_links
        self.collection_resource_import_results = self.collection.collection_resource_import_results
        self.collection_literature_memberships = self.collection.collection_literature_memberships
        self.collection_imports = self.collection.collection_imports
        self.collection_diffs = self.collection.collection_diffs
        self.expert_profiles = self.expert.expert_profiles
        self.expert_opinions = self.expert.expert_opinions
        self.expert_attributions = self.expert.expert_attributions
        self.expert_knowledge_inputs = self.expert.expert_knowledge_inputs
        self.expert_knowledge_chunks = self.expert.expert_knowledge_chunks
        self.expert_knowledge_proposals = self.expert.expert_knowledge_proposals
        self.expert_knowledge_groundings = self.expert.expert_knowledge_groundings
        self.expert_knowledge_compilations = self.expert.expert_knowledge_compilations
        self.relations = KnowledgeRelationRepository(root)
        self.curations = KnowledgeCurationRepository(root)
        self.source_quotes = self.scientific_records.source_quotes
        self.source_claims = self.scientific_records.source_claims
        self.reported_results = self.scientific_records.reported_results
        self.method_facts = self.scientific_records.method_facts
        self.model_facts = self.scientific_records.model_facts
        self.literature_knowledge_inputs = self.literature.literature_knowledge_inputs
        self.literature_knowledge_chunks = self.literature.literature_knowledge_chunks
        self.literature_knowledge_proposals = self.literature.literature_knowledge_proposals
        self.literature_knowledge_groundings = self.literature.literature_knowledge_groundings
        self.literature_knowledge_compilations = self.literature.literature_knowledge_compilations

    def load_expert_cases(self, records: Iterable[ExpertCase]) -> None:
        for record in records:
            self.expert_cases.put(record.case_id, record)

    def load_workflow_patterns(self, records: Iterable[LiteratureWorkflowPattern]) -> None:
        for record in records:
            self.workflow_patterns.put(record.pattern_id, record)

    def load_capabilities(self, records: Iterable[ScientificCapability]) -> None:
        for record in records:
            self.capabilities.put(record.capability_id, record)

    def load_literature_documents(
        self, records: Iterable[LiteratureDocument]
    ) -> None:
        for record in records:
            self.literature_documents.put(record.literature_id, record)

    def load_literature_ingestions(
        self, records: Iterable[LiteratureIngestionRecord]
    ) -> None:
        for record in records:
            self.literature_ingestions.put(record.ingestion_id, record)

    def load_literature_representation_selections(
        self, records: Iterable[LiteratureRepresentationSelection]
    ) -> None:
        for record in records:
            self.literature_representation_selections.put(
                record.selection_id, record
            )

    def load_historical_evidence_authorizations(
        self, records: Iterable[HistoricalLiteratureEvidenceAuthorization]
    ) -> None:
        for record in records:
            self.historical_evidence_authorizations.put(
                record.authorization_id, record
            )

    def load_acquisition_requests(
        self, records: Iterable[LiteratureAcquisitionRequest]
    ) -> None:
        for record in records:
            self.acquisition_requests.put(record.request_id, record)

    def load_resolved_literature_resources(
        self, records: Iterable[ResolvedLiteratureResource]
    ) -> None:
        for record in records:
            self.resolved_literature_resources.put(record.resource_id, record)

    def load_literature_acquisitions(
        self, records: Iterable[LiteratureAcquisitionRecord]
    ) -> None:
        for record in records:
            self.literature_acquisitions.put(record.acquisition_id, record)

    def load_expert_profiles(self, records: Iterable[ExpertProfile]) -> None:
        for record in records:
            self.expert_profiles.put(record.expert_id, record)

    def load_expert_opinions(self, records: Iterable[ExpertOpinion]) -> None:
        for record in records:
            self.expert_opinions.put(record.opinion_id, record)

    def load_expert_attributions(
        self, records: Iterable[ExpertAttributionRecord]
    ) -> None:
        for record in records:
            self.expert_attributions.put(record.attribution_id, record)

    def load_relations(self, records: Iterable[KnowledgeRelation]) -> None:
        for record in records:
            self.relations.put(record.relation_id, record)

    def load_curations(self, records: Iterable[KnowledgeCurationRecord]) -> None:
        for record in records:
            self.curations.put(record.curation_id, record)

    def create_snapshot(
        self,
        evidence_store: EvidenceStore,
        domain_profile: DomainProfile,
        *,
        _validated_read: _ValidatedTrustedRead | None = None,
    ) -> KnowledgeSnapshot:
        return create_snapshot(self, evidence_store, domain_profile, _validated_read=_validated_read)
