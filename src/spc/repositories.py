"""Compatibility exports for the original SPC repositories module."""

from __future__ import annotations

import hashlib as hashlib
from io import BytesIO as BytesIO
import os as os
import shutil as shutil
import tempfile as tempfile
from pathlib import Path as Path
from pathlib import PurePosixPath as PurePosixPath
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Generic as Generic,
    Iterable as Iterable,
    Protocol as Protocol,
    TypeVar as TypeVar,
    cast as cast,
)
from zipfile import BadZipFile as BadZipFile, ZipFile as ZipFile

from pydantic import BaseModel as BaseModel

from .models import (
    AcquisitionAttemptRecord as AcquisitionAttemptRecord,
    CanonicalTextArtifact as CanonicalTextArtifact,
    CanonicalTextBlock as CanonicalTextBlock,
    CanonicalHTMLTextArtifact as CanonicalHTMLTextArtifact,
    CollectionAcquisitionLink as CollectionAcquisitionLink,
    CollectionDefinition as CollectionDefinition,
    CollectionDiff as CollectionDiff,
    CollectionImportRecord as CollectionImportRecord,
    CollectionLiteratureMembership as CollectionLiteratureMembership,
    CollectionPageArtifact as CollectionPageArtifact,
    CollectionPageRecord as CollectionPageRecord,
    CollectionResourceImportResult as CollectionResourceImportResult,
    CollectionResourceOccurrence as CollectionResourceOccurrence,
    CollectionSnapshot as CollectionSnapshot,
    DiscoveredCollectionResource as DiscoveredCollectionResource,
    DocumentStructureArtifact as DocumentStructureArtifact,
    DocumentStructureBlock as DocumentStructureBlock,
    DocumentStructureSelection as DocumentStructureSelection,
    DomainProfile as DomainProfile,
    EvidenceSpan as EvidenceSpan,
    ExpertAttributionRecord as ExpertAttributionRecord,
    ExpertCase as ExpertCase,
    ExpertOpinion as ExpertOpinion,
    ExpertProfile as ExpertProfile,
    HistoricalLiteratureEvidenceAuthorization as HistoricalLiteratureEvidenceAuthorization,
    HTMLLiteratureIngestionRecord as HTMLLiteratureIngestionRecord,
    KnowledgeSnapshot as KnowledgeSnapshot,
    KnowledgeCurationRecord as KnowledgeCurationRecord,
    KnowledgeRelation as KnowledgeRelation,
    LiteratureAcquisitionRecord as LiteratureAcquisitionRecord,
    LiteratureAcquisitionRequest as LiteratureAcquisitionRequest,
    LiteratureDocument as LiteratureDocument,
    LiteratureIngestionRecord as LiteratureIngestionRecord,
    LiteratureRepresentationSelection as LiteratureRepresentationSelection,
    LiteratureRepresentationReference as LiteratureRepresentationReference,
    LiteratureSupplementarySource as LiteratureSupplementarySource,
    LiteratureWorkflowPattern as LiteratureWorkflowPattern,
    MethodFact as MethodFact,
    MetadataMergeManifest as MetadataMergeManifest,
    MetadataRetrievalRecord as MetadataRetrievalRecord,
    ModelFact as ModelFact,
    RawLiteratureArtifact as RawLiteratureArtifact,
    RawHTMLLiteratureArtifact as RawHTMLLiteratureArtifact,
    ReportedResult as ReportedResult,
    ResolvedLiteratureResource as ResolvedLiteratureResource,
    ScientificCapability as ScientificCapability,
    SourceClaim as SourceClaim,
    SourceDocument as SourceDocument,
    SourceQuote as SourceQuote,
    SourceType as SourceType,
    StructuredEvidenceLocator as StructuredEvidenceLocator,
    TableCellStructure as TableCellStructure,
    TableStructure as TableStructure,
    FigureStructure as FigureStructure,
)
from .serialization import (
    content_hash as content_hash,
    dump_json as dump_json,
    dump_yaml as dump_yaml,
    load_data as load_data,
    load_model as load_model,
    require_safe_path_component as require_safe_path_component,
)

from .repository_components.base import (
    IdentityBoundRepository as IdentityBoundRepository,
    ModelRepository as ModelRepository,
    ModelT as ModelT,
    STATE_DIRECTORIES as STATE_DIRECTORIES,
    initialize_state as initialize_state,
)
from .repository_components.collection import (
    CollectionAcquisitionLinkRepository as CollectionAcquisitionLinkRepository,
    CollectionDefinitionRepository as CollectionDefinitionRepository,
    CollectionDiffRepository as CollectionDiffRepository,
    CollectionImportRepository as CollectionImportRepository,
    CollectionLiteratureMembershipRepository as CollectionLiteratureMembershipRepository,
    CollectionPageArtifactRepository as CollectionPageArtifactRepository,
    CollectionPageRepository as CollectionPageRepository,
    CollectionResourceImportResultRepository as CollectionResourceImportResultRepository,
    CollectionResourceOccurrenceRepository as CollectionResourceOccurrenceRepository,
    CollectionSnapshotRepository as CollectionSnapshotRepository,
    DiscoveredCollectionResourceRepository as DiscoveredCollectionResourceRepository,
)
from .repository_components.evidence import (
    CompositeEvidenceStore as CompositeEvidenceStore,
    EvidenceStore as EvidenceStore,
    FilesystemEvidenceStore as FilesystemEvidenceStore,
    KnowledgeEvidenceStore as KnowledgeEvidenceStore,
    ProjectEvidenceStore as ProjectEvidenceStore,
    SourceEvidenceStore as SourceEvidenceStore,
    _MAX_LIST_SOURCE_BYTES as _MAX_LIST_SOURCE_BYTES,
    _STANDARD_SOURCE_INTEGRITY_CHECK as _STANDARD_SOURCE_INTEGRITY_CHECK,
    source_documents_equivalent as source_documents_equivalent,
)
from .repository_components.expert import (
    ExpertAttributionRepository as ExpertAttributionRepository,
    ExpertCaseRepository as ExpertCaseRepository,
    ExpertOpinionRepository as ExpertOpinionRepository,
    ExpertProfileRepository as ExpertProfileRepository,
)
from .repository_components.knowledge import (
    KnowledgeCurationRepository as KnowledgeCurationRepository,
    KnowledgeRelationRepository as KnowledgeRelationRepository,
    KnowledgeRepositories as KnowledgeRepositories,
)
from .repository_components.literature import (
    AcquisitionAttemptRepository as AcquisitionAttemptRepository,
    CanonicalHTMLTextArtifactRepository as CanonicalHTMLTextArtifactRepository,
    CanonicalTextArtifactRepository as CanonicalTextArtifactRepository,
    DocumentStructureArtifactRepository as DocumentStructureArtifactRepository,
    DocumentStructureBlockRepository as DocumentStructureBlockRepository,
    DocumentStructureSelectionRepository as DocumentStructureSelectionRepository,
    FigureStructureRepository as FigureStructureRepository,
    HTMLLiteratureIngestionRepository as HTMLLiteratureIngestionRepository,
    HistoricalLiteratureEvidenceAuthorizationRepository as HistoricalLiteratureEvidenceAuthorizationRepository,
    LiteratureAcquisitionRepository as LiteratureAcquisitionRepository,
    LiteratureAcquisitionRequestRepository as LiteratureAcquisitionRequestRepository,
    LiteratureArtifactStore as LiteratureArtifactStore,
    LiteratureDocumentRepository as LiteratureDocumentRepository,
    LiteratureIngestionRepository as LiteratureIngestionRepository,
    LiteratureRepresentationReferenceRepository as LiteratureRepresentationReferenceRepository,
    LiteratureRepresentationSelectionRepository as LiteratureRepresentationSelectionRepository,
    LiteratureWorkflowRepository as LiteratureWorkflowRepository,
    MetadataMergeManifestRepository as MetadataMergeManifestRepository,
    MetadataRetrievalRepository as MetadataRetrievalRepository,
    RawHTMLLiteratureArtifactRepository as RawHTMLLiteratureArtifactRepository,
    RawLiteratureArtifactRepository as RawLiteratureArtifactRepository,
    ResolvedLiteratureResourceRepository as ResolvedLiteratureResourceRepository,
    StructuredEvidenceLocatorRepository as StructuredEvidenceLocatorRepository,
    TableCellStructureRepository as TableCellStructureRepository,
    TableStructureRepository as TableStructureRepository,
)
from .repository_components.scientific_records import (
    ScientificCapabilityRepository as ScientificCapabilityRepository,
)
