"""Persistent knowledge graph views built from immutable repositories."""

from .acquisition import (
    AcquisitionError,
    ArticleURLResolver,
    DOIResolver,
    HTMLLiteratureTextExtractor,
    HTMLTextExtraction,
    HTTPResponse,
    LiteratureAcquisitionService,
    LocalPDFResolver,
    PinnedHTTPTransport,
    ResourceResolver,
    SafeHTTPError,
    SafeHTTPFetcher,
    detect_acquisition_input,
    normalize_doi,
    validate_literature_representation,
)
from .graph import KnowledgeGraphBuilder, KnowledgeGraphError
from .ingestion import (
    LiteratureIngestionService,
    LiteratureExtractionError,
    LiteratureMetadata,
    LiteratureTextExtractor,
    PypdfLiteratureTextExtractor,
    LiteratureRepresentationSelector,
    create_evidence_span_from_canonical_text,
    validate_literature_ingestion_chain,
)
from .mermaid import knowledge_graph_to_mermaid
from .trust import TrustedKnowledgeError, TrustedKnowledgeValidator

__all__ = [
    "AcquisitionError",
    "ArticleURLResolver",
    "DOIResolver",
    "HTMLLiteratureTextExtractor",
    "HTMLTextExtraction",
    "HTTPResponse",
    "KnowledgeGraphBuilder",
    "KnowledgeGraphError",
    "TrustedKnowledgeError",
    "TrustedKnowledgeValidator",
    "LiteratureIngestionService",
    "LiteratureAcquisitionService",
    "LiteratureExtractionError",
    "LiteratureMetadata",
    "LiteratureTextExtractor",
    "PypdfLiteratureTextExtractor",
    "LiteratureRepresentationSelector",
    "LocalPDFResolver",
    "PinnedHTTPTransport",
    "ResourceResolver",
    "SafeHTTPError",
    "SafeHTTPFetcher",
    "create_evidence_span_from_canonical_text",
    "validate_literature_ingestion_chain",
    "detect_acquisition_input",
    "knowledge_graph_to_mermaid",
    "normalize_doi",
    "validate_literature_representation",
]
