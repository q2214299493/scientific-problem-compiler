"""Persistent knowledge graph views built from immutable repositories."""

from importlib import import_module as _import_module


# Resolve public names on demand so reading a document does not load network or structure services.
_EXPORT_MODULE = {
    "AcquisitionError": "acquisition",
    "ArticleURLResolver": "acquisition",
    "DOIResolver": "acquisition",
    "HTMLLiteratureTextExtractor": "ingestion",
    "HTMLTextExtraction": "literature_text",
    "HTTPResponse": "acquisition",
    "CollectionConnector": "collection",
    "CollectionError": "collection",
    "CollectionImportService": "collection",
    "GenericHTMLCollectionConnector": "collection",
    "KnowledgeGraphBuilder": "graph",
    "KnowledgeGraphError": "graph",
    "KnowledgeEvidenceMigrationResult": "evidence_migration",
    "TrustedKnowledgeError": "trust",
    "TrustedKnowledgeValidator": "trust",
    "DocumentStructureExtractionResult": "structure",
    "DocumentStructureExtractor": "structure",
    "DocumentStructureInput": "structure",
    "DocumentStructureService": "structure",
    "DocumentStructureSelector": "structure_selection",
    "HTMLDocumentStructureExtractor": "structure",
    "PDFDocumentStructureExtractor": "structure",
    "LiteratureIngestionService": "ingestion",
    "LiteratureAcquisitionService": "acquisition",
    "LiteratureExtractionError": "ingestion",
    "LiteratureMetadata": "ingestion",
    "LiteratureTextExtractor": "ingestion",
    "PypdfLiteratureTextExtractor": "ingestion",
    "LiteratureRepresentationSelector": "representation",
    "LocalPDFResolver": "acquisition",
    "PinnedHTTPTransport": "acquisition",
    "ResourceResolver": "acquisition",
    "SafeHTTPError": "acquisition",
    "SafeHTTPFetcher": "acquisition",
    "create_evidence_span_from_canonical_text": "ingestion",
    "create_evidence_from_block": "structure",
    "create_evidence_from_figure_caption": "structure",
    "create_evidence_from_table_cell": "structure",
    "validate_literature_ingestion_chain": "ingestion",
    "detect_acquisition_input": "acquisition",
    "knowledge_graph_to_mermaid": "mermaid",
    "migrate_knowledge_evidence": "evidence_migration",
    "normalize_doi": "acquisition",
    "validate_literature_representation": "representation",
    "canonicalize_url": "collection",
    "diff_collection_snapshots": "collection",
    "make_collection_definition": "collection",
    "make_collection_scope_policy": "collection",
    "path_matches_prefix": "collection",
    "format_structured_evidence_locator": "structure",
    "inspect_document_structure": "structure",
    "validate_document_structure": "structure",
    "validate_structure_selection": "structure_selection",
    "verify_structured_evidence_locator": "structure",
    "require_current_structure": "structure_selection",
    "resolve_current_structure_selection": "structure_selection",
}
__all__ = list(_EXPORT_MODULE)


def __getattr__(name: str):
    owner = _EXPORT_MODULE.get(name)
    if owner is not None:
        value = getattr(_import_module(f".{owner}", __name__), name)
        globals()[name] = value
        return value
    if name in _EXPORT_MODULE.values():
        return _import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_EXPORT_MODULE.values()))
