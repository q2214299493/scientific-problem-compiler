"""Public structure interface and historical import names."""

from __future__ import annotations

from collections import defaultdict as defaultdict, deque as deque
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor
from dataclasses import dataclass as dataclass, field as field
from functools import partial as partial
from html.parser import HTMLParser as HTMLParser
from pathlib import Path as Path, PurePosixPath as PurePosixPath
from typing import Protocol as Protocol
import hashlib as hashlib
import re as re

from ...models import (
    CanonicalHTMLTextArtifact as CanonicalHTMLTextArtifact,
    CanonicalTextArtifact as CanonicalTextArtifact,
    CanonicalTextBlock as CanonicalTextBlock,
    DocumentBlockType as DocumentBlockType,
    DocumentContentRegion as DocumentContentRegion,
    DocumentStructureArtifact as DocumentStructureArtifact,
    DocumentStructureBlock as DocumentStructureBlock,
    DocumentStructureSelection as DocumentStructureSelection,
    EvidenceSpan as EvidenceSpan,
    FigureStructure as FigureStructure,
    LiteratureRepresentationKind as LiteratureRepresentationKind,
    LiteratureRepresentationReference as LiteratureRepresentationReference,
    StructureExtractionStatus as StructureExtractionStatus,
    StructuredEvidenceLocator as StructuredEvidenceLocator,
    TableCellStructure as TableCellStructure,
    TableStructure as TableStructure,
)
from ...repositories import (
    CanonicalHTMLTextArtifactRepository as CanonicalHTMLTextArtifactRepository,
    CanonicalTextArtifactRepository as CanonicalTextArtifactRepository,
    EvidenceStore as EvidenceStore,
    KnowledgeRepositories as KnowledgeRepositories,
)
from ...serialization import content_hash as content_hash
from ..ingestion.evidence import (
    create_evidence_span_from_canonical_text as create_evidence_span_from_canonical_text,
)
from ..representation import validate_literature_representation as validate_literature_representation
from .evidence import (
    _create_locator as _create_locator,
    _display_for_block as _display_for_block,
    _verify_locator_against_validated_structure as _verify_locator_against_validated_structure,
    create_evidence_from_block as create_evidence_from_block,
    create_evidence_from_figure_caption as create_evidence_from_figure_caption,
    create_evidence_from_table_cell as create_evidence_from_table_cell,
    format_structured_evidence_locator as format_structured_evidence_locator,
    verify_structured_evidence_locator as verify_structured_evidence_locator,
)
from .html import (
    HTMLDocumentStructureExtractor as HTMLDocumentStructureExtractor,
    _HTMLElement as _HTMLElement,
    _HTMLStructureParser as _HTMLStructureParser,
    _HTMLTableCell as _HTMLTableCell,
    _label as _label,
    _table_occupancy_layout as _table_occupancy_layout,
)
from .pdf import (
    PDFDocumentStructureExtractor as PDFDocumentStructureExtractor,
    _PDF_BARE_TABLE_RE as _PDF_BARE_TABLE_RE,
    _PDF_CAPTION_RE as _PDF_CAPTION_RE,
    _heading_level as _heading_level,
    _pdf_caption as _pdf_caption,
)
from .records import (
    CanonicalArtifact as CanonicalArtifact,
    DocumentStructureExtractionResult as DocumentStructureExtractionResult,
    DocumentStructureExtractor as DocumentStructureExtractor,
    DocumentStructureInput as DocumentStructureInput,
    ExtractedDocumentStructure as ExtractedDocumentStructure,
    _make_block as _make_block,
    _make_cell as _make_cell,
    _make_figure as _make_figure,
    _make_table as _make_table,
    _text_hash as _text_hash,
)
from .service import (
    DocumentStructureService as DocumentStructureService,
    _structure_id as _structure_id,
    inspect_document_structure as inspect_document_structure,
)
from .validation import (
    _CANONICAL_READERS as _CANONICAL_READERS,
    _validate_structure_records as _validate_structure_records,
    validate_document_structure as validate_document_structure,
    validate_table_cell_topology as validate_table_cell_topology,
)
from .evidence import create_structured_evidence_locator as create_structured_evidence_locator
