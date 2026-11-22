"""Public ingestion interface and historical import names."""

from __future__ import annotations

from dataclasses import dataclass as dataclass
from pathlib import Path as Path, PurePosixPath as PurePosixPath
from typing import Any as Any, Mapping as Mapping, Protocol as Protocol
from xml.etree import ElementTree as ElementTree
from zipfile import BadZipFile as BadZipFile, ZipFile as ZipFile
import hashlib as hashlib
import pypdf as pypdf
import re as re
import unicodedata as unicodedata

from pydantic import Field as Field

from ...models import (
    CanonicalTextArtifact as CanonicalTextArtifact,
    CanonicalTextBlock as CanonicalTextBlock,
    EvidenceSpan as EvidenceSpan,
    LiteratureDocument as LiteratureDocument,
    LiteratureIngestionOutcome as LiteratureIngestionOutcome,
    LiteratureIngestionRecord as LiteratureIngestionRecord,
    LiteratureIngestionStatus as LiteratureIngestionStatus,
    LiteratureRepresentationKind as LiteratureRepresentationKind,
    LiteratureRepresentationReference as LiteratureRepresentationReference,
    LiteratureRepresentationSelection as LiteratureRepresentationSelection,
    LiteratureRepresentationSelectionOutcome as LiteratureRepresentationSelectionOutcome,
    LiteratureSupplementarySource as LiteratureSupplementarySource,
    NonBlankStr as NonBlankStr,
    RawLiteratureArtifact as RawLiteratureArtifact,
    SourceDocument as SourceDocument,
    SourceRole as SourceRole,
    SourceType as SourceType,
    StrictModel as StrictModel,
    literature_identity_id as literature_identity_id,
)
from ...repositories import (
    EvidenceStore as EvidenceStore,
    KnowledgeEvidenceStore as KnowledgeEvidenceStore,
    KnowledgeRepositories as KnowledgeRepositories,
)
from ...serialization import content_hash as content_hash
from ..literature_text import (
    CANONICALIZATION_POLICY as CANONICALIZATION_POLICY,
    LiteratureExtractionError as LiteratureExtractionError,
    LiteratureMetadata as LiteratureMetadata,
    LiteratureTextExtraction as LiteratureTextExtraction,
    LiteratureTextExtractor as LiteratureTextExtractor,
    PAGE_SEPARATOR as PAGE_SEPARATOR,
)
from ..representation import (
    LiteratureRepresentationSelector as LiteratureRepresentationSelector,
    ValidatedLiteratureIngestionChain as ValidatedLiteratureIngestionChain,
    validate_literature_ingestion_chain as validate_literature_ingestion_chain,
)
from .docx import DocxSupplementaryTextExtractor as DocxSupplementaryTextExtractor
from .evidence import (
    create_evidence_span_from_canonical_text as create_evidence_span_from_canonical_text,
    create_supplementary_evidence_span as create_supplementary_evidence_span,
)
from .pdf import PypdfLiteratureTextExtractor as PypdfLiteratureTextExtractor
from .service import LiteratureIngestionService as LiteratureIngestionService
from .supplementary import (
    _ingest_literature_supplementary as _ingest_literature_supplementary,
    ingest_literature_supplementary_docx as ingest_literature_supplementary_docx,
    ingest_literature_supplementary_pdf as ingest_literature_supplementary_pdf,
    validate_literature_supplementary_source as validate_literature_supplementary_source,
)
from ..literature_text import CanonicalTextExtractor as CanonicalTextExtractor, CanonicalTextResult as CanonicalTextResult
from .html import HTMLLiteratureTextExtractor as HTMLLiteratureTextExtractor
