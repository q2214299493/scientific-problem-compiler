"""Public acquisition interface and historical import names."""

from __future__ import annotations

from dataclasses import dataclass as dataclass
from html.parser import HTMLParser as HTMLParser
from io import BytesIO as BytesIO
from pathlib import Path as Path
from types import MappingProxyType as MappingProxyType
from typing import Callable as Callable, Mapping as Mapping, Protocol as Protocol
from urllib.parse import quote as quote, urljoin as urljoin, urlparse as urlparse
import hashlib as hashlib
import http.client  # noqa: F401
import ipaddress as ipaddress
import json as json
import re as re
import shutil as shutil
import socket as socket
import ssl as ssl
import tempfile as tempfile

from pypdf import PdfReader as PdfReader
from pypdf.errors import PdfReadError as PdfReadError

from ...models import (
    AcquisitionAttemptRecord as AcquisitionAttemptRecord,
    AcquisitionInputKind as AcquisitionInputKind,
    AcquisitionStatus as AcquisitionStatus,
    CanonicalHTMLTextArtifact as CanonicalHTMLTextArtifact,
    CanonicalTextBlock as CanonicalTextBlock,
    FullTextAccessStatus as FullTextAccessStatus,
    FullTextCandidate as FullTextCandidate,
    FullTextSourceKind as FullTextSourceKind,
    HTMLLiteratureIngestionRecord as HTMLLiteratureIngestionRecord,
    LiteratureAcquisitionOutcome as LiteratureAcquisitionOutcome,
    LiteratureAcquisitionRecord as LiteratureAcquisitionRecord,
    LiteratureAcquisitionRequest as LiteratureAcquisitionRequest,
    LiteratureDocument as LiteratureDocument,
    LiteratureIngestionStatus as LiteratureIngestionStatus,
    LiteratureRepresentationKind as LiteratureRepresentationKind,
    LiteratureRepresentationReference as LiteratureRepresentationReference,
    MetadataAlternative as MetadataAlternative,
    MetadataFieldDecision as MetadataFieldDecision,
    MetadataMergeManifest as MetadataMergeManifest,
    MetadataRetrievalRecord as MetadataRetrievalRecord,
    MetadataValueOrigin as MetadataValueOrigin,
    RawHTMLLiteratureArtifact as RawHTMLLiteratureArtifact,
    ResolvedLiteratureResource as ResolvedLiteratureResource,
    SourceDocument as SourceDocument,
    literature_identity_id as literature_identity_id,
)
from ...repositories import EvidenceStore as EvidenceStore, KnowledgeRepositories as KnowledgeRepositories
from ...serialization import content_hash as content_hash
from ..ingestion.html import (
    HTMLLiteratureTextExtractor as HTMLLiteratureTextExtractor,
    _CanonicalHTMLParser as _CanonicalHTMLParser,
)
from ..ingestion.service import LiteratureIngestionService as LiteratureIngestionService
from ..literature_text import (
    AcquisitionError as AcquisitionError,
    HTMLTextExtraction as HTMLTextExtraction,
    LiteratureMetadata as LiteratureMetadata,
)
from ..representation import (
    validate_literature_ingestion_chain as validate_literature_ingestion_chain,
    validate_literature_representation as validate_literature_representation,
)
from .metadata import (
    DOI_PATTERN as DOI_PATTERN,
    METADATA_FIELDS as METADATA_FIELDS,
    SUPPORTED_HTML_TYPES as SUPPORTED_HTML_TYPES,
    SUPPORTED_PDF_TYPES as SUPPORTED_PDF_TYPES,
    _MetadataHTMLParser as _MetadataHTMLParser,
    _make_candidate as _make_candidate,
    _make_metadata_retrieval as _make_metadata_retrieval,
    _make_request as _make_request,
    _make_resource as _make_resource,
    _merge_metadata as _merge_metadata,
    _metadata_from_resource as _metadata_from_resource,
    _normalized_metadata_value as _normalized_metadata_value,
    _pdf_metadata as _pdf_metadata,
    _reprioritize_candidates as _reprioritize_candidates,
    detect_acquisition_input as detect_acquisition_input,
    normalize_doi as normalize_doi,
)
from .resolvers import (
    ArticleURLResolver as ArticleURLResolver,
    DOIResolver as DOIResolver,
    LocalPDFResolver as LocalPDFResolver,
    ResourceResolver as ResourceResolver,
)
from .service import (
    LiteratureAcquisitionService as LiteratureAcquisitionService,
    _CandidateResult as _CandidateResult,
)
from .transport import (
    HTTPResponse as HTTPResponse,
    HTTPTransport as HTTPTransport,
    PinnedHTTPTransport as PinnedHTTPTransport,
    SafeHTTPError as SafeHTTPError,
    SafeHTTPFetcher as SafeHTTPFetcher,
    _PinnedHTTPConnection as _PinnedHTTPConnection,
    _PinnedHTTPSConnection as _PinnedHTTPSConnection,
)
