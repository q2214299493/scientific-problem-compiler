"""Public collection API and compatibility imports."""
from __future__ import annotations

import hashlib as hashlib
import posixpath as posixpath
import re as re
from dataclasses import dataclass as dataclass
from datetime import (
    datetime as datetime,
    timezone as timezone,
)
from html.parser import HTMLParser as HTMLParser
from typing import (
    Callable as Callable,
    Protocol as Protocol,
)
from urllib.parse import (
    parse_qsl as parse_qsl,
    quote as quote,
    unquote as unquote,
    urlencode as urlencode,
    urljoin as urljoin,
    urlparse as urlparse,
    urlunparse as urlunparse,
)

from pydantic import ValidationError as ValidationError

from ...immutable import FrozenDict as FrozenDict
from ...models import (
    AcquisitionStatus as AcquisitionStatus,
    CollectionAcquisitionLink as CollectionAcquisitionLink,
    CollectionCompletenessStatus as CollectionCompletenessStatus,
    CollectionDefinition as CollectionDefinition,
    CollectionDiff as CollectionDiff,
    CollectionDiscoveryConfidence as CollectionDiscoveryConfidence,
    CollectionDiscoveryContext as CollectionDiscoveryContext,
    CollectionDiscoveryResult as CollectionDiscoveryResult,
    CollectionImportOutcome as CollectionImportOutcome,
    CollectionImportRecord as CollectionImportRecord,
    CollectionImportStatus as CollectionImportStatus,
    CollectionLiteratureMembership as CollectionLiteratureMembership,
    CollectionMembershipDecision as CollectionMembershipDecision,
    CollectionPageRecord as CollectionPageRecord,
    CollectionResourceImportResult as CollectionResourceImportResult,
    CollectionResourceKind as CollectionResourceKind,
    CollectionResourceOccurrence as CollectionResourceOccurrence,
    CollectionScopePolicy as CollectionScopePolicy,
    CollectionSnapshot as CollectionSnapshot,
    CollectionSourceKind as CollectionSourceKind,
    CollectionTraversalStatus as CollectionTraversalStatus,
    DiscoveredCollectionResource as DiscoveredCollectionResource,
)
from ...repositories import (
    EvidenceStore as EvidenceStore,
    KnowledgeRepositories as KnowledgeRepositories,
)
from ...serialization import content_hash as content_hash
from ..acquisition import (
    SUPPORTED_HTML_TYPES as SUPPORTED_HTML_TYPES,
    AcquisitionError as AcquisitionError,
    LiteratureAcquisitionService as LiteratureAcquisitionService,
    SafeHTTPError as SafeHTTPError,
    SafeHTTPFetcher as SafeHTTPFetcher,
    normalize_doi as normalize_doi,
)
from .connectors import (
    CollectionConnector as CollectionConnector,
    GenericHTMLCollectionConnector as GenericHTMLCollectionConnector,
)
from .discovery import (
    ARTICLE_PATH_MARKERS as ARTICLE_PATH_MARKERS,
    CONTEXT_TOKEN_PATTERN as CONTEXT_TOKEN_PATTERN,
    DOI_DISCOVERY_PATTERN as DOI_DISCOVERY_PATTERN,
    NAVIGATION_CONTEXTS as NAVIGATION_CONTEXTS,
    NEXT_LABELS as NEXT_LABELS,
    POSITIVE_MEMBERSHIP_CONTEXTS as POSITIVE_MEMBERSHIP_CONTEXTS,
    REFERENCE_CONTEXTS as REFERENCE_CONTEXTS,
    VOID_HTML_TAGS as VOID_HTML_TAGS,
    _Candidate as _Candidate,
    _clean_doi_candidate as _clean_doi_candidate,
    _CollectionHTMLParser as _CollectionHTMLParser,
    _doi_inside as _doi_inside,
    _extract_page as _extract_page,
    _Link as _Link,
    _membership_context as _membership_context,
    _resource_candidate as _resource_candidate,
    _TextFragment as _TextFragment,
)
from .importer import (
    EXPECTED_COLLECTION_ACQUISITION_EXCEPTIONS as EXPECTED_COLLECTION_ACQUISITION_EXCEPTIONS,
    CollectionImportService as CollectionImportService,
)
from .snapshots import (
    _make_bound as _make_bound,
    diff_collection_snapshots as diff_collection_snapshots,
)
from .urls import (
    CollectionError as CollectionError,
    _canonical_path_prefix as _canonical_path_prefix,
    _default_path_prefix as _default_path_prefix,
    canonical_origin as canonical_origin,
    canonicalize_url as canonicalize_url,
    make_collection_definition as make_collection_definition,
    make_collection_scope_policy as make_collection_scope_policy,
    path_matches_prefix as path_matches_prefix,
)
