from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from ..immutable import FrozenDict

from .base import NonBlankStr, Sha256Str, StrictModel
from .literature import AcquisitionStatus


class CollectionSourceKind(StrEnum):
    PROJECT_URL = "project_url"
    COLLECTION_URL = "collection_url"
    LOCAL_MANIFEST = "local_manifest"


class CollectionImportStatus(StrEnum):
    DISCOVERED = "discovered"
    IMPORTING = "importing"
    COMPLETE = "complete"
    PARTIAL = "partial"
    REQUIRES_AUTHENTICATION = "requires_authentication"
    FAILED = "failed"


class CollectionResourceKind(StrEnum):
    DOI = "doi"
    ARTICLE_URL = "article_url"
    PDF_URL = "pdf_url"
    METADATA_RECORD = "metadata_record"


class CollectionTraversalStatus(StrEnum):
    EXHAUSTED = "exhausted"
    LIMITED = "limited"
    REQUIRES_AUTHENTICATION = "requires_authentication"
    UNSUPPORTED_DYNAMIC = "unsupported_dynamic"
    FAILED = "failed"


class CollectionCompletenessStatus(StrEnum):
    PROVEN_COMPLETE = "proven_complete"
    POLICY_EXHAUSTED_UNVERIFIED = "policy_exhausted_unverified"
    PARTIAL = "partial"
    REQUIRES_AUTHENTICATION = "requires_authentication"
    UNSUPPORTED_DYNAMIC = "unsupported_dynamic"
    FAILED = "failed"


class CollectionDiscoveryConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CollectionDiscoveryContext(StrEnum):
    CITATION_METADATA = "citation_metadata"
    EXPLICIT_DOI_LINK = "explicit_doi_link"
    DIRECT_PDF_LINK = "direct_pdf_link"
    ARTICLE_LINK = "article_link"
    PUBLICATION_LIST_TEXT = "publication_list_text"
    ARBITRARY_BODY_TEXT = "arbitrary_body_text"
    PUBLICATION_MEMBER_CONTEXT = "publication_member_context"
    REFERENCE_CONTEXT = "reference_context"
    NAVIGATION_CONTEXT = "navigation_context"
    UNVERIFIED_CONTEXT = "unverified_context"


class CollectionMembershipDecision(StrEnum):
    ELIGIBLE = "eligible"
    UNVERIFIED = "unverified"
    EXCLUDED_REFERENCE_CONTEXT = "excluded_reference_context"
    EXCLUDED_NAVIGATION_CONTEXT = "excluded_navigation_context"


class CollectionScopePolicy(StrictModel):
    allowed_origins: tuple[NonBlankStr, ...]
    allowed_path_prefixes: tuple[NonBlankStr, ...]
    allow_external_literature_links: bool = False
    max_pages: int = Field(gt=0)
    max_depth: int = Field(ge=0)
    max_resources: int = Field(gt=0)
    pagination_policy: NonBlankStr = "explicit_next_only"
    content_types: tuple[NonBlankStr, ...] = (
        "text/html",
        "application/xhtml+xml",
    )
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_policy(self) -> CollectionScopePolicy:
        from urllib.parse import urlparse

        from ..serialization import content_hash

        for field_name in (
            "allowed_origins",
            "allowed_path_prefixes",
            "content_types",
        ):
            values = getattr(self, field_name)
            if not values or len(set(values)) != len(values) or tuple(sorted(values)) != values:
                raise ValueError(f"CollectionScopePolicy {field_name} must be sorted and unique")
        for origin in self.allowed_origins:
            parsed = urlparse(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.path not in {"", "/"}
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("CollectionScopePolicy allowed_origins must be HTTP(S) origins")
        if any(not prefix.startswith("/") for prefix in self.allowed_path_prefixes):
            raise ValueError("CollectionScopePolicy path prefixes must be absolute URL paths")
        identity = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(identity):
            raise ValueError("CollectionScopePolicy content_hash is invalid")
        return self


class CollectionDefinition(StrictModel):
    collection_id: NonBlankStr
    source_kind: CollectionSourceKind
    entry_url: NonBlankStr
    name: NonBlankStr
    domain: NonBlankStr
    connector_id: NonBlankStr
    connector_version: NonBlankStr
    connector_completeness_contract: NonBlankStr
    scope_policy: CollectionScopePolicy
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionDefinition:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"collection_id", "content_hash"}
        )
        expected_id = f"literature-collection-{content_hash(identity)[:24]}"
        if self.collection_id != expected_id:
            raise ValueError("CollectionDefinition collection_id is not content-bound")
        payload = {"collection_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionDefinition content_hash is invalid")
        return self


class CollectionPageRecord(StrictModel):
    page_id: NonBlankStr
    collection_id: NonBlankStr
    requested_url: NonBlankStr
    resolved_url: NonBlankStr
    http_status: int = Field(ge=0, le=599)
    media_type: NonBlankStr
    response_sha256: Sha256Str
    page_artifact_id: NonBlankStr | None = None
    page_artifact_hash: Sha256Str | None = None
    connector_id: NonBlankStr
    connector_version: NonBlankStr
    discovered_resource_refs: tuple[NonBlankStr, ...] = ()
    pagination_refs: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionPageRecord:
        from ..serialization import content_hash

        artifact_binding = (self.page_artifact_id, self.page_artifact_hash)
        if any(value is None for value in artifact_binding) and any(
            value is not None for value in artifact_binding
        ):
            raise ValueError("CollectionPageRecord page artifact binding must be complete")
        if 200 <= self.http_status < 300 and self.page_artifact_id is None:
            raise ValueError("successful CollectionPageRecord requires exact page bytes")
        for field_name in ("discovered_resource_refs", "pagination_refs"):
            values = getattr(self, field_name)
            if len(set(values)) != len(values) or tuple(sorted(values)) != values:
                raise ValueError(f"CollectionPageRecord {field_name} must be sorted and unique")
        identity = self.model_dump(
            mode="json",
            exclude={"page_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"collection-page-{content_hash(identity)[:24]}"
        if self.page_id != expected_id:
            raise ValueError("CollectionPageRecord page_id is not content-bound")
        payload = {"page_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionPageRecord content_hash is invalid")
        return self


class CollectionPageArtifact(StrictModel):
    artifact_id: NonBlankStr
    collection_id: NonBlankStr
    requested_url: NonBlankStr
    resolved_url: NonBlankStr
    media_type: NonBlankStr
    byte_size: int = Field(ge=0)
    sha256: Sha256Str
    stored_path: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionPageArtifact:
        from ..serialization import content_hash

        stable_identity = {
            "collection_id": self.collection_id,
            "requested_url": self.requested_url,
            "resolved_url": self.resolved_url,
            "media_type": self.media_type,
            "sha256": self.sha256,
        }
        expected_id = f"collection-page-artifact-{content_hash(stable_identity)[:24]}"
        if self.artifact_id != expected_id:
            raise ValueError("CollectionPageArtifact artifact_id is not content-bound")
        expected_path = f"collection_page_artifacts/{expected_id}/content.bin"
        if self.stored_path != expected_path:
            raise ValueError("CollectionPageArtifact stored_path is not canonical")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionPageArtifact content_hash is invalid")
        return self


class DiscoveredCollectionResource(StrictModel):
    discovered_resource_id: NonBlankStr
    collection_id: NonBlankStr
    resource_kind: CollectionResourceKind
    normalized_identifier: NonBlankStr
    doi: NonBlankStr | None = None
    url: NonBlankStr | None = None
    media_type: NonBlankStr | None = None
    highest_discovery_confidence: CollectionDiscoveryConfidence
    membership_eligible: bool
    membership_decision: CollectionMembershipDecision
    membership_reason: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> DiscoveredCollectionResource:
        from ..serialization import content_hash

        if (
            self.highest_discovery_confidence == CollectionDiscoveryConfidence.LOW
            and self.membership_eligible
        ):
            raise ValueError("low-confidence discovery cannot establish collection membership")
        if self.membership_eligible != (
            self.membership_decision == CollectionMembershipDecision.ELIGIBLE
        ):
            raise ValueError("membership eligibility and decision are inconsistent")
        if self.resource_kind == CollectionResourceKind.DOI:
            if self.doi != self.normalized_identifier or self.url is not None:
                raise ValueError("DOI collection resource binding is invalid")
        elif self.url != self.normalized_identifier or self.doi is not None:
            raise ValueError("URL collection resource binding is invalid")
        stable_identity = {
            "collection_id": self.collection_id,
            "resource_kind": self.resource_kind,
            "normalized_identifier": self.normalized_identifier,
        }
        expected_id = f"collection-resource-{content_hash(stable_identity)[:24]}"
        if self.discovered_resource_id != expected_id:
            raise ValueError("DiscoveredCollectionResource ID is not content-bound")
        payload = self.model_dump(mode="json", exclude={"content_hash"}, exclude_none=True)
        if self.content_hash != content_hash(payload):
            raise ValueError("DiscoveredCollectionResource content_hash is invalid")
        return self


class CollectionResourceOccurrence(StrictModel):
    occurrence_id: NonBlankStr
    collection_id: NonBlankStr
    discovered_resource_id: NonBlankStr
    discovery_page_id: NonBlankStr
    discovery_index: int = Field(ge=0)
    original_identifier: NonBlankStr
    discovery_method: NonBlankStr
    discovery_confidence: CollectionDiscoveryConfidence
    discovery_context: CollectionDiscoveryContext
    membership_eligible: bool
    membership_decision: CollectionMembershipDecision
    membership_reason: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionResourceOccurrence:
        from ..serialization import content_hash

        if (
            self.discovery_confidence == CollectionDiscoveryConfidence.LOW
            and self.membership_eligible
        ):
            raise ValueError("low-confidence occurrence cannot establish collection membership")
        if self.membership_eligible != (
            self.membership_decision == CollectionMembershipDecision.ELIGIBLE
        ):
            raise ValueError("occurrence eligibility and decision are inconsistent")
        identity = self.model_dump(mode="json", exclude={"occurrence_id", "content_hash"})
        expected_id = f"collection-occurrence-{content_hash(identity)[:24]}"
        if self.occurrence_id != expected_id:
            raise ValueError("CollectionResourceOccurrence ID is not content-bound")
        payload = {"occurrence_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionResourceOccurrence content_hash is invalid")
        return self


class CollectionSnapshot(StrictModel):
    snapshot_id: NonBlankStr
    collection_id: NonBlankStr
    connector_id: NonBlankStr
    connector_version: NonBlankStr
    connector_completeness_contract: NonBlankStr
    scope_policy_hash: Sha256Str
    page_record_hashes: FrozenDict = Field(default_factory=FrozenDict)
    discovered_resource_hashes: FrozenDict = Field(default_factory=FrozenDict)
    resource_provenance_hashes: FrozenDict
    occurrence_hashes: FrozenDict = Field(default_factory=FrozenDict)
    visited_page_count: int = Field(ge=0)
    unique_resource_count: int = Field(ge=0)
    traversal_status: CollectionTraversalStatus
    completeness_status: CollectionCompletenessStatus
    completeness_basis: tuple[NonBlankStr, ...]
    completeness: CollectionImportStatus
    completeness_reasons: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionSnapshot:
        from ..serialization import content_hash

        for field_name in (
            "page_record_hashes",
            "discovered_resource_hashes",
            "resource_provenance_hashes",
            "occurrence_hashes",
        ):
            values = getattr(self, field_name)
            if any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for key, value in values.items()
            ):
                raise ValueError(f"CollectionSnapshot {field_name} is invalid")
        if self.visited_page_count != len(self.page_record_hashes):
            raise ValueError("CollectionSnapshot visited page count is inconsistent")
        if self.unique_resource_count != len(self.discovered_resource_hashes):
            raise ValueError("CollectionSnapshot resource count is inconsistent")
        if set(self.resource_provenance_hashes) != set(
            self.discovered_resource_hashes
        ):
            raise ValueError(
                "CollectionSnapshot resource provenance inventory is incomplete"
            )
        if set(self.occurrence_hashes) and not self.discovered_resource_hashes:
            raise ValueError("CollectionSnapshot occurrences require resources")
        if len(set(self.completeness_reasons)) != len(self.completeness_reasons):
            raise ValueError("CollectionSnapshot completeness reasons must be unique")
        if not self.completeness_basis or len(set(self.completeness_basis)) != len(
            self.completeness_basis
        ):
            raise ValueError("CollectionSnapshot completeness basis must be nonempty and unique")
        legacy_status = {
            CollectionCompletenessStatus.PROVEN_COMPLETE: CollectionImportStatus.COMPLETE,
            CollectionCompletenessStatus.REQUIRES_AUTHENTICATION: CollectionImportStatus.REQUIRES_AUTHENTICATION,
            CollectionCompletenessStatus.FAILED: CollectionImportStatus.FAILED,
        }.get(self.completeness_status, CollectionImportStatus.PARTIAL)
        if self.completeness != legacy_status:
            raise ValueError("CollectionSnapshot legacy completeness status is inconsistent")
        expected_traversal = {
            CollectionCompletenessStatus.PROVEN_COMPLETE: (
                CollectionTraversalStatus.EXHAUSTED
            ),
            CollectionCompletenessStatus.POLICY_EXHAUSTED_UNVERIFIED: (
                CollectionTraversalStatus.EXHAUSTED
            ),
            CollectionCompletenessStatus.PARTIAL: CollectionTraversalStatus.LIMITED,
            CollectionCompletenessStatus.REQUIRES_AUTHENTICATION: (
                CollectionTraversalStatus.REQUIRES_AUTHENTICATION
            ),
            CollectionCompletenessStatus.UNSUPPORTED_DYNAMIC: (
                CollectionTraversalStatus.UNSUPPORTED_DYNAMIC
            ),
            CollectionCompletenessStatus.FAILED: CollectionTraversalStatus.FAILED,
        }[self.completeness_status]
        if self.traversal_status != expected_traversal:
            raise ValueError(
                "CollectionSnapshot traversal and completeness statuses are inconsistent"
            )
        if self.completeness_status == CollectionCompletenessStatus.PROVEN_COMPLETE:
            if self.traversal_status != CollectionTraversalStatus.EXHAUSTED:
                raise ValueError("proven completeness requires exhausted traversal")
            if self.completeness_reasons:
                raise ValueError("proven complete CollectionSnapshot cannot have failure reasons")
        elif not self.completeness_reasons:
            raise ValueError("unproven CollectionSnapshot requires an explicit reason")
        identity = self.model_dump(mode="json", exclude={"snapshot_id", "content_hash"})
        expected_id = f"collection-snapshot-{content_hash(identity)[:24]}"
        if self.snapshot_id != expected_id:
            raise ValueError("CollectionSnapshot snapshot_id is not content-bound")
        payload = {"snapshot_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionSnapshot content_hash is invalid")
        return self


class CollectionDiscoveryResult(StrictModel):
    definition: CollectionDefinition
    pages: tuple[CollectionPageRecord, ...]
    resources: tuple[DiscoveredCollectionResource, ...]
    occurrences: tuple[CollectionResourceOccurrence, ...]
    snapshot: CollectionSnapshot

    @model_validator(mode="after")
    def validate_bindings(self) -> CollectionDiscoveryResult:
        from ..serialization import content_hash

        if self.snapshot.collection_id != self.definition.collection_id:
            raise ValueError("collection discovery snapshot belongs to another collection")
        if self.snapshot.scope_policy_hash != self.definition.scope_policy.content_hash:
            raise ValueError("collection snapshot scope policy hash is stale")
        if (
            self.snapshot.connector_completeness_contract
            != self.definition.connector_completeness_contract
        ):
            raise ValueError("collection snapshot completeness contract is stale")
        page_hashes = {item.page_id: item.content_hash for item in self.pages}
        resource_hashes = {
            item.discovered_resource_id: item.content_hash for item in self.resources
        }
        occurrence_hashes = {
            item.occurrence_id: item.content_hash for item in self.occurrences
        }
        resource_provenance_hashes = {
            resource_id: content_hash(
                {
                    "resource_hash": resource_hashes[resource_id],
                    "occurrence_hashes": tuple(
                        sorted(
                            item.content_hash
                            for item in self.occurrences
                            if item.discovered_resource_id == resource_id
                        )
                    ),
                }
            )
            for resource_id in resource_hashes
        }
        if dict(self.snapshot.page_record_hashes) != page_hashes:
            raise ValueError("collection discovery page inventory is incomplete")
        if dict(self.snapshot.discovered_resource_hashes) != resource_hashes:
            raise ValueError("collection discovery resource inventory is incomplete")
        if dict(self.snapshot.occurrence_hashes) != occurrence_hashes:
            raise ValueError("collection discovery occurrence inventory is incomplete")
        if (
            dict(self.snapshot.resource_provenance_hashes)
            != resource_provenance_hashes
        ):
            raise ValueError("collection discovery resource provenance is incomplete")
        known_pages = set(page_hashes)
        known_resources = set(resource_hashes)
        if any(
            item.collection_id != self.definition.collection_id
            or item.discovery_page_id not in known_pages
            or item.discovered_resource_id not in known_resources
            for item in self.occurrences
        ):
            raise ValueError("collection occurrence references an unknown page or resource")
        if any(
            item.collection_id != self.definition.collection_id
            or not set(item.discovered_resource_refs).issubset(known_resources)
            for item in self.pages
        ):
            raise ValueError("collection page references an unknown resource")
        return self


class CollectionAcquisitionLink(StrictModel):
    link_id: NonBlankStr
    collection_id: NonBlankStr
    snapshot_id: NonBlankStr
    discovered_resource_id: NonBlankStr
    discovered_resource_hash: Sha256Str
    acquisition_id: NonBlankStr
    acquisition_hash: Sha256Str
    resulting_literature_id: NonBlankStr | None = None
    acquisition_status: AcquisitionStatus
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionAcquisitionLink:
        from ..serialization import content_hash

        if (
            self.acquisition_status == AcquisitionStatus.INGESTED
        ) != (self.resulting_literature_id is not None):
            raise ValueError("collection acquisition literature binding is inconsistent")
        identity = self.model_dump(mode="json", exclude={"link_id", "content_hash"}, exclude_none=True)
        expected_id = f"collection-acquisition-link-{content_hash(identity)[:24]}"
        if self.link_id != expected_id:
            raise ValueError("CollectionAcquisitionLink link_id is not content-bound")
        payload = {"link_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionAcquisitionLink content_hash is invalid")
        return self


class CollectionResourceImportResult(StrictModel):
    result_id: NonBlankStr
    collection_id: NonBlankStr
    snapshot_id: NonBlankStr
    discovered_resource_id: NonBlankStr
    discovered_resource_hash: Sha256Str
    acquisition_link_id: NonBlankStr | None = None
    acquisition_link_hash: Sha256Str | None = None
    acquisition_status: AcquisitionStatus
    failure_code: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionResourceImportResult:
        from ..serialization import content_hash

        binding = (self.acquisition_link_id, self.acquisition_link_hash)
        if any(value is None for value in binding) and any(
            value is not None for value in binding
        ):
            raise ValueError("CollectionResourceImportResult link binding must be complete")
        if self.acquisition_link_id is None and self.failure_code is None:
            raise ValueError("resource import without acquisition link requires a failure code")
        if (
            self.acquisition_link_id is None
            and self.acquisition_status == AcquisitionStatus.INGESTED
        ):
            raise ValueError("ingested resource import requires acquisition provenance")
        if self.acquisition_link_id is not None and self.failure_code is not None:
            raise ValueError("resource import with acquisition provenance cannot add a failure code")
        identity = self.model_dump(
            mode="json", exclude={"result_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"collection-resource-import-{content_hash(identity)[:24]}"
        if self.result_id != expected_id:
            raise ValueError("CollectionResourceImportResult result_id is not content-bound")
        payload = {"result_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionResourceImportResult content_hash is invalid")
        return self


class CollectionLiteratureMembership(StrictModel):
    membership_id: NonBlankStr
    collection_id: NonBlankStr
    snapshot_id: NonBlankStr
    literature_id: NonBlankStr
    discovered_resource_refs: tuple[NonBlankStr, ...]
    acquisition_refs: tuple[NonBlankStr, ...]
    occurrence_refs: tuple[NonBlankStr, ...]
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionLiteratureMembership:
        from ..serialization import content_hash

        for field_name in (
            "discovered_resource_refs",
            "acquisition_refs",
            "occurrence_refs",
        ):
            values = getattr(self, field_name)
            if (
                not values
                or tuple(sorted(values)) != values
                or len(set(values)) != len(values)
            ):
                raise ValueError(
                    f"CollectionLiteratureMembership {field_name} must be nonempty, sorted, unique"
                )
        identity = self.model_dump(
            mode="json", exclude={"membership_id", "content_hash"}
        )
        expected_id = f"collection-membership-{content_hash(identity)[:24]}"
        if self.membership_id != expected_id:
            raise ValueError("CollectionLiteratureMembership membership_id is not content-bound")
        payload = {"membership_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionLiteratureMembership content_hash is invalid")
        return self


class CollectionImportRecord(StrictModel):
    import_id: NonBlankStr
    executed_at: datetime
    collection_id: NonBlankStr
    snapshot_id: NonBlankStr
    snapshot_hash: Sha256Str
    acquisition_link_hashes: FrozenDict = Field(default_factory=FrozenDict)
    resource_result_hashes: FrozenDict = Field(default_factory=FrozenDict)
    membership_hashes: FrozenDict = Field(default_factory=FrozenDict)
    total_resources: int = Field(ge=0)
    discovery_occurrence_count: int = Field(ge=0)
    discovered_resource_count: int = Field(ge=0)
    logical_literature_count: int = Field(ge=0)
    ingested_literature_count: int = Field(ge=0)
    ingested_count: int = Field(ge=0)
    metadata_only_count: int = Field(ge=0)
    auth_required_count: int = Field(ge=0)
    unavailable_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    unverified_membership_count: int = Field(ge=0)
    status: CollectionImportStatus
    warnings: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionImportRecord:
        from ..serialization import content_hash

        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for mapping in (
                self.acquisition_link_hashes,
                self.resource_result_hashes,
                self.membership_hashes,
            )
            for key, value in mapping.items()
        ):
            raise ValueError("CollectionImportRecord bound hash inventory is invalid")
        if self.total_resources != self.discovered_resource_count:
            raise ValueError("CollectionImportRecord discovered resource count is inconsistent")
        if self.total_resources != len(self.resource_result_hashes):
            raise ValueError("CollectionImportRecord resource result inventory is incomplete")
        if len(self.acquisition_link_hashes) > self.total_resources:
            raise ValueError("CollectionImportRecord has too many acquisition links")
        if self.logical_literature_count != len(self.membership_hashes):
            raise ValueError("CollectionImportRecord membership count is inconsistent")
        if self.ingested_literature_count != self.logical_literature_count:
            raise ValueError("ingested and logical literature counts must agree")
        if self.ingested_literature_count > self.ingested_count:
            raise ValueError("ingested literature count exceeds ingested resource count")
        if self.total_resources != sum(
            (
                self.ingested_count,
                self.metadata_only_count,
                self.auth_required_count,
                self.unavailable_count,
                self.failed_count,
                self.unverified_membership_count,
            )
        ):
            raise ValueError("CollectionImportRecord result counts are inconsistent")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("CollectionImportRecord warnings must be unique")
        identity = self.model_dump(mode="json", exclude={"import_id", "content_hash"})
        expected_id = f"collection-import-{content_hash(identity)[:24]}"
        if self.import_id != expected_id:
            raise ValueError("CollectionImportRecord import_id is not content-bound")
        payload = {"import_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionImportRecord content_hash is invalid")
        return self


class CollectionDiff(StrictModel):
    diff_id: NonBlankStr
    old_snapshot_id: NonBlankStr
    old_snapshot_hash: Sha256Str
    new_snapshot_id: NonBlankStr
    new_snapshot_hash: Sha256Str
    added_resource_ids: tuple[NonBlankStr, ...] = ()
    removed_resource_ids: tuple[NonBlankStr, ...] = ()
    changed_resource_ids: tuple[NonBlankStr, ...] = ()
    unchanged_resource_ids: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CollectionDiff:
        from ..serialization import content_hash

        groups = (
            self.added_resource_ids,
            self.removed_resource_ids,
            self.changed_resource_ids,
            self.unchanged_resource_ids,
        )
        if any(tuple(sorted(group)) != group or len(set(group)) != len(group) for group in groups):
            raise ValueError("CollectionDiff resource IDs must be sorted and unique")
        if any(set(left) & set(right) for index, left in enumerate(groups) for right in groups[index + 1 :]):
            raise ValueError("CollectionDiff resource groups must be disjoint")
        identity = self.model_dump(mode="json", exclude={"diff_id", "content_hash"})
        expected_id = f"collection-diff-{content_hash(identity)[:24]}"
        if self.diff_id != expected_id:
            raise ValueError("CollectionDiff diff_id is not content-bound")
        payload = {"diff_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("CollectionDiff content_hash is invalid")
        return self


class CollectionImportOutcome(StrictModel):
    collection_id: NonBlankStr
    snapshot_id: NonBlankStr
    completeness: CollectionImportStatus
    traversal_status: CollectionTraversalStatus
    completeness_status: CollectionCompletenessStatus
    completeness_basis: tuple[NonBlankStr, ...]
    visited_pages: int = Field(ge=0)
    discovered_resources: int = Field(ge=0)
    unique_resources: int = Field(ge=0)
    discovery_occurrence_count: int = Field(ge=0)
    discovered_resource_count: int = Field(ge=0)
    logical_literature_count: int = Field(ge=0)
    ingested_literature_count: int = Field(ge=0)
    ingested: int = Field(ge=0)
    metadata_only: int = Field(ge=0)
    requires_authentication: int = Field(ge=0)
    unavailable: int = Field(ge=0)
    failed: int = Field(ge=0)
    unverified_membership: int = Field(ge=0)
    import_id: NonBlankStr
    warnings: tuple[NonBlankStr, ...] = ()
