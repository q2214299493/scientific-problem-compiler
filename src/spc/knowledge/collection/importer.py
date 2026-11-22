"""Import eligible resources and persist collection memberships and import history."""
from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from typing import Callable

from pydantic import ValidationError

from ...immutable import FrozenDict
from ...models import (
    AcquisitionStatus,
    CollectionAcquisitionLink,
    CollectionCompletenessStatus,
    CollectionDefinition,
    CollectionDiscoveryResult,
    CollectionImportOutcome,
    CollectionImportRecord,
    CollectionImportStatus,
    CollectionLiteratureMembership,
    CollectionResourceImportResult,
    CollectionSnapshot,
    DiscoveredCollectionResource,
)
from ...repositories import (
    EvidenceStore,
    KnowledgeRepositories,
)
from ..acquisition import (
    AcquisitionError,
    LiteratureAcquisitionService,
    SafeHTTPFetcher,
)
from ..acquisition.transport import restrict_http_urls
from .connectors import (
    CollectionConnector,
    GenericHTMLCollectionConnector,
)
from .snapshots import (
    _make_bound,
    diff_collection_snapshots,
)
from .urls import CollectionRequestScope

EXPECTED_COLLECTION_ACQUISITION_EXCEPTIONS = (
    AcquisitionError,
    FileExistsError,
    OSError,
    ValidationError,
)


class CollectionImportService:
    def __init__(
        self,
        *,
        fetcher: SafeHTTPFetcher | None = None,
        connector: CollectionConnector | None = None,
        acquisition_service: LiteratureAcquisitionService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.fetcher = fetcher or SafeHTTPFetcher()
        self.connector = connector or GenericHTMLCollectionConnector()
        self.acquisition_service = acquisition_service or LiteratureAcquisitionService(
            self.fetcher
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def run(
        self,
        definition: CollectionDefinition,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        *,
        previous_snapshot: CollectionSnapshot | None = None,
    ) -> CollectionImportOutcome:
        with restrict_http_urls(CollectionRequestScope(definition.scope_policy)):
            discovery = self.connector.discover(definition, repositories, self.fetcher)
        resource_scope = CollectionRequestScope(definition.scope_policy, literature_resource=True)
        links: list[CollectionAcquisitionLink] = []
        results: list[CollectionResourceImportResult] = []
        warnings = list(discovery.snapshot.completeness_reasons)
        counts = {
            "ingested": 0,
            "metadata_only": 0,
            "auth": 0,
            "unavailable": 0,
            "failed": 0,
            "unverified": 0,
        }
        for resource in discovery.resources:
            if not resource.membership_eligible:
                result = self._failed_resource_result(
                    definition,
                    discovery.snapshot,
                    resource,
                    status=AcquisitionStatus.DISCOVERED,
                    failure_code="UNVERIFIED_COLLECTION_MEMBERSHIP",
                )
                repositories.collection_resource_import_results.put(
                    result.result_id, result
                )
                results.append(result)
                counts["unverified"] += 1
                continue
            source = resource.doi or resource.url
            if source is None:
                result = self._failed_resource_result(
                    definition,
                    discovery.snapshot,
                    resource,
                    status=AcquisitionStatus.FAILED,
                    failure_code="MISSING_ACQUISITION_INPUT",
                )
                repositories.collection_resource_import_results.put(
                    result.result_id, result
                )
                results.append(result)
                counts["failed"] += 1
                continue
            try:
                with restrict_http_urls(resource_scope):
                    outcome = self.acquisition_service.add(
                        source, definition.domain, repositories, evidence_store,
                    )
                acquisition = repositories.literature_acquisitions.get(
                    outcome.acquisition_id
                )
            except EXPECTED_COLLECTION_ACQUISITION_EXCEPTIONS as error:
                failure_code = f"EXPECTED_ACQUISITION_FAILURE:{type(error).__name__}"
                result = self._failed_resource_result(
                    definition,
                    discovery.snapshot,
                    resource,
                    status=AcquisitionStatus.FAILED,
                    failure_code=failure_code,
                )
                repositories.collection_resource_import_results.put(
                    result.result_id, result
                )
                results.append(result)
                counts["failed"] += 1
                warnings.append(f"{resource.discovered_resource_id}:{failure_code}")
                continue
            link = _make_bound(
                CollectionAcquisitionLink,
                "collection-acquisition-link",
                {
                    "collection_id": definition.collection_id,
                    "snapshot_id": discovery.snapshot.snapshot_id,
                    "discovered_resource_id": resource.discovered_resource_id,
                    "discovered_resource_hash": resource.content_hash,
                    "acquisition_id": acquisition.acquisition_id,
                    "acquisition_hash": acquisition.content_hash,
                    "resulting_literature_id": outcome.literature_id,
                    "acquisition_status": outcome.acquisition_status,
                },
            )
            repositories.collection_acquisition_links.put(link.link_id, link)
            links.append(link)
            result = _make_bound(
                CollectionResourceImportResult,
                "collection-resource-import",
                {
                    "collection_id": definition.collection_id,
                    "snapshot_id": discovery.snapshot.snapshot_id,
                    "discovered_resource_id": resource.discovered_resource_id,
                    "discovered_resource_hash": resource.content_hash,
                    "acquisition_link_id": link.link_id,
                    "acquisition_link_hash": link.content_hash,
                    "acquisition_status": outcome.acquisition_status,
                },
            )
            repositories.collection_resource_import_results.put(result.result_id, result)
            results.append(result)
            self._increment_count(counts, outcome.acquisition_status)
            warnings.extend(
                f"{resource.discovered_resource_id}:{warning}"
                for warning in outcome.warnings
            )
        memberships = self._build_memberships(discovery, links, repositories)
        total = len(discovery.resources)
        if (
            discovery.snapshot.completeness_status
            == CollectionCompletenessStatus.REQUIRES_AUTHENTICATION
            and total == 0
        ):
            status = CollectionImportStatus.REQUIRES_AUTHENTICATION
        elif (
            discovery.snapshot.completeness_status
            == CollectionCompletenessStatus.PROVEN_COMPLETE
            and counts["auth"]
            == counts["unavailable"]
            == counts["failed"]
            == counts["unverified"]
            == 0
        ):
            status = CollectionImportStatus.COMPLETE
        elif (
            discovery.snapshot.completeness_status
            == CollectionCompletenessStatus.FAILED
            and total == 0
        ):
            status = CollectionImportStatus.FAILED
        else:
            status = CollectionImportStatus.PARTIAL
        execution_time = self.clock().astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        record = _make_bound(
            CollectionImportRecord,
            "collection-import",
            {
                "executed_at": execution_time,
                "collection_id": definition.collection_id,
                "snapshot_id": discovery.snapshot.snapshot_id,
                "snapshot_hash": discovery.snapshot.content_hash,
                "acquisition_link_hashes": FrozenDict(
                    {item.link_id: item.content_hash for item in links}
                ),
                "resource_result_hashes": FrozenDict(
                    {item.result_id: item.content_hash for item in results}
                ),
                "membership_hashes": FrozenDict(
                    {item.membership_id: item.content_hash for item in memberships}
                ),
                "total_resources": total,
                "discovery_occurrence_count": len(discovery.occurrences),
                "discovered_resource_count": total,
                "logical_literature_count": len(memberships),
                "ingested_literature_count": len(memberships),
                "ingested_count": counts["ingested"],
                "metadata_only_count": counts["metadata_only"],
                "auth_required_count": counts["auth"],
                "unavailable_count": counts["unavailable"],
                "failed_count": counts["failed"],
                "unverified_membership_count": counts["unverified"],
                "status": status,
                "warnings": tuple(dict.fromkeys(warnings)),
            },
        )
        repositories.collection_imports.put(record.import_id, record)
        if previous_snapshot is not None:
            diff = diff_collection_snapshots(previous_snapshot, discovery.snapshot)
            repositories.collection_diffs.put(diff.diff_id, diff)
        return CollectionImportOutcome(
            collection_id=definition.collection_id,
            snapshot_id=discovery.snapshot.snapshot_id,
            completeness=discovery.snapshot.completeness,
            traversal_status=discovery.snapshot.traversal_status,
            completeness_status=discovery.snapshot.completeness_status,
            completeness_basis=discovery.snapshot.completeness_basis,
            visited_pages=discovery.snapshot.visited_page_count,
            discovered_resources=len(discovery.occurrences),
            unique_resources=total,
            discovery_occurrence_count=len(discovery.occurrences),
            discovered_resource_count=total,
            logical_literature_count=len(memberships),
            ingested_literature_count=len(memberships),
            ingested=counts["ingested"],
            metadata_only=counts["metadata_only"],
            requires_authentication=counts["auth"],
            unavailable=counts["unavailable"],
            failed=counts["failed"],
            unverified_membership=counts["unverified"],
            import_id=record.import_id,
            warnings=record.warnings,
        )

    @staticmethod
    def _failed_resource_result(
        definition: CollectionDefinition,
        snapshot: CollectionSnapshot,
        resource: DiscoveredCollectionResource,
        *,
        status: AcquisitionStatus,
        failure_code: str,
    ) -> CollectionResourceImportResult:
        return _make_bound(
            CollectionResourceImportResult,
            "collection-resource-import",
            {
                "collection_id": definition.collection_id,
                "snapshot_id": snapshot.snapshot_id,
                "discovered_resource_id": resource.discovered_resource_id,
                "discovered_resource_hash": resource.content_hash,
                "acquisition_status": status,
                "failure_code": failure_code,
            },
        )

    @staticmethod
    def _increment_count(counts: dict[str, int], status: AcquisitionStatus) -> None:
        if status == AcquisitionStatus.INGESTED:
            counts["ingested"] += 1
        elif status == AcquisitionStatus.METADATA_ONLY:
            counts["metadata_only"] += 1
        elif status == AcquisitionStatus.REQUIRES_AUTHENTICATION:
            counts["auth"] += 1
        elif status in {
            AcquisitionStatus.FULLTEXT_UNAVAILABLE,
            AcquisitionStatus.UNSUPPORTED_MEDIA,
        }:
            counts["unavailable"] += 1
        else:
            counts["failed"] += 1

    @staticmethod
    def _build_memberships(
        discovery: CollectionDiscoveryResult,
        links: list[CollectionAcquisitionLink],
        repositories: KnowledgeRepositories,
    ) -> tuple[CollectionLiteratureMembership, ...]:
        grouped: dict[str, list[CollectionAcquisitionLink]] = {}
        for link in links:
            if link.resulting_literature_id is not None:
                grouped.setdefault(link.resulting_literature_id, []).append(link)
        memberships: list[CollectionLiteratureMembership] = []
        for literature_id, literature_links in sorted(grouped.items()):
            resource_ids = {
                link.discovered_resource_id for link in literature_links
            }
            occurrence_ids = {
                occurrence.occurrence_id
                for occurrence in discovery.occurrences
                if occurrence.discovered_resource_id in resource_ids
            }
            membership = _make_bound(
                CollectionLiteratureMembership,
                "collection-membership",
                {
                    "collection_id": discovery.definition.collection_id,
                    "snapshot_id": discovery.snapshot.snapshot_id,
                    "literature_id": literature_id,
                    "discovered_resource_refs": tuple(sorted(resource_ids)),
                    "acquisition_refs": tuple(
                        sorted({link.acquisition_id for link in literature_links})
                    ),
                    "occurrence_refs": tuple(sorted(occurrence_ids)),
                },
            )
            repositories.collection_literature_memberships.put(
                membership.membership_id, membership
            )
            memberships.append(membership)
        return tuple(memberships)
