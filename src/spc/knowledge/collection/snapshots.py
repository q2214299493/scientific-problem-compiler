"""Bind collection records, finalize discovery snapshots and compare crawls."""
from __future__ import annotations

from collections.abc import Iterable

from ...immutable import FrozenDict
from ...models import (
    CollectionCompletenessStatus,
    CollectionDefinition,
    CollectionDiff,
    CollectionDiscoveryResult,
    CollectionImportStatus,
    CollectionPageRecord,
    CollectionResourceOccurrence,
    CollectionSnapshot,
    CollectionTraversalStatus,
    DiscoveredCollectionResource,
)
from ...repositories import KnowledgeRepositories
from ...serialization import content_hash
from .urls import CollectionError


def _make_bound(model_type, prefix: str, identity: dict):
    identity = {key: value for key, value in identity.items() if value is not None}
    record_id = f"{prefix}-{content_hash(identity)[:24]}"
    id_field = {
        "collection-page": "page_id",
        "collection-resource": "discovered_resource_id",
        "collection-occurrence": "occurrence_id",
        "collection-snapshot": "snapshot_id",
        "collection-acquisition-link": "link_id",
        "collection-resource-import": "result_id",
        "collection-membership": "membership_id",
        "collection-import": "import_id",
        "collection-diff": "diff_id",
    }[prefix]
    payload = {id_field: record_id, **identity}
    return model_type(**payload, content_hash=content_hash(payload))


def diff_collection_snapshots(
    old: CollectionSnapshot,
    new: CollectionSnapshot,
) -> CollectionDiff:
    if old.collection_id != new.collection_id:
        raise CollectionError("collection snapshots belong to different collections")
    old_ids = set(old.discovered_resource_hashes)
    new_ids = set(new.discovered_resource_hashes)
    shared_ids = old_ids & new_ids
    changed_ids = {
        resource_id
        for resource_id in shared_ids
        if (
            old.discovered_resource_hashes[resource_id]
            != new.discovered_resource_hashes[resource_id]
            or old.resource_provenance_hashes.get(
                resource_id, old.discovered_resource_hashes[resource_id]
            )
            != new.resource_provenance_hashes.get(
                resource_id, new.discovered_resource_hashes[resource_id]
            )
        )
    }
    return _make_bound(
        CollectionDiff,
        "collection-diff",
        {
            "old_snapshot_id": old.snapshot_id,
            "old_snapshot_hash": old.content_hash,
            "new_snapshot_id": new.snapshot_id,
            "new_snapshot_hash": new.content_hash,
            "added_resource_ids": tuple(sorted(new_ids - old_ids)),
            "removed_resource_ids": tuple(sorted(old_ids - new_ids)),
            "changed_resource_ids": tuple(sorted(changed_ids)),
            "unchanged_resource_ids": tuple(sorted(shared_ids - changed_ids)),
        },
    )


def finish_discovery(
    definition: CollectionDefinition,
    repositories: KnowledgeRepositories,
    pages: list[CollectionPageRecord],
    resources: Iterable[DiscoveredCollectionResource],
    occurrences: list[CollectionResourceOccurrence],
    *,
    reasons: list[str],
    authentication_blocked: bool,
    saw_explicit_pagination: bool,
) -> CollectionDiscoveryResult:
    unique_reasons = tuple(dict.fromkeys(reasons))
    successful_pages = sum(200 <= page.http_status < 300 for page in pages)
    if authentication_blocked and successful_pages == 0:
        traversal_status = CollectionTraversalStatus.REQUIRES_AUTHENTICATION
        completeness_status = CollectionCompletenessStatus.REQUIRES_AUTHENTICATION
        completeness_basis = ("authentication_prevented_scope_exhaustion",)
    elif successful_pages == 0 and unique_reasons:
        traversal_status = CollectionTraversalStatus.FAILED
        completeness_status = CollectionCompletenessStatus.FAILED
        completeness_basis = ("no_collection_page_was_retrieved",)
    elif "unsupported_dynamic_collection" in unique_reasons:
        traversal_status = CollectionTraversalStatus.UNSUPPORTED_DYNAMIC
        completeness_status = CollectionCompletenessStatus.UNSUPPORTED_DYNAMIC
        completeness_basis = ("generic_connector_cannot_execute_dynamic_continuation",)
    elif unique_reasons:
        traversal_status = CollectionTraversalStatus.LIMITED
        completeness_status = CollectionCompletenessStatus.PARTIAL
        completeness_basis = tuple(f"scope_not_exhausted:{reason}" for reason in unique_reasons)
    elif saw_explicit_pagination:
        traversal_status = CollectionTraversalStatus.EXHAUSTED
        completeness_status = CollectionCompletenessStatus.PROVEN_COMPLETE
        completeness_basis = ("explicit_pagination_chain_reached_terminal_page",)
    else:
        traversal_status = CollectionTraversalStatus.EXHAUSTED
        completeness_status = CollectionCompletenessStatus.POLICY_EXHAUSTED_UNVERIFIED
        completeness_basis = ("no_explicit_completeness_evidence",)
        unique_reasons = ("collection_completeness_unverified",)
    completeness = {
        CollectionCompletenessStatus.PROVEN_COMPLETE: CollectionImportStatus.COMPLETE,
        CollectionCompletenessStatus.REQUIRES_AUTHENTICATION: CollectionImportStatus.REQUIRES_AUTHENTICATION,
        CollectionCompletenessStatus.FAILED: CollectionImportStatus.FAILED,
    }.get(completeness_status, CollectionImportStatus.PARTIAL)
    ordered_pages = tuple(sorted(pages, key=lambda item: item.page_id))
    ordered_resources = tuple(
        sorted(resources, key=lambda item: item.discovered_resource_id)
    )
    for resource in ordered_resources:
        repositories.collection_resources.put(
            resource.discovered_resource_id, resource
        )
    ordered_occurrences = tuple(
        sorted(occurrences, key=lambda item: item.occurrence_id)
    )
    snapshot = _make_bound(
        CollectionSnapshot,
        "collection-snapshot",
        {
            "collection_id": definition.collection_id,
            "connector_id": definition.connector_id,
            "connector_version": definition.connector_version,
            "connector_completeness_contract": definition.connector_completeness_contract,
            "scope_policy_hash": definition.scope_policy.content_hash,
            "page_record_hashes": FrozenDict(
                {item.page_id: item.content_hash for item in ordered_pages}
            ),
            "discovered_resource_hashes": FrozenDict(
                {
                    item.discovered_resource_id: item.content_hash
                    for item in ordered_resources
                }
            ),
            "resource_provenance_hashes": FrozenDict(
                {
                    resource.discovered_resource_id: content_hash(
                        {
                            "resource_hash": resource.content_hash,
                            "occurrence_hashes": tuple(
                                sorted(
                                    occurrence.content_hash
                                    for occurrence in ordered_occurrences
                                    if occurrence.discovered_resource_id
                                    == resource.discovered_resource_id
                                )
                            ),
                        }
                    )
                    for resource in ordered_resources
                }
            ),
            "occurrence_hashes": FrozenDict(
                {item.occurrence_id: item.content_hash for item in ordered_occurrences}
            ),
            "visited_page_count": len(ordered_pages),
            "unique_resource_count": len(ordered_resources),
            "traversal_status": traversal_status,
            "completeness_status": completeness_status,
            "completeness_basis": completeness_basis,
            "completeness": completeness,
            "completeness_reasons": unique_reasons,
        },
    )
    repositories.collection_snapshots.put(snapshot.snapshot_id, snapshot)
    return CollectionDiscoveryResult(
        definition=definition,
        pages=ordered_pages,
        resources=ordered_resources,
        occurrences=ordered_occurrences,
        snapshot=snapshot,
    )
