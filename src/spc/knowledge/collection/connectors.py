"""Traverse scoped collection pages and record discovery provenance."""
from __future__ import annotations

import hashlib
from typing import Protocol

from ...models import (
    CollectionDefinition,
    CollectionDiscoveryConfidence,
    CollectionDiscoveryResult,
    CollectionMembershipDecision,
    CollectionPageRecord,
    CollectionResourceOccurrence,
    CollectionSourceKind,
    DiscoveredCollectionResource,
)
from ...repositories import KnowledgeRepositories
from ...serialization import content_hash
from ..acquisition import (
    SafeHTTPError,
    SafeHTTPFetcher,
)
from ..acquisition.transport import restrict_http_urls
from .discovery import (
    _Candidate,
    _extract_page,
)
from .snapshots import (
    _make_bound,
    finish_discovery,
)
from .urls import (
    CollectionError,
    CollectionRequestScope,
    canonicalize_url,
    url_in_scope,
)


class CollectionConnector(Protocol):
    connector_id: str
    connector_version: str

    def discover(
        self,
        definition: CollectionDefinition,
        repositories: KnowledgeRepositories,
        fetcher: SafeHTTPFetcher,
    ) -> CollectionDiscoveryResult:
        ...


class GenericHTMLCollectionConnector:
    connector_id = "generic-html-collection"
    connector_version = "1.1.0"
    completeness_contract = "explicit-pagination-terminal-v1"

    def discover(
        self,
        definition: CollectionDefinition,
        repositories: KnowledgeRepositories,
        fetcher: SafeHTTPFetcher,
    ) -> CollectionDiscoveryResult:
        if (
            definition.connector_id != self.connector_id
            or definition.connector_version != self.connector_version
            or definition.connector_completeness_contract
            != self.completeness_contract
        ):
            raise CollectionError("collection definition does not bind this connector")
        if definition.source_kind == CollectionSourceKind.LOCAL_MANIFEST:
            raise CollectionError("GenericHTMLCollectionConnector does not read local manifests")
        repositories.collection_definitions.put(definition.collection_id, definition)
        policy = definition.scope_policy
        page_scope = CollectionRequestScope(policy)
        queue: list[tuple[str, int]] = [(canonicalize_url(definition.entry_url), 0)]
        queued = {queue[0][0]}
        visited: set[str] = set()
        pages: list[CollectionPageRecord] = []
        resources: dict[tuple[int, str], DiscoveredCollectionResource] = {}
        occurrences: list[CollectionResourceOccurrence] = []
        reasons: list[str] = []
        authentication_blocked = False
        resource_limit_reached = False
        saw_explicit_pagination = False
        while queue and len(pages) < policy.max_pages:
            requested_url, depth = queue.pop(0)
            queued.discard(requested_url)
            if requested_url in visited:
                if "pagination_cycle_detected" not in reasons:
                    reasons.append("pagination_cycle_detected")
                continue
            visited.add(requested_url)
            if not self._page_in_scope(requested_url, policy):
                reasons.append("unresolved_pagination_outside_scope")
                continue
            try:
                with restrict_http_urls(page_scope):
                    response = fetcher.fetch(
                        requested_url, allowed_content_types=frozenset(policy.content_types),
                    )
            except SafeHTTPError as error:
                authentication_blocked = error.code == "REQUIRES_AUTHENTICATION"
                reason = (
                    "collection_requires_authentication"
                    if authentication_blocked
                    else f"page_retrieval_failed:{error.code}"
                )
                reasons.append(reason)
                page = self._page_record(
                    definition,
                    requested_url=requested_url,
                    resolved_url=requested_url,
                    http_status=error.status or 0,
                    media_type="unavailable",
                    response_sha256=hashlib.sha256(b"").hexdigest(),
                    resource_ids=(),
                    pagination=(),
                )
                repositories.collection_pages.put(page.page_id, page)
                pages.append(page)
                continue
            try:
                candidates, pagination, dynamic_only, unresolved_pagination = _extract_page(
                    response.body, response.url, policy
                )
            except CollectionError:
                candidates, pagination, dynamic_only, unresolved_pagination = (), (), False, False
                reasons.append("unsupported_page_encoding")
            if pagination:
                saw_explicit_pagination = True
            page_resource_ids: list[str] = []
            accepted_candidates: list[_Candidate] = []
            for candidate in candidates:
                if candidate.key not in resources and len(resources) >= policy.max_resources:
                    resource_limit_reached = True
                    continue
                resource = resources.get(candidate.key)
                if resource is None:
                    stable_identity = {
                        "collection_id": definition.collection_id,
                        "resource_kind": candidate.kind,
                        "normalized_identifier": candidate.normalized_identifier,
                    }
                    resource_id = f"collection-resource-{content_hash(stable_identity)[:24]}"
                    payload = {
                        "discovered_resource_id": resource_id,
                        "collection_id": definition.collection_id,
                        "resource_kind": candidate.kind,
                        "normalized_identifier": candidate.normalized_identifier,
                        "doi": candidate.doi,
                        "url": candidate.url,
                        "media_type": candidate.media_type,
                        "highest_discovery_confidence": candidate.discovery_confidence,
                        "membership_eligible": candidate.membership_eligible,
                        "membership_decision": candidate.membership_decision,
                        "membership_reason": candidate.membership_reason,
                    }
                    payload = {key: value for key, value in payload.items() if value is not None}
                    resource = DiscoveredCollectionResource(
                        **payload, content_hash=content_hash(payload)
                    )
                    resources[candidate.key] = resource
                else:
                    confidence_rank = {
                        CollectionDiscoveryConfidence.LOW: 0,
                        CollectionDiscoveryConfidence.MEDIUM: 1,
                        CollectionDiscoveryConfidence.HIGH: 2,
                    }
                    highest = (
                        candidate.discovery_confidence
                        if confidence_rank[candidate.discovery_confidence]
                        > confidence_rank[resource.highest_discovery_confidence]
                        else resource.highest_discovery_confidence
                    )
                    membership_eligible = (
                        resource.membership_eligible or candidate.membership_eligible
                    )
                    decision_rank = {
                        CollectionMembershipDecision.EXCLUDED_NAVIGATION_CONTEXT: 0,
                        CollectionMembershipDecision.EXCLUDED_REFERENCE_CONTEXT: 0,
                        CollectionMembershipDecision.UNVERIFIED: 1,
                        CollectionMembershipDecision.ELIGIBLE: 2,
                    }
                    current_decision = resource.membership_decision
                    candidate_wins = decision_rank[candidate.membership_decision] > (
                        decision_rank[current_decision]
                    )
                    membership_decision = (
                        candidate.membership_decision
                        if candidate_wins
                        else current_decision
                    )
                    membership_reason = (
                        candidate.membership_reason
                        if candidate_wins
                        else resource.membership_reason
                    )
                    if (
                        highest != resource.highest_discovery_confidence
                        or membership_eligible != resource.membership_eligible
                        or membership_decision != resource.membership_decision
                        or membership_reason != resource.membership_reason
                    ):
                        payload = resource.model_dump(
                            mode="json", exclude={"content_hash"}, exclude_none=True
                        )
                        payload.update(
                            highest_discovery_confidence=highest,
                            membership_eligible=membership_eligible,
                            membership_decision=membership_decision,
                            membership_reason=membership_reason,
                        )
                        resource = DiscoveredCollectionResource(
                            **payload, content_hash=content_hash(payload)
                        )
                        resources[candidate.key] = resource
                page_resource_ids.append(resource.discovered_resource_id)
                accepted_candidates.append(candidate)
            if dynamic_only:
                reasons.append("unsupported_dynamic_collection")
            if unresolved_pagination:
                reasons.append("unresolved_pagination")
            valid_pagination: list[str] = []
            for next_url in pagination:
                if not self._page_in_scope(next_url, policy):
                    reasons.append("unresolved_pagination_outside_scope")
                    continue
                if next_url in visited or next_url in queued:
                    reasons.append("pagination_cycle_detected")
                    continue
                if depth >= policy.max_depth:
                    reasons.append("max_depth_reached")
                    continue
                valid_pagination.append(next_url)
                queue.append((next_url, depth + 1))
                queued.add(next_url)
            queue.sort(key=lambda item: (item[1], item[0]))
            media_type = (
                response.headers.get("content-type", "").split(";", 1)[0]
                or "unknown"
            )
            page_artifact = repositories.collection_page_artifacts.put(
                response.body,
                collection_id=definition.collection_id,
                requested_url=canonicalize_url(requested_url),
                resolved_url=canonicalize_url(response.url),
                media_type=media_type,
            )
            page = self._page_record(
                definition,
                requested_url=requested_url,
                resolved_url=response.url,
                http_status=response.status,
                media_type=media_type,
                response_sha256=hashlib.sha256(response.body).hexdigest(),
                page_artifact_id=page_artifact.artifact_id,
                page_artifact_hash=page_artifact.content_hash,
                resource_ids=tuple(sorted(set(page_resource_ids))),
                pagination=tuple(sorted(set(pagination))),
            )
            repositories.collection_pages.put(page.page_id, page)
            pages.append(page)
            for index, (candidate, resource_id) in enumerate(
                zip(accepted_candidates, page_resource_ids)
            ):
                occurrence = _make_bound(
                    CollectionResourceOccurrence,
                    "collection-occurrence",
                    {
                        "collection_id": definition.collection_id,
                        "discovered_resource_id": resource_id,
                        "discovery_page_id": page.page_id,
                        "discovery_index": index,
                        "original_identifier": candidate.original_identifier,
                        "discovery_method": candidate.discovery_method,
                        "discovery_confidence": candidate.discovery_confidence,
                        "discovery_context": candidate.discovery_context,
                        "membership_eligible": candidate.membership_eligible,
                        "membership_decision": candidate.membership_decision,
                        "membership_reason": candidate.membership_reason,
                    },
                )
                repositories.collection_occurrences.put(occurrence.occurrence_id, occurrence)
                occurrences.append(occurrence)
        if queue:
            reasons.append("max_pages_reached")
        if resource_limit_reached:
            reasons.append("max_resources_reached")
        return finish_discovery(
            definition, repositories, pages, resources.values(), occurrences,
            reasons=reasons,
            authentication_blocked=authentication_blocked,
            saw_explicit_pagination=saw_explicit_pagination,
        )

    _page_in_scope = staticmethod(url_in_scope)

    @staticmethod
    def _page_record(
        definition: CollectionDefinition,
        *,
        requested_url: str,
        resolved_url: str,
        http_status: int,
        media_type: str,
        response_sha256: str,
        page_artifact_id: str | None = None,
        page_artifact_hash: str | None = None,
        resource_ids: tuple[str, ...],
        pagination: tuple[str, ...],
    ) -> CollectionPageRecord:
        return _make_bound(
            CollectionPageRecord,
            "collection-page",
            {
                "collection_id": definition.collection_id,
                "requested_url": canonicalize_url(requested_url),
                "resolved_url": canonicalize_url(resolved_url),
                "http_status": http_status,
                "media_type": media_type,
                "response_sha256": response_sha256,
                "page_artifact_id": page_artifact_id,
                "page_artifact_hash": page_artifact_hash,
                "connector_id": definition.connector_id,
                "connector_version": definition.connector_version,
                "discovered_resource_refs": tuple(sorted(set(resource_ids))),
                "pagination_refs": tuple(sorted(set(pagination))),
            },
        )
