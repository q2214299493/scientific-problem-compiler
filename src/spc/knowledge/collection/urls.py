"""Canonical collection URLs, scope policy and definitions."""
from __future__ import annotations

import posixpath
from dataclasses import dataclass
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urlparse,
    urlunparse,
)

from ...models import (
    CollectionDefinition,
    CollectionScopePolicy,
    CollectionSourceKind,
)
from ...serialization import content_hash
from ..acquisition import SUPPORTED_HTML_TYPES, SafeHTTPError


class CollectionError(ValueError):
    pass


def canonicalize_url(value: str) -> str:
    parsed = urlparse(value.strip())
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise CollectionError("collection URLs must use HTTP/HTTPS and include a host")
    host = parsed.hostname.rstrip(".").casefold()
    try:
        port = parsed.port
    except ValueError as error:
        raise CollectionError("collection URL port is invalid") from error
    default_port = 443 if scheme == "https" else 80
    netloc = host if port in {None, default_port} else f"{host}:{port}"
    decoded_path = unquote(parsed.path or "/")
    normalized_path = posixpath.normpath(decoded_path)
    if decoded_path.endswith("/") and normalized_path != "/":
        normalized_path += "/"
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    path = quote(normalized_path, safe="/%:@+~!$&'()*,-.;=")
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)), doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def canonical_origin(value: str) -> str:
    parsed = urlparse(canonicalize_url(value))
    return f"{parsed.scheme}://{parsed.netloc}"


def _canonical_path_prefix(value: str) -> str:
    path = unquote(value.strip())
    if not path.startswith("/"):
        path = f"/{path}"
    normalized = posixpath.normpath(path)
    if path.endswith("/") and normalized != "/":
        normalized += "/"
    return normalized


def _default_path_prefix(entry_path: str) -> str:
    canonical = _canonical_path_prefix(entry_path or "/")
    if canonical == "/" or canonical.endswith("/"):
        return canonical
    leaf = posixpath.basename(canonical)
    if "." in leaf:
        parent = posixpath.dirname(canonical) or "/"
        return parent if parent == "/" else f"{parent}/"
    return canonical


def path_matches_prefix(path: str, prefix: str) -> bool:
    candidate = _canonical_path_prefix(path)
    boundary = _canonical_path_prefix(prefix)
    if boundary == "/":
        return True
    base = boundary.rstrip("/")
    return candidate == base or candidate.startswith(f"{base}/")


def make_collection_scope_policy(
    entry_url: str,
    *,
    max_pages: int = 100,
    max_depth: int = 10,
    max_resources: int = 1000,
    allow_external_literature_links: bool = False,
    allowed_origins: tuple[str, ...] | None = None,
    allowed_path_prefixes: tuple[str, ...] | None = None,
) -> CollectionScopePolicy:
    canonical_entry = canonicalize_url(entry_url)
    parsed = urlparse(canonical_entry)
    origin_values = tuple(
        sorted({canonical_origin(item) for item in (allowed_origins or (canonical_entry,))})
    )
    prefix_values = allowed_path_prefixes or (_default_path_prefix(parsed.path),)
    prefixes = tuple(sorted({_canonical_path_prefix(item) for item in prefix_values}))
    identity = {
        "allowed_origins": origin_values,
        "allowed_path_prefixes": prefixes,
        "allow_external_literature_links": allow_external_literature_links,
        "max_pages": max_pages,
        "max_depth": max_depth,
        "max_resources": max_resources,
        "pagination_policy": "explicit_next_only",
        "content_types": tuple(sorted(SUPPORTED_HTML_TYPES)),
    }
    return CollectionScopePolicy(**identity, content_hash=content_hash(identity))


def make_collection_definition(
    entry_url: str,
    domain: str,
    *,
    name: str | None = None,
    source_kind: CollectionSourceKind = CollectionSourceKind.COLLECTION_URL,
    connector_id: str = "generic-html-collection",
    connector_version: str = "1.1.0",
    connector_completeness_contract: str = "explicit-pagination-terminal-v1",
    scope_policy: CollectionScopePolicy | None = None,
) -> CollectionDefinition:
    canonical_entry = canonicalize_url(entry_url)
    policy = scope_policy or make_collection_scope_policy(canonical_entry)
    identity = {
        "source_kind": source_kind,
        "entry_url": canonical_entry,
        "name": name or canonical_entry,
        "domain": domain,
        "connector_id": connector_id,
        "connector_version": connector_version,
        "connector_completeness_contract": connector_completeness_contract,
        "scope_policy": policy,
    }
    collection_id = f"literature-collection-{content_hash(identity)[:24]}"
    payload = {"collection_id": collection_id, **identity}
    return CollectionDefinition(**payload, content_hash=content_hash(payload))


def url_in_scope(
    url: str, policy: CollectionScopePolicy, *, literature_resource: bool = False,
) -> bool:
    parsed = urlparse(canonicalize_url(url))
    allowed_origin = canonical_origin(url) in policy.allowed_origins
    if not allowed_origin and not (literature_resource and policy.allow_external_literature_links):
        return False
    return any(path_matches_prefix(parsed.path, prefix) for prefix in policy.allowed_path_prefixes)


@dataclass(frozen=True)
class CollectionRequestScope:
    policy: CollectionScopePolicy
    literature_resource: bool = False

    def __call__(self, url: str) -> None:
        try:
            allowed = url_in_scope(url, self.policy, literature_resource=self.literature_resource)
        except ValueError as error:
            raise SafeHTTPError("COLLECTION_URL_OUTSIDE_SCOPE", "invalid collection URL") from error
        if not allowed:
            raise SafeHTTPError("COLLECTION_URL_OUTSIDE_SCOPE", "URL is outside collection scope")
