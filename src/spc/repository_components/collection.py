from __future__ import annotations

import hashlib
from pathlib import Path

from ..models import (
    CollectionAcquisitionLink,
    CollectionDefinition,
    CollectionDiff,
    CollectionImportRecord,
    CollectionLiteratureMembership,
    CollectionPageArtifact,
    CollectionPageRecord,
    CollectionResourceImportResult,
    CollectionResourceOccurrence,
    CollectionSnapshot,
    DiscoveredCollectionResource,
    RawLiteratureArtifact,
)
from ..serialization import content_hash, load_model, require_safe_path_component

from .base import (
    ArtifactDirectory,
    IdentityBoundRepository,
    put_immutable_model,
    verify_artifact_bytes,
)
from .literature import LiteratureRepositories, raw_artifact_metadata


class CollectionPageArtifactRepository(ArtifactDirectory[CollectionPageArtifact]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root, "collection_page_artifacts")

    def put(
        self,
        content: bytes,
        *,
        collection_id: str,
        requested_url: str,
        resolved_url: str,
        media_type: str,
    ) -> CollectionPageArtifact:
        require_safe_path_component(collection_id, field="collection_id")
        digest = hashlib.sha256(content).hexdigest()
        stable_identity = {
            "collection_id": collection_id,
            "requested_url": requested_url,
            "resolved_url": resolved_url,
            "media_type": media_type,
            "sha256": digest,
        }
        artifact_id = f"collection-page-artifact-{content_hash(stable_identity)[:24]}"
        payload = {
            "artifact_id": artifact_id,
            "collection_id": collection_id,
            "requested_url": requested_url,
            "resolved_url": resolved_url,
            "media_type": media_type,
            "byte_size": len(content),
            "sha256": digest,
            "stored_path": f"collection_page_artifacts/{artifact_id}/content.bin",
        }
        record = CollectionPageArtifact(**payload, content_hash=content_hash(payload))
        destination = self.root / artifact_id
        if destination.exists():
            existing = self.get(artifact_id)
            if existing != record:
                raise FileExistsError(
                    f"refusing to overwrite different collection page artifact: {artifact_id}"
                )
            return existing
        self._write_directory(destination, "content.bin", content, record)
        return self.get(artifact_id)

    def get(self, artifact_id: str) -> CollectionPageArtifact:
        require_safe_path_component(artifact_id, field="artifact_id")
        directory = self.root / artifact_id
        content_path = directory / "content.bin"
        metadata_path = directory / "metadata.json"
        self._validate_paths(directory, content_path, metadata_path)
        record = load_model(metadata_path, CollectionPageArtifact)
        if record.artifact_id != artifact_id:
            raise ValueError("collection page artifact ID does not match its directory")
        content = content_path.read_bytes()
        verify_artifact_bytes(content, record.byte_size, record.sha256, "collection page artifact")
        return record

    def read_bytes(self, artifact_id: str) -> bytes:
        record = self.get(artifact_id)
        return (self.root / record.artifact_id / "content.bin").read_bytes()

    def verify_page_record(self, page: CollectionPageRecord) -> CollectionPageArtifact:
        if page.page_artifact_id is None or page.page_artifact_hash is None:
            raise ValueError("collection page has no exact-byte artifact binding")
        artifact = self.get(page.page_artifact_id)
        if artifact.content_hash != page.page_artifact_hash:
            raise ValueError("collection page artifact record hash changed")
        if artifact.sha256 != page.response_sha256:
            raise ValueError("collection page response hash does not match stored bytes")
        if (
            artifact.collection_id != page.collection_id
            or artifact.requested_url != page.requested_url
            or artifact.resolved_url != page.resolved_url
            or artifact.media_type != page.media_type
        ):
            raise ValueError("collection page artifact provenance binding is invalid")
        return artifact

    def list_metadata(self) -> tuple[RawLiteratureArtifact, ...]:
        # Preserve the inherited raw-literature metadata API for legacy callers.
        return raw_artifact_metadata(self)


class CollectionDefinitionRepository(IdentityBoundRepository[CollectionDefinition]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_definitions",
            CollectionDefinition,
            "collection_id",
        )


class CollectionPageRepository(IdentityBoundRepository[CollectionPageRecord]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_pages",
            CollectionPageRecord,
            "page_id",
        )


class DiscoveredCollectionResourceRepository:
    """Preserve immutable versions when one logical resource changes."""

    def __init__(self, knowledge_root: Path) -> None:
        self.root = knowledge_root / "collection_resources"

    def put(self, key: str, model: DiscoveredCollectionResource) -> Path:
        require_safe_path_component(key, field="repository key")
        if key != model.discovered_resource_id:
            raise ValueError("repository key does not match discovered_resource_id")
        path = self.root / key / f"{model.content_hash}.json"
        return put_immutable_model(path, model, DiscoveredCollectionResource)

    def get(
        self,
        key: str,
        content_hash_value: str | None = None,
    ) -> DiscoveredCollectionResource:
        require_safe_path_component(key, field="repository key")
        directory = self.root / key
        if content_hash_value is not None:
            require_safe_path_component(content_hash_value, field="content hash")
            return load_model(
                directory / f"{content_hash_value}.json",
                DiscoveredCollectionResource,
            )
        paths = sorted(directory.glob("*.json"))
        if len(paths) != 1:
            raise ValueError(
                f"logical collection resource has {len(paths)} versions; content hash is required"
            )
        return load_model(paths[0], DiscoveredCollectionResource)

    def list(self) -> tuple[DiscoveredCollectionResource, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            load_model(path, DiscoveredCollectionResource)
            for path in sorted(self.root.glob("*/*.json"))
        )


class CollectionResourceOccurrenceRepository(
    IdentityBoundRepository[CollectionResourceOccurrence]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_occurrences",
            CollectionResourceOccurrence,
            "occurrence_id",
        )


class CollectionSnapshotRepository(IdentityBoundRepository[CollectionSnapshot]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_snapshots",
            CollectionSnapshot,
            "snapshot_id",
        )


class CollectionAcquisitionLinkRepository(
    IdentityBoundRepository[CollectionAcquisitionLink]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_acquisition_links",
            CollectionAcquisitionLink,
            "link_id",
        )


class CollectionResourceImportResultRepository(
    IdentityBoundRepository[CollectionResourceImportResult]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_resource_import_results",
            CollectionResourceImportResult,
            "result_id",
        )


class CollectionLiteratureMembershipRepository(
    IdentityBoundRepository[CollectionLiteratureMembership]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_literature_memberships",
            CollectionLiteratureMembership,
            "membership_id",
        )


class CollectionImportRepository(IdentityBoundRepository[CollectionImportRecord]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_imports",
            CollectionImportRecord,
            "import_id",
        )


class CollectionDiffRepository(IdentityBoundRepository[CollectionDiff]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_diffs",
            CollectionDiff,
            "diff_id",
        )


class CollectionRepositories:
    """Collection records and the literature stores used to acquire their members."""

    def __init__(self, literature: LiteratureRepositories) -> None:
        root = literature.root
        self.root = root
        self.evidence_store = literature.evidence_store
        self.acquisition_attempts = literature.acquisition_attempts
        self.acquisition_requests = literature.acquisition_requests
        self.canonical_html_text_artifacts = literature.canonical_html_text_artifacts
        self.canonical_text_artifacts = literature.canonical_text_artifacts
        self.html_literature_ingestions = literature.html_literature_ingestions
        self.literature_acquisitions = literature.literature_acquisitions
        self.literature_documents = literature.literature_documents
        self.literature_ingestions = literature.literature_ingestions
        self.literature_representation_refs = literature.literature_representation_refs
        self.metadata_merge_manifests = literature.metadata_merge_manifests
        self.metadata_retrievals = literature.metadata_retrievals
        self.raw_html_literature_artifacts = literature.raw_html_literature_artifacts
        self.raw_literature_artifacts = literature.raw_literature_artifacts
        self.resolved_literature_resources = literature.resolved_literature_resources
        self.collection_definitions = CollectionDefinitionRepository(root)
        self.collection_pages = CollectionPageRepository(root)
        self.collection_page_artifacts = CollectionPageArtifactRepository(root)
        self.collection_resources = DiscoveredCollectionResourceRepository(root)
        self.collection_occurrences = CollectionResourceOccurrenceRepository(root)
        self.collection_snapshots = CollectionSnapshotRepository(root)
        self.collection_acquisition_links = CollectionAcquisitionLinkRepository(root)
        self.collection_resource_import_results = (
            CollectionResourceImportResultRepository(root)
        )
        self.collection_literature_memberships = (
            CollectionLiteratureMembershipRepository(root)
        )
        self.collection_imports = CollectionImportRepository(root)
        self.collection_diffs = CollectionDiffRepository(root)
