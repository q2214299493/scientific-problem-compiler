from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .contracts import (
    BackendAdapter,
    BackendCapability,
    BackendInputBinding,
    BackendOutputType,
    BackendRunRecord,
    BackendRunStatus,
    BackendRuntimeArtifactEntry,
    BackendRuntimeArtifactManifest,
    BackendRuntimeComponentVersion,
    BackendRuntimeIdentity,
    BackendRuntimeAvailability,
    ExternalBackendDescriptor,
)
from ..repositories import IdentityBoundRepository
from ..serialization import content_hash, file_sha256


MAX_RUNTIME_ARTIFACT_FILES = 100_000
MAX_RUNTIME_ARTIFACT_BYTES = 100 * 1024 * 1024 * 1024


class BackendInvocationError(RuntimeError):
    """Sanitized SPC error raised when an external backend boundary fails."""


class BackendDescriptorRepository(IdentityBoundRepository[ExternalBackendDescriptor]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "backend_descriptors",
            ExternalBackendDescriptor,
            "content_hash",
        )


class BackendRuntimeIdentityRepository(IdentityBoundRepository[BackendRuntimeIdentity]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "backend_runtime_identities",
            BackendRuntimeIdentity,
            "runtime_identity_id",
        )


class BackendRuntimeArtifactManifestRepository(
    IdentityBoundRepository[BackendRuntimeArtifactManifest]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "backend_runtime_artifact_manifests",
            BackendRuntimeArtifactManifest,
            "manifest_id",
        )


def build_backend_runtime_artifact_manifest(
    root: Path,
    *,
    backend_id: str,
    artifact_role: str,
    max_files: int = MAX_RUNTIME_ARTIFACT_FILES,
    max_total_bytes: int = MAX_RUNTIME_ARTIFACT_BYTES,
) -> BackendRuntimeArtifactManifest:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("runtime artifact root must be a regular absolute directory")
    resolved_root = root.resolve(strict=True)
    files: list[Path] = []
    directories = [root]
    while directories:
        directory = directories.pop()
        if directory.is_symlink() or not directory.resolve(strict=True).is_relative_to(resolved_root):
            raise ValueError("runtime artifact directory is unsafe")
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                raise ValueError("runtime artifact tree must not contain symlinks")
            resolved = child.resolve(strict=True)
            if not resolved.is_relative_to(resolved_root):
                raise ValueError("runtime artifact path escapes configured root")
            if child.is_dir():
                directories.append(child)
            elif child.is_file():
                files.append(child)
            else:
                raise ValueError("runtime artifact tree contains a non-regular entry")
    files.sort(key=lambda item: item.resolve().relative_to(resolved_root).as_posix())
    if len(files) > max_files:
        raise ValueError("runtime artifact file-count limit exceeded")
    entries: list[BackendRuntimeArtifactEntry] = []
    total_size = 0
    for path in files:
        before = path.stat()
        size = before.st_size
        total_size += size
        if total_size > max_total_bytes:
            raise ValueError("runtime artifact total-size limit exceeded")
        digest = file_sha256(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or path.is_symlink()
        ):
            raise ValueError("runtime artifact changed during integrity inspection")
        entries.append(
            BackendRuntimeArtifactEntry(
                relative_path=path.resolve().relative_to(resolved_root).as_posix(),
                file_sha256=digest,
                file_size=size,
            )
        )
    entry_payload = [item.model_dump(mode="json") for item in entries]
    identity = {
        "backend_id": backend_id,
        "artifact_role": artifact_role,
        "root_identity": f"{backend_id}:{artifact_role}",
        "entries": entry_payload,
        "aggregate_hash": content_hash(entry_payload),
    }
    manifest_id = f"backend-runtime-artifacts-{content_hash(identity)[:24]}"
    payload = {"manifest_id": manifest_id, **identity}
    return BackendRuntimeArtifactManifest(**payload, content_hash=content_hash(payload))


class BackendRunRepository(IdentityBoundRepository[BackendRunRecord]):
    def __init__(self, knowledge_root: Path) -> None:
        self.knowledge_root = knowledge_root
        super().__init__(
            knowledge_root / "backend_runs",
            BackendRunRecord,
            "run_id",
        )

    def put(self, key: str, model: BackendRunRecord) -> Path:
        descriptor = BackendDescriptorRepository(self.knowledge_root).get(model.backend_descriptor_hash)
        runtime = BackendRuntimeIdentityRepository(self.knowledge_root).get(model.runtime_identity_id)
        if descriptor.backend_id != model.backend_id:
            raise ValueError("backend run descriptor binding is invalid")
        if runtime.content_hash != model.runtime_identity_hash:
            raise ValueError("backend run runtime identity hash is invalid")
        validate_runtime_identity(descriptor, runtime)
        validate_runtime_artifact_binding(self.knowledge_root, runtime)
        return super().put(key, model)

    def get(self, key: str) -> BackendRunRecord:
        record = super().get(key)
        descriptor = BackendDescriptorRepository(self.knowledge_root).get(record.backend_descriptor_hash)
        runtime = BackendRuntimeIdentityRepository(self.knowledge_root).get(record.runtime_identity_id)
        if descriptor.backend_id != record.backend_id:
            raise ValueError("backend run descriptor binding is invalid")
        if (
            runtime.content_hash != record.runtime_identity_hash
            or runtime.backend_id != record.backend_id
            or runtime.backend_descriptor_hash != record.backend_descriptor_hash
        ):
            raise ValueError("backend run runtime identity binding is invalid")
        validate_runtime_artifact_binding(self.knowledge_root, runtime)
        return record


def build_backend_runtime_identity(
    descriptor: ExternalBackendDescriptor,
    *,
    resolved_backend_version: str,
    runtime_provider: str,
    runtime_config_hash: str,
    runtime_components: Mapping[str, str] | None = None,
    runtime_artifact_manifest_hash: str | None = None,
) -> BackendRuntimeIdentity:
    components = tuple(
        BackendRuntimeComponentVersion(component=component, version=version)
        for component, version in sorted((runtime_components or {}).items())
    )
    component_payload = [item.model_dump(mode="json") for item in components]
    identity = {
        "backend_id": descriptor.backend_id,
        "backend_descriptor_hash": descriptor.content_hash,
        "resolved_backend_version": resolved_backend_version,
        "adapter_version": descriptor.adapter_version,
        "integration_mode": descriptor.integration_mode,
        "runtime_provider": runtime_provider,
        "runtime_config_hash": runtime_config_hash,
        "runtime_components": component_payload,
        "runtime_components_hash": content_hash(component_payload),
    }
    if runtime_artifact_manifest_hash is not None:
        identity["runtime_artifact_manifest_hash"] = runtime_artifact_manifest_hash
    runtime_identity_id = f"backend-runtime-{content_hash(identity)[:24]}"
    payload = {"runtime_identity_id": runtime_identity_id, **identity}
    return BackendRuntimeIdentity(**payload, content_hash=content_hash(payload))


def validate_runtime_identity(
    descriptor: ExternalBackendDescriptor,
    runtime: BackendRuntimeIdentity,
) -> BackendRuntimeIdentity:
    if (
        runtime.backend_id != descriptor.backend_id
        or runtime.backend_descriptor_hash != descriptor.content_hash
        or runtime.adapter_version != descriptor.adapter_version
        or runtime.integration_mode != descriptor.integration_mode
    ):
        raise ValueError("backend runtime identity does not match descriptor")
    return runtime


def validate_runtime_artifact_binding(
    knowledge_root: Path,
    runtime: BackendRuntimeIdentity,
) -> BackendRuntimeArtifactManifest | None:
    manifest_hash = runtime.runtime_artifact_manifest_hash
    if manifest_hash is None:
        return None
    manifests = BackendRuntimeArtifactManifestRepository(knowledge_root).list()
    matching = tuple(item for item in manifests if item.content_hash == manifest_hash)
    if len(matching) != 1 or matching[0].backend_id != runtime.backend_id:
        raise ValueError("backend runtime artifact manifest binding is invalid")
    return matching[0]


def persist_backend_identity(
    knowledge_root: Path,
    descriptor: ExternalBackendDescriptor,
    runtime: BackendRuntimeIdentity,
) -> None:
    validate_runtime_identity(descriptor, runtime)
    BackendDescriptorRepository(knowledge_root).put(descriptor.content_hash, descriptor)
    BackendRuntimeIdentityRepository(knowledge_root).put(runtime.runtime_identity_id, runtime)


def unresolved_runtime_identity(
    descriptor: ExternalBackendDescriptor,
    *,
    category: str,
    detected_version: str | None = None,
) -> BackendRuntimeIdentity:
    return build_backend_runtime_identity(
        descriptor,
        resolved_backend_version=detected_version or "unresolved",
        runtime_provider=f"unresolved:{category}",
        runtime_config_hash=content_hash({"category": category}),
    )


def resolve_backend_runtime_identity(
    backend: BackendAdapter,
) -> BackendRuntimeIdentity:
    runtime = BackendRuntimeIdentity.model_validate(backend.resolve_runtime_identity())
    return validate_runtime_identity(backend.descriptor, runtime)


@dataclass(frozen=True)
class BackendRuntimeProbe:
    runtime_identity: BackendRuntimeIdentity
    availability: BackendRuntimeAvailability | None
    failure_status: BackendRunStatus | None = None
    failure_category: str | None = None


def probe_backend_runtime(
    backend: BackendAdapter,
    knowledge_root: Path,
) -> BackendRuntimeProbe:
    descriptor = backend.descriptor
    BackendDescriptorRepository(knowledge_root).put(descriptor.content_hash, descriptor)
    try:
        availability = BackendRuntimeAvailability.model_validate(backend.inspect_availability())
        if availability.backend_id != descriptor.backend_id:
            raise ValueError("backend availability identity mismatch")
    except Exception:
        runtime = unresolved_runtime_identity(descriptor, category="availability_inspection_failed")
        persist_backend_identity(knowledge_root, descriptor, runtime)
        return BackendRuntimeProbe(
            runtime_identity=runtime,
            availability=None,
            failure_status=BackendRunStatus.FAILED,
            failure_category="availability_inspection_failed",
        )
    if not availability.available:
        runtime = unresolved_runtime_identity(
            descriptor,
            category="runtime_unavailable",
            detected_version=availability.detected_version,
        )
        persist_backend_identity(knowledge_root, descriptor, runtime)
        return BackendRuntimeProbe(
            runtime_identity=runtime,
            availability=availability,
            failure_status=BackendRunStatus.UNAVAILABLE,
            failure_category="runtime_unavailable",
        )
    try:
        runtime = resolve_backend_runtime_identity(backend)
        if runtime.runtime_artifact_manifest_hash is not None:
            manifest_builder = getattr(backend, "build_runtime_artifact_manifest", None)
            if not callable(manifest_builder):
                raise ValueError("backend runtime artifact manifest cannot be reconstructed")
            manifest = BackendRuntimeArtifactManifest.model_validate(manifest_builder())
            if (
                manifest.backend_id != descriptor.backend_id
                or manifest.content_hash != runtime.runtime_artifact_manifest_hash
            ):
                raise ValueError("backend runtime artifact manifest changed during inspection")
            BackendRuntimeArtifactManifestRepository(knowledge_root).put(manifest.manifest_id, manifest)
    except Exception:
        runtime = unresolved_runtime_identity(
            descriptor,
            category="runtime_identity_resolution_failed",
            detected_version=availability.detected_version,
        )
        persist_backend_identity(knowledge_root, descriptor, runtime)
        return BackendRuntimeProbe(
            runtime_identity=runtime,
            availability=availability,
            failure_status=BackendRunStatus.FAILED,
            failure_category="runtime_identity_resolution_failed",
        )
    persist_backend_identity(knowledge_root, descriptor, runtime)
    return BackendRuntimeProbe(runtime_identity=runtime, availability=availability)


def create_backend_run_record(
    *,
    descriptor: ExternalBackendDescriptor,
    runtime_identity: BackendRuntimeIdentity,
    capability: BackendCapability,
    input_bindings: tuple[BackendInputBinding, ...],
    config_hash: str,
    output_hash: str | None,
    output_id: str | None = None,
    output_type: BackendOutputType | None = None,
    status: BackendRunStatus,
    warnings: tuple[str, ...] = (),
    output_count: int = 0,
    candidate_count: int = 0,
    resolved_count: int = 0,
    unresolved_count: int = 0,
) -> BackendRunRecord:
    validate_runtime_identity(descriptor, runtime_identity)
    identity = {
        "backend_id": descriptor.backend_id,
        "backend_descriptor_hash": descriptor.content_hash,
        "runtime_identity_id": runtime_identity.runtime_identity_id,
        "runtime_identity_hash": runtime_identity.content_hash,
        "capability": capability,
        "input_bindings": input_bindings,
        "config_hash": config_hash,
        "output_id": output_id,
        "output_type": output_type,
        "output_hash": output_hash,
        "status": status,
        "warnings": warnings,
        "output_count": output_count,
        "candidate_count": candidate_count,
        "resolved_count": resolved_count,
        "unresolved_count": unresolved_count,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    run_id = f"backend-run-{content_hash(identity)[:24]}"
    payload = {"run_id": run_id, **identity}
    return BackendRunRecord(
        **payload,
        content_hash=content_hash(payload),
    )


@dataclass(frozen=True)
class ValidatedBackendRun:
    run: BackendRunRecord
    descriptor: ExternalBackendDescriptor
    runtime_identity: BackendRuntimeIdentity
    output: object | None
    runtime_artifact_manifest: BackendRuntimeArtifactManifest | None


def validate_backend_run(
    run: str | BackendRunRecord,
    knowledge_root: Path,
) -> ValidatedBackendRun:
    repository = BackendRunRepository(knowledge_root)
    record = repository.get(run) if isinstance(run, str) else run
    stored = repository.get(record.run_id)
    if stored != record:
        raise ValueError("backend run differs from repository record")
    descriptor = BackendDescriptorRepository(knowledge_root).get(record.backend_descriptor_hash)
    runtime = BackendRuntimeIdentityRepository(knowledge_root).get(record.runtime_identity_id)
    validate_runtime_identity(descriptor, runtime)
    manifest = validate_runtime_artifact_binding(knowledge_root, runtime)
    output = None
    if record.output_type is not None and record.output_id is not None:
        if record.output_type == BackendOutputType.EXTERNAL_DOCUMENT_PARSE_PROPOSAL:
            from .rebinding import ExternalDocumentProposalRepository

            output = ExternalDocumentProposalRepository(knowledge_root).get(record.output_id)
        elif record.output_type == BackendOutputType.EXTERNAL_LITERATURE_RETRIEVAL_RESULT:
            from .retrieval.service import ExternalRetrievalResultRepository

            output = ExternalRetrievalResultRepository(knowledge_root).get(record.output_id)
        elif record.output_type == BackendOutputType.SCHOLARLY_METADATA_PROPOSAL:
            from .metadata.grobid import ScholarlyMetadataProposalRepository

            output = ScholarlyMetadataProposalRepository(knowledge_root).get(record.output_id)
        else:
            raise ValueError("unsupported backend run output type")
        if (
            getattr(output, "content_hash", None) != record.output_hash
            or getattr(output, "backend_id", None) != record.backend_id
            or getattr(output, "backend_descriptor_hash", None) != record.backend_descriptor_hash
            or getattr(output, "runtime_identity_hash", None) != record.runtime_identity_hash
        ):
            raise ValueError("backend run output binding is invalid")
    elif record.status in {BackendRunStatus.SUCCEEDED, BackendRunStatus.PARTIAL}:
        raise ValueError("successful backend run output is not traceable")
    return ValidatedBackendRun(record, descriptor, runtime, output, manifest)
