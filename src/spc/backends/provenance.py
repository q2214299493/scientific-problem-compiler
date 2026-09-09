from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import (
    BackendAdapter,
    BackendCapability,
    BackendInputBinding,
    BackendRunRecord,
    BackendRunStatus,
    BackendRuntimeIdentity,
    BackendRuntimeAvailability,
    ExternalBackendDescriptor,
)
from ..repositories import IdentityBoundRepository
from ..serialization import content_hash


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
        return record


def build_backend_runtime_identity(
    descriptor: ExternalBackendDescriptor,
    *,
    resolved_backend_version: str,
    runtime_provider: str,
    runtime_config_hash: str,
) -> BackendRuntimeIdentity:
    identity = {
        "backend_id": descriptor.backend_id,
        "backend_descriptor_hash": descriptor.content_hash,
        "resolved_backend_version": resolved_backend_version,
        "adapter_version": descriptor.adapter_version,
        "integration_mode": descriptor.integration_mode,
        "runtime_provider": runtime_provider,
        "runtime_config_hash": runtime_config_hash,
    }
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
