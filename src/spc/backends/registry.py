from __future__ import annotations

from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from .contracts import (
    BackendAdapter,
    BackendCapability,
    BackendRuntimeAvailability,
    ExternalBackendDescriptor,
)
from .descriptors import (
    kag_descriptor,
    lightrag_descriptor,
    mineru_descriptor,
    ragflow_descriptor,
)
from .document import BuiltinDocumentParsingBackend, DoclingDocumentParsingBackend
from .metadata import GROBIDScholarlyMetadataBackend
from .retrieval import PaperQALiteratureRetrievalBackend
from ..serialization import require_safe_path_component


class BackendRegistryError(ValueError):
    pass


def validate_backend_descriptor(
    descriptor: ExternalBackendDescriptor,
) -> ExternalBackendDescriptor:
    require_safe_path_component(descriptor.backend_id, field="backend_id")
    return ExternalBackendDescriptor.model_validate(descriptor.model_dump(mode="json"))


class UnavailableBackend:
    def __init__(self, descriptor: ExternalBackendDescriptor, reason: str) -> None:
        self.descriptor = descriptor
        self._reason = reason

    def inspect_availability(self) -> BackendRuntimeAvailability:
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=False,
            reason=self._reason,
        )


class ConfiguredSubprocessBackend:
    """Availability boundary only; K1F0 does not execute subprocess parsers."""

    def __init__(
        self,
        descriptor: ExternalBackendDescriptor,
        executable: Path | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.executable = executable

    def inspect_availability(self) -> BackendRuntimeAvailability:
        if self.executable is None:
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                reason="no explicit subprocess executable is configured",
            )
        if not self.executable.is_absolute():
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                reason="configured subprocess executable must be an absolute path",
            )
        if not self.executable.is_file():
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                reason="configured subprocess executable does not exist",
            )
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=True,
            detected_version="configured-executable",
        )


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, BackendAdapter] = {}

    def register(self, backend: BackendAdapter) -> None:
        descriptor = validate_backend_descriptor(backend.descriptor)
        existing = self._backends.get(descriptor.backend_id)
        if existing is not None:
            if existing.descriptor != descriptor:
                raise BackendRegistryError(f"conflicting backend descriptor: {descriptor.backend_id}")
            return
        self._backends[descriptor.backend_id] = backend

    def list_descriptors(self) -> tuple[ExternalBackendDescriptor, ...]:
        return tuple(self._backends[key].descriptor for key in sorted(self._backends))

    def resolve(self, backend_id: str) -> BackendAdapter:
        try:
            return self._backends[backend_id]
        except KeyError as error:
            raise BackendRegistryError(f"unknown backend: {backend_id}") from error

    def resolve_capability(
        self,
        capability: BackendCapability,
        *,
        backend_id: str | None = None,
    ) -> BackendAdapter:
        if backend_id is not None:
            backend = self.resolve(backend_id)
            if capability not in backend.descriptor.capability_types:
                raise BackendRegistryError(f"backend {backend_id} does not provide {capability.value}")
            return backend
        matches = tuple(
            backend for backend in self._backends.values() if capability in backend.descriptor.capability_types
        )
        if len(matches) != 1:
            raise BackendRegistryError(f"capability {capability.value} requires an explicit backend ID")
        return matches[0]

    def inspect_runtime(self, backend_id: str) -> BackendRuntimeAvailability:
        backend = self.resolve(backend_id)
        try:
            availability = backend.inspect_availability()
        except Exception as error:
            return BackendRuntimeAvailability(
                backend_id=backend_id,
                available=False,
                reason=f"availability inspection failed: {type(error).__name__}",
            )
        if availability.backend_id != backend_id:
            raise BackendRegistryError("backend availability identity mismatch")
        return availability

    def inspect_all(
        self,
    ) -> Mapping[str, BackendRuntimeAvailability]:
        return MappingProxyType(
            {
                descriptor.backend_id: self.inspect_runtime(descriptor.backend_id)
                for descriptor in self.list_descriptors()
            }
        )


def load_backend_license_manifest() -> tuple[Mapping[str, object], ...]:
    resource = resources.files("spc.backends").joinpath("backend_licenses.yaml")
    data = yaml.safe_load(resource.read_text(encoding="utf-8"))
    records = data.get("backends") if isinstance(data, dict) else None
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("backend license manifest is invalid")
    if any(item.get("vendored") is not False for item in records):
        raise ValueError("backend license manifest must not declare vendored source")
    return tuple(MappingProxyType(dict(item)) for item in records)


def default_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(BuiltinDocumentParsingBackend())
    registry.register(DoclingDocumentParsingBackend())
    registry.register(PaperQALiteratureRetrievalBackend())
    registry.register(GROBIDScholarlyMetadataBackend())
    registry.register(ConfiguredSubprocessBackend(mineru_descriptor()))
    registry.register(UnavailableBackend(lightrag_descriptor(), "future graph adapter is not configured"))
    registry.register(UnavailableBackend(kag_descriptor(), "future graph adapter is not configured"))
    registry.register(UnavailableBackend(ragflow_descriptor(), "future retrieval adapter is not configured"))
    return registry
