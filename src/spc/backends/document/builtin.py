from __future__ import annotations

from ..contracts import (
    BackendCapability,
    BackendIntegrationMode,
    BackendLicenseStatus,
    BackendRuntimeAvailability,
    BackendRuntimeIdentity,
    ExternalBackendDescriptor,
)
from ..provenance import build_backend_runtime_identity
from ...serialization import content_hash


def _descriptor() -> ExternalBackendDescriptor:
    payload = {
        "backend_id": "builtin-document-parser",
        "backend_name": "SPC builtin document parser",
        "backend_version": "3.0.0",
        "adapter_version": "1.0.0",
        "capability_types": (BackendCapability.BUILTIN_STRUCTURE,),
        "integration_mode": BackendIntegrationMode.BUILTIN,
        "source_project": "https://github.com/q2214299493/scientific-problem-compiler",
        "license_id": "project-license",
        "license_status": BackendLicenseStatus.CONFIRMED,
        "deterministic": True,
        "requires_network": False,
        "optional_dependency": False,
        "vendored": False,
        "redistributable_confirmed": False,
    }
    return ExternalBackendDescriptor(**payload, content_hash=content_hash(payload))


class BuiltinDocumentParsingBackend:
    descriptor = _descriptor()

    def inspect_availability(self) -> BackendRuntimeAvailability:
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=True,
            detected_version=self.descriptor.backend_version,
        )

    def resolve_runtime_identity(self) -> BackendRuntimeIdentity:
        return build_backend_runtime_identity(
            self.descriptor,
            resolved_backend_version=self.descriptor.backend_version,
            runtime_provider="spc-builtin",
            runtime_config_hash=content_hash({}),
        )

    def structure(self, literature_id, representation_id, repositories, evidence_store):
        from ...knowledge.structure import DocumentStructureService

        return DocumentStructureService().extract(
            literature_id,
            representation_id,
            repositories,
            evidence_store,
        )
