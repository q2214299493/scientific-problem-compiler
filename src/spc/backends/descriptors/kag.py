from __future__ import annotations

from ..contracts import (
    BackendCapability,
    BackendIntegrationMode,
    BackendLicenseStatus,
    ExternalBackendDescriptor,
)
from ._common import make_descriptor


def descriptor() -> ExternalBackendDescriptor:
    return make_descriptor(
        backend_id="kag",
        backend_name="KAG",
        backend_version="future-service",
        adapter_version="1.0.0",
        capability_types=(
            BackendCapability.KNOWLEDGE_EXTRACTION,
            BackendCapability.GRAPH_RETRIEVAL,
        ),
        integration_mode=BackendIntegrationMode.HTTP_SERVICE,
        package_name=None,
        package_version=None,
        source_project="https://github.com/OpenSPG/KAG",
        license_id="Apache-2.0",
        license_status=BackendLicenseStatus.CONFIRMED,
        deterministic=False,
        requires_network=True,
        optional_dependency=True,
        vendored=False,
        redistributable_confirmed=False,
    )
