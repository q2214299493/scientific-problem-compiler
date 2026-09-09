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
        backend_id="lightrag",
        backend_name="LightRAG",
        backend_version="future-service",
        adapter_version="1.0.0",
        capability_types=(BackendCapability.GRAPH_RETRIEVAL,),
        integration_mode=BackendIntegrationMode.HTTP_SERVICE,
        package_name=None,
        package_version=None,
        source_project="https://github.com/HKUDS/LightRAG",
        license_id="MIT",
        license_status=BackendLicenseStatus.CONFIRMED,
        deterministic=False,
        requires_network=True,
        optional_dependency=True,
        vendored=False,
        redistributable_confirmed=False,
    )
