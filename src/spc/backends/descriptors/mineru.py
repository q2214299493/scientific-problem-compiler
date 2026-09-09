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
        backend_id="mineru",
        backend_name="MinerU",
        backend_version="runtime-resolved",
        adapter_version="1.0.0",
        capability_types=(BackendCapability.DOCUMENT_PARSING,),
        integration_mode=BackendIntegrationMode.SUBPROCESS,
        package_name="magic-pdf",
        package_version=None,
        source_project="https://github.com/opendatalab/MinerU",
        license_id="NOASSERTION",
        license_status=BackendLicenseStatus.UNVERIFIED,
        deterministic=False,
        requires_network=False,
        optional_dependency=True,
        vendored=False,
        redistributable_confirmed=False,
    )
