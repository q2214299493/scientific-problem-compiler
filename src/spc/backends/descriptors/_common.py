from __future__ import annotations

from ..contracts import ExternalBackendDescriptor
from ...serialization import content_hash


def make_descriptor(**values: object) -> ExternalBackendDescriptor:
    payload = {key: value for key, value in values.items() if value is not None}
    return ExternalBackendDescriptor(
        **payload,
        content_hash=content_hash(payload),
    )
