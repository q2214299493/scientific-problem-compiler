from __future__ import annotations

from pathlib import Path

from .contracts import (
    BackendCapability,
    BackendInputBinding,
    BackendRunRecord,
    BackendRunStatus,
    ExternalBackendDescriptor,
)
from ..repositories import IdentityBoundRepository
from ..serialization import content_hash


class BackendRunRepository(IdentityBoundRepository[BackendRunRecord]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "backend_runs",
            BackendRunRecord,
            "run_id",
        )


def create_backend_run_record(
    *,
    descriptor: ExternalBackendDescriptor,
    capability: BackendCapability,
    input_bindings: tuple[BackendInputBinding, ...],
    config_hash: str,
    output_hash: str | None,
    status: BackendRunStatus,
    warnings: tuple[str, ...] = (),
    resolved_count: int = 0,
    unresolved_count: int = 0,
) -> BackendRunRecord:
    identity = {
        "backend_id": descriptor.backend_id,
        "backend_descriptor_hash": descriptor.content_hash,
        "capability": capability,
        "input_bindings": input_bindings,
        "config_hash": config_hash,
        "output_hash": output_hash,
        "status": status,
        "warnings": warnings,
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
