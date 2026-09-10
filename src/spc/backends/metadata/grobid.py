from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping

from ..contracts import (
    BackendCapability,
    BackendIntegrationMode,
    BackendLicenseStatus,
    BackendOutputType,
    BackendInputBinding,
    BackendRunStatus,
    BackendRuntimeAvailability,
    BackendRuntimeIdentity,
    ExternalBackendDescriptor,
    ScholarlyMetadataProposal,
)
from ..provenance import (
    BackendInvocationError,
    BackendRunRecord,
    BackendRunRepository,
    build_backend_runtime_identity,
    create_backend_run_record,
    probe_backend_runtime,
)
from ...repositories import IdentityBoundRepository
from ...serialization import content_hash


GROBIDRunner = Callable[
    [bytes, str, Mapping[str, object]],
    Mapping[str, object],
]


def _descriptor() -> ExternalBackendDescriptor:
    payload = {
        "backend_id": "grobid",
        "backend_name": "GROBID",
        "backend_version": "service-resolved",
        "adapter_version": "1.1.0",
        "capability_types": (BackendCapability.SCHOLARLY_METADATA,),
        "integration_mode": BackendIntegrationMode.HTTP_SERVICE,
        "source_project": "https://github.com/grobidOrg/grobid",
        "license_id": "Apache-2.0",
        "license_status": BackendLicenseStatus.CONFIRMED,
        "deterministic": False,
        "requires_network": True,
        "optional_dependency": True,
        "vendored": False,
        "redistributable_confirmed": False,
    }
    return ExternalBackendDescriptor(**payload, content_hash=content_hash(payload))


class GROBIDScholarlyMetadataBackend:
    """GROBID normalization boundary using an explicitly configured safe runner."""

    descriptor = _descriptor()

    def __init__(
        self,
        runner: GROBIDRunner | None = None,
        *,
        runner_id: str | None = None,
        runner_version: str | None = None,
        service_version: str | None = None,
    ) -> None:
        if runner is not None and (not runner_id or not runner_version or not service_version):
            raise ValueError("configured GROBID runner requires runner ID/version and service version")
        self._runner = runner
        self._runner_id = runner_id
        self._runner_version = runner_version
        self._service_version = service_version

    def inspect_availability(self) -> BackendRuntimeAvailability:
        if self._runner is None:
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                reason="no GROBID service runner is explicitly configured",
            )
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=True,
            detected_version=self._service_version,
        )

    def resolve_runtime_identity(self) -> BackendRuntimeIdentity:
        availability = self.inspect_availability()
        if (
            not availability.available
            or availability.detected_version is None
            or self._runner_id is None
            or self._runner_version is None
        ):
            raise RuntimeError(availability.reason or "GROBID service is unavailable")
        return build_backend_runtime_identity(
            self.descriptor,
            resolved_backend_version=availability.detected_version,
            runtime_provider=f"runner:{self._runner_id}@{self._runner_version}",
            runtime_config_hash=content_hash(
                {
                    "runner_id": self._runner_id,
                    "runner_version": self._runner_version,
                }
            ),
        )

    def extract_metadata(
        self, source: bytes, *, media_type: str, config: Mapping[str, object]
    ) -> ScholarlyMetadataProposal:
        if self._runner is None:
            raise RuntimeError(self.inspect_availability().reason)
        runtime = self.resolve_runtime_identity()
        candidate = self._runner(source, media_type, config)
        allowed = {
            "title",
            "authors",
            "doi",
            "abstract",
            "references",
            "section_hints",
        }
        identity = {
            "backend_id": self.descriptor.backend_id,
            "backend_descriptor_hash": self.descriptor.content_hash,
            "runtime_identity_hash": runtime.content_hash,
            "authors": tuple(candidate.get("authors", ())),
            "references": tuple(candidate.get("references", ())),
            "section_hints": tuple(candidate.get("section_hints", ())),
            "trust_class": "external_proposal",
        }
        for key in allowed - {"authors", "references", "section_hints"}:
            if candidate.get(key) is not None:
                identity[key] = candidate[key]
        proposal_id = f"scholarly-metadata-proposal-{content_hash(identity)[:24]}"
        payload = {"proposal_id": proposal_id, **identity}
        return ScholarlyMetadataProposal(
            **payload,
            content_hash=content_hash(payload),
        )


class ScholarlyMetadataProposalRepository(IdentityBoundRepository[ScholarlyMetadataProposal]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "backend_metadata_proposals",
            ScholarlyMetadataProposal,
            "proposal_id",
        )


@dataclass(frozen=True)
class ScholarlyMetadataOutcome:
    proposal: ScholarlyMetadataProposal
    run_record: BackendRunRecord


class ScholarlyMetadataService:
    def extract(
        self,
        backend: GROBIDScholarlyMetadataBackend,
        source: bytes,
        *,
        media_type: str,
        config: Mapping[str, object],
        knowledge_root: Path,
    ) -> ScholarlyMetadataOutcome:
        descriptor = backend.descriptor
        source_hash = hashlib.sha256(source).hexdigest()
        config_hash = content_hash(dict(config))
        inputs = (
            BackendInputBinding(
                input_id=f"metadata-source-{source_hash[:24]}",
                input_hash=source_hash,
            ),
        )
        runtime_probe = probe_backend_runtime(backend, knowledge_root)
        runtime = runtime_probe.runtime_identity
        if runtime_probe.failure_status is not None:
            run = create_backend_run_record(
                descriptor=descriptor,
                runtime_identity=runtime,
                capability=BackendCapability.SCHOLARLY_METADATA,
                input_bindings=inputs,
                config_hash=config_hash,
                output_hash=None,
                status=runtime_probe.failure_status,
                warnings=(runtime_probe.failure_category or "runtime_probe_failed",),
            )
            BackendRunRepository(knowledge_root).put(run.run_id, run)
            raise BackendInvocationError(f"backend {descriptor.backend_id} runtime probe failed; run_id={run.run_id}")
        try:
            proposal = ScholarlyMetadataProposal.model_validate(
                backend.extract_metadata(source, media_type=media_type, config=config)
            )
            if (
                proposal.backend_id != descriptor.backend_id
                or proposal.backend_descriptor_hash != descriptor.content_hash
                or proposal.runtime_identity_hash != runtime.content_hash
            ):
                raise ValueError("scholarly metadata proposal backend binding is invalid")
        except Exception as error:
            run = create_backend_run_record(
                descriptor=descriptor,
                runtime_identity=runtime,
                capability=BackendCapability.SCHOLARLY_METADATA,
                input_bindings=inputs,
                config_hash=config_hash,
                output_hash=None,
                status=BackendRunStatus.FAILED,
                warnings=("backend invocation or output validation failed",),
            )
            BackendRunRepository(knowledge_root).put(run.run_id, run)
            raise BackendInvocationError(f"scholarly metadata backend failed; run_id={run.run_id}") from error
        ScholarlyMetadataProposalRepository(knowledge_root).put(proposal.proposal_id, proposal)
        run = create_backend_run_record(
            descriptor=descriptor,
            runtime_identity=runtime,
            capability=BackendCapability.SCHOLARLY_METADATA,
            input_bindings=inputs,
            config_hash=config_hash,
            output_hash=proposal.content_hash,
            output_id=proposal.proposal_id,
            output_type=BackendOutputType.SCHOLARLY_METADATA_PROPOSAL,
            status=BackendRunStatus.SUCCEEDED,
            output_count=1,
        )
        BackendRunRepository(knowledge_root).put(run.run_id, run)
        return ScholarlyMetadataOutcome(proposal=proposal, run_record=run)
