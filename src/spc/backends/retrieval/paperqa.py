from __future__ import annotations

from importlib import metadata, util
from typing import Callable, Iterable, Mapping

from ..contracts import (
    BackendCapability,
    BackendIntegrationMode,
    BackendLicenseStatus,
    BackendRuntimeAvailability,
    ExternalBackendDescriptor,
    ExternalLiteratureRetrievalHit,
    ExternalLiteratureRetrievalQuery,
    ExternalLiteratureRetrievalResult,
    ExternalProposalStatus,
)
from ...serialization import content_hash


PaperQARunner = Callable[
    [ExternalLiteratureRetrievalQuery],
    Iterable[Mapping[str, str | int | float | bool | None]],
]


def _descriptor() -> ExternalBackendDescriptor:
    payload = {
        "backend_id": "paperqa2",
        "backend_name": "PaperQA2",
        "backend_version": "runtime-resolved",
        "adapter_version": "1.0.0",
        "capability_types": (BackendCapability.LITERATURE_RETRIEVAL,),
        "integration_mode": BackendIntegrationMode.PYTHON_LIBRARY,
        "package_name": "paper-qa",
        "source_project": "https://github.com/Future-House/paper-qa",
        "license_id": "Apache-2.0",
        "license_status": BackendLicenseStatus.CONFIRMED,
        "deterministic": False,
        "requires_network": False,
        "optional_dependency": True,
        "vendored": False,
        "redistributable_confirmed": False,
    }
    return ExternalBackendDescriptor(**payload, content_hash=content_hash(payload))


def _hit(
    record: Mapping[str, str | int | float | bool | None],
) -> ExternalLiteratureRetrievalHit:
    known = {
        "external_document_id",
        "doi",
        "title",
        "source_url",
        "page_hint",
        "text_snippet",
        "score",
    }
    identity = {key: record[key] for key in known if key in record and record[key] is not None}
    identity["backend_metadata"] = {key: value for key, value in record.items() if key not in known}
    hit_id = f"external-retrieval-hit-{content_hash(identity)[:24]}"
    payload = {"hit_id": hit_id, **identity}
    return ExternalLiteratureRetrievalHit(
        **payload,
        content_hash=content_hash(payload),
    )


class PaperQALiteratureRetrievalBackend:
    """Optional PaperQA boundary; callers inject an API-compatible runner."""

    descriptor = _descriptor()

    def __init__(self, runner: PaperQARunner | None = None) -> None:
        self._runner = runner

    def inspect_availability(self) -> BackendRuntimeAvailability:
        if self._runner is not None:
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=True,
                detected_version="configured-runner",
            )
        try:
            installed = util.find_spec("paperqa") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=False,
                reason="optional package 'paper-qa' is not installed",
            )
        try:
            version = metadata.version("paper-qa")
        except metadata.PackageNotFoundError:
            version = "installed-version-unresolved"
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=False,
            detected_version=version,
            reason="PaperQA is installed but no explicit SPC runner is configured",
        )

    def retrieve(self, query: ExternalLiteratureRetrievalQuery) -> ExternalLiteratureRetrievalResult:
        if self._runner is None:
            raise RuntimeError(self.inspect_availability().reason)
        hits = tuple(_hit(item) for item in self._runner(query))
        identity = {
            "backend_id": self.descriptor.backend_id,
            "backend_descriptor_hash": self.descriptor.content_hash,
            "query_hash": content_hash(query.model_dump(mode="json")),
            "hits": hits,
            "status": ExternalProposalStatus.COMPLETE,
            "warnings": (),
            "trust_class": "external_proposal",
        }
        result_id = f"external-retrieval-result-{content_hash(identity)[:24]}"
        payload = {"result_id": result_id, **identity}
        return ExternalLiteratureRetrievalResult(
            **payload,
            content_hash=content_hash(payload),
        )
