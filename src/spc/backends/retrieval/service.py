from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..contracts import (
    BackendCapability,
    BackendInputBinding,
    BackendRunStatus,
    ExternalLiteratureRetrievalQuery,
    ExternalLiteratureRetrievalResult,
    ScientificRetrievalBackend,
)
from ..provenance import BackendRunRecord, BackendRunRepository, create_backend_run_record
from ...repositories import IdentityBoundRepository
from ...serialization import content_hash


class ExternalRetrievalResultRepository(IdentityBoundRepository[ExternalLiteratureRetrievalResult]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "backend_retrieval_results",
            ExternalLiteratureRetrievalResult,
            "result_id",
        )


@dataclass(frozen=True)
class ExternalRetrievalOutcome:
    result: ExternalLiteratureRetrievalResult
    run_record: BackendRunRecord


class ExternalLiteratureRetrievalService:
    def retrieve(
        self,
        backend: ScientificRetrievalBackend,
        query: ExternalLiteratureRetrievalQuery,
        knowledge_root: Path,
    ) -> ExternalRetrievalOutcome:
        descriptor = backend.descriptor
        if BackendCapability.LITERATURE_RETRIEVAL not in descriptor.capability_types:
            raise ValueError("backend does not provide literature_retrieval")
        query_hash = content_hash(query.model_dump(mode="json"))
        input_bindings = (
            BackendInputBinding(
                input_id=f"external-retrieval-query-{query_hash[:24]}",
                input_hash=query_hash,
            ),
        )
        availability = backend.inspect_availability()
        if not availability.available:
            run = create_backend_run_record(
                descriptor=descriptor,
                capability=BackendCapability.LITERATURE_RETRIEVAL,
                input_bindings=input_bindings,
                config_hash=content_hash(
                    {
                        "corpus_scope": query.corpus_scope,
                        "metadata_filters": query.metadata_filters,
                        "max_results": query.max_results,
                    }
                ),
                output_hash=None,
                status=BackendRunStatus.UNAVAILABLE,
                warnings=(availability.reason or "backend is unavailable",),
            )
            BackendRunRepository(knowledge_root).put(run.run_id, run)
            raise RuntimeError(f"backend {descriptor.backend_id} is unavailable; run_id={run.run_id}")
        config_hash = content_hash(
            {
                "corpus_scope": query.corpus_scope,
                "metadata_filters": query.metadata_filters,
                "max_results": query.max_results,
            }
        )
        try:
            result = ExternalLiteratureRetrievalResult.model_validate(backend.retrieve(query))
            if (
                result.backend_id != descriptor.backend_id
                or result.backend_descriptor_hash != descriptor.content_hash
                or result.query_hash != query_hash
            ):
                raise ValueError("external retrieval result binding is invalid")
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            run = create_backend_run_record(
                descriptor=descriptor,
                capability=BackendCapability.LITERATURE_RETRIEVAL,
                input_bindings=input_bindings,
                config_hash=config_hash,
                output_hash=None,
                status=BackendRunStatus.FAILED,
                warnings=("backend invocation or output validation failed",),
            )
            BackendRunRepository(knowledge_root).put(run.run_id, run)
            raise ValueError(f"external retrieval backend failed; run_id={run.run_id}") from error
        ExternalRetrievalResultRepository(knowledge_root).put(result.result_id, result)
        run = create_backend_run_record(
            descriptor=descriptor,
            capability=BackendCapability.LITERATURE_RETRIEVAL,
            input_bindings=input_bindings,
            config_hash=config_hash,
            output_hash=result.content_hash,
            status=BackendRunStatus.SUCCEEDED,
            warnings=result.warnings,
            resolved_count=len(result.hits),
        )
        BackendRunRepository(knowledge_root).put(run.run_id, run)
        return ExternalRetrievalOutcome(result=result, run_record=run)
