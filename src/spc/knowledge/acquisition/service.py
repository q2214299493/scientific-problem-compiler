"""Coordinate literature acquisition attempts and record their provenance."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Mapping
import hashlib
import shutil
import tempfile

from ...models import (
    AcquisitionAttemptRecord,
    AcquisitionInputKind,
    AcquisitionStatus,
    CanonicalHTMLTextArtifact,
    CanonicalTextBlock,
    FullTextAccessStatus,
    FullTextCandidate,
    HTMLLiteratureIngestionRecord,
    LiteratureAcquisitionOutcome,
    LiteratureAcquisitionRecord,
    LiteratureAcquisitionRequest,
    LiteratureDocument,
    LiteratureIngestionStatus,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    MetadataMergeManifest,
    RawHTMLLiteratureArtifact,
    ResolvedLiteratureResource,
    SourceDocument,
    literature_identity_id,
)
from ...repositories import EvidenceStore, KnowledgeRepositories
from ...serialization import content_hash
from ..ingestion import LiteratureIngestionService, LiteratureMetadata, validate_literature_ingestion_chain
from ..literature_text import AcquisitionError
from .metadata import (
    SUPPORTED_HTML_TYPES,
    SUPPORTED_PDF_TYPES,
    _make_request,
    _make_resource,
    _merge_metadata,
    _metadata_from_resource,
    _pdf_metadata,
    _reprioritize_candidates,
    detect_acquisition_input,
)
from .resolvers import ArticleURLResolver, DOIResolver, LocalPDFResolver, ResourceResolver
from .transport import SafeHTTPError, SafeHTTPFetcher


@dataclass(frozen=True)
class HTMLTextExtraction:
    source_url: str
    artifact_sha256: str
    canonical_text: str
    text_sha256: str
    blocks: tuple[CanonicalTextBlock, ...]


@dataclass(frozen=True)
class _CandidateResult:
    status: AcquisitionStatus
    failure_code: str | None
    http_status: int | None
    manifest: MetadataMergeManifest
    warnings: tuple[str, ...] = ()
    literature_id: str | None = None
    raw_artifact_id: str | None = None
    raw_artifact_hash: str | None = None
    canonical_text_id: str | None = None
    canonical_text_hash: str | None = None
    ingestion_id: str | None = None
    ingestion_hash: str | None = None
    representation: LiteratureRepresentationReference | None = None


class LiteratureAcquisitionService:
    def __init__(
        self,
        fetcher: SafeHTTPFetcher | None = None,
        *,
        html_extractor: HTMLLiteratureTextExtractor | None = None,
    ) -> None:
        self.fetcher = fetcher or SafeHTTPFetcher()
        self.html_extractor = html_extractor or HTMLLiteratureTextExtractor()

    def add(
        self,
        original_input: str,
        domain: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        explicit_metadata: Mapping[str, object] | None = None,
    ) -> LiteratureAcquisitionOutcome:
        input_kind = detect_acquisition_input(original_input)
        request = _make_request(original_input, input_kind, domain)
        repositories.acquisition_requests.put(request.request_id, request)
        resolver: ResourceResolver
        if input_kind == AcquisitionInputKind.LOCAL_FILE:
            resolver = LocalPDFResolver()
        elif input_kind == AcquisitionInputKind.DOI:
            resolver = DOIResolver(self.fetcher)
        else:
            resolver = ArticleURLResolver(self.fetcher)
        try:
            resource = resolver.resolve(request)
        except (AcquisitionError, SafeHTTPError) as error:
            status = AcquisitionStatus.FAILED
            if isinstance(error, SafeHTTPError):
                if error.code == "REQUIRES_AUTHENTICATION":
                    status = AcquisitionStatus.REQUIRES_AUTHENTICATION
                elif error.code == "UNSUPPORTED_CONTENT_TYPE":
                    status = AcquisitionStatus.UNSUPPORTED_MEDIA
            failure_retrievals = tuple(
                getattr(resolver, "metadata_retrievals", ())
            )
            for retrieval in failure_retrievals:
                repositories.metadata_retrievals.put(
                    retrieval.retrieval_id, retrieval
                )
            resource = _make_resource(
                canonical_identifier=request.original_input,
                authors=(),
                metadata_source=f"{resolver.resolver_id}:failed",
                metadata_retrieval_refs=tuple(
                    item.retrieval_id for item in failure_retrievals
                ),
                metadata_retrieval_hashes=tuple(
                    item.content_hash for item in failure_retrievals
                ),
                fulltext_candidates=(),
                resolution_status=status,
            )
            repositories.resolved_literature_resources.put(
                resource.resource_id, resource
            )
            return self._finish(
                request,
                resolver,
                resource,
                repositories,
                status=status,
                warnings=(str(error),),
            )
        retrievals = tuple(getattr(resolver, "metadata_retrievals", ()))
        expected_retrieval_ids = tuple(item.retrieval_id for item in retrievals)
        expected_retrieval_hashes = tuple(item.content_hash for item in retrievals)
        if (
            resource.metadata_retrieval_refs != expected_retrieval_ids
            or resource.metadata_retrieval_hashes != expected_retrieval_hashes
        ):
            raise AcquisitionError("resolved metadata provenance binding is incomplete")
        for retrieval in retrievals:
            repositories.metadata_retrievals.put(
                retrieval.retrieval_id, retrieval
            )
        repositories.resolved_literature_resources.put(resource.resource_id, resource)
        attempts: list[AcquisitionAttemptRecord] = []
        results: list[_CandidateResult] = []
        attempted_keys: set[tuple[str | None, str | None, str]] = set()
        discovery_failure_codes: list[str] = []
        active_resource = resource
        stages = 2 if input_kind == AcquisitionInputKind.DOI else 1
        for stage_index in range(stages):
            if stage_index == 1:
                active_resource, discovery_code = self._discover_doi_landing(
                    request, active_resource, repositories
                )
                if discovery_code is not None:
                    discovery_failure_codes.append(discovery_code)
            candidates = tuple(
                item
                for item in active_resource.fulltext_candidates
                if item.access_status
                in {
                    FullTextAccessStatus.DISCOVERED,
                    FullTextAccessStatus.ACCESSIBLE,
                }
                and self._candidate_key(item) not in attempted_keys
            )
            for candidate in candidates:
                attempted_keys.add(self._candidate_key(candidate))
                result = self._attempt_candidate(
                    candidate,
                    active_resource,
                    request,
                    repositories,
                    evidence_store,
                    explicit_metadata,
                )
                repositories.metadata_merge_manifests.put(
                    result.manifest.manifest_id, result.manifest
                )
                attempt = self._make_attempt(
                    request, candidate, len(attempts), result
                )
                repositories.acquisition_attempts.put(attempt.attempt_id, attempt)
                attempts.append(attempt)
                results.append(result)
                if result.status == AcquisitionStatus.INGESTED:
                    return self._finish(
                        request,
                        resolver,
                        active_resource,
                        repositories,
                        status=AcquisitionStatus.INGESTED,
                        candidate=candidate,
                        manifest=result.manifest,
                        attempts=tuple(attempts),
                        literature_id=result.literature_id,
                        ingestion_id=result.ingestion_id,
                        canonical_text_id=result.canonical_text_id,
                        representation=result.representation,
                        warnings=result.warnings,
                    )
        final_status = self._final_failure_status(
            results, discovery_failure_codes, active_resource
        )
        warnings = tuple(
            dict.fromkeys(
                (
                    *(
                        warning
                        for result in results
                        for warning in (
                            *result.warnings,
                            *((result.failure_code,) if result.failure_code else ()),
                        )
                    ),
                    *discovery_failure_codes,
                )
            )
        )
        return self._finish(
            request,
            resolver,
            active_resource,
            repositories,
            status=final_status,
            attempts=tuple(attempts),
            warnings=warnings,
        )

    @staticmethod
    def _candidate_key(
        candidate: FullTextCandidate,
    ) -> tuple[str | None, str | None, str]:
        return (candidate.url, candidate.local_path_ref, candidate.media_type)

    def _discover_doi_landing(
        self,
        request: LiteratureAcquisitionRequest,
        resource: ResolvedLiteratureResource,
        repositories: KnowledgeRepositories,
    ) -> tuple[ResolvedLiteratureResource, str | None]:
        if not resource.landing_url:
            return resource, None
        resolver = ArticleURLResolver(self.fetcher)
        fallback_request = _make_request(
            resource.landing_url,
            AcquisitionInputKind.URL,
            request.requested_domain,
        )
        try:
            fallback = resolver.resolve(fallback_request)
        except SafeHTTPError as error:
            return resource, error.code
        except AcquisitionError:
            return resource, "LANDING_DISCOVERY_FAILED"
        for retrieval in resolver.metadata_retrievals:
            repositories.metadata_retrievals.put(
                retrieval.retrieval_id, retrieval
            )
        retrieval_pairs = tuple(
            dict.fromkeys(
                (
                    *zip(
                        resource.metadata_retrieval_refs,
                        resource.metadata_retrieval_hashes,
                    ),
                    *(
                        (item.retrieval_id, item.content_hash)
                        for item in resolver.metadata_retrievals
                    ),
                )
            )
        )
        candidates = _reprioritize_candidates(
            (*resource.fulltext_candidates, *fallback.fulltext_candidates)
        )
        enriched = _make_resource(
            canonical_identifier=resource.canonical_identifier,
            doi=resource.doi or fallback.doi,
            title=resource.title or fallback.title,
            authors=resource.authors or fallback.authors,
            year=resource.year or fallback.year,
            journal=resource.journal or fallback.journal,
            landing_url=resource.landing_url,
            metadata_source=f"{resource.metadata_source}+article-url",
            metadata_retrieval_refs=tuple(item[0] for item in retrieval_pairs),
            metadata_retrieval_hashes=tuple(item[1] for item in retrieval_pairs),
            fulltext_candidates=candidates,
            resolution_status=(
                AcquisitionStatus.FULLTEXT_FOUND
                if candidates
                else AcquisitionStatus.FULLTEXT_UNAVAILABLE
            ),
        )
        repositories.resolved_literature_resources.put(
            enriched.resource_id, enriched
        )
        return enriched, None

    @staticmethod
    def _final_failure_status(
        results: list[_CandidateResult],
        discovery_failure_codes: list[str],
        resource: ResolvedLiteratureResource,
    ) -> AcquisitionStatus:
        if any(item.status == AcquisitionStatus.METADATA_ONLY for item in results):
            return AcquisitionStatus.METADATA_ONLY
        failure_codes = [
            item.failure_code for item in results if item.failure_code is not None
        ] + discovery_failure_codes
        if failure_codes and all(
            code == "REQUIRES_AUTHENTICATION" for code in failure_codes
        ):
            return AcquisitionStatus.REQUIRES_AUTHENTICATION
        if results and all(
            item.status == AcquisitionStatus.UNSUPPORTED_MEDIA for item in results
        ):
            return AcquisitionStatus.UNSUPPORTED_MEDIA
        hard_failures = {
            "CONTENT_HASH_CHANGED",
            "HTML_EXTRACTION_FAILED",
            "LANDING_DISCOVERY_FAILED",
            "NETWORK_ERROR",
            "PDF_INGESTION_FAILED",
            "REQUIRES_OCR",
        }
        if any(code in hard_failures for code in failure_codes):
            return AcquisitionStatus.FAILED
        if resource.resolution_status == AcquisitionStatus.UNSUPPORTED_MEDIA:
            return AcquisitionStatus.UNSUPPORTED_MEDIA
        return AcquisitionStatus.FULLTEXT_UNAVAILABLE

    def _attempt_candidate(
        self,
        candidate: FullTextCandidate,
        resource: ResolvedLiteratureResource,
        request: LiteratureAcquisitionRequest,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        explicit_metadata: Mapping[str, object] | None,
    ) -> _CandidateResult:
        if candidate.media_type == "application/pdf":
            return self._attempt_pdf(
                candidate,
                resource,
                request,
                repositories,
                evidence_store,
                explicit_metadata,
            )
        if candidate.media_type in SUPPORTED_HTML_TYPES:
            return self._attempt_html(
                candidate,
                resource,
                request,
                repositories,
                evidence_store,
                explicit_metadata,
            )
        _, manifest = _merge_metadata(
            explicit_metadata,
            _metadata_from_resource(resource),
            {},
            domain=request.requested_domain,
        )
        return _CandidateResult(
            status=AcquisitionStatus.UNSUPPORTED_MEDIA,
            failure_code="UNSUPPORTED_MEDIA",
            http_status=None,
            manifest=manifest,
            warnings=("Selected full text uses unsupported media.",),
        )

    def _attempt_pdf(
        self,
        candidate: FullTextCandidate,
        resource: ResolvedLiteratureResource,
        request: LiteratureAcquisitionRequest,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        explicit_metadata: Mapping[str, object] | None,
    ) -> _CandidateResult:
        http_status: int | None = None
        if candidate.local_path_ref is not None:
            pdf_path = Path(candidate.local_path_ref)
            if pdf_path.is_symlink() or not pdf_path.is_file():
                _, manifest = _merge_metadata(
                    explicit_metadata,
                    _metadata_from_resource(resource),
                    {},
                    domain=request.requested_domain,
                )
                return _CandidateResult(
                    status=AcquisitionStatus.FULLTEXT_UNAVAILABLE,
                    failure_code="LOCAL_FILE_UNAVAILABLE",
                    http_status=None,
                    manifest=manifest,
                )
            pdf_bytes = pdf_path.read_bytes()
        else:
            try:
                response = self.fetcher.fetch(
                    candidate.url or "", allowed_content_types=SUPPORTED_PDF_TYPES
                )
                pdf_bytes = response.body
                http_status = response.status
            except SafeHTTPError as error:
                _, manifest = _merge_metadata(
                    explicit_metadata,
                    _metadata_from_resource(resource),
                    {},
                    domain=request.requested_domain,
                )
                return _CandidateResult(
                    status=AcquisitionStatus.FULLTEXT_UNAVAILABLE,
                    failure_code=error.code,
                    http_status=error.status,
                    manifest=manifest,
                    warnings=(str(error),),
                )
        if not pdf_bytes.startswith(b"%PDF-"):
            _, manifest = _merge_metadata(
                explicit_metadata,
                _metadata_from_resource(resource),
                {},
                domain=request.requested_domain,
            )
            return _CandidateResult(
                status=AcquisitionStatus.UNSUPPORTED_MEDIA,
                failure_code="INVALID_PDF_SIGNATURE",
                http_status=http_status,
                manifest=manifest,
            )
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        if candidate.content_sha256 is not None and candidate.content_sha256 != digest:
            _, manifest = _merge_metadata(
                explicit_metadata,
                _metadata_from_resource(resource),
                {},
                domain=request.requested_domain,
            )
            return _CandidateResult(
                status=AcquisitionStatus.FAILED,
                failure_code="CONTENT_HASH_CHANGED",
                http_status=http_status,
                manifest=manifest,
            )
        try:
            embedded = _pdf_metadata(pdf_bytes)
        except AcquisitionError:
            embedded = {}
        metadata, manifest = _merge_metadata(
            explicit_metadata,
            _metadata_from_resource(resource),
            embedded,
            domain=request.requested_domain,
        )
        if metadata is None:
            return _CandidateResult(
                status=AcquisitionStatus.METADATA_ONLY,
                failure_code="INCOMPLETE_METADATA",
                http_status=http_status,
                manifest=manifest,
            )
        if candidate.local_path_ref is not None:
            return self._run_k1b(
                pdf_path,
                pdf_bytes,
                metadata,
                manifest,
                repositories,
                evidence_store,
                http_status,
            )
        staging_root = repositories.root / ".acquisition_staging"
        if staging_root.exists() and staging_root.is_symlink():
            raise AcquisitionError("acquisition staging root cannot be a symlink")
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="download-", dir=staging_root))
        try:
            pdf_path = staging / "resource.pdf"
            pdf_path.write_bytes(pdf_bytes)
            if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != digest:
                raise AcquisitionError("staged PDF integrity check failed")
            return self._run_k1b(
                pdf_path,
                pdf_bytes,
                metadata,
                manifest,
                repositories,
                evidence_store,
                http_status,
            )
        finally:
            if staging.exists():
                if not staging.resolve().is_relative_to(staging_root.resolve()):
                    raise AcquisitionError("acquisition staging path escaped its root")
                shutil.rmtree(staging)

    def _run_k1b(
        self,
        pdf_path: Path,
        pdf_bytes: bytes,
        metadata: LiteratureMetadata,
        manifest: MetadataMergeManifest,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        http_status: int | None,
    ) -> _CandidateResult:
        if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != hashlib.sha256(
            pdf_bytes
        ).hexdigest():
            raise AcquisitionError("staged PDF integrity check failed")
        outcome = LiteratureIngestionService().ingest(
            pdf_path, metadata, repositories, evidence_store
        )
        ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
        raw = repositories.raw_literature_artifacts.get(outcome.artifact_id)
        if outcome.canonical_text_id is None:
            return _CandidateResult(
                status=AcquisitionStatus.FAILED,
                failure_code=(
                    "REQUIRES_OCR" if outcome.requires_ocr else "PDF_INGESTION_FAILED"
                ),
                http_status=http_status,
                manifest=manifest,
                warnings=outcome.warnings,
                literature_id=outcome.literature_id,
                raw_artifact_id=raw.artifact_id,
                raw_artifact_hash=raw.content_hash,
                ingestion_id=ingestion.ingestion_id,
                ingestion_hash=ingestion.content_hash,
            )
        canonical = repositories.canonical_text_artifacts.get(
            outcome.canonical_text_id
        )
        representation = self._representation(
            kind=LiteratureRepresentationKind.PDF,
            literature_id=outcome.literature_id,
            raw_artifact_id=raw.artifact_id,
            raw_artifact_hash=raw.content_hash,
            canonical_text_id=canonical.canonical_text_id,
            canonical_text_hash=canonical.content_hash,
            ingestion_id=ingestion.ingestion_id,
            ingestion_hash=ingestion.content_hash,
            source_id=outcome.source_id,
            source_version=outcome.source_version,
        )
        repositories.literature_representation_refs.put(
            representation.representation_id, representation
        )
        validate_literature_representation(
            representation, repositories, evidence_store
        )
        return _CandidateResult(
            status=AcquisitionStatus.INGESTED,
            failure_code=None,
            http_status=http_status,
            manifest=manifest,
            warnings=outcome.warnings,
            literature_id=outcome.literature_id,
            raw_artifact_id=raw.artifact_id,
            raw_artifact_hash=raw.content_hash,
            canonical_text_id=canonical.canonical_text_id,
            canonical_text_hash=canonical.content_hash,
            ingestion_id=ingestion.ingestion_id,
            ingestion_hash=ingestion.content_hash,
            representation=representation,
        )

    def _attempt_html(
        self,
        candidate: FullTextCandidate,
        resource: ResolvedLiteratureResource,
        request: LiteratureAcquisitionRequest,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
        explicit_metadata: Mapping[str, object] | None,
    ) -> _CandidateResult:
        _, base_manifest = _merge_metadata(
            explicit_metadata,
            _metadata_from_resource(resource),
            {},
            domain=request.requested_domain,
        )
        try:
            response = self.fetcher.fetch(
                candidate.url or "", allowed_content_types=SUPPORTED_HTML_TYPES
            )
            extraction = self.html_extractor.extract(response.url, response.body)
        except (AcquisitionError, SafeHTTPError) as error:
            return _CandidateResult(
                status=AcquisitionStatus.FULLTEXT_UNAVAILABLE,
                failure_code=(error.code if isinstance(error, SafeHTTPError) else "HTML_EXTRACTION_FAILED"),
                http_status=(error.status if isinstance(error, SafeHTTPError) else None),
                manifest=base_manifest,
                warnings=(str(error),),
            )
        if (
            candidate.content_sha256 is not None
            and candidate.content_sha256 != extraction.artifact_sha256
        ):
            return _CandidateResult(
                status=AcquisitionStatus.FAILED,
                failure_code="CONTENT_HASH_CHANGED",
                http_status=response.status,
                manifest=base_manifest,
            )
        metadata, manifest = _merge_metadata(
            explicit_metadata,
            _metadata_from_resource(resource),
            {},
            domain=request.requested_domain,
        )
        if metadata is None:
            return _CandidateResult(
                status=AcquisitionStatus.METADATA_ONLY,
                failure_code="INCOMPLETE_METADATA",
                http_status=response.status,
                manifest=manifest,
            )
        literature_id = literature_identity_id(
            title=metadata.title,
            authors=metadata.authors,
            year=metadata.year,
            doi=metadata.doi,
        )
        media_type = response.headers.get("content-type", "").split(";", 1)[0]
        raw = repositories.raw_html_literature_artifacts.put(
            response.body,
            source_url=response.url,
            literature_id=literature_id,
            media_type=media_type,
        )
        canonical = repositories.canonical_html_text_artifacts.put(
            extraction.canonical_text,
            raw_artifact=raw,
            extractor_id=self.html_extractor.extractor_id,
            extractor_version=self.html_extractor.extractor_version,
            extractor_config_hash=self.html_extractor.extractor_config_hash,
            blocks=extraction.blocks,
        )
        canonical_path = repositories.root.joinpath(
            *Path(canonical.stored_path).parts
        )
        source = evidence_store.ingest_source(
            canonical_path,
            f"source-{literature_id}",
            f"html-{canonical.canonical_text_id.removeprefix('canonical-html-text-')}",
            metadata.title,
            source_role="literature_author",
            source_type="literature_article",
        )
        ingestion = self._html_ingestion(raw, canonical, source)
        repositories.html_literature_ingestions.put(
            ingestion.ingestion_id, ingestion
        )
        self._finalize_html_document(
            metadata, raw, canonical, source, repositories
        )
        representation = self._representation(
            kind=LiteratureRepresentationKind.HTML,
            literature_id=literature_id,
            raw_artifact_id=raw.artifact_id,
            raw_artifact_hash=raw.content_hash,
            canonical_text_id=canonical.canonical_text_id,
            canonical_text_hash=canonical.content_hash,
            ingestion_id=ingestion.ingestion_id,
            ingestion_hash=ingestion.content_hash,
            source_id=source.source_id,
            source_version=source.version,
        )
        repositories.literature_representation_refs.put(
            representation.representation_id, representation
        )
        validate_literature_representation(
            representation, repositories, evidence_store
        )
        return _CandidateResult(
            status=AcquisitionStatus.INGESTED,
            failure_code=None,
            http_status=response.status,
            manifest=manifest,
            literature_id=literature_id,
            raw_artifact_id=raw.artifact_id,
            raw_artifact_hash=raw.content_hash,
            canonical_text_id=canonical.canonical_text_id,
            canonical_text_hash=canonical.content_hash,
            ingestion_id=ingestion.ingestion_id,
            ingestion_hash=ingestion.content_hash,
            representation=representation,
        )

    def _html_ingestion(
        self,
        raw: RawHTMLLiteratureArtifact,
        canonical: CanonicalHTMLTextArtifact,
        source: SourceDocument,
    ) -> HTMLLiteratureIngestionRecord:
        identity = {
            "literature_id": raw.literature_id,
            "raw_artifact_id": raw.artifact_id,
            "raw_artifact_hash": raw.content_hash,
            "canonical_text_id": canonical.canonical_text_id,
            "canonical_text_hash": canonical.content_hash,
            "source_id": source.source_id,
            "source_version": source.version,
            "extractor_id": canonical.extractor_id,
            "extractor_version": canonical.extractor_version,
            "extractor_config_hash": canonical.extractor_config_hash,
            "ingestion_status": LiteratureIngestionStatus.ACCEPTED,
        }
        ingestion_id = f"html-literature-ingestion-{content_hash(identity)[:24]}"
        payload = {"ingestion_id": ingestion_id, **identity}
        return HTMLLiteratureIngestionRecord(
            **payload, content_hash=content_hash(payload)
        )

    @staticmethod
    def _finalize_html_document(
        metadata: LiteratureMetadata,
        raw: RawHTMLLiteratureArtifact,
        canonical: CanonicalHTMLTextArtifact,
        source: SourceDocument,
        repositories: KnowledgeRepositories,
    ) -> LiteratureDocument:
        try:
            return repositories.literature_documents.get(raw.literature_id)
        except FileNotFoundError:
            pass
        identity = {
            **metadata.model_dump(mode="json", exclude_none=True),
            "literature_id": raw.literature_id,
            "raw_artifact_ref": raw.stored_path,
            "canonical_text_ref": canonical.stored_path,
            "source_id": source.source_id,
            "source_version": source.version,
        }
        document = LiteratureDocument(
            **identity, content_hash=content_hash(identity)
        )
        repositories.literature_documents.put(document.literature_id, document)
        return document

    @staticmethod
    def _representation(
        *,
        kind: LiteratureRepresentationKind,
        literature_id: str,
        raw_artifact_id: str,
        raw_artifact_hash: str,
        canonical_text_id: str | None,
        canonical_text_hash: str | None,
        ingestion_id: str,
        ingestion_hash: str,
        source_id: str | None,
        source_version: str | None,
    ) -> LiteratureRepresentationReference:
        identity = {
            "representation_kind": kind,
            "literature_id": literature_id,
            "raw_artifact_id": raw_artifact_id,
            "raw_artifact_hash": raw_artifact_hash,
            "canonical_text_id": canonical_text_id,
            "canonical_text_hash": canonical_text_hash,
            "ingestion_id": ingestion_id,
            "ingestion_hash": ingestion_hash,
            "source_id": source_id,
            "source_version": source_version,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        representation_id = f"literature-representation-{content_hash(identity)[:24]}"
        payload = {"representation_id": representation_id, **identity}
        return LiteratureRepresentationReference(
            **payload, content_hash=content_hash(payload)
        )

    @staticmethod
    def _make_attempt(
        request: LiteratureAcquisitionRequest,
        candidate: FullTextCandidate,
        attempt_index: int,
        result: _CandidateResult,
    ) -> AcquisitionAttemptRecord:
        identity = {
            "request_id": request.request_id,
            "request_hash": request.content_hash,
            "candidate_id": candidate.candidate_id,
            "candidate_hash": candidate.content_hash,
            "attempt_index": attempt_index,
            "status": result.status,
            "http_status": result.http_status,
            "media_type": candidate.media_type,
            "failure_code": result.failure_code,
            "raw_artifact_id": result.raw_artifact_id,
            "raw_artifact_hash": result.raw_artifact_hash,
            "canonical_text_id": result.canonical_text_id,
            "canonical_text_hash": result.canonical_text_hash,
            "ingestion_id": result.ingestion_id,
            "ingestion_hash": result.ingestion_hash,
            "representation_id": (
                result.representation.representation_id
                if result.representation is not None
                else None
            ),
            "representation_hash": (
                result.representation.content_hash
                if result.representation is not None
                else None
            ),
            "metadata_merge_manifest_id": result.manifest.manifest_id,
            "metadata_merge_manifest_hash": result.manifest.content_hash,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        attempt_id = f"acquisition-attempt-{content_hash(identity)[:24]}"
        payload = {"attempt_id": attempt_id, **identity}
        return AcquisitionAttemptRecord(
            **payload, content_hash=content_hash(payload)
        )

    @staticmethod
    def _request(
        original_input: str,
        input_kind: AcquisitionInputKind,
        domain: str,
    ) -> LiteratureAcquisitionRequest:
        return _make_request(original_input, input_kind, domain)

    @staticmethod
    def _finish(
        request: LiteratureAcquisitionRequest,
        resolver: ResourceResolver,
        resource: ResolvedLiteratureResource,
        repositories: KnowledgeRepositories,
        *,
        status: AcquisitionStatus,
        candidate: FullTextCandidate | None = None,
        manifest: MetadataMergeManifest | None = None,
        attempts: tuple[AcquisitionAttemptRecord, ...] = (),
        literature_id: str | None = None,
        ingestion_id: str | None = None,
        canonical_text_id: str | None = None,
        representation: LiteratureRepresentationReference | None = None,
        warnings: tuple[str, ...] = (),
    ) -> LiteratureAcquisitionOutcome:
        identity = {
            "request_id": request.request_id,
            "request_hash": request.content_hash,
            "resolver_id": resolver.resolver_id,
            "resolver_version": resolver.resolver_version,
            "resolved_resource_id": resource.resource_id,
            "resolved_resource_hash": resource.content_hash,
            "selected_candidate_id": candidate.candidate_id if candidate else None,
            "selected_candidate_hash": candidate.content_hash if candidate else None,
            "metadata_merge_manifest_id": manifest.manifest_id if manifest else None,
            "metadata_merge_manifest_hash": manifest.content_hash if manifest else None,
            "attempt_refs": tuple(item.attempt_id for item in attempts),
            "attempt_hashes": tuple(item.content_hash for item in attempts),
            "resulting_literature_id": literature_id,
            "resulting_ingestion_id": ingestion_id,
            "resulting_canonical_text_id": canonical_text_id,
            "resulting_representation_id": (
                representation.representation_id if representation else None
            ),
            "resulting_representation_hash": (
                representation.content_hash if representation else None
            ),
            "status": status,
            "warnings": warnings,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        acquisition_id = f"literature-acquisition-{content_hash(identity)[:24]}"
        payload = {"acquisition_id": acquisition_id, **identity}
        record = LiteratureAcquisitionRecord(
            **payload, content_hash=content_hash(payload)
        )
        repositories.literature_acquisitions.put(record.acquisition_id, record)
        return LiteratureAcquisitionOutcome(
            input_kind=request.input_kind,
            canonical_identifier=resource.canonical_identifier,
            doi=resource.doi,
            title=resource.title,
            acquisition_status=status,
            fulltext_status=(
                AcquisitionStatus.FULLTEXT_FOUND
                if status == AcquisitionStatus.INGESTED
                else resource.resolution_status
                if status == AcquisitionStatus.METADATA_ONLY
                else status
            ),
            literature_id=literature_id,
            ingestion_id=ingestion_id,
            canonical_text_id=canonical_text_id,
            representation_id=(
                representation.representation_id if representation else None
            ),
            acquisition_id=record.acquisition_id,
            attempt_ids=tuple(item.attempt_id for item in attempts),
            warnings=warnings,
        )


class _CanonicalHTMLParser(HTMLParser):
    BLOCK_TAGS = frozenset(
        {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "li",
            "blockquote",
            "caption",
            "th",
            "td",
            "figcaption",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.active: list[tuple[str, list[str]]] = []
        self.blocks: list[tuple[str, str]] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and normalized in self.BLOCK_TAGS:
            self.active.append((normalized, []))

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
            return
        if self.ignored_depth or normalized not in self.BLOCK_TAGS:
            return
        for index in range(len(self.active) - 1, -1, -1):
            active_tag, content = self.active[index]
            if active_tag == normalized:
                text = " ".join("".join(content).split())
                if text:
                    self.blocks.append((normalized, text))
                del self.active[index]
                break

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth and self.active:
            self.active[-1][1].append(data)


class HTMLLiteratureTextExtractor:
    extractor_id = "html-canonical-text"
    extractor_version = "2.0.0"
    extractor_config_hash = content_hash(
        {
            "block_tags": tuple(sorted(_CanonicalHTMLParser.BLOCK_TAGS)),
            "ignored_tags": ("noscript", "script", "style"),
            "encoding": "utf-8",
            "block_separator": "two_lf",
        }
    )

    def extract(self, source_url: str, html_bytes: bytes) -> HTMLTextExtraction:
        try:
            html_text = html_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AcquisitionError("HTML resource is not UTF-8") from error
        parser = _CanonicalHTMLParser()
        parser.feed(html_text)
        if not parser.blocks:
            raise AcquisitionError("HTML article contains no canonical text blocks")
        texts = [text for _, text in parser.blocks]
        canonical_text = "\n\n".join(texts)
        blocks: list[CanonicalTextBlock] = []
        cursor = 0
        for block_type, block_text in parser.blocks:
            end = cursor + len(block_text)
            identity = {
                "page_number": 1,
                "block_type": f"html_{block_type}",
                "section_path": (),
                "start_offset": cursor,
                "end_offset": end,
                "text_hash": hashlib.sha256(block_text.encode("utf-8")).hexdigest(),
            }
            blocks.append(
                CanonicalTextBlock(
                    block_id=f"canonical-block-{content_hash(identity)[:24]}",
                    **identity,
                )
            )
            cursor = end + 2
        return HTMLTextExtraction(
            source_url=source_url,
            artifact_sha256=hashlib.sha256(html_bytes).hexdigest(),
            canonical_text=canonical_text,
            text_sha256=hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
            blocks=tuple(blocks),
        )


def validate_literature_representation(
    representation: LiteratureRepresentationReference,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
) -> LiteratureRepresentationReference:
    stored = repositories.literature_representation_refs.get(
        representation.representation_id
    )
    if stored != representation:
        raise ValueError("literature representation differs from repository record")
    if representation.representation_kind == LiteratureRepresentationKind.PDF:
        ingestion = repositories.literature_ingestions.get(
            representation.ingestion_id
        )
        chain = validate_literature_ingestion_chain(
            ingestion, repositories, evidence_store
        )
        if (
            representation.ingestion_hash != ingestion.content_hash
            or representation.raw_artifact_id != chain.raw_artifact.artifact_id
            or representation.raw_artifact_hash != chain.raw_artifact.content_hash
            or representation.canonical_text_id
            != chain.canonical_text.canonical_text_id
            or representation.canonical_text_hash
            != chain.canonical_text.content_hash
            or representation.source_id != chain.source.source_id
            or representation.source_version != chain.source.version
        ):
            raise ValueError("PDF representation binding is invalid")
        return representation
    ingestion = repositories.html_literature_ingestions.get(
        representation.ingestion_id
    )
    raw = repositories.raw_html_literature_artifacts.get(
        representation.raw_artifact_id
    )
    canonical = repositories.canonical_html_text_artifacts.get(
        representation.canonical_text_id or ""
    )
    source = evidence_store.get_source(
        representation.source_id, representation.source_version
    )
    evidence_store.verify_source_integrity(source)
    if (
        representation.ingestion_hash != ingestion.content_hash
        or representation.literature_id != ingestion.literature_id
        or representation.raw_artifact_hash != raw.content_hash
        or representation.canonical_text_hash != canonical.content_hash
        or ingestion.raw_artifact_id != raw.artifact_id
        or ingestion.raw_artifact_hash != raw.content_hash
        or ingestion.canonical_text_id != canonical.canonical_text_id
        or ingestion.canonical_text_hash != canonical.content_hash
        or canonical.raw_artifact_id != raw.artifact_id
        or canonical.raw_artifact_hash != raw.content_hash
        or ingestion.source_id != source.source_id
        or ingestion.source_version != source.version
        or source.content_sha256 != canonical.text_sha256
    ):
        raise ValueError("HTML representation binding is invalid")
    return representation
