from __future__ import annotations

from enum import StrEnum
import unicodedata

from pydantic import Field, model_validator

from .base import NonBlankStr, Sha256Str, StrictModel


def _normalize_identity_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalize_doi(value: str) -> str:
    normalized = _normalize_identity_text(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix).strip()
            break
    return normalized


class LiteratureIngestionStatus(StrEnum):
    ACCEPTED = "accepted"
    REQUIRES_OCR = "requires_ocr"
    FAILED = "failed"


class AcquisitionInputKind(StrEnum):
    DOI = "doi"
    URL = "url"
    LOCAL_FILE = "local_file"


class AcquisitionStatus(StrEnum):
    DISCOVERED = "discovered"
    METADATA_RESOLVED = "metadata_resolved"
    FULLTEXT_FOUND = "fulltext_found"
    INGESTED = "ingested"
    METADATA_ONLY = "metadata_only"
    FULLTEXT_UNAVAILABLE = "fulltext_unavailable"
    REQUIRES_AUTHENTICATION = "requires_authentication"
    UNSUPPORTED_MEDIA = "unsupported_media"
    FAILED = "failed"


class FullTextAccessStatus(StrEnum):
    DISCOVERED = "discovered"
    ACCESSIBLE = "accessible"
    REQUIRES_AUTHENTICATION = "requires_authentication"
    UNAVAILABLE = "unavailable"


class FullTextSourceKind(StrEnum):
    LOCAL_FILE = "local_file"
    DIRECT_PDF = "direct_pdf"
    HTML_ARTICLE = "html_article"
    LANDING_PAGE_PDF = "landing_page_pdf"
    METADATA_LINK = "metadata_link"


class LiteratureRepresentationKind(StrEnum):
    PDF = "pdf"
    HTML = "html"


class DocumentBlockType(StrEnum):
    PAGE = "page"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_CAPTION = "table_caption"
    TABLE_CELL = "table_cell"
    FIGURE_CAPTION = "figure_caption"
    OTHER_TEXT = "other_text"


class DocumentContentRegion(StrEnum):
    MAIN_CONTENT = "main_content"
    REFERENCES = "references"
    NAVIGATION = "navigation"
    FOOTER = "footer"
    RELATED_CONTENT = "related_content"
    SUPPLEMENTARY_CONTEXT = "supplementary_context"
    UNKNOWN = "unknown"


class StructureExtractionStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


class MetadataValueOrigin(StrEnum):
    EXPLICIT = "explicit"
    RESOLVED = "resolved"
    EMBEDDED = "embedded"
    UNRESOLVED = "unresolved"


class LiteratureWorkflowPattern(StrictModel):
    pattern_id: str
    domain: str
    trigger: NonBlankStr
    workflow_capabilities: tuple[str, ...]
    limitations: tuple[NonBlankStr, ...] = ()
    evidence_refs: tuple[str, ...] = ()


class LiteratureDocument(StrictModel):
    literature_id: NonBlankStr
    title: NonBlankStr
    authors: tuple[NonBlankStr, ...] = Field(min_length=1)
    year: int = Field(ge=1000, le=9999)
    journal: NonBlankStr | None = None
    doi: NonBlankStr | None = None
    url: NonBlankStr | None = None
    domain: NonBlankStr
    topics: tuple[NonBlankStr, ...] = ()
    keywords: tuple[NonBlankStr, ...] = ()
    raw_artifact_ref: NonBlankStr
    canonical_text_ref: NonBlankStr
    source_id: NonBlankStr
    source_version: NonBlankStr
    citation_refs: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureDocument:
        from ..serialization import content_hash

        for field_name in ("authors", "topics", "keywords", "citation_refs"):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"LiteratureDocument {field_name} must be unique")
        expected_id = literature_identity_id(
            title=self.title,
            authors=self.authors,
            year=self.year,
            doi=self.doi,
        )
        if self.literature_id != expected_id:
            raise ValueError("LiteratureDocument literature_id is not content-bound")
        payload = self.model_dump(mode="json", exclude={"content_hash"}, exclude_none=True)
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureDocument content_hash is invalid")
        return self


def literature_identity_id(
    *,
    title: str,
    authors: tuple[str, ...],
    year: int,
    doi: str | None,
) -> str:
    from ..serialization import content_hash

    if doi is not None:
        stable_identity = {"doi": _normalize_doi(doi)}
    else:
        stable_identity = {
            "title": _normalize_identity_text(title),
            "authors": tuple(_normalize_identity_text(item) for item in authors),
            "year": year,
        }
    return f"literature-{content_hash(stable_identity)[:24]}"


class LiteratureAcquisitionRequest(StrictModel):
    request_id: NonBlankStr
    original_input: NonBlankStr
    input_kind: AcquisitionInputKind
    requested_domain: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureAcquisitionRequest:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"request_id", "content_hash"}
        )
        expected_id = f"literature-acquisition-request-{content_hash(identity)[:24]}"
        if self.request_id != expected_id:
            raise ValueError("LiteratureAcquisitionRequest request_id is not content-bound")
        payload = {"request_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureAcquisitionRequest content_hash is invalid")
        return self


class FullTextCandidate(StrictModel):
    candidate_id: NonBlankStr
    url: NonBlankStr | None = None
    local_path_ref: NonBlankStr | None = None
    media_type: NonBlankStr
    access_status: FullTextAccessStatus
    source_kind: FullTextSourceKind
    discovered_by: NonBlankStr = "unspecified"
    discovery_source_url: NonBlankStr | None = None
    priority: int = Field(ge=0)
    content_sha256: Sha256Str | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> FullTextCandidate:
        from ..serialization import content_hash

        if (self.url is None) == (self.local_path_ref is None):
            raise ValueError("FullTextCandidate requires exactly one URL or local path")
        identity = self.model_dump(
            mode="json",
            exclude={"candidate_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"fulltext-candidate-{content_hash(identity)[:24]}"
        if self.candidate_id != expected_id:
            raise ValueError("FullTextCandidate candidate_id is not content-bound")
        payload = {"candidate_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("FullTextCandidate content_hash is invalid")
        return self


class ResolvedLiteratureResource(StrictModel):
    resource_id: NonBlankStr
    canonical_identifier: NonBlankStr
    doi: NonBlankStr | None = None
    title: NonBlankStr | None = None
    authors: tuple[NonBlankStr, ...] = ()
    year: int | None = Field(default=None, ge=1000, le=9999)
    journal: NonBlankStr | None = None
    landing_url: NonBlankStr | None = None
    metadata_source: NonBlankStr
    metadata_retrieval_refs: tuple[NonBlankStr, ...] = ()
    metadata_retrieval_hashes: tuple[Sha256Str, ...] = ()
    fulltext_candidates: tuple[FullTextCandidate, ...] = ()
    resolution_status: AcquisitionStatus
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ResolvedLiteratureResource:
        from ..serialization import content_hash

        candidate_ids = [item.candidate_id for item in self.fulltext_candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("ResolvedLiteratureResource candidate IDs must be unique")
        if len(set(self.metadata_retrieval_refs)) != len(
            self.metadata_retrieval_refs
        ):
            raise ValueError("metadata retrieval refs must be unique")
        if len(self.metadata_retrieval_refs) != len(
            self.metadata_retrieval_hashes
        ):
            raise ValueError("metadata retrieval ID/hash bindings must align")
        if tuple(candidate_ids) != tuple(
            item.candidate_id
            for item in sorted(
                self.fulltext_candidates,
                key=lambda item: (item.priority, item.candidate_id),
            )
        ):
            raise ValueError("full-text candidates must be deterministically ordered")
        identity = self.model_dump(
            mode="json", exclude={"resource_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"resolved-literature-{content_hash(identity)[:24]}"
        if self.resource_id != expected_id:
            raise ValueError("ResolvedLiteratureResource resource_id is not content-bound")
        payload = {"resource_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ResolvedLiteratureResource content_hash is invalid")
        return self


class LiteratureAcquisitionRecord(StrictModel):
    acquisition_id: NonBlankStr
    request_id: NonBlankStr
    request_hash: Sha256Str
    resolver_id: NonBlankStr
    resolver_version: NonBlankStr
    resolved_resource_id: NonBlankStr
    resolved_resource_hash: Sha256Str
    selected_candidate_id: NonBlankStr | None = None
    selected_candidate_hash: Sha256Str | None = None
    metadata_merge_manifest_id: NonBlankStr | None = None
    metadata_merge_manifest_hash: Sha256Str | None = None
    attempt_refs: tuple[NonBlankStr, ...] = ()
    attempt_hashes: tuple[Sha256Str, ...] = ()
    resulting_literature_id: NonBlankStr | None = None
    resulting_ingestion_id: NonBlankStr | None = None
    resulting_canonical_text_id: NonBlankStr | None = None
    resulting_representation_id: NonBlankStr | None = None
    resulting_representation_hash: Sha256Str | None = None
    status: AcquisitionStatus
    warnings: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureAcquisitionRecord:
        from ..serialization import content_hash

        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("LiteratureAcquisitionRecord warnings must be unique")
        if len(set(self.attempt_refs)) != len(self.attempt_refs):
            raise ValueError("LiteratureAcquisitionRecord attempt_refs must be unique")
        if len(self.attempt_refs) != len(self.attempt_hashes):
            raise ValueError("acquisition attempt ID/hash bindings must align")
        candidate_binding = (
            self.selected_candidate_id,
            self.selected_candidate_hash,
        )
        if any(value is None for value in candidate_binding) and any(
            value is not None for value in candidate_binding
        ):
            raise ValueError("selected full-text candidate binding must be complete")
        for name, binding in (
            (
                "metadata merge manifest",
                (
                    self.metadata_merge_manifest_id,
                    self.metadata_merge_manifest_hash,
                ),
            ),
            (
                "literature representation",
                (
                    self.resulting_representation_id,
                    self.resulting_representation_hash,
                ),
            ),
        ):
            if any(value is None for value in binding) and any(
                value is not None for value in binding
            ):
                raise ValueError(f"{name} binding must be complete")
        required_result_binding = (
            self.resulting_literature_id,
            self.resulting_ingestion_id,
        )
        result_binding = (*required_result_binding, self.resulting_canonical_text_id)
        if self.status == AcquisitionStatus.INGESTED and any(
            value is None for value in required_result_binding
        ):
            raise ValueError("ingested acquisition requires literature and ingestion bindings")
        if self.status != AcquisitionStatus.INGESTED and any(
            value is not None for value in result_binding
        ):
            raise ValueError("non-ingested acquisition cannot bind K1B results")
        identity = self.model_dump(
            mode="json",
            exclude={"acquisition_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"literature-acquisition-{content_hash(identity)[:24]}"
        if self.acquisition_id != expected_id:
            raise ValueError("LiteratureAcquisitionRecord acquisition_id is not content-bound")
        payload = {"acquisition_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureAcquisitionRecord content_hash is invalid")
        return self


class LiteratureAcquisitionOutcome(StrictModel):
    input_kind: AcquisitionInputKind
    canonical_identifier: NonBlankStr
    doi: NonBlankStr | None = None
    title: NonBlankStr | None = None
    acquisition_status: AcquisitionStatus
    fulltext_status: AcquisitionStatus
    literature_id: NonBlankStr | None = None
    ingestion_id: NonBlankStr | None = None
    canonical_text_id: NonBlankStr | None = None
    representation_id: NonBlankStr | None = None
    acquisition_id: NonBlankStr
    attempt_ids: tuple[NonBlankStr, ...] = ()
    warnings: tuple[NonBlankStr, ...] = ()


class MetadataRetrievalRecord(StrictModel):
    retrieval_id: NonBlankStr
    source_url: NonBlankStr
    response_sha256: Sha256Str
    response_byte_size: int = Field(gt=0)
    media_type: NonBlankStr
    resolver_id: NonBlankStr
    resolver_version: NonBlankStr
    response_payload_utf8: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> MetadataRetrievalRecord:
        import hashlib

        from ..serialization import content_hash

        if self.response_payload_utf8 is not None:
            encoded = self.response_payload_utf8.encode("utf-8")
            if (
                len(encoded) != self.response_byte_size
                or hashlib.sha256(encoded).hexdigest() != self.response_sha256
            ):
                raise ValueError("metadata response payload does not match its hash")
        identity = self.model_dump(
            mode="json",
            exclude={"retrieval_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"metadata-retrieval-{content_hash(identity)[:24]}"
        if self.retrieval_id != expected_id:
            raise ValueError("MetadataRetrievalRecord retrieval_id is not content-bound")
        payload = {"retrieval_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("MetadataRetrievalRecord content_hash is invalid")
        return self


class MetadataAlternative(StrictModel):
    origin: MetadataValueOrigin
    value: NonBlankStr | int | tuple[NonBlankStr, ...]


class MetadataFieldDecision(StrictModel):
    field_name: NonBlankStr
    selected_origin: MetadataValueOrigin
    selected_value: NonBlankStr | int | tuple[NonBlankStr, ...] | None = None
    alternatives: tuple[MetadataAlternative, ...] = ()

    @model_validator(mode="after")
    def validate_decision(self) -> MetadataFieldDecision:
        if self.selected_origin == MetadataValueOrigin.UNRESOLVED:
            if self.selected_value is not None:
                raise ValueError("unresolved metadata cannot have a selected value")
        elif self.selected_value is None:
            raise ValueError("resolved metadata requires a selected value")
        origins = [item.origin for item in self.alternatives]
        if len(set(origins)) != len(origins):
            raise ValueError("metadata alternatives must have unique origins")
        return self


class MetadataMergeManifest(StrictModel):
    manifest_id: NonBlankStr
    decisions: tuple[MetadataFieldDecision, ...]
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> MetadataMergeManifest:
        from ..serialization import content_hash

        fields = [item.field_name for item in self.decisions]
        if len(set(fields)) != len(fields) or fields != sorted(fields):
            raise ValueError("metadata decisions must be unique and sorted")
        identity = self.model_dump(
            mode="json", exclude={"manifest_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"metadata-merge-{content_hash(identity)[:24]}"
        if self.manifest_id != expected_id:
            raise ValueError("MetadataMergeManifest manifest_id is not content-bound")
        payload = {"manifest_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("MetadataMergeManifest content_hash is invalid")
        return self


class AcquisitionAttemptRecord(StrictModel):
    attempt_id: NonBlankStr
    request_id: NonBlankStr
    request_hash: Sha256Str
    candidate_id: NonBlankStr
    candidate_hash: Sha256Str
    attempt_index: int = Field(ge=0)
    status: AcquisitionStatus
    http_status: int | None = Field(default=None, ge=100, le=599)
    media_type: NonBlankStr
    failure_code: NonBlankStr | None = None
    raw_artifact_id: NonBlankStr | None = None
    raw_artifact_hash: Sha256Str | None = None
    canonical_text_id: NonBlankStr | None = None
    canonical_text_hash: Sha256Str | None = None
    ingestion_id: NonBlankStr | None = None
    ingestion_hash: Sha256Str | None = None
    representation_id: NonBlankStr | None = None
    representation_hash: Sha256Str | None = None
    metadata_merge_manifest_id: NonBlankStr | None = None
    metadata_merge_manifest_hash: Sha256Str | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> AcquisitionAttemptRecord:
        from ..serialization import content_hash

        for name, binding in (
            ("raw artifact", (self.raw_artifact_id, self.raw_artifact_hash)),
            (
                "canonical text",
                (self.canonical_text_id, self.canonical_text_hash),
            ),
            ("ingestion", (self.ingestion_id, self.ingestion_hash)),
            (
                "representation",
                (self.representation_id, self.representation_hash),
            ),
            (
                "metadata merge manifest",
                (
                    self.metadata_merge_manifest_id,
                    self.metadata_merge_manifest_hash,
                ),
            ),
        ):
            if any(value is None for value in binding) and any(
                value is not None for value in binding
            ):
                raise ValueError(f"attempt {name} binding must be complete")
        if self.status == AcquisitionStatus.INGESTED:
            if any(
                value is None
                for value in (
                    self.raw_artifact_id,
                    self.ingestion_id,
                    self.representation_id,
                )
            ):
                raise ValueError("successful attempt requires artifact and representation")
            if self.failure_code is not None:
                raise ValueError("successful attempt cannot have a failure code")
        elif self.failure_code is None:
            raise ValueError("unsuccessful attempt requires a failure code")
        identity = self.model_dump(
            mode="json", exclude={"attempt_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"acquisition-attempt-{content_hash(identity)[:24]}"
        if self.attempt_id != expected_id:
            raise ValueError("AcquisitionAttemptRecord attempt_id is not content-bound")
        payload = {"attempt_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("AcquisitionAttemptRecord content_hash is invalid")
        return self


class RawLiteratureArtifact(StrictModel):
    artifact_id: NonBlankStr
    literature_id: NonBlankStr
    original_filename: NonBlankStr
    media_type: NonBlankStr = "application/pdf"
    byte_size: int = Field(gt=0)
    sha256: Sha256Str
    stored_path: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> RawLiteratureArtifact:
        from pathlib import PurePosixPath, PureWindowsPath

        from ..serialization import content_hash

        if (
            PurePosixPath(self.original_filename).name != self.original_filename
            or PureWindowsPath(self.original_filename).name != self.original_filename
        ):
            raise ValueError("RawLiteratureArtifact original_filename must be a basename")
        extensions = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        }
        if self.media_type not in extensions:
            raise ValueError("RawLiteratureArtifact media type is unsupported")
        identity = {"literature_id": self.literature_id, "sha256": self.sha256}
        expected_id = f"literature-artifact-{content_hash(identity)[:24]}"
        if self.artifact_id != expected_id:
            raise ValueError("RawLiteratureArtifact artifact_id is not content-bound")
        expected_path = (
            f"literature_artifacts/{expected_id}/artifact.{extensions[self.media_type]}"
        )
        if self.stored_path != expected_path:
            raise ValueError("RawLiteratureArtifact stored_path is not canonical")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("RawLiteratureArtifact content_hash is invalid")
        return self


class CanonicalTextBlock(StrictModel):
    block_id: NonBlankStr
    page_number: int = Field(ge=1)
    block_type: NonBlankStr
    section_path: tuple[NonBlankStr, ...] = ()
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CanonicalTextBlock:
        from ..serialization import content_hash

        if self.end_offset <= self.start_offset:
            raise ValueError("CanonicalTextBlock end_offset must exceed start_offset")
        identity = self.model_dump(mode="json", exclude={"block_id"})
        expected_id = f"canonical-block-{content_hash(identity)[:24]}"
        if self.block_id != expected_id:
            raise ValueError("CanonicalTextBlock block_id is not content-bound")
        return self


class CanonicalTextArtifact(StrictModel):
    canonical_text_id: NonBlankStr
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    parser_id: NonBlankStr
    parser_version: NonBlankStr
    parser_config_hash: Sha256Str
    text_sha256: Sha256Str
    stored_path: NonBlankStr
    character_count: int = Field(gt=0)
    page_count: int = Field(gt=0)
    blocks: tuple[CanonicalTextBlock, ...] = Field(min_length=1)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CanonicalTextArtifact:
        from ..serialization import content_hash

        block_ids = [block.block_id for block in self.blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("CanonicalTextArtifact block IDs must be unique")
        ordering = [
            (block.page_number, block.start_offset, block.end_offset)
            for block in self.blocks
        ]
        if ordering != sorted(ordering):
            raise ValueError("CanonicalTextArtifact blocks must be deterministically ordered")
        if any(
            right.start_offset < left.end_offset
            for left, right in zip(self.blocks, self.blocks[1:])
        ):
            raise ValueError("CanonicalTextArtifact blocks must not overlap")
        if any(
            block.page_number > self.page_count
            or block.end_offset > self.character_count
            for block in self.blocks
        ):
            raise ValueError("CanonicalTextArtifact block bounds are invalid")
        identity = self.model_dump(
            mode="json",
            exclude={"canonical_text_id", "stored_path", "content_hash"},
        )
        expected_id = f"canonical-text-{content_hash(identity)[:24]}"
        if self.canonical_text_id != expected_id:
            raise ValueError("CanonicalTextArtifact canonical_text_id is not content-bound")
        expected_path = f"canonical_text/{expected_id}/content.txt"
        if self.stored_path != expected_path:
            raise ValueError("CanonicalTextArtifact stored_path is not canonical")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("CanonicalTextArtifact content_hash is invalid")
        return self


class LiteratureSupplementarySource(StrictModel):
    supplement_id: NonBlankStr
    literature_id: NonBlankStr
    literature_hash: Sha256Str
    main_selection_id: NonBlankStr
    main_selection_hash: Sha256Str
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    canonical_text_id: NonBlankStr
    canonical_text_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    source_sha256: Sha256Str
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureSupplementarySource:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"supplement_id", "content_hash"}
        )
        expected_id = f"literature-supplement-{content_hash(identity)[:24]}"
        if self.supplement_id != expected_id:
            raise ValueError("LiteratureSupplementarySource ID is not content-bound")
        payload = {"supplement_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureSupplementarySource content_hash is invalid")
        return self


class RawHTMLLiteratureArtifact(StrictModel):
    artifact_id: NonBlankStr
    source_url: NonBlankStr
    literature_id: NonBlankStr
    media_type: NonBlankStr
    byte_size: int = Field(gt=0)
    sha256: Sha256Str
    stored_path: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> RawHTMLLiteratureArtifact:
        from ..serialization import content_hash

        if self.media_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("RawHTMLLiteratureArtifact requires an HTML media type")
        identity = {
            "source_url": self.source_url,
            "literature_id": self.literature_id,
            "sha256": self.sha256,
        }
        expected_id = f"html-artifact-{content_hash(identity)[:24]}"
        if self.artifact_id != expected_id:
            raise ValueError("RawHTMLLiteratureArtifact artifact_id is not content-bound")
        expected_path = f"html_literature_artifacts/{expected_id}/artifact.html"
        if self.stored_path != expected_path:
            raise ValueError("RawHTMLLiteratureArtifact stored_path is not canonical")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("RawHTMLLiteratureArtifact content_hash is invalid")
        return self


class CanonicalHTMLTextArtifact(StrictModel):
    canonical_text_id: NonBlankStr
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    source_url: NonBlankStr
    extractor_id: NonBlankStr
    extractor_version: NonBlankStr
    extractor_config_hash: Sha256Str
    text_sha256: Sha256Str
    stored_path: NonBlankStr
    character_count: int = Field(gt=0)
    blocks: tuple[CanonicalTextBlock, ...] = Field(min_length=1)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> CanonicalHTMLTextArtifact:
        from ..serialization import content_hash

        block_ids = [block.block_id for block in self.blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("CanonicalHTMLTextArtifact block IDs must be unique")
        ordering = [
            (block.start_offset, block.end_offset) for block in self.blocks
        ]
        if ordering != sorted(ordering):
            raise ValueError("HTML canonical blocks must be ordered")
        if any(
            right.start_offset < left.end_offset
            for left, right in zip(self.blocks, self.blocks[1:])
        ):
            raise ValueError("HTML canonical blocks must not overlap")
        if any(block.end_offset > self.character_count for block in self.blocks):
            raise ValueError("HTML canonical block bounds are invalid")
        identity = self.model_dump(
            mode="json",
            exclude={"canonical_text_id", "stored_path", "content_hash"},
        )
        expected_id = f"canonical-html-text-{content_hash(identity)[:24]}"
        if self.canonical_text_id != expected_id:
            raise ValueError("CanonicalHTMLTextArtifact ID is not content-bound")
        expected_path = f"canonical_html_text/{expected_id}/content.txt"
        if self.stored_path != expected_path:
            raise ValueError("CanonicalHTMLTextArtifact stored_path is not canonical")
        payload = self.model_dump(mode="json", exclude={"content_hash"})
        if self.content_hash != content_hash(payload):
            raise ValueError("CanonicalHTMLTextArtifact content_hash is invalid")
        return self


class HTMLLiteratureIngestionRecord(StrictModel):
    ingestion_id: NonBlankStr
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    canonical_text_id: NonBlankStr
    canonical_text_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    extractor_id: NonBlankStr
    extractor_version: NonBlankStr
    extractor_config_hash: Sha256Str
    ingestion_status: LiteratureIngestionStatus = LiteratureIngestionStatus.ACCEPTED
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> HTMLLiteratureIngestionRecord:
        from ..serialization import content_hash

        if self.ingestion_status != LiteratureIngestionStatus.ACCEPTED:
            raise ValueError("HTML ingestion records represent successful extraction only")
        identity = self.model_dump(
            mode="json", exclude={"ingestion_id", "content_hash"}
        )
        expected_id = f"html-literature-ingestion-{content_hash(identity)[:24]}"
        if self.ingestion_id != expected_id:
            raise ValueError("HTMLLiteratureIngestionRecord ID is not content-bound")
        payload = {"ingestion_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("HTMLLiteratureIngestionRecord content_hash is invalid")
        return self


class LiteratureRepresentationReference(StrictModel):
    representation_id: NonBlankStr
    representation_kind: LiteratureRepresentationKind
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    canonical_text_id: NonBlankStr
    canonical_text_hash: Sha256Str
    ingestion_id: NonBlankStr
    ingestion_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureRepresentationReference:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"representation_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"literature-representation-{content_hash(identity)[:24]}"
        if self.representation_id != expected_id:
            raise ValueError("LiteratureRepresentationReference ID is not content-bound")
        payload = {"representation_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureRepresentationReference content_hash is invalid")
        return self


class DocumentStructureBlock(StrictModel):
    block_id: NonBlankStr
    structure_id: NonBlankStr
    block_type: DocumentBlockType
    ordinal: int = Field(ge=0)
    parent_block_id: NonBlankStr | None = None
    page_number: int | None = Field(default=None, ge=1)
    heading_level: int | None = Field(default=None, ge=1, le=6)
    section_path: tuple[NonBlankStr, ...] = ()
    content_region: DocumentContentRegion = DocumentContentRegion.UNKNOWN
    label: NonBlankStr | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text_hash: Sha256Str
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> DocumentStructureBlock:
        from ..serialization import content_hash

        if self.end_offset <= self.start_offset:
            raise ValueError("document structure block range is invalid")
        if self.block_type == DocumentBlockType.HEADING:
            if self.heading_level is None or not self.section_path:
                raise ValueError("heading block requires level and section path")
        elif self.heading_level is not None:
            raise ValueError("only heading blocks may set heading_level")
        if self.block_type == DocumentBlockType.PAGE and self.page_number is None:
            raise ValueError("page block requires page_number")
        identity = self.model_dump(
            mode="json", exclude={"block_id", "content_hash"}, exclude_none=True
        )
        if self.content_region == DocumentContentRegion.UNKNOWN:
            identity.pop("content_region", None)
        expected_id = f"document-block-{content_hash(identity)[:24]}"
        if self.block_id != expected_id:
            raise ValueError("DocumentStructureBlock block_id is not content-bound")
        payload = {"block_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("DocumentStructureBlock content_hash is invalid")
        return self


class TableCellStructure(StrictModel):
    cell_id: NonBlankStr
    table_id: NonBlankStr
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    is_header: bool = False
    canonical_block_refs: tuple[NonBlankStr, ...] = ()
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, gt=0)
    text_hash: Sha256Str | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> TableCellStructure:
        from ..serialization import content_hash

        range_binding = (self.start_offset, self.end_offset, self.text_hash)
        if any(value is not None for value in range_binding) and not all(
            value is not None for value in range_binding
        ):
            raise ValueError("table cell text range binding must be complete")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError("table cell range is invalid")
        if self.canonical_block_refs and self.start_offset is None:
            raise ValueError("table cell canonical blocks require a text range")
        if len(set(self.canonical_block_refs)) != len(self.canonical_block_refs):
            raise ValueError("table cell canonical_block_refs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"cell_id", "content_hash"},
            exclude_none=True,
        )
        # K1E cells predate explicit canonical block ownership. Preserve their
        # content-bound identity while all K1E.1 cells bind exact regions.
        if not self.canonical_block_refs and self.start_offset is not None:
            identity.pop("canonical_block_refs", None)
        expected_id = f"table-cell-{content_hash(identity)[:24]}"
        if self.cell_id != expected_id:
            raise ValueError("TableCellStructure cell_id is not content-bound")
        payload = {"cell_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("TableCellStructure content_hash is invalid")
        return self


class TableStructure(StrictModel):
    table_id: NonBlankStr
    structure_id: NonBlankStr
    ordinal: int = Field(ge=0)
    label: NonBlankStr | None = None
    caption_block_ref: NonBlankStr | None = None
    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[NonBlankStr, ...] = ()
    cell_refs: tuple[NonBlankStr, ...] = ()
    extraction_status: StructureExtractionStatus
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> TableStructure:
        from ..serialization import content_hash

        if len(set(self.cell_refs)) != len(self.cell_refs):
            raise ValueError("TableStructure cell_refs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"table_id", "cell_refs", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"table-structure-{content_hash(identity)[:24]}"
        if self.table_id != expected_id:
            raise ValueError("TableStructure table_id is not content-bound")
        payload = {
            "table_id": expected_id,
            **identity,
            "cell_refs": self.cell_refs,
        }
        if self.content_hash != content_hash(payload):
            raise ValueError("TableStructure content_hash is invalid")
        return self


class FigureStructure(StrictModel):
    figure_id: NonBlankStr
    structure_id: NonBlankStr
    ordinal: int = Field(ge=0)
    label: NonBlankStr | None = None
    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[NonBlankStr, ...] = ()
    caption_block_ref: NonBlankStr
    extraction_status: StructureExtractionStatus
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> FigureStructure:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json", exclude={"figure_id", "content_hash"}, exclude_none=True
        )
        expected_id = f"figure-structure-{content_hash(identity)[:24]}"
        if self.figure_id != expected_id:
            raise ValueError("FigureStructure figure_id is not content-bound")
        payload = {"figure_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("FigureStructure content_hash is invalid")
        return self


class DocumentStructureArtifact(StrictModel):
    structure_id: NonBlankStr
    literature_id: NonBlankStr
    representation_id: NonBlankStr
    representation_hash: Sha256Str
    representation_kind: LiteratureRepresentationKind
    canonical_text_id: NonBlankStr
    canonical_text_hash: Sha256Str
    source_id: NonBlankStr
    source_version: NonBlankStr
    extractor_id: NonBlankStr
    extractor_version: NonBlankStr
    extractor_config_hash: Sha256Str
    block_hashes: dict[NonBlankStr, Sha256Str] = Field(min_length=1)
    table_hashes: dict[NonBlankStr, Sha256Str]
    figure_hashes: dict[NonBlankStr, Sha256Str]
    page_count: int | None = Field(default=None, ge=1)
    extraction_status: StructureExtractionStatus
    warnings: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> DocumentStructureArtifact:
        from ..serialization import content_hash

        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("DocumentStructureArtifact warnings must be unique")
        identity = self.model_dump(
            mode="json",
            include={
                "literature_id",
                "representation_id",
                "representation_hash",
                "representation_kind",
                "canonical_text_id",
                "canonical_text_hash",
                "source_id",
                "source_version",
                "extractor_id",
                "extractor_version",
                "extractor_config_hash",
            },
        )
        expected_id = f"document-structure-{content_hash(identity)[:24]}"
        if self.structure_id != expected_id:
            raise ValueError("DocumentStructureArtifact structure_id is not content-bound")
        payload = self.model_dump(
            mode="json", exclude={"content_hash"}, exclude_none=True
        )
        if self.content_hash != content_hash(payload):
            raise ValueError("DocumentStructureArtifact content_hash is invalid")
        return self


class DocumentStructureSelection(StrictModel):
    selection_id: NonBlankStr
    literature_id: NonBlankStr
    representation_id: NonBlankStr
    representation_hash: Sha256Str
    structure_id: NonBlankStr
    structure_hash: Sha256Str
    selected_by_policy: NonBlankStr
    selector_version: NonBlankStr
    rationale: NonBlankStr
    supersedes_selection_id: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> DocumentStructureSelection:
        from ..serialization import content_hash

        identity = self.model_dump(
            mode="json",
            exclude={"selection_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"document-structure-selection-{content_hash(identity)[:24]}"
        if self.selection_id != expected_id:
            raise ValueError("DocumentStructureSelection selection_id is not content-bound")
        payload = {"selection_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("DocumentStructureSelection content_hash is invalid")
        return self


class StructuredEvidenceLocator(StrictModel):
    locator_id: NonBlankStr
    evidence_id: NonBlankStr
    source_id: NonBlankStr
    source_version: NonBlankStr
    literature_id: NonBlankStr
    representation_id: NonBlankStr
    structure_id: NonBlankStr
    block_id: NonBlankStr
    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[NonBlankStr, ...] = ()
    content_region: DocumentContentRegion = DocumentContentRegion.UNKNOWN
    table_id: NonBlankStr | None = None
    table_cell_id: NonBlankStr | None = None
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    figure_id: NonBlankStr | None = None
    canonical_start_offset: int = Field(ge=0)
    canonical_end_offset: int = Field(gt=0)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> StructuredEvidenceLocator:
        from ..serialization import content_hash

        if self.canonical_end_offset <= self.canonical_start_offset:
            raise ValueError("structured evidence locator range is invalid")
        cell_values = (
            self.table_id,
            self.table_cell_id,
            self.row_index,
            self.column_index,
        )
        if any(value is not None for value in cell_values) and not all(
            value is not None for value in cell_values
        ):
            raise ValueError("table-cell locator fields must be complete")
        if self.figure_id is not None and self.table_id is not None:
            raise ValueError("locator cannot target both a table cell and figure")
        identity = self.model_dump(
            mode="json", exclude={"locator_id", "content_hash"}, exclude_none=True
        )
        if self.content_region == DocumentContentRegion.UNKNOWN:
            identity.pop("content_region", None)
        expected_id = f"structured-locator-{content_hash(identity)[:24]}"
        if self.locator_id != expected_id:
            raise ValueError("StructuredEvidenceLocator locator_id is not content-bound")
        payload = {"locator_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("StructuredEvidenceLocator content_hash is invalid")
        return self


class LiteratureIngestionRecord(StrictModel):
    ingestion_id: NonBlankStr
    literature_id: NonBlankStr
    raw_artifact_id: NonBlankStr
    raw_artifact_hash: Sha256Str
    canonical_text_id: NonBlankStr | None = None
    canonical_text_hash: Sha256Str | None = None
    source_id: NonBlankStr | None = None
    source_version: NonBlankStr | None = None
    parser_id: NonBlankStr
    parser_version: NonBlankStr
    parser_config_hash: Sha256Str
    ingestion_status: LiteratureIngestionStatus
    total_pages: int = Field(ge=0)
    pages_with_text: int = Field(ge=0)
    pages_without_text: int = Field(ge=0)
    text_coverage_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    warnings: tuple[NonBlankStr, ...] = ()
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureIngestionRecord:
        from ..serialization import content_hash

        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("LiteratureIngestionRecord warnings must be unique")
        if self.pages_with_text + self.pages_without_text != self.total_pages:
            raise ValueError("literature ingestion page metrics are inconsistent")
        expected_coverage = (
            self.pages_with_text / self.total_pages if self.total_pages else 0.0
        )
        if abs(self.text_coverage_ratio - expected_coverage) > 1e-12:
            raise ValueError("literature ingestion text coverage ratio is invalid")
        canonical_binding = (
            self.canonical_text_id,
            self.canonical_text_hash,
            self.source_id,
            self.source_version,
        )
        if self.ingestion_status == LiteratureIngestionStatus.ACCEPTED:
            if any(value is None for value in canonical_binding):
                raise ValueError("accepted literature ingestion requires canonical/source binding")
            if self.pages_with_text == 0:
                raise ValueError("accepted literature ingestion requires extractable text")
        elif self.ingestion_status == LiteratureIngestionStatus.REQUIRES_OCR:
            if self.pages_with_text != 0:
                raise ValueError("requires_ocr ingestion cannot contain extractable text")
            if any(value is not None for value in canonical_binding):
                raise ValueError("non-accepted literature ingestion cannot bind canonical/source records")
        elif any(value is not None for value in canonical_binding):
            raise ValueError("non-accepted literature ingestion cannot bind canonical/source records")
        identity = self.model_dump(
            mode="json",
            exclude={"ingestion_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"literature-ingestion-{content_hash(identity)[:24]}"
        if self.ingestion_id != expected_id:
            raise ValueError("LiteratureIngestionRecord ingestion_id is not content-bound")
        payload = {"ingestion_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("LiteratureIngestionRecord content_hash is invalid")
        return self


class LiteratureRepresentationSelection(StrictModel):
    selection_id: NonBlankStr
    literature_id: NonBlankStr
    representation_id: NonBlankStr | None = None
    representation_hash: Sha256Str | None = None
    ingestion_id: NonBlankStr | None = None
    ingestion_hash: Sha256Str | None = None
    canonical_text_id: NonBlankStr | None = None
    canonical_text_hash: Sha256Str | None = None
    source_id: NonBlankStr | None = None
    source_version: NonBlankStr | None = None
    selected_by: NonBlankStr
    rationale: NonBlankStr
    supersedes_selection_id: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> LiteratureRepresentationSelection:
        from ..serialization import content_hash

        representation_binding = (self.representation_id, self.representation_hash)
        legacy_binding = (
            self.ingestion_id,
            self.ingestion_hash,
            self.canonical_text_id,
            self.canonical_text_hash,
            self.source_id,
            self.source_version,
        )
        has_representation = all(value is not None for value in representation_binding)
        has_legacy = all(value is not None for value in legacy_binding)
        if has_representation == has_legacy or (
            any(value is not None for value in representation_binding)
            and not has_representation
        ) or (any(value is not None for value in legacy_binding) and not has_legacy):
            raise ValueError(
                "selection requires exactly one complete representation or legacy binding"
            )
        identity = self.model_dump(
            mode="json",
            exclude={"selection_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"literature-selection-{content_hash(identity)[:24]}"
        if self.selection_id != expected_id:
            raise ValueError(
                "LiteratureRepresentationSelection selection_id is not content-bound"
            )
        payload = {"selection_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError(
                "LiteratureRepresentationSelection content_hash is invalid"
            )
        return self


class LiteratureRepresentationSelectionOutcome(StrictModel):
    selection: LiteratureRepresentationSelection
    ingestion_status: LiteratureIngestionStatus
    total_pages: int | None = Field(default=None, ge=0)
    pages_with_text: int | None = Field(default=None, ge=0)
    pages_without_text: int | None = Field(default=None, ge=0)
    text_coverage_ratio: float | None = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    warnings: tuple[NonBlankStr, ...] = ()


class HistoricalLiteratureEvidenceAuthorization(StrictModel):
    authorization_id: NonBlankStr
    literature_id: NonBlankStr
    representation_id: NonBlankStr | None = None
    representation_hash: Sha256Str | None = None
    ingestion_id: NonBlankStr | None = None
    ingestion_hash: Sha256Str | None = None
    source_id: NonBlankStr
    source_version: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    authorized_by: NonBlankStr
    rationale: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> HistoricalLiteratureEvidenceAuthorization:
        from ..serialization import content_hash

        representation_binding = (self.representation_id, self.representation_hash)
        legacy_binding = (self.ingestion_id, self.ingestion_hash)
        has_representation = all(value is not None for value in representation_binding)
        has_legacy = all(value is not None for value in legacy_binding)
        if has_representation == has_legacy or (
            any(value is not None for value in representation_binding)
            and not has_representation
        ) or (any(value is not None for value in legacy_binding) and not has_legacy):
            raise ValueError(
                "historical authorization requires one representation or legacy ingestion binding"
            )
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError(
                "HistoricalLiteratureEvidenceAuthorization evidence_refs must be unique"
            )
        identity = self.model_dump(
            mode="json",
            exclude={"authorization_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"historical-evidence-authorization-{content_hash(identity)[:24]}"
        if self.authorization_id != expected_id:
            raise ValueError(
                "HistoricalLiteratureEvidenceAuthorization authorization_id is not content-bound"
            )
        payload = {"authorization_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError(
                "HistoricalLiteratureEvidenceAuthorization content_hash is invalid"
            )
        return self


class LiteratureIngestionOutcome(StrictModel):
    literature_id: NonBlankStr
    artifact_id: NonBlankStr
    canonical_text_id: NonBlankStr | None = None
    ingestion_id: NonBlankStr
    selection_id: NonBlankStr | None = None
    source_id: NonBlankStr | None = None
    source_version: NonBlankStr | None = None
    warnings: tuple[NonBlankStr, ...] = ()
    requires_ocr: bool = False
