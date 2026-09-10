from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol

from pydantic import Field, model_validator

from ..models import (
    DocumentContentRegion,
    NonBlankStr,
    Sha256Str,
    StrictModel,
)
from ..serialization import content_hash


_SENSITIVE_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "access_token",
    "refresh_token",
)


def _reject_sensitive_mapping(values: Mapping[str, object]) -> None:
    for key in values:
        normalized = key.casefold().replace("-", "_")
        if any(marker in normalized for marker in _SENSITIVE_MARKERS):
            raise ValueError("backend metadata must not contain secret-bearing keys")


def _reject_sensitive_warnings(values: tuple[str, ...]) -> None:
    for value in values:
        normalized = value.casefold().replace("-", "_")
        if any(marker in normalized for marker in _SENSITIVE_MARKERS):
            raise ValueError("backend warnings must not contain secret-bearing text")


class BackendCapability(StrEnum):
    DOCUMENT_PARSING = "document_parsing"
    BUILTIN_STRUCTURE = "builtin_structure"
    SCHOLARLY_METADATA = "scholarly_metadata"
    LITERATURE_RETRIEVAL = "literature_retrieval"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    GRAPH_RETRIEVAL = "graph_retrieval"


class BackendIntegrationMode(StrEnum):
    PYTHON_LIBRARY = "python_library"
    SUBPROCESS = "subprocess"
    HTTP_SERVICE = "http_service"
    BUILTIN = "builtin"


class BackendLicenseStatus(StrEnum):
    CONFIRMED = "confirmed"
    UNVERIFIED = "unverified"
    INCOMPATIBLE = "incompatible"


class ExternalProposalTrust(StrEnum):
    EXTERNAL_PROPOSAL = "external_proposal"


class ExternalProposalStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


class ExternalDocumentElementKind(StrEnum):
    PAGE = "page"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_CAPTION = "table_caption"
    TABLE_CELL = "table_cell"
    FIGURE_CAPTION = "figure_caption"
    OTHER_TEXT = "other_text"


class ExternalBindingStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


class BackendRunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class BackendOutputType(StrEnum):
    EXTERNAL_DOCUMENT_PARSE_PROPOSAL = "external_document_parse_proposal"
    EXTERNAL_LITERATURE_RETRIEVAL_RESULT = "external_literature_retrieval_result"
    SCHOLARLY_METADATA_PROPOSAL = "scholarly_metadata_proposal"


class ExternalBackendDescriptor(StrictModel):
    backend_id: NonBlankStr
    backend_name: NonBlankStr
    backend_version: NonBlankStr
    adapter_version: NonBlankStr
    capability_types: tuple[BackendCapability, ...] = Field(min_length=1)
    integration_mode: BackendIntegrationMode
    package_name: NonBlankStr | None = None
    package_version: NonBlankStr | None = None
    source_project: NonBlankStr
    license_id: NonBlankStr
    license_status: BackendLicenseStatus
    deterministic: bool
    requires_network: bool
    optional_dependency: bool
    vendored: bool = False
    redistributable_confirmed: bool = False
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExternalBackendDescriptor:
        if len(set(self.capability_types)) != len(self.capability_types):
            raise ValueError("backend capabilities must be unique")
        if self.vendored:
            raise ValueError("SPC backend descriptors must not vendor upstream source")
        if self.license_status != BackendLicenseStatus.CONFIRMED and self.redistributable_confirmed:
            raise ValueError("unverified backend license cannot claim redistributable status")
        payload = self.model_dump(mode="json", exclude={"content_hash"}, exclude_none=True)
        if self.content_hash != content_hash(payload):
            raise ValueError("ExternalBackendDescriptor content_hash is invalid")
        return self


class BackendRuntimeArtifactEntry(StrictModel):
    relative_path: NonBlankStr
    file_sha256: Sha256Str
    file_size: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_path(self) -> BackendRuntimeArtifactEntry:
        path = PurePosixPath(self.relative_path)
        if (
            "\\" in self.relative_path
            or path.is_absolute()
            or self.relative_path != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("runtime artifact relative_path is unsafe")
        return self


class BackendRuntimeArtifactManifest(StrictModel):
    manifest_id: NonBlankStr
    backend_id: NonBlankStr
    artifact_role: NonBlankStr
    root_identity: NonBlankStr
    entries: tuple[BackendRuntimeArtifactEntry, ...]
    aggregate_hash: Sha256Str
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> BackendRuntimeArtifactManifest:
        paths = tuple(item.relative_path for item in self.entries)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("runtime artifact entries must be unique and sorted")
        entry_payload = [item.model_dump(mode="json") for item in self.entries]
        if self.aggregate_hash != content_hash(entry_payload):
            raise ValueError("runtime artifact aggregate_hash is invalid")
        identity = self.model_dump(
            mode="json",
            exclude={"manifest_id", "content_hash"},
        )
        expected_id = f"backend-runtime-artifacts-{content_hash(identity)[:24]}"
        if self.manifest_id != expected_id:
            raise ValueError("BackendRuntimeArtifactManifest ID is not content-bound")
        payload = {"manifest_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("BackendRuntimeArtifactManifest content_hash is invalid")
        return self


class BackendRuntimeComponentVersion(StrictModel):
    component: NonBlankStr
    version: NonBlankStr


class BackendRuntimeIdentity(StrictModel):
    runtime_identity_id: NonBlankStr
    backend_id: NonBlankStr
    backend_descriptor_hash: Sha256Str
    resolved_backend_version: NonBlankStr
    adapter_version: NonBlankStr
    integration_mode: BackendIntegrationMode
    runtime_provider: NonBlankStr
    runtime_config_hash: Sha256Str
    runtime_components: tuple[BackendRuntimeComponentVersion, ...] = ()
    runtime_components_hash: Sha256Str = content_hash([])
    runtime_artifact_manifest_hash: Sha256Str | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> BackendRuntimeIdentity:
        component_names = tuple(item.component for item in self.runtime_components)
        if component_names != tuple(sorted(set(component_names))):
            raise ValueError("backend runtime components must be unique and sorted")
        component_payload = [item.model_dump(mode="json") for item in self.runtime_components]
        if self.runtime_components_hash != content_hash(component_payload):
            raise ValueError("backend runtime component hash is invalid")
        identity = self.model_dump(
            mode="json",
            exclude={"runtime_identity_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"backend-runtime-{content_hash(identity)[:24]}"
        if self.runtime_identity_id != expected_id:
            raise ValueError("BackendRuntimeIdentity ID is not content-bound")
        payload = {"runtime_identity_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("BackendRuntimeIdentity content_hash is invalid")
        return self


class BackendRuntimeAvailability(StrictModel):
    backend_id: NonBlankStr
    available: bool
    detected_version: NonBlankStr | None = None
    reason: NonBlankStr | None = None

    @model_validator(mode="after")
    def validate_state(self) -> BackendRuntimeAvailability:
        if self.available and self.reason is not None:
            raise ValueError("available backend must not carry an unavailable reason")
        if not self.available and self.reason is None:
            raise ValueError("unavailable backend requires a reason")
        return self


class ExternalDocumentParseInput(StrictModel):
    artifact_id: NonBlankStr
    artifact_path: NonBlankStr
    artifact_sha256: Sha256Str
    media_type: NonBlankStr
    parsing_config: dict[str, str | int | float | bool] = Field(default_factory=dict)
    config_hash: Sha256Str

    @model_validator(mode="after")
    def validate_config(self) -> ExternalDocumentParseInput:
        _reject_sensitive_mapping(self.parsing_config)
        if self.config_hash != content_hash(self.parsing_config):
            raise ValueError("external parsing config_hash is invalid")
        return self


class ExternalDocumentElement(StrictModel):
    element_id: NonBlankStr
    kind: ExternalDocumentElementKind
    text: NonBlankStr
    reading_order: int = Field(ge=0)
    page_hint: int | None = Field(default=None, ge=1)
    claimed_start_offset: int | None = Field(default=None, ge=0)
    claimed_end_offset: int | None = Field(default=None, gt=0)
    heading_level: int | None = Field(default=None, ge=1, le=6)
    proposed_content_region: DocumentContentRegion = DocumentContentRegion.UNKNOWN
    table_ref: NonBlankStr | None = None
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    is_header: bool = False
    backend_native_ref: NonBlankStr | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    backend_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExternalDocumentElement:
        _reject_sensitive_mapping(self.backend_metadata)
        if self.kind == ExternalDocumentElementKind.HEADING:
            if self.heading_level is None:
                raise ValueError("external heading requires heading_level")
        elif self.heading_level is not None:
            raise ValueError("only an external heading may set heading_level")
        cell_fields = (self.table_ref, self.row_index, self.column_index)
        if self.kind == ExternalDocumentElementKind.TABLE_CELL:
            if any(value is None for value in cell_fields):
                raise ValueError("external table cell binding must be complete")
        elif self.kind == ExternalDocumentElementKind.TABLE_CAPTION:
            if self.table_ref is None or any(value is not None for value in (self.row_index, self.column_index)):
                raise ValueError("external table caption requires only table_ref")
        elif any(value is not None for value in cell_fields):
            raise ValueError("only an external table cell may set cell coordinates")
        if self.kind != ExternalDocumentElementKind.TABLE_CELL and (
            self.row_span != 1 or self.column_span != 1 or self.is_header
        ):
            raise ValueError("only an external table cell may set topology fields")
        if (self.claimed_start_offset is None) != (self.claimed_end_offset is None):
            raise ValueError("external claimed offset binding must be complete")
        identity = self.model_dump(mode="json", exclude={"element_id", "content_hash"}, exclude_none=True)
        expected_id = f"external-document-element-{content_hash(identity)[:24]}"
        if self.element_id != expected_id:
            raise ValueError("ExternalDocumentElement element_id is not content-bound")
        payload = {"element_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExternalDocumentElement content_hash is invalid")
        return self


class ExternalDocumentParseProposal(StrictModel):
    proposal_id: NonBlankStr
    backend_id: NonBlankStr
    backend_descriptor_hash: Sha256Str
    runtime_identity_hash: Sha256Str
    artifact_id: NonBlankStr
    artifact_sha256: Sha256Str
    media_type: NonBlankStr
    elements: tuple[ExternalDocumentElement, ...]
    status: ExternalProposalStatus
    warnings: tuple[NonBlankStr, ...] = ()
    trust_class: ExternalProposalTrust = ExternalProposalTrust.EXTERNAL_PROPOSAL
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExternalDocumentParseProposal:
        _reject_sensitive_warnings(self.warnings)
        element_ids = [item.element_id for item in self.elements]
        orders = [item.reading_order for item in self.elements]
        if len(set(element_ids)) != len(element_ids):
            raise ValueError("external document proposal element IDs must be unique")
        if len(set(orders)) != len(orders) or orders != sorted(orders):
            raise ValueError("external document reading order must be unique and sorted")
        identity = self.model_dump(mode="json", exclude={"proposal_id", "content_hash"}, exclude_none=True)
        expected_id = f"external-document-proposal-{content_hash(identity)[:24]}"
        if self.proposal_id != expected_id:
            raise ValueError("ExternalDocumentParseProposal proposal_id is not content-bound")
        payload = {"proposal_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExternalDocumentParseProposal content_hash is invalid")
        return self


class ReboundDocumentElement(StrictModel):
    binding_id: NonBlankStr
    source_element_id: NonBlankStr
    kind: ExternalDocumentElementKind
    status: ExternalBindingStatus
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, gt=0)
    text_hash: Sha256Str | None = None
    page_number: int | None = Field(default=None, ge=1)
    page_hint_consistent: bool | None = None
    heading_level: int | None = Field(default=None, ge=1, le=6)
    proposed_content_region: DocumentContentRegion = DocumentContentRegion.UNKNOWN
    content_region: DocumentContentRegion = DocumentContentRegion.UNKNOWN
    table_ref: NonBlankStr | None = None
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    is_header: bool = False
    reason: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ReboundDocumentElement:
        if self.kind == ExternalDocumentElementKind.HEADING:
            if self.heading_level is None:
                raise ValueError("rebound heading requires heading_level")
        elif self.heading_level is not None:
            raise ValueError("only a rebound heading may set heading_level")
        cell_fields = (self.table_ref, self.row_index, self.column_index)
        if self.kind == ExternalDocumentElementKind.TABLE_CELL:
            if any(value is None for value in cell_fields):
                raise ValueError("rebound table cell binding must be complete")
        elif self.kind == ExternalDocumentElementKind.TABLE_CAPTION:
            if self.table_ref is None or any(value is not None for value in (self.row_index, self.column_index)):
                raise ValueError("rebound table caption requires only table_ref")
        elif any(value is not None for value in cell_fields):
            raise ValueError("only rebound table records may set table fields")
        if self.kind != ExternalDocumentElementKind.TABLE_CELL and (
            self.row_span != 1 or self.column_span != 1 or self.is_header
        ):
            raise ValueError("only a rebound table cell may set topology fields")
        exact_binding = (self.start_offset, self.end_offset, self.text_hash)
        if self.status == ExternalBindingStatus.RESOLVED:
            if any(value is None for value in exact_binding) or self.reason is not None:
                raise ValueError("resolved external element requires exact offsets and hash")
        elif any(value is not None for value in exact_binding) or self.reason is None:
            raise ValueError("unresolved external element must not claim exact offsets")
        identity = self.model_dump(mode="json", exclude={"binding_id", "content_hash"}, exclude_none=True)
        expected_id = f"rebound-document-element-{content_hash(identity)[:24]}"
        if self.binding_id != expected_id:
            raise ValueError("ReboundDocumentElement binding_id is not content-bound")
        payload = {"binding_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ReboundDocumentElement content_hash is invalid")
        return self


class ExternalStructureRebindingResult(StrictModel):
    rebinding_id: NonBlankStr
    proposal_id: NonBlankStr
    proposal_hash: Sha256Str
    canonical_text_id: NonBlankStr
    canonical_text_hash: Sha256Str
    bindings: tuple[ReboundDocumentElement, ...]
    resolved_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExternalStructureRebindingResult:
        resolved = sum(item.status == ExternalBindingStatus.RESOLVED for item in self.bindings)
        if self.resolved_count != resolved:
            raise ValueError("external structure resolved_count is invalid")
        if self.unresolved_count != len(self.bindings) - resolved:
            raise ValueError("external structure unresolved_count is invalid")
        identity = self.model_dump(
            mode="json",
            exclude={"rebinding_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"external-structure-rebinding-{content_hash(identity)[:24]}"
        if self.rebinding_id != expected_id:
            raise ValueError("ExternalStructureRebindingResult ID is not content-bound")
        payload = {"rebinding_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExternalStructureRebindingResult content_hash is invalid")
        return self


class ExternalLiteratureRetrievalQuery(StrictModel):
    query: NonBlankStr
    corpus_scope: tuple[NonBlankStr, ...] = ()
    metadata_filters: dict[str, str | int | bool] = Field(default_factory=dict)
    max_results: int = Field(default=10, ge=1, le=100)


class ExternalLiteratureRetrievalHit(StrictModel):
    hit_id: NonBlankStr
    external_document_id: NonBlankStr | None = None
    doi: NonBlankStr | None = None
    title: NonBlankStr | None = None
    source_url: NonBlankStr | None = None
    page_hint: int | None = Field(default=None, ge=1)
    text_snippet: NonBlankStr | None = None
    score: float | None = Field(default=None, allow_inf_nan=False)
    backend_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExternalLiteratureRetrievalHit:
        _reject_sensitive_mapping(self.backend_metadata)
        if not any((self.external_document_id, self.doi, self.title, self.source_url)):
            raise ValueError("external retrieval hit requires document identity")
        identity = self.model_dump(mode="json", exclude={"hit_id", "content_hash"}, exclude_none=True)
        expected_id = f"external-retrieval-hit-{content_hash(identity)[:24]}"
        if self.hit_id != expected_id:
            raise ValueError("ExternalLiteratureRetrievalHit hit_id is not content-bound")
        payload = {"hit_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExternalLiteratureRetrievalHit content_hash is invalid")
        return self


class ExternalLiteratureRetrievalResult(StrictModel):
    result_id: NonBlankStr
    backend_id: NonBlankStr
    backend_descriptor_hash: Sha256Str
    runtime_identity_hash: Sha256Str
    query_hash: Sha256Str
    hits: tuple[ExternalLiteratureRetrievalHit, ...]
    status: ExternalProposalStatus
    warnings: tuple[NonBlankStr, ...] = ()
    trust_class: ExternalProposalTrust = ExternalProposalTrust.EXTERNAL_PROPOSAL
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExternalLiteratureRetrievalResult:
        _reject_sensitive_warnings(self.warnings)
        if len({item.hit_id for item in self.hits}) != len(self.hits):
            raise ValueError("external retrieval hit IDs must be unique")
        identity = self.model_dump(
            mode="json",
            exclude={"result_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"external-retrieval-result-{content_hash(identity)[:24]}"
        if self.result_id != expected_id:
            raise ValueError("ExternalLiteratureRetrievalResult ID is not content-bound")
        payload = {"result_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExternalLiteratureRetrievalResult content_hash is invalid")
        return self


class ExternalRetrievalResolution(StrictModel):
    resolution_id: NonBlankStr
    hit_id: NonBlankStr
    resolved: bool
    literature_id: NonBlankStr | None = None
    representation_id: NonBlankStr | None = None
    structure_id: NonBlankStr | None = None
    evidence_id: NonBlankStr | None = None
    evidence_hash: Sha256Str | None = None
    locator_id: NonBlankStr | None = None
    locator_hash: Sha256Str | None = None
    reason: NonBlankStr | None = None
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_state(self) -> ExternalRetrievalResolution:
        bindings = (
            self.literature_id,
            self.representation_id,
            self.structure_id,
            self.evidence_id,
            self.evidence_hash,
            self.locator_id,
            self.locator_hash,
        )
        if self.resolved:
            if any(value is None for value in bindings) or self.reason is not None:
                raise ValueError("resolved retrieval hit requires complete SPC binding")
        elif any(value is not None for value in bindings) or self.reason is None:
            raise ValueError("unresolved retrieval hit must not claim SPC bindings")
        identity = self.model_dump(
            mode="json",
            exclude={"resolution_id", "content_hash"},
            exclude_none=True,
        )
        expected_id = f"external-retrieval-resolution-{content_hash(identity)[:24]}"
        if self.resolution_id != expected_id:
            raise ValueError("ExternalRetrievalResolution ID is not content-bound")
        payload = {"resolution_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExternalRetrievalResolution content_hash is invalid")
        return self


class ExternalRetrievalAuthorityBinding(StrictModel):
    literature_id: NonBlankStr
    literature_hash: Sha256Str
    representation_selection_id: NonBlankStr
    representation_selection_hash: Sha256Str
    representation_id: NonBlankStr
    representation_hash: Sha256Str
    structure_selection_id: NonBlankStr
    structure_selection_hash: Sha256Str
    structure_id: NonBlankStr
    structure_hash: Sha256Str


class ExternalRetrievalResolutionBatch(StrictModel):
    batch_id: NonBlankStr
    external_result_id: NonBlankStr
    external_result_hash: Sha256Str
    authority_bindings: tuple[ExternalRetrievalAuthorityBinding, ...]
    resolutions: tuple[ExternalRetrievalResolution, ...]
    resolved_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    resolver_id: NonBlankStr
    resolver_version: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExternalRetrievalResolutionBatch:
        if len({item.resolution_id for item in self.resolutions}) != len(self.resolutions):
            raise ValueError("retrieval resolution IDs must be unique")
        resolved = sum(item.resolved for item in self.resolutions)
        if self.resolved_count != resolved or self.unresolved_count != len(self.resolutions) - resolved:
            raise ValueError("retrieval resolution batch counts are invalid")
        authority_ids = tuple(item.literature_id for item in self.authority_bindings)
        if authority_ids != tuple(sorted(set(authority_ids))):
            raise ValueError("retrieval authority bindings must be unique and sorted")
        identity = self.model_dump(
            mode="json",
            exclude={"batch_id", "content_hash"},
        )
        expected_id = f"external-retrieval-resolution-batch-{content_hash(identity)[:24]}"
        if self.batch_id != expected_id:
            raise ValueError("ExternalRetrievalResolutionBatch ID is not content-bound")
        payload = {"batch_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExternalRetrievalResolutionBatch content_hash is invalid")
        return self


class ScholarlyMetadataProposal(StrictModel):
    proposal_id: NonBlankStr
    backend_id: NonBlankStr
    backend_descriptor_hash: Sha256Str
    runtime_identity_hash: Sha256Str
    title: NonBlankStr | None = None
    authors: tuple[NonBlankStr, ...] = ()
    doi: NonBlankStr | None = None
    abstract: NonBlankStr | None = None
    references: tuple[NonBlankStr, ...] = ()
    section_hints: tuple[NonBlankStr, ...] = ()
    trust_class: ExternalProposalTrust = ExternalProposalTrust.EXTERNAL_PROPOSAL
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ScholarlyMetadataProposal:
        identity = self.model_dump(mode="json", exclude={"proposal_id", "content_hash"}, exclude_none=True)
        expected_id = f"scholarly-metadata-proposal-{content_hash(identity)[:24]}"
        if self.proposal_id != expected_id:
            raise ValueError("ScholarlyMetadataProposal ID is not content-bound")
        payload = {"proposal_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ScholarlyMetadataProposal content_hash is invalid")
        return self


class BackendInputBinding(StrictModel):
    input_id: NonBlankStr
    input_hash: Sha256Str


class BackendRunRecord(StrictModel):
    run_id: NonBlankStr
    backend_id: NonBlankStr
    backend_descriptor_hash: Sha256Str
    runtime_identity_id: NonBlankStr
    runtime_identity_hash: Sha256Str
    capability: BackendCapability
    input_bindings: tuple[BackendInputBinding, ...]
    config_hash: Sha256Str
    output_id: NonBlankStr | None = None
    output_type: BackendOutputType | None = None
    output_hash: Sha256Str | None = None
    status: BackendRunStatus
    warnings: tuple[NonBlankStr, ...] = ()
    output_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    resolved_count: int = Field(default=0, ge=0)
    unresolved_count: int = Field(default=0, ge=0)
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> BackendRunRecord:
        _reject_sensitive_warnings(self.warnings)
        if len({item.input_id for item in self.input_bindings}) != len(self.input_bindings):
            raise ValueError("backend run input IDs must be unique")
        input_ids = tuple(item.input_id for item in self.input_bindings)
        if input_ids != tuple(sorted(input_ids)):
            raise ValueError("backend run inputs must be deterministically sorted")
        if self.status in {BackendRunStatus.SUCCEEDED, BackendRunStatus.PARTIAL}:
            if any(value is None for value in (self.output_id, self.output_type, self.output_hash)):
                raise ValueError("successful backend run requires complete output identity")
        elif any(value is not None for value in (self.output_id, self.output_type, self.output_hash)) and not all(
            value is not None for value in (self.output_id, self.output_type, self.output_hash)
        ):
            raise ValueError("backend run output identity must be complete")
        identity = self.model_dump(mode="json", exclude={"run_id", "content_hash"}, exclude_none=True)
        expected_id = f"backend-run-{content_hash(identity)[:24]}"
        if self.run_id != expected_id:
            raise ValueError("BackendRunRecord run_id is not content-bound")
        payload = {"run_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("BackendRunRecord content_hash is invalid")
        return self


class ExternalStructurePromotionRecord(StrictModel):
    promotion_id: NonBlankStr
    backend_run_id: NonBlankStr
    backend_run_hash: Sha256Str
    proposal_id: NonBlankStr
    proposal_hash: Sha256Str
    rebinding_id: NonBlankStr
    rebinding_hash: Sha256Str
    structure_id: NonBlankStr
    structure_hash: Sha256Str
    selection_id: NonBlankStr
    selection_hash: Sha256Str
    promotion_policy: NonBlankStr
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ExternalStructurePromotionRecord:
        identity = self.model_dump(
            mode="json",
            exclude={"promotion_id", "content_hash"},
        )
        expected_id = f"external-structure-promotion-{content_hash(identity)[:24]}"
        if self.promotion_id != expected_id:
            raise ValueError("ExternalStructurePromotionRecord ID is not content-bound")
        payload = {"promotion_id": expected_id, **identity}
        if self.content_hash != content_hash(payload):
            raise ValueError("ExternalStructurePromotionRecord content_hash is invalid")
        return self


class BackendAdapter(Protocol):
    descriptor: ExternalBackendDescriptor

    def inspect_availability(self) -> BackendRuntimeAvailability: ...

    def resolve_runtime_identity(self) -> BackendRuntimeIdentity: ...


class DocumentParsingBackend(BackendAdapter, Protocol):
    def parse(self, request: ExternalDocumentParseInput) -> ExternalDocumentParseProposal: ...


class BuiltinStructureBackend(BackendAdapter, Protocol):
    def structure(self, *args: Any, **kwargs: Any) -> Any: ...


class ScientificRetrievalBackend(BackendAdapter, Protocol):
    def retrieve(self, query: ExternalLiteratureRetrievalQuery) -> ExternalLiteratureRetrievalResult: ...


class ScholarlyMetadataBackend(BackendAdapter, Protocol):
    def extract_metadata(
        self, source: bytes, *, media_type: str, config: Mapping[str, Any]
    ) -> ScholarlyMetadataProposal: ...


class KnowledgeExtractionBackend(BackendAdapter, Protocol):
    def extract_knowledge(self, proposal: Mapping[str, Any]) -> Mapping[str, Any]: ...


class GraphRetrievalBackend(BackendAdapter, Protocol):
    def retrieve_graph(self, query: Mapping[str, Any]) -> Mapping[str, Any]: ...
