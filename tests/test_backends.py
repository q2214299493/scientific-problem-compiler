from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

import spc
from spc.backends import (
    BackendCapability,
    BackendDescriptorRepository,
    BackendInvocationError,
    BackendInputBinding,
    BackendRegistry,
    BackendRegistryError,
    BackendRunRepository,
    BackendRunStatus,
    BackendRuntimeAvailability,
    BackendRuntimeIdentityRepository,
    BackendUnavailableError,
    DoclingDocumentParsingBackend,
    ExternalBackendDescriptor,
    ExternalDocumentElementKind,
    ExternalDocumentProposalRepository,
    ExternalDocumentStructureService,
    ExternalLiteratureRetrievalService,
    ExternalProposalStatus,
    ExternalStructureRebinder,
    ExternalStructurePromotionRepository,
    ExternalStructureRebindingRepository,
    GROBIDScholarlyMetadataBackend,
    ScholarlyMetadataService,
    build_external_document_element,
    build_external_document_parse_proposal,
    build_external_retrieval_hit,
    build_backend_runtime_identity,
    create_backend_run_record,
    default_backend_registry,
    load_backend_license_manifest,
)
from spc.backends.provenance import persist_backend_identity
from spc.backends.contracts import (
    BackendIntegrationMode,
    BackendLicenseStatus,
    ExternalDocumentParseInput,
    ExternalLiteratureRetrievalQuery,
)
from spc.backends.retrieval import (
    ExternalRetrievalEvidenceResolver,
    ExternalRetrievalResolutionBatchRepository,
    PaperQALiteratureRetrievalBackend,
    validate_resolution_batch_current,
)
from spc.backends.document.docling import normalize_docling_document
from spc.cli import app
from spc.knowledge.acquisition import (
    HTTPResponse,
    LiteratureAcquisitionService,
    SafeHTTPFetcher,
)
from spc.knowledge.ingestion import LiteratureRepresentationSelector
from spc.knowledge.structure import (
    DocumentStructureService,
    validate_document_structure,
)
from spc.knowledge.structure_selection import DocumentStructureSelector
from spc.models import CurationStatus, DocumentContentRegion, KnowledgeCurationRecord
from spc.repositories import KnowledgeEvidenceStore, KnowledgeRepositories
from spc.serialization import content_hash, file_sha256


ARTICLE_URL = "https://journal.example/backend-paper"
HTML = b"""<html><head>
<meta name="citation_title" content="Backend Boundary Paper">
<meta name="citation_author" content="A. Author">
<meta name="citation_publication_date" content="2026">
<meta name="citation_doi" content="10.0000/backend.1">
</head><body><article>
<h1>Results</h1><p>Unique scientific paragraph.</p>
<p>Repeated passage.</p><p>Repeated passage.</p>
<table><caption>Table 1 Values</caption><tr><td>Unique cell value</td></tr></table>
<figure><figcaption>Figure 1 unique caption.</figcaption></figure>
<div class="references"><p>Reference-only statement.</p></div>
</article></body></html>"""


class StaticHTMLTransport:
    def request(
        self,
        url: str,
        *,
        validated_ips: tuple[str, ...],
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse:
        assert validated_ips and timeout > 0 and max_bytes > len(HTML)
        return HTTPResponse(
            url=url,
            status=200,
            headers={"content-type": "text/html"},
            body=HTML,
            connected_ip="93.184.216.34",
        )


def descriptor(
    backend_id: str = "fake-document",
    backend_version: str = "1.0.0",
    capability: BackendCapability = BackendCapability.DOCUMENT_PARSING,
) -> ExternalBackendDescriptor:
    payload = {
        "backend_id": backend_id,
        "backend_name": "Deterministic fake backend",
        "backend_version": backend_version,
        "adapter_version": "1.0.0",
        "capability_types": (capability,),
        "integration_mode": BackendIntegrationMode.PYTHON_LIBRARY,
        "source_project": "https://example.invalid/fake-backend",
        "license_id": "test-only",
        "license_status": BackendLicenseStatus.UNVERIFIED,
        "deterministic": True,
        "requires_network": False,
        "optional_dependency": True,
        "vendored": False,
        "redistributable_confirmed": False,
    }
    return ExternalBackendDescriptor(**payload, content_hash=content_hash(payload))


def runtime_identity(
    record_descriptor: ExternalBackendDescriptor,
    version: str | None = None,
):
    return build_backend_runtime_identity(
        record_descriptor,
        resolved_backend_version=version or record_descriptor.backend_version,
        runtime_provider="test-runner:test@1.0.0",
        runtime_config_hash=content_hash({"fixture": "backend-tests"}),
    )


class FakeDocumentBackend:
    def __init__(self) -> None:
        self.descriptor = descriptor()

    def inspect_availability(self) -> BackendRuntimeAvailability:
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=True,
            detected_version=self.descriptor.backend_version,
        )

    def resolve_runtime_identity(self):
        return runtime_identity(self.descriptor)

    def parse(self, request: ExternalDocumentParseInput):
        elements = (
            build_external_document_element(
                kind=ExternalDocumentElementKind.HEADING,
                text="Results",
                reading_order=0,
                heading_level=1,
                proposed_content_region=DocumentContentRegion.REFERENCES,
            ),
            build_external_document_element(
                kind=ExternalDocumentElementKind.PARAGRAPH,
                text="Unique   scientific paragraph.",
                reading_order=1,
                page_hint=99,
                claimed_start_offset=999,
                claimed_end_offset=1000,
            ),
            build_external_document_element(
                kind=ExternalDocumentElementKind.PARAGRAPH,
                text="Repeated passage.",
                reading_order=2,
            ),
            build_external_document_element(
                kind=ExternalDocumentElementKind.PARAGRAPH,
                text="Missing backend passage.",
                reading_order=3,
            ),
            build_external_document_element(
                kind=ExternalDocumentElementKind.TABLE_CAPTION,
                text="Table 1 Values",
                reading_order=4,
                table_ref="table-one",
            ),
            build_external_document_element(
                kind=ExternalDocumentElementKind.TABLE_CELL,
                text="Unique cell value",
                reading_order=5,
                table_ref="table-one",
                row_index=0,
                column_index=0,
                column_span=2,
                is_header=True,
            ),
            build_external_document_element(
                kind=ExternalDocumentElementKind.FIGURE_CAPTION,
                text="Figure 1 unique caption.",
                reading_order=6,
            ),
        )
        return build_external_document_parse_proposal(
            descriptor=self.descriptor,
            runtime_identity=self.resolve_runtime_identity(),
            request=request,
            elements=elements,
            status=ExternalProposalStatus.PARTIAL,
            warnings=("fake backend advisory output",),
        )


class MalformedDocumentBackend(FakeDocumentBackend):
    def parse(self, request: ExternalDocumentParseInput):
        return {"backend_id": self.descriptor.backend_id, "untrusted": "not a proposal"}


class FakeRegistryBackend:
    def __init__(self, record: ExternalBackendDescriptor) -> None:
        self.descriptor = record

    def inspect_availability(self) -> BackendRuntimeAvailability:
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=True,
            detected_version="test-installed",
        )

    def resolve_runtime_identity(self):
        return runtime_identity(self.descriptor, "test-installed")

    def parse(self, _request):
        raise NotImplementedError


def setup_literature(tmp_path: Path):
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = KnowledgeEvidenceStore(tmp_path / "knowledge")
    fetcher = SafeHTTPFetcher(
        StaticHTMLTransport(),
        dns_resolver=lambda _host: ("93.184.216.34",),
    )
    outcome = LiteratureAcquisitionService(fetcher).add(ARTICLE_URL, "base", repositories, store)
    LiteratureRepresentationSelector().select(
        outcome.literature_id or "",
        outcome.representation_id or "",
        "backend-test",
        "Select exact HTML representation.",
        repositories,
        store,
    )
    return repositories, store, outcome


def accept(
    repositories: KnowledgeRepositories,
    target_type: str,
    target_id: str,
    target_hash: str,
) -> KnowledgeCurationRecord:
    identity = {
        "target_type": target_type,
        "target_id": target_id,
        "target_hash": target_hash,
        "status": CurationStatus.ACCEPTED,
        "curator_id": "backend-test-curator",
        "rationale": "Accepted for deterministic backend rebinding test.",
        "evidence_refs": (),
    }
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    record = KnowledgeCurationRecord(**payload, content_hash=content_hash(payload))
    repositories.curations.put(record.curation_id, record)
    return record


def test_base_import_registry_and_license_manifest_are_optional_safe() -> None:
    assert spc.__version__ == "0.1.0"
    registry = default_backend_registry()
    descriptors = {item.backend_id: item for item in registry.list_descriptors()}

    assert descriptors["builtin-document-parser"].optional_dependency is False
    assert registry.inspect_runtime("builtin-document-parser").available
    assert registry.inspect_runtime("docling").available is False
    assert all(record["vendored"] is False for record in load_backend_license_manifest())


def test_registry_availability_conflicts_unknown_and_lazy_failure() -> None:
    registry = BackendRegistry()
    first = descriptor()
    registry.register(FakeRegistryBackend(first))
    assert registry.inspect_runtime(first.backend_id).available
    registry.register(FakeRegistryBackend(first))

    with pytest.raises(BackendRegistryError, match="conflicting"):
        registry.register(FakeRegistryBackend(descriptor(backend_version="2.0.0")))
    with pytest.raises(BackendRegistryError, match="unknown backend"):
        registry.resolve("missing-backend")

    command = CliRunner().invoke(app, ["backend-info", "docling"])
    assert command.exit_code == 0, command.output
    assert json.loads(command.output)["runtime"]["available"] is False


def test_external_rebinding_is_exact_conservative_and_ignores_fake_offsets(
    tmp_path: Path,
) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    representation = repositories.literature_representation_refs.get(outcome.representation_id or "")
    canonical = repositories.canonical_html_text_artifacts.get(representation.canonical_text_id)
    text = repositories.canonical_html_text_artifacts.read_text(representation.canonical_text_id)
    request = ExternalDocumentParseInput(
        artifact_id=representation.raw_artifact_id,
        artifact_path="ignored-by-rebinder",
        artifact_sha256=repositories.raw_html_literature_artifacts.get(representation.raw_artifact_id).sha256,
        media_type="text/html",
        config_hash=content_hash({}),
    )
    proposal = FakeDocumentBackend().parse(request)
    rebound = ExternalStructureRebinder().rebind(proposal, canonical, text)
    by_source = {item.source_element_id: item for item in rebound.bindings}

    unique = by_source[proposal.elements[1].element_id]
    assert unique.status.value == "resolved"
    assert unique.start_offset == text.index("Unique scientific paragraph.")
    assert unique.start_offset != 999
    assert unique.page_number is None
    assert unique.page_hint_consistent is False
    assert by_source[proposal.elements[2].element_id].status.value == "ambiguous"
    assert by_source[proposal.elements[3].element_id].status.value == "missing"
    assert by_source[proposal.elements[0].element_id].proposed_content_region == DocumentContentRegion.REFERENCES
    assert by_source[proposal.elements[0].element_id].content_region == DocumentContentRegion.UNKNOWN
    assert rebound.resolved_count == 5
    assert rebound.unresolved_count == 2


def test_external_document_backend_materializes_valid_audit_structure(
    tmp_path: Path,
) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    external = ExternalDocumentStructureService().structure(
        outcome.literature_id or "",
        outcome.representation_id or "",
        FakeDocumentBackend(),
        repositories,
        store,
    )

    assert external.selection is None
    assert external.proposal.trust_class.value == "external_proposal"
    assert external.rebinding.unresolved_count == 2
    assert external.run_record.status == BackendRunStatus.PARTIAL
    assert len(external.structure.tables) == 1
    assert len(external.structure.table_cells) == 1
    assert len(external.structure.figures) == 1
    assert validate_document_structure(external.structure.artifact, repositories, store)
    assert BackendRunRepository(repositories.root).get(external.run_record.run_id) == external.run_record
    assert ExternalDocumentProposalRepository(repositories.root).get(external.proposal.proposal_id) == external.proposal
    assert (
        ExternalStructureRebindingRepository(repositories.root).get(external.rebinding.rebinding_id)
        == external.rebinding
    )


def test_malformed_external_document_output_fails_with_audit_record(tmp_path: Path) -> None:
    repositories, store, outcome = setup_literature(tmp_path)

    with pytest.raises(BackendInvocationError, match="external document backend failed; run_id="):
        ExternalDocumentStructureService().structure(
            outcome.literature_id or "",
            outcome.representation_id or "",
            MalformedDocumentBackend(),
            repositories,
            store,
        )

    runs = BackendRunRepository(repositories.root).list()
    assert len(runs) == 1
    assert runs[0].status == BackendRunStatus.FAILED
    assert runs[0].output_hash is None
    assert runs[0].warnings == ("backend invocation or output validation failed",)


def test_paperqa_runner_normalizes_untrusted_results() -> None:
    backend = PaperQALiteratureRetrievalBackend(
        lambda _query: (
            {
                "doi": "10.0000/backend.1",
                "title": "Backend Boundary Paper",
                "text_snippet": "Unique scientific paragraph.",
                "score": 0.9,
                "native_rank": 1,
            },
        ),
        runner_id="paperqa-test-runner",
        runner_version="1.0.0",
    )
    query = ExternalLiteratureRetrievalQuery(query="What is reported?")
    result = backend.retrieve(query)
    assert result.trust_class.value == "external_proposal"
    assert result.hits[0].backend_metadata == {"native_rank": 1}
    assert result.hits[0].text_snippet == "Unique scientific paragraph."


def test_retrieval_service_persists_result_and_backend_run(tmp_path: Path) -> None:
    backend = PaperQALiteratureRetrievalBackend(
        lambda _query: (
            {
                "doi": "10.0000/backend.1",
                "text_snippet": "Unique scientific paragraph.",
            },
        ),
        runner_id="paperqa-test-runner",
        runner_version="1.0.0",
    )
    outcome = ExternalLiteratureRetrievalService().retrieve(
        backend,
        ExternalLiteratureRetrievalQuery(query="What is reported?"),
        tmp_path / "knowledge",
    )

    assert outcome.run_record.output_hash == outcome.result.content_hash
    assert outcome.run_record.candidate_count == 1
    assert outcome.run_record.resolved_count == 0
    assert outcome.result.runtime_identity_hash == outcome.run_record.runtime_identity_hash
    assert BackendRunRepository(tmp_path / "knowledge").get(outcome.run_record.run_id) == outcome.run_record


def test_grobid_metadata_remains_untrusted_and_run_bound(tmp_path: Path) -> None:
    backend = GROBIDScholarlyMetadataBackend(
        lambda _source, _media_type, _config: {
            "title": "Candidate metadata only",
            "authors": ("A. Author",),
            "doi": "10.0000/candidate",
            "references": ("Reference A",),
        },
        runner_id="grobid-test-runner",
        runner_version="1.0.0",
        service_version="0.8.2",
    )
    outcome = ScholarlyMetadataService().extract(
        backend,
        b"mock PDF bytes",
        media_type="application/pdf",
        config={"consolidate": False},
        knowledge_root=tmp_path / "knowledge",
    )

    assert outcome.proposal.trust_class.value == "external_proposal"
    assert outcome.proposal.title == "Candidate metadata only"
    assert outcome.proposal.backend_descriptor_hash == outcome.run_record.backend_descriptor_hash
    assert outcome.proposal.runtime_identity_hash == outcome.run_record.runtime_identity_hash
    assert outcome.run_record.output_hash == outcome.proposal.content_hash
    assert KnowledgeRepositories(tmp_path / "knowledge").literature_documents.list() == ()


def test_retrieval_hit_rebinds_only_to_current_accepted_spc_evidence(
    tmp_path: Path,
) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    structure = DocumentStructureService().extract(
        outcome.literature_id or "",
        outcome.representation_id or "",
        repositories,
        store,
    )
    document = repositories.literature_documents.get(outcome.literature_id or "")
    representation_selection = repositories.literature_representation_selections.resolve_current(document.literature_id)
    accept(
        repositories,
        "literature_document",
        document.literature_id,
        document.content_hash,
    )
    accept(
        repositories,
        "literature_representation_selection",
        representation_selection.selection_id,
        representation_selection.content_hash,
    )
    resolver = ExternalRetrievalEvidenceResolver()
    hit = build_external_retrieval_hit(
        doi="https://doi.org/10.0000/backend.1",
        page_hint=99,
        text_snippet="Unique scientific paragraph.",
    )
    resolved = resolver.resolve(hit, repositories, store)

    assert resolved.resolved
    assert resolved.literature_id == document.literature_id
    assert resolved.structure_id == structure.artifact.structure_id
    evidence = store.get_evidence(resolved.evidence_id or "")
    locator = repositories.structured_evidence_locators.get(resolved.locator_id or "")
    assert evidence.text == "Unique scientific paragraph."
    assert locator.page_number is None

    reference = resolver.resolve(
        build_external_retrieval_hit(
            doi="10.0000/backend.1",
            text_snippet="Reference-only statement.",
        ),
        repositories,
        store,
    )
    assert reference.resolved
    assert (
        repositories.structured_evidence_locators.get(reference.locator_id or "").content_region.value == "references"
    )


def test_retrieval_ambiguity_wrong_document_and_page_only_remain_unresolved(
    tmp_path: Path,
) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    DocumentStructureService().extract(
        outcome.literature_id or "",
        outcome.representation_id or "",
        repositories,
        store,
    )
    document = repositories.literature_documents.get(outcome.literature_id or "")
    unaccepted = ExternalRetrievalEvidenceResolver().resolve(
        build_external_retrieval_hit(
            doi="10.0000/backend.1",
            text_snippet="Unique scientific paragraph.",
        ),
        repositories,
        store,
    )
    assert not unaccepted.resolved
    selection = repositories.literature_representation_selections.resolve_current(document.literature_id)
    accept(repositories, "literature_document", document.literature_id, document.content_hash)
    accept(
        repositories,
        "literature_representation_selection",
        selection.selection_id,
        selection.content_hash,
    )
    resolver = ExternalRetrievalEvidenceResolver()

    ambiguous = resolver.resolve(
        build_external_retrieval_hit(doi="10.0000/backend.1", text_snippet="Repeated passage."),
        repositories,
        store,
    )
    wrong = resolver.resolve(
        build_external_retrieval_hit(
            external_document_id="literature-wrong",
            doi="10.0000/backend.1",
            text_snippet="Unique scientific paragraph.",
        ),
        repositories,
        store,
    )
    page_only = resolver.resolve(
        build_external_retrieval_hit(
            doi="10.0000/backend.1",
            page_hint=1,
        ),
        repositories,
        store,
    )

    assert not ambiguous.resolved
    assert not wrong.resolved
    assert not page_only.resolved
    assert store.list_evidence() == ()


def test_retrieval_cannot_use_a_noncurrent_structure_block(tmp_path: Path) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    DocumentStructureService().extract(
        outcome.literature_id or "",
        outcome.representation_id or "",
        repositories,
        store,
    )
    external = ExternalDocumentStructureService().structure(
        outcome.literature_id or "",
        outcome.representation_id or "",
        FakeDocumentBackend(),
        repositories,
        store,
    )
    DocumentStructureSelector().select(
        outcome.literature_id or "",
        outcome.representation_id or "",
        external.structure.artifact.structure_id,
        repositories,
        store,
        rationale="Select reviewed external structure without reference block.",
    )
    document = repositories.literature_documents.get(outcome.literature_id or "")
    representation_selection = repositories.literature_representation_selections.resolve_current(document.literature_id)
    accept(repositories, "literature_document", document.literature_id, document.content_hash)
    accept(
        repositories,
        "literature_representation_selection",
        representation_selection.selection_id,
        representation_selection.content_hash,
    )

    result = ExternalRetrievalEvidenceResolver().resolve(
        build_external_retrieval_hit(
            doi="10.0000/backend.1",
            text_snippet="Reference-only statement.",
        ),
        repositories,
        store,
    )

    assert not result.resolved
    assert "outside current structure" in (result.reason or "")


def test_backend_run_provenance_is_content_bound_secret_free_and_tamper_evident(
    tmp_path: Path,
) -> None:
    record_descriptor = descriptor()
    record_runtime = runtime_identity(record_descriptor)
    bindings = (BackendInputBinding(input_id="artifact-1", input_hash="1" * 64),)
    first = create_backend_run_record(
        descriptor=record_descriptor,
        runtime_identity=record_runtime,
        capability=BackendCapability.DOCUMENT_PARSING,
        input_bindings=bindings,
        config_hash=content_hash({"api_key": "not-persisted"}),
        output_hash="2" * 64,
        status=BackendRunStatus.SUCCEEDED,
        resolved_count=1,
    )
    changed_config = create_backend_run_record(
        descriptor=record_descriptor,
        runtime_identity=record_runtime,
        capability=BackendCapability.DOCUMENT_PARSING,
        input_bindings=bindings,
        config_hash=content_hash({"mode": "other"}),
        output_hash="2" * 64,
        status=BackendRunStatus.SUCCEEDED,
        resolved_count=1,
    )
    changed_descriptor = descriptor(backend_version="2.0.0")
    changed_version = create_backend_run_record(
        descriptor=changed_descriptor,
        runtime_identity=runtime_identity(changed_descriptor),
        capability=BackendCapability.DOCUMENT_PARSING,
        input_bindings=bindings,
        config_hash=first.config_hash,
        output_hash="2" * 64,
        status=BackendRunStatus.SUCCEEDED,
        resolved_count=1,
    )
    repository = BackendRunRepository(tmp_path / "knowledge")
    persist_backend_identity(tmp_path / "knowledge", record_descriptor, record_runtime)
    repository.put(first.run_id, first)

    assert first.run_id != changed_config.run_id
    assert first.run_id != changed_version.run_id
    assert "api_key" not in first.model_dump_json()
    path = repository.root / f"{first.run_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output_hash"] = "3" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        repository.get(first.run_id)


def test_backend_outputs_reject_secret_bearing_audit_metadata() -> None:
    with pytest.raises(ValueError, match="secret-bearing"):
        build_external_document_element(
            kind=ExternalDocumentElementKind.PARAGRAPH,
            text="Candidate text.",
            reading_order=0,
            backend_metadata={"api_key": "must-not-persist"},
        )


def test_backend_cli_lists_and_inspects_run(tmp_path: Path) -> None:
    command = CliRunner().invoke(app, ["backends"])
    assert command.exit_code == 0, command.output
    rows = json.loads(command.output)
    assert any(item["backend_id"] == "builtin-document-parser" for item in rows)

    record_descriptor = descriptor()
    record_runtime = runtime_identity(record_descriptor)
    persist_backend_identity(tmp_path / "knowledge", record_descriptor, record_runtime)
    run = create_backend_run_record(
        descriptor=record_descriptor,
        runtime_identity=record_runtime,
        capability=BackendCapability.DOCUMENT_PARSING,
        input_bindings=(BackendInputBinding(input_id="artifact-cli", input_hash="4" * 64),),
        config_hash="5" * 64,
        output_hash="6" * 64,
        status=BackendRunStatus.SUCCEEDED,
    )
    BackendRunRepository(tmp_path / "knowledge").put(run.run_id, run)
    inspected = CliRunner().invoke(
        app,
        [
            "inspect-backend-run",
            run.run_id,
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
        ],
    )
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.output)["run"]["run_id"] == run.run_id


def test_runtime_versions_change_docling_and_paperqa_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "docling-models"
    artifacts.mkdir()
    docling = DoclingDocumentParsingBackend(artifacts)
    monkeypatch.setattr(docling, "_installed_version", lambda: "2.125.0")
    docling_v1 = docling.resolve_runtime_identity()
    monkeypatch.setattr(docling, "_installed_version", lambda: "2.126.0")
    docling_v2 = docling.resolve_runtime_identity()
    assert docling_v1.runtime_identity_id != docling_v2.runtime_identity_id
    docling_run_v1 = create_backend_run_record(
        descriptor=docling.descriptor,
        runtime_identity=docling_v1,
        capability=BackendCapability.DOCUMENT_PARSING,
        input_bindings=(),
        config_hash=content_hash({}),
        output_hash=content_hash({"output": "docling"}),
        status=BackendRunStatus.SUCCEEDED,
    )
    docling_run_v2 = create_backend_run_record(
        descriptor=docling.descriptor,
        runtime_identity=docling_v2,
        capability=BackendCapability.DOCUMENT_PARSING,
        input_bindings=(),
        config_hash=content_hash({}),
        output_hash=content_hash({"output": "docling"}),
        status=BackendRunStatus.SUCCEEDED,
    )
    assert docling_run_v1.run_id != docling_run_v2.run_id

    monkeypatch.setattr("spc.backends.retrieval.paperqa.util.find_spec", lambda _name: object())
    versions = iter(("2026.3.18", "2026.8.12"))
    monkeypatch.setattr("spc.backends.retrieval.paperqa.metadata.version", lambda _name: next(versions))

    def runner(_query):
        return ()

    paperqa_v1 = PaperQALiteratureRetrievalBackend(
        runner,
        runner_id="paperqa-runtime-test",
        runner_version="1.0.0",
    ).resolve_runtime_identity()
    paperqa_v2 = PaperQALiteratureRetrievalBackend(
        runner,
        runner_id="paperqa-runtime-test",
        runner_version="1.0.0",
    ).resolve_runtime_identity()
    assert paperqa_v1.runtime_identity_id != paperqa_v2.runtime_identity_id
    paperqa_run_v1 = create_backend_run_record(
        descriptor=PaperQALiteratureRetrievalBackend.descriptor,
        runtime_identity=paperqa_v1,
        capability=BackendCapability.LITERATURE_RETRIEVAL,
        input_bindings=(),
        config_hash=content_hash({}),
        output_hash=content_hash({"output": "paperqa"}),
        status=BackendRunStatus.SUCCEEDED,
    )
    paperqa_run_v2 = create_backend_run_record(
        descriptor=PaperQALiteratureRetrievalBackend.descriptor,
        runtime_identity=paperqa_v2,
        capability=BackendCapability.LITERATURE_RETRIEVAL,
        input_bindings=(),
        config_hash=content_hash({}),
        output_hash=content_hash({"output": "paperqa"}),
        status=BackendRunStatus.SUCCEEDED,
    )
    assert paperqa_run_v1.run_id != paperqa_run_v2.run_id


def test_descriptor_and_runtime_identity_snapshots_are_reopenable(
    tmp_path: Path,
) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    external = ExternalDocumentStructureService().structure(
        outcome.literature_id or "",
        outcome.representation_id or "",
        FakeDocumentBackend(),
        repositories,
        store,
    )
    run = external.run_record

    descriptor_record = BackendDescriptorRepository(repositories.root).get(run.backend_descriptor_hash)
    runtime_record = BackendRuntimeIdentityRepository(repositories.root).get(run.runtime_identity_id)
    assert descriptor_record.content_hash == run.backend_descriptor_hash
    assert runtime_record.content_hash == run.runtime_identity_hash


def test_optional_dependency_availability_is_deterministic_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("spc.backends.document.docling.util.find_spec", lambda _name: None)
    monkeypatch.setattr("spc.backends.retrieval.paperqa.util.find_spec", lambda _name: None)
    registry = default_backend_registry()
    assert not registry.inspect_runtime("docling").available
    assert not registry.inspect_runtime("paperqa2").available


def test_docling_requires_local_artifacts_and_audits_unavailable_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    backend = DoclingDocumentParsingBackend(tmp_path / "missing-models")
    monkeypatch.setattr(backend, "_installed_version", lambda: "2.126.0")

    with pytest.raises(BackendUnavailableError, match="runtime probe failed"):
        ExternalDocumentStructureService().structure(
            outcome.literature_id or "",
            outcome.representation_id or "",
            backend,
            repositories,
            store,
        )

    run = BackendRunRepository(repositories.root).list()[0]
    assert run.status == BackendRunStatus.UNAVAILABLE
    assert run.warnings == ("runtime_unavailable",)


def test_docling_converter_is_configured_for_offline_local_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakePipelineOptions:
        def __init__(self, **values):
            captured["pipeline_options"] = values

    class FakeFormatOption:
        def __init__(self, **values):
            captured["format_option"] = values

    class FakeConverter:
        def __init__(self, **values):
            captured["converter"] = values

        def convert(self, path):
            captured["path"] = path
            return SimpleNamespace(document="typed-document")

    base_models = ModuleType("docling.datamodel.base_models")
    base_models.InputFormat = SimpleNamespace(PDF="pdf")
    pipeline_options = ModuleType("docling.datamodel.pipeline_options")
    pipeline_options.PdfPipelineOptions = FakePipelineOptions
    document_converter = ModuleType("docling.document_converter")
    document_converter.DocumentConverter = FakeConverter
    document_converter.PdfFormatOption = FakeFormatOption
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", base_models)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", pipeline_options)
    monkeypatch.setitem(sys.modules, "docling.document_converter", document_converter)
    artifacts = tmp_path / "models"
    artifacts.mkdir()
    result = DoclingDocumentParsingBackend(artifacts)._convert_document(tmp_path / "paper.pdf")

    assert result == "typed-document"
    assert captured["pipeline_options"] == {
        "artifacts_path": artifacts,
        "enable_remote_services": False,
        "allow_external_plugins": False,
    }


def _typed_docling_fixture():
    heading = SimpleNamespace(
        label="section_header",
        text="Results",
        self_ref="#/texts/0",
        level=1,
        prov=(SimpleNamespace(page_no=1),),
        content_region="references",
    )
    paragraph = SimpleNamespace(
        label="text",
        text="Scientific paragraph.",
        self_ref="#/texts/1",
        prov=(SimpleNamespace(page_no=1),),
        content_layer="body",
    )
    table = SimpleNamespace(
        label="table",
        self_ref="#/tables/0",
        prov=(SimpleNamespace(page_no=2),),
        captions=(SimpleNamespace(text="Table caption", self_ref="#/texts/2"),),
        data=SimpleNamespace(
            table_cells=(
                SimpleNamespace(
                    text="Header value",
                    start_row_offset_idx=0,
                    start_col_offset_idx=0,
                    row_span=1,
                    col_span=2,
                    column_header=True,
                    row_header=False,
                    row_section=False,
                ),
                SimpleNamespace(
                    text="Body value",
                    start_row_offset_idx=1,
                    start_col_offset_idx=0,
                    row_span=2,
                    col_span=1,
                    column_header=False,
                    row_header=True,
                    row_section=False,
                ),
            )
        ),
    )
    picture = SimpleNamespace(
        label="picture",
        self_ref="#/pictures/0",
        prov=(SimpleNamespace(page_no=2),),
        captions=(SimpleNamespace(text="Figure caption", self_ref="#/texts/3"),),
    )
    return SimpleNamespace(iterate_items=lambda: iter(((heading, 1), (paragraph, 1), (table, 1), (picture, 1))))


def test_real_style_docling_items_preserve_table_topology_and_proposals() -> None:
    record_descriptor = DoclingDocumentParsingBackend.descriptor
    runtime = build_backend_runtime_identity(
        record_descriptor,
        resolved_backend_version="2.126.0",
        runtime_provider="python-package:docling",
        runtime_config_hash=content_hash({"fixture": "typed-docling"}),
    )
    request = ExternalDocumentParseInput(
        artifact_id="artifact-typed-docling",
        artifact_path="C:/fixture/paper.pdf",
        artifact_sha256="1" * 64,
        media_type="application/pdf",
        config_hash=content_hash({}),
    )
    proposal = normalize_docling_document(
        _typed_docling_fixture(),
        descriptor=record_descriptor,
        runtime_identity=runtime,
        request=request,
    )
    cells = tuple(item for item in proposal.elements if item.kind == ExternalDocumentElementKind.TABLE_CELL)

    assert proposal.status == ExternalProposalStatus.COMPLETE
    assert proposal.runtime_identity_hash == runtime.content_hash
    assert len(cells) == 2
    assert (cells[0].row_index, cells[0].column_index) == (0, 0)
    assert (cells[0].row_span, cells[0].column_span, cells[0].is_header) == (
        1,
        2,
        True,
    )
    assert (cells[1].row_span, cells[1].column_span, cells[1].is_header) == (
        2,
        1,
        True,
    )
    assert proposal.elements[0].proposed_content_region.value == "references"


def test_external_table_span_rebinds_and_validates_topology(tmp_path: Path) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    external = ExternalDocumentStructureService().structure(
        outcome.literature_id or "",
        outcome.representation_id or "",
        FakeDocumentBackend(),
        repositories,
        store,
    )

    cell = external.structure.table_cells[0]
    assert (cell.row_span, cell.column_span, cell.is_header) == (1, 2, True)
    assert validate_document_structure(external.structure.artifact, repositories, store)


class RaisingDocumentBackend(FakeDocumentBackend):
    def __init__(self, error_type: type[Exception]) -> None:
        super().__init__()
        self.error_type = error_type

    def parse(self, request: ExternalDocumentParseInput):
        raise self.error_type("third-party details must not be persisted")


class AvailabilityRaisingBackend(FakeDocumentBackend):
    def inspect_availability(self):
        raise OSError("availability secret must not be persisted")


@pytest.mark.parametrize("error_type", [OSError, TimeoutError])
def test_third_party_exceptions_create_sanitized_failed_run(tmp_path: Path, error_type: type[Exception]) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    with pytest.raises(BackendInvocationError):
        ExternalDocumentStructureService().structure(
            outcome.literature_id or "",
            outcome.representation_id or "",
            RaisingDocumentBackend(error_type),
            repositories,
            store,
        )
    run = BackendRunRepository(repositories.root).list()[0]
    assert run.status == BackendRunStatus.FAILED
    assert run.warnings == ("backend invocation or output validation failed",)
    assert "third-party details" not in run.model_dump_json()


def test_availability_exception_creates_sanitized_failed_run(tmp_path: Path) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    with pytest.raises(BackendInvocationError, match="runtime probe failed"):
        ExternalDocumentStructureService().structure(
            outcome.literature_id or "",
            outcome.representation_id or "",
            AvailabilityRaisingBackend(),
            repositories,
            store,
        )
    run = BackendRunRepository(repositories.root).list()[0]
    assert run.status == BackendRunStatus.FAILED
    assert run.warnings == ("availability_inspection_failed",)
    assert "availability secret" not in run.model_dump_json()


def test_registry_rejects_capability_without_callable_contract() -> None:
    class MissingParser:
        descriptor = descriptor()

        def inspect_availability(self):
            return BackendRuntimeAvailability(
                backend_id=self.descriptor.backend_id,
                available=True,
                detected_version="1.0.0",
            )

        def resolve_runtime_identity(self):
            return runtime_identity(self.descriptor)

    with pytest.raises(BackendRegistryError, match="does not implement parse"):
        BackendRegistry().register(MissingParser())


class TypedDoclingBackend(DoclingDocumentParsingBackend):
    def _installed_version(self) -> str | None:
        return "2.126.0"

    def _convert_document(self, path: Path) -> object:
        return _typed_docling_fixture()


def test_nonempty_docling_config_is_rejected_and_audited(tmp_path: Path) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    artifacts = tmp_path / "models"
    artifacts.mkdir()
    with pytest.raises(BackendInvocationError):
        ExternalDocumentStructureService().structure(
            outcome.literature_id or "",
            outcome.representation_id or "",
            TypedDoclingBackend(artifacts),
            repositories,
            store,
            config={"do_ocr": False},
        )
    run = BackendRunRepository(repositories.root).list()[0]
    assert run.status == BackendRunStatus.FAILED


def test_resolution_batch_uses_true_spc_resolution_counts_and_current_authority(
    tmp_path: Path,
) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    structure = DocumentStructureService().extract(
        outcome.literature_id or "",
        outcome.representation_id or "",
        repositories,
        store,
    )
    document = repositories.literature_documents.get(outcome.literature_id or "")
    representation_selection = repositories.literature_representation_selections.resolve_current(document.literature_id)
    accept(repositories, "literature_document", document.literature_id, document.content_hash)
    accept(
        repositories,
        "literature_representation_selection",
        representation_selection.selection_id,
        representation_selection.content_hash,
    )
    backend = PaperQALiteratureRetrievalBackend(
        lambda _query: (
            {
                "doi": document.doi,
                "text_snippet": "Unique scientific paragraph.",
            },
            {"doi": document.doi, "text_snippet": "Repeated passage."},
        ),
        runner_id="paperqa-batch-test",
        runner_version="1.0.0",
    )
    retrieval = ExternalLiteratureRetrievalService().retrieve(
        backend,
        ExternalLiteratureRetrievalQuery(query="Find exact evidence"),
        repositories.root,
    )
    batch = ExternalRetrievalEvidenceResolver().resolve_batch(retrieval.result, repositories, store)

    assert retrieval.run_record.candidate_count == 2
    assert retrieval.run_record.resolved_count == 0
    assert retrieval.run_record.unresolved_count == 0
    assert (batch.resolved_count, batch.unresolved_count) == (1, 1)
    assert batch.authority_bindings[0].structure_id == structure.artifact.structure_id
    assert ExternalRetrievalResolutionBatchRepository(repositories.root).get(batch.batch_id) == batch
    assert validate_resolution_batch_current(batch, repositories, store) == batch

    alternate = ExternalDocumentStructureService().structure(
        outcome.literature_id or "",
        outcome.representation_id or "",
        FakeDocumentBackend(),
        repositories,
        store,
    )
    DocumentStructureSelector().select(
        document.literature_id,
        outcome.representation_id or "",
        alternate.structure.artifact.structure_id,
        repositories,
        store,
        rationale="Select a new reviewed structure for staleness test.",
    )
    with pytest.raises(ValueError, match="authority binding is stale"):
        validate_resolution_batch_current(batch, repositories, store)


def test_external_structure_promotion_preserves_full_external_chain(
    tmp_path: Path,
) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    external = ExternalDocumentStructureService().structure(
        outcome.literature_id or "",
        outcome.representation_id or "",
        FakeDocumentBackend(),
        repositories,
        store,
        promote=True,
    )

    assert external.selection is not None
    assert external.selection.rationale == "external_exact_rebinding_promotion"
    assert external.promotion is not None
    assert external.promotion.backend_run_id == external.run_record.run_id
    assert external.promotion.proposal_id == external.proposal.proposal_id
    assert external.promotion.rebinding_id == external.rebinding.rebinding_id
    assert external.promotion.structure_id == external.structure.artifact.structure_id
    assert (
        ExternalStructurePromotionRepository(repositories.root).get(external.promotion.promotion_id)
        == external.promotion
    )


def test_external_structure_downgrade_is_not_recorded_as_promotion(tmp_path: Path) -> None:
    repositories, store, outcome = setup_literature(tmp_path)
    current = DocumentStructureService().extract(
        outcome.literature_id or "",
        outcome.representation_id or "",
        repositories,
        store,
    )
    external = ExternalDocumentStructureService().structure(
        outcome.literature_id or "",
        outcome.representation_id or "",
        FakeDocumentBackend(),
        repositories,
        store,
        promote=True,
    )

    assert current.artifact.extraction_status.value == "complete"
    assert external.structure.artifact.extraction_status.value == "partial"
    assert external.selection is None
    assert external.promotion is None
    selected = repositories.document_structure_selections.resolve_current(outcome.representation_id or "")
    assert selected.structure_id == current.artifact.structure_id


def test_real_docling_smoke_only_with_preconfigured_local_artifacts() -> None:
    artifacts_value = os.environ.get("SPC_DOCLING_ARTIFACTS_PATH")
    if not artifacts_value:
        pytest.skip("SPC_DOCLING_ARTIFACTS_PATH is not configured")
    artifacts = Path(artifacts_value)
    backend = DoclingDocumentParsingBackend(artifacts)
    if not backend.inspect_availability().available:
        pytest.skip("compatible Docling and local artifacts are unavailable")
    fixture = Path(__file__).parent / "fixtures" / "generic-born-digital.pdf"
    request = ExternalDocumentParseInput(
        artifact_id="optional-real-docling-fixture",
        artifact_path=str(fixture.resolve()),
        artifact_sha256=file_sha256(fixture),
        media_type="application/pdf",
        config_hash=content_hash({}),
    )
    proposal = backend.parse(request)
    assert proposal.backend_id == "docling"
    assert proposal.elements
