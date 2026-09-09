from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import spc
from spc.backends import (
    BackendCapability,
    BackendInputBinding,
    BackendRegistry,
    BackendRegistryError,
    BackendRunRepository,
    BackendRunStatus,
    BackendRuntimeAvailability,
    ExternalBackendDescriptor,
    ExternalDocumentElementKind,
    ExternalDocumentProposalRepository,
    ExternalDocumentStructureService,
    ExternalLiteratureRetrievalService,
    ExternalProposalStatus,
    ExternalStructureRebinder,
    ExternalStructureRebindingRepository,
    GROBIDScholarlyMetadataBackend,
    ScholarlyMetadataService,
    build_external_document_element,
    build_external_document_parse_proposal,
    build_external_retrieval_hit,
    create_backend_run_record,
    default_backend_registry,
    load_backend_license_manifest,
)
from spc.backends.contracts import (
    BackendIntegrationMode,
    BackendLicenseStatus,
    ExternalDocumentParseInput,
    ExternalLiteratureRetrievalQuery,
)
from spc.backends.retrieval import (
    ExternalRetrievalEvidenceResolver,
    PaperQALiteratureRetrievalBackend,
)
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
from spc.models import CurationStatus, KnowledgeCurationRecord
from spc.repositories import KnowledgeEvidenceStore, KnowledgeRepositories
from spc.serialization import content_hash


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


class FakeDocumentBackend:
    def __init__(self) -> None:
        self.descriptor = descriptor()

    def inspect_availability(self) -> BackendRuntimeAvailability:
        return BackendRuntimeAvailability(
            backend_id=self.descriptor.backend_id,
            available=True,
            detected_version=self.descriptor.backend_version,
        )

    def parse(self, request: ExternalDocumentParseInput):
        elements = (
            build_external_document_element(
                kind=ExternalDocumentElementKind.HEADING,
                text="Results",
                reading_order=0,
                heading_level=1,
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
            ),
            build_external_document_element(
                kind=ExternalDocumentElementKind.FIGURE_CAPTION,
                text="Figure 1 unique caption.",
                reading_order=6,
            ),
        )
        return build_external_document_parse_proposal(
            descriptor=self.descriptor,
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

    with pytest.raises(ValueError, match="external document backend failed; run_id="):
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
        )
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
        )
    )
    outcome = ExternalLiteratureRetrievalService().retrieve(
        backend,
        ExternalLiteratureRetrievalQuery(query="What is reported?"),
        tmp_path / "knowledge",
    )

    assert outcome.run_record.output_hash == outcome.result.content_hash
    assert BackendRunRepository(tmp_path / "knowledge").get(outcome.run_record.run_id) == outcome.run_record


def test_grobid_metadata_remains_untrusted_and_run_bound(tmp_path: Path) -> None:
    backend = GROBIDScholarlyMetadataBackend(
        lambda _source, _media_type, _config: {
            "title": "Candidate metadata only",
            "authors": ("A. Author",),
            "doi": "10.0000/candidate",
            "references": ("Reference A",),
        }
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
    bindings = (BackendInputBinding(input_id="artifact-1", input_hash="1" * 64),)
    first = create_backend_run_record(
        descriptor=record_descriptor,
        capability=BackendCapability.DOCUMENT_PARSING,
        input_bindings=bindings,
        config_hash=content_hash({"api_key": "not-persisted"}),
        output_hash="2" * 64,
        status=BackendRunStatus.SUCCEEDED,
        resolved_count=1,
    )
    changed_config = create_backend_run_record(
        descriptor=record_descriptor,
        capability=BackendCapability.DOCUMENT_PARSING,
        input_bindings=bindings,
        config_hash=content_hash({"mode": "other"}),
        output_hash="2" * 64,
        status=BackendRunStatus.SUCCEEDED,
        resolved_count=1,
    )
    changed_version = create_backend_run_record(
        descriptor=descriptor(backend_version="2.0.0"),
        capability=BackendCapability.DOCUMENT_PARSING,
        input_bindings=bindings,
        config_hash=first.config_hash,
        output_hash="2" * 64,
        status=BackendRunStatus.SUCCEEDED,
        resolved_count=1,
    )
    repository = BackendRunRepository(tmp_path / "knowledge")
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

    run = create_backend_run_record(
        descriptor=descriptor(),
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
    assert json.loads(inspected.output)["run_id"] == run.run_id
