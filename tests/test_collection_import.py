from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import stat

from typer.testing import CliRunner

from spc.cli import app
import pytest

from spc.knowledge.acquisition import AcquisitionError, HTTPResponse, SafeHTTPFetcher
from spc.knowledge.collection import (
    CollectionImportService,
    GenericHTMLCollectionConnector,
    diff_collection_snapshots,
    make_collection_definition,
    make_collection_scope_policy,
    path_matches_prefix,
)
from spc.models import (
    AcquisitionInputKind,
    AcquisitionStatus,
    CollectionCompletenessStatus,
    CollectionDiscoveryConfidence,
    CollectionImportStatus,
    CollectionResourceKind,
    LiteratureAcquisitionOutcome,
    LiteratureAcquisitionRecord,
)
from spc.repositories import KnowledgeRepositories, SourceEvidenceStore
from spc.serialization import content_hash


FIXTURES = Path(__file__).parent / "fixtures"
PDF_BYTES = (FIXTURES / "generic-born-digital.pdf").read_bytes()
PUBLIC_IP = "93.184.216.34"


class FakeHTTPTransport:
    def __init__(self, responses: dict[str, HTTPResponse | list[HTTPResponse]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def request(
        self,
        url: str,
        *,
        validated_ips: tuple[str, ...],
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse:
        assert validated_ips == (PUBLIC_IP,)
        assert timeout > 0
        assert max_bytes > 0
        self.calls.append(url)
        configured = self.responses[url]
        if isinstance(configured, list):
            if not configured:
                raise AssertionError(f"no responses remain for {url}")
            return configured.pop(0)
        return configured


def response(
    url: str,
    body: bytes,
    media_type: str = "text/html",
    *,
    status: int = 200,
) -> HTTPResponse:
    return HTTPResponse(
        url=url,
        status=status,
        headers={"content-type": media_type},
        body=body,
        connected_ip=PUBLIC_IP,
    )


def fetcher(responses: dict[str, HTTPResponse | list[HTTPResponse]]) -> SafeHTTPFetcher:
    return SafeHTTPFetcher(
        FakeHTTPTransport(responses),
        dns_resolver=lambda _host: (PUBLIC_IP,),
    )


def repositories(tmp_path: Path) -> tuple[KnowledgeRepositories, SourceEvidenceStore]:
    return KnowledgeRepositories(tmp_path / "knowledge"), SourceEvidenceStore(tmp_path / ".spc")


def discover(
    tmp_path: Path,
    entry_url: str,
    responses: dict[str, HTTPResponse | list[HTTPResponse]],
    *,
    max_pages: int = 100,
    max_resources: int = 1000,
    allow_external: bool = False,
    allowed_origins: tuple[str, ...] | None = None,
):
    repos, _ = repositories(tmp_path)
    policy = make_collection_scope_policy(
        entry_url,
        max_pages=max_pages,
        max_resources=max_resources,
        allow_external_literature_links=allow_external,
        allowed_origins=allowed_origins,
    )
    definition = make_collection_definition(entry_url, "base", scope_policy=policy)
    result = GenericHTMLCollectionConnector().discover(
        definition, repos, fetcher(responses)
    )
    return result, repos


def test_one_page_collection_discovers_multiple_dois(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    html = b"<html><body>10.1234/alpha <a href='https://doi.org/10.1234/beta'>paper</a></body></html>"
    result, _ = discover(tmp_path, url, {url: response(url, html)})

    assert result.snapshot.completeness == CollectionImportStatus.PARTIAL
    assert (
        result.snapshot.completeness_status
        == CollectionCompletenessStatus.POLICY_EXHAUSTED_UNVERIFIED
    )
    assert [item.doi for item in result.resources] == ["10.1234/alpha", "10.1234/beta"]


def test_multi_page_pagination_and_cross_page_occurrences(tmp_path: Path) -> None:
    first = "https://collection.example/project?page=1"
    second = "https://collection.example/project?page=2"
    responses = {
        first: response(
            first,
            f"<p>10.1234/shared</p><a rel='next' href='{second}'>Next</a>".encode(),
        ),
        second: response(second, b"<p>10.1234/shared 10.1234/new</p>"),
    }
    result, _ = discover(tmp_path, first, responses)

    assert result.snapshot.visited_page_count == 2
    assert result.snapshot.unique_resource_count == 2
    assert (
        result.snapshot.completeness_status
        == CollectionCompletenessStatus.PROVEN_COMPLETE
    )
    shared = next(item for item in result.resources if item.doi == "10.1234/shared")
    assert sum(
        item.discovered_resource_id == shared.discovered_resource_id
        for item in result.occurrences
    ) == 2


def test_article_url_with_embedded_doi_deduplicates_against_doi(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    html = b"<p>10.1234/shared</p><a href='/doi/10.1234/shared'>Article</a>"
    result, _ = discover(tmp_path, url, {url: response(url, html)})

    assert len(result.resources) == 1
    assert result.resources[0].resource_kind == CollectionResourceKind.DOI


def test_citation_article_url_is_deduplicated_by_citation_doi(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    html = b"""
    <meta name='citation_doi' content='10.1234/shared'>
    <meta name='citation_fulltext_html_url' content='https://journal.example/article/abc'>
    """
    result, _ = discover(tmp_path, url, {url: response(url, html)})

    assert len(result.resources) == 1
    assert result.resources[0].doi == "10.1234/shared"


def test_direct_pdf_is_discovered(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    result, _ = discover(
        tmp_path,
        url,
        {url: response(url, b"<a href='/project/paper.pdf'>Download PDF</a>")},
    )

    assert len(result.resources) == 1
    assert result.resources[0].resource_kind == CollectionResourceKind.PDF_URL
    assert result.resources[0].url == "https://collection.example/project/paper.pdf"


def test_cyclic_pagination_terminates_safely(tmp_path: Path) -> None:
    first = "https://collection.example/project?page=1"
    second = "https://collection.example/project?page=2"
    result, _ = discover(
        tmp_path,
        first,
        {
            first: response(first, f"<a rel='next' href='{second}'>Next</a>".encode()),
            second: response(second, f"<a rel='next' href='{first}'>Next</a>".encode()),
        },
    )

    assert result.snapshot.visited_page_count == 2
    assert "pagination_cycle_detected" in result.snapshot.completeness_reasons


def test_max_pages_with_more_pagination_is_partial(tmp_path: Path) -> None:
    first = "https://collection.example/project?page=1"
    second = "https://collection.example/project?page=2"
    result, _ = discover(
        tmp_path,
        first,
        {first: response(first, f"<a rel='next' href='{second}'>Next</a>".encode())},
        max_pages=1,
    )

    assert result.snapshot.completeness == CollectionImportStatus.PARTIAL
    assert "max_pages_reached" in result.snapshot.completeness_reasons


def test_max_resources_is_enforced_and_marks_partial(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    result, _ = discover(
        tmp_path,
        url,
        {url: response(url, b"10.1234/one 10.1234/two")},
        max_resources=1,
    )

    assert result.snapshot.unique_resource_count == 1
    assert result.snapshot.completeness == CollectionImportStatus.PARTIAL
    assert "max_resources_reached" in result.snapshot.completeness_reasons


def test_out_of_scope_next_page_is_partial_and_not_fetched(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    outside = "https://unrelated.example/pages/2"
    transport = FakeHTTPTransport(
        {url: response(url, f"<a rel='next' href='{outside}'>Next</a>".encode())}
    )
    safe = SafeHTTPFetcher(transport, dns_resolver=lambda _host: (PUBLIC_IP,))
    repos, _ = repositories(tmp_path)
    definition = make_collection_definition(url, "base")
    result = GenericHTMLCollectionConnector().discover(definition, repos, safe)

    assert transport.calls == [url]
    assert result.snapshot.completeness == CollectionImportStatus.PARTIAL
    assert "unresolved_pagination_outside_scope" in result.snapshot.completeness_reasons


def test_unresolved_script_pagination_is_partial(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    result, _ = discover(
        tmp_path,
        url,
        {url: response(url, b"<a rel='next' href='javascript:loadMore()'>Next</a>")},
    )

    assert result.snapshot.completeness == CollectionImportStatus.PARTIAL
    assert "unresolved_pagination" in result.snapshot.completeness_reasons


def test_non_authentication_entry_failure_is_failed(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    result, _ = discover(
        tmp_path,
        url,
        {url: response(url, b"server error", "text/plain", status=500)},
    )

    assert result.snapshot.completeness == CollectionImportStatus.FAILED


def test_authentication_wall_is_recorded(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    result, _ = discover(
        tmp_path,
        url,
        {url: response(url, b"forbidden", status=403)},
    )

    assert result.snapshot.completeness == CollectionImportStatus.REQUIRES_AUTHENTICATION
    assert result.pages[0].http_status == 403


def test_authentication_wall_import_record_is_non_fabricated_and_terminal(
    tmp_path: Path,
) -> None:
    url = "https://collection.example/project"
    repos, store = repositories(tmp_path)
    definition = make_collection_definition(url, "base")
    outcome = CollectionImportService(
        fetcher=fetcher({url: response(url, b"forbidden", status=403)})
    ).run(definition, repos, store)
    record = repos.collection_imports.get(outcome.import_id)

    assert outcome.completeness == CollectionImportStatus.REQUIRES_AUTHENTICATION
    assert record.status == CollectionImportStatus.REQUIRES_AUTHENTICATION
    assert record.total_resources == 0
    assert record.acquisition_link_hashes == {}


def test_javascript_only_collection_is_partial(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    html = b"<html><body>Please enable JavaScript</body><script src='app.js'></script></html>"
    result, _ = discover(tmp_path, url, {url: response(url, html)})

    assert result.snapshot.completeness == CollectionImportStatus.PARTIAL
    assert "unsupported_dynamic_collection" in result.snapshot.completeness_reasons


def test_external_unrelated_link_is_not_crawled_or_imported(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    result, _ = discover(
        tmp_path,
        url,
        {url: response(url, b"<a href='https://unrelated.example/home'>Home</a>")},
    )

    assert result.resources == ()
    assert result.snapshot.visited_page_count == 1


def test_allowed_external_pdf_is_discovered_but_not_crawled(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    pdf = "https://papers.example/paper.pdf"
    transport = FakeHTTPTransport(
        {url: response(url, f"<a href='{pdf}'>PDF</a>".encode())}
    )
    safe = SafeHTTPFetcher(transport, dns_resolver=lambda _host: (PUBLIC_IP,))
    repos, _ = repositories(tmp_path)
    policy = make_collection_scope_policy(
        url,
        allow_external_literature_links=True,
    )
    definition = make_collection_definition(url, "base", scope_policy=policy)
    result = GenericHTMLCollectionConnector().discover(definition, repos, safe)

    assert [item.url for item in result.resources] == [pdf]
    assert transport.calls == [url]


def test_collection_snapshot_is_deterministic(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    html = b"<p>10.1234/stable</p>"
    first, _ = discover(tmp_path / "one", url, {url: response(url, html)})
    second, _ = discover(tmp_path / "two", url, {url: response(url, html)})

    assert first.snapshot == second.snapshot


def test_changed_second_crawl_creates_snapshot_and_diff(tmp_path: Path) -> None:
    url = "https://collection.example/project"
    repos, _ = repositories(tmp_path)
    definition = make_collection_definition(url, "base")
    first = GenericHTMLCollectionConnector().discover(
        definition,
        repos,
        fetcher({url: response(url, b"10.1234/old 10.1234/same")}),
    )
    second = GenericHTMLCollectionConnector().discover(
        definition,
        repos,
        fetcher({url: response(url, b"10.1234/new 10.1234/same")}),
    )
    diff = diff_collection_snapshots(first.snapshot, second.snapshot)

    assert first.snapshot.snapshot_id != second.snapshot.snapshot_id
    assert len(repos.collection_snapshots.list()) == 2
    assert len(diff.added_resource_ids) == 1
    assert len(diff.removed_resource_ids) == 1
    assert len(diff.changed_resource_ids) == 1
    assert diff.unchanged_resource_ids == ()


def test_import_isolates_failed_pdf_and_preserves_counts_and_trust_boundary(
    tmp_path: Path,
) -> None:
    entry = "https://collection.example/project"
    good = "https://collection.example/project/good.pdf"
    bad = "https://collection.example/project/bad.pdf"
    page = f"<a href='{good}'>good</a><a href='{bad}'>bad</a>".encode()
    responses: dict[str, HTTPResponse | list[HTTPResponse]] = {
        entry: response(entry, page),
        good: [
            response(good, PDF_BYTES, "application/pdf"),
            response(good, PDF_BYTES, "application/pdf"),
        ],
        bad: response(bad, b"failure", "text/plain", status=500),
    }
    repos, store = repositories(tmp_path)
    definition = make_collection_definition(entry, "base")
    outcome = CollectionImportService(fetcher=fetcher(responses)).run(
        definition, repos, store
    )
    record = repos.collection_imports.get(outcome.import_id)

    assert outcome.unique_resources == 2
    assert outcome.ingested == 1
    assert outcome.failed == 1
    assert record.total_resources == 2
    assert len(repos.collection_acquisition_links.list()) == 2
    assert len(repos.collection_resource_import_results.list()) == 2
    assert repos.literature_representation_selections.list() == ()
    assert repos.curations.list() == ()
    assert repos.source_claims.list() == ()
    assert repos.method_facts.list() == ()
    assert repos.model_facts.list() == ()
    assert repos.reported_results.list() == ()


def test_cli_exposes_bounded_collection_import_options() -> None:
    result = CliRunner().invoke(app, ["import-literature-collection", "--help"])
    plain_output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)

    assert result.exit_code == 0
    assert "--max-pages" in plain_output
    assert "--max-resources" in plain_output
    assert "--max-depth" in plain_output
    assert "--allowed-origin" in plain_output
    assert "--allowed-path-prefix" in plain_output
    assert "--allow-external-lit" in plain_output
    assert "--connector" in plain_output


class FakeAcquisitionService:
    def __init__(self, outcomes: dict[str, str | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def add(
        self,
        source: str,
        domain: str,
        repositories: KnowledgeRepositories,
        _evidence_store: SourceEvidenceStore,
    ) -> LiteratureAcquisitionOutcome:
        self.calls.append(source)
        configured = self.outcomes[source]
        if isinstance(configured, Exception):
            raise configured
        source_hash = content_hash({"source": source, "domain": domain})
        identity = {
            "request_id": f"request-{source_hash[:20]}",
            "request_hash": content_hash({"request": source}),
            "resolver_id": "fake-collection-test-resolver",
            "resolver_version": "1.0.0",
            "resolved_resource_id": f"resolved-{source_hash[:20]}",
            "resolved_resource_hash": content_hash({"resolved": source}),
            "attempt_refs": (),
            "attempt_hashes": (),
            "resulting_literature_id": configured,
            "resulting_ingestion_id": f"ingestion-{source_hash[:20]}",
            "status": AcquisitionStatus.INGESTED,
            "warnings": (),
        }
        acquisition_id = f"literature-acquisition-{content_hash(identity)[:24]}"
        payload = {"acquisition_id": acquisition_id, **identity}
        record = LiteratureAcquisitionRecord(
            **payload,
            content_hash=content_hash(payload),
        )
        repositories.literature_acquisitions.put(record.acquisition_id, record)
        return LiteratureAcquisitionOutcome(
            input_kind=(
                AcquisitionInputKind.DOI
                if source.startswith("10.")
                else AcquisitionInputKind.URL
            ),
            canonical_identifier=source,
            acquisition_status=AcquisitionStatus.INGESTED,
            fulltext_status=AcquisitionStatus.FULLTEXT_FOUND,
            literature_id=configured,
            ingestion_id=identity["resulting_ingestion_id"],
            acquisition_id=record.acquisition_id,
        )


def test_path_scope_uses_segment_boundaries() -> None:
    assert path_matches_prefix("/project", "/project")
    assert path_matches_prefix("/project/", "/project")
    assert path_matches_prefix("/project/page2", "/project")
    assert not path_matches_prefix("/project-old", "/project")
    assert not path_matches_prefix("/project-admin", "/project")


def test_pagination_child_path_is_accepted_but_prefix_collision_is_rejected(
    tmp_path: Path,
) -> None:
    entry = "https://collection.example/project"
    child = "https://collection.example/project/page2"
    collision = "https://collection.example/project-old/page2"
    transport = FakeHTTPTransport(
        {
            entry: response(
                entry,
                (
                    f"<a rel='next' href='{child}'>Next</a>"
                    f"<a rel='next' href='{collision}'>Next</a>"
                ).encode(),
            ),
            child: response(child, b"<p>terminal</p>"),
        }
    )
    safe = SafeHTTPFetcher(transport, dns_resolver=lambda _host: (PUBLIC_IP,))
    repos, _ = repositories(tmp_path)
    result = GenericHTMLCollectionConnector().discover(
        make_collection_definition(entry, "base"), repos, safe
    )

    assert transport.calls == [entry, child]
    assert "unresolved_pagination_outside_scope" in result.snapshot.completeness_reasons


def test_default_filename_entry_scope_uses_conservative_parent() -> None:
    policy = make_collection_scope_policy(
        "https://collection.example/project/index.html"
    )

    assert policy.allowed_path_prefixes == ("/project/",)
    assert path_matches_prefix("/project/page2", policy.allowed_path_prefixes[0])
    assert not path_matches_prefix("/project-old", policy.allowed_path_prefixes[0])


def test_scope_and_completeness_contract_are_snapshot_bound(tmp_path: Path) -> None:
    url = "https://collection.example/project/index.html"
    narrow = make_collection_scope_policy(url, allowed_path_prefixes=("/project/a",))
    broad = make_collection_scope_policy(url, allowed_path_prefixes=("/project",))
    narrow_definition = make_collection_definition(url, "base", scope_policy=narrow)
    broad_definition = make_collection_definition(url, "base", scope_policy=broad)
    first, _ = discover(tmp_path, url, {url: response(url, b"<p>empty</p>")})

    assert narrow_definition.collection_id != broad_definition.collection_id
    assert first.snapshot.scope_policy_hash == first.definition.scope_policy.content_hash
    assert (
        first.snapshot.connector_completeness_contract
        == first.definition.connector_completeness_contract
    )
    assert first.snapshot.completeness_basis == ("no_explicit_completeness_evidence",)


def test_cli_custom_collection_scope_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeOutcome:
        @staticmethod
        def model_dump_json(*, indent: int) -> str:
            assert indent == 2
            return "{}"

    def fake_run(_self, definition, _repositories, _evidence_store):
        captured["definition"] = definition
        return FakeOutcome()

    monkeypatch.setattr(CollectionImportService, "run", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "import-literature-collection",
            "https://collection.example/project/index.html",
            "--domain",
            "base",
            "--allowed-origin",
            "https://collection.example",
            "--allowed-path-prefix",
            "/project/publications",
            "--allow-external-literature-links",
            "--max-depth",
            "3",
        ],
    )

    assert result.exit_code == 0
    definition = captured["definition"]
    assert definition.scope_policy.allowed_origins == ("https://collection.example",)
    assert definition.scope_policy.allowed_path_prefixes == (
        "/project/publications",
    )
    assert definition.scope_policy.allow_external_literature_links is True
    assert definition.scope_policy.max_depth == 3


def test_post_resolution_deduplicates_three_resources_to_one_literature(
    tmp_path: Path,
) -> None:
    entry = "https://collection.example/project"
    article = "https://collection.example/project/article/paper-one"
    pdf = "https://collection.example/project/paper-one.pdf"
    doi = "10.1234/paper-one"
    page = (
        f"<a href='https://doi.org/{doi}'>DOI</a>"
        f"<a href='{article}'>Article</a>"
        f"<a href='{pdf}'>PDF</a>"
    ).encode()
    fake = FakeAcquisitionService(
        {doi: "literature-shared", article: "literature-shared", pdf: "literature-shared"}
    )
    repos, store = repositories(tmp_path)
    outcome = CollectionImportService(
        fetcher=fetcher({entry: response(entry, page)}),
        acquisition_service=fake,
    ).run(make_collection_definition(entry, "base"), repos, store)
    membership = repos.collection_literature_memberships.list()[0]

    assert outcome.discovery_occurrence_count == 3
    assert outcome.discovered_resource_count == 3
    assert outcome.logical_literature_count == 1
    assert outcome.ingested_literature_count == 1
    assert len(membership.discovered_resource_refs) == 3
    assert len(membership.acquisition_refs) == 3
    assert len(membership.occurrence_refs) == 3


def test_expected_acquisition_failure_does_not_abort_later_resource(
    tmp_path: Path,
) -> None:
    entry = "https://collection.example/project"
    first = "https://collection.example/project/a.pdf"
    second = "https://collection.example/project/b.pdf"
    page = f"<a href='{first}'>A</a><a href='{second}'>B</a>".encode()
    fake = FakeAcquisitionService(
        {first: AcquisitionError("expected failure"), second: "literature-b"}
    )
    repos, store = repositories(tmp_path)
    outcome = CollectionImportService(
        fetcher=fetcher({entry: response(entry, page)}),
        acquisition_service=fake,
    ).run(make_collection_definition(entry, "base"), repos, store)

    assert set(fake.calls) == {first, second}
    assert outcome.failed == 1
    assert outcome.ingested == 1
    assert len(repos.collection_resource_import_results.list()) == 2
    assert len(repos.collection_literature_memberships.list()) == 1


def test_unexpected_programming_error_is_not_swallowed(tmp_path: Path) -> None:
    entry = "https://collection.example/project"
    pdf = "https://collection.example/project/a.pdf"
    fake = FakeAcquisitionService({pdf: RuntimeError("programming defect")})
    repos, store = repositories(tmp_path)
    service = CollectionImportService(
        fetcher=fetcher({entry: response(entry, f"<a href='{pdf}'>PDF</a>".encode())}),
        acquisition_service=fake,
    )

    with pytest.raises(RuntimeError, match="programming defect"):
        service.run(make_collection_definition(entry, "base"), repos, store)


def test_collection_page_exact_bytes_are_hash_verifiable(tmp_path: Path) -> None:
    entry = "https://collection.example/project"
    body = b"<html><body>exact collection bytes</body></html>"
    result, repos = discover(tmp_path, entry, {entry: response(entry, body)})
    page = result.pages[0]
    artifact = repos.collection_page_artifacts.verify_page_record(page)

    assert repos.collection_page_artifacts.read_bytes(artifact.artifact_id) == body
    content_path = (
        tmp_path
        / "knowledge"
        / "collection_page_artifacts"
        / artifact.artifact_id
        / "content.bin"
    )
    content_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    content_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash changed|byte size changed"):
        repos.collection_page_artifacts.verify_page_record(page)


def test_same_resource_id_with_changed_hash_is_reported_changed(
    tmp_path: Path,
) -> None:
    entry = "https://collection.example/project"
    repos, _ = repositories(tmp_path)
    definition = make_collection_definition(entry, "base")
    first = GenericHTMLCollectionConnector().discover(
        definition,
        repos,
        fetcher({entry: response(entry, b"<p>10.1234/shared</p>")}),
    )
    second = GenericHTMLCollectionConnector().discover(
        definition,
        repos,
        fetcher(
            {
                entry: response(
                    entry,
                    b"<meta name='citation_doi' content='10.1234/shared'>",
                )
            }
        ),
    )
    diff = diff_collection_snapshots(first.snapshot, second.snapshot)

    assert first.resources[0].discovered_resource_id == second.resources[0].discovered_resource_id
    assert first.resources[0].content_hash != second.resources[0].content_hash
    assert diff.changed_resource_ids == (first.resources[0].discovered_resource_id,)
    assert diff.unchanged_resource_ids == ()


def test_same_resource_with_changed_discovery_provenance_is_reported_changed(
    tmp_path: Path,
) -> None:
    entry = "https://collection.example/project"
    repos, _ = repositories(tmp_path)
    definition = make_collection_definition(entry, "base")
    first = GenericHTMLCollectionConnector().discover(
        definition,
        repos,
        fetcher(
            {
                entry: response(
                    entry,
                    b"<meta name='citation_doi' content='10.1234/shared'>",
                )
            }
        ),
    )
    second = GenericHTMLCollectionConnector().discover(
        definition,
        repos,
        fetcher(
            {
                entry: response(
                    entry,
                    b"<a href='https://doi.org/10.1234/shared'>DOI</a>",
                )
            }
        ),
    )
    diff = diff_collection_snapshots(first.snapshot, second.snapshot)

    assert first.resources[0].content_hash == second.resources[0].content_hash
    assert diff.changed_resource_ids == (first.resources[0].discovered_resource_id,)
    assert diff.unchanged_resource_ids == ()


def test_identical_snapshot_still_creates_distinct_import_events(tmp_path: Path) -> None:
    entry = "https://collection.example/project"
    body = b"<p>10.1234/reference-only</p>"
    times = iter(
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    )
    repos, store = repositories(tmp_path)
    service = CollectionImportService(
        fetcher=fetcher({entry: response(entry, body)}),
        clock=lambda: next(times),
    )
    definition = make_collection_definition(entry, "base")
    first = service.run(definition, repos, store)
    second = service.run(definition, repos, store)

    assert first.snapshot_id == second.snapshot_id
    assert first.import_id != second.import_id
    assert len(repos.collection_snapshots.list()) == 1
    assert len(repos.collection_imports.list()) == 2


def test_arbitrary_body_doi_is_audited_but_not_imported_as_membership(
    tmp_path: Path,
) -> None:
    entry = "https://collection.example/project"
    fake = FakeAcquisitionService({})
    repos, store = repositories(tmp_path)
    outcome = CollectionImportService(
        fetcher=fetcher(
            {entry: response(entry, b"References include 10.1234/not-a-member")}
        ),
        acquisition_service=fake,
    ).run(make_collection_definition(entry, "base"), repos, store)
    resource = repos.collection_resources.list()[0]

    assert resource.highest_discovery_confidence == CollectionDiscoveryConfidence.LOW
    assert resource.membership_eligible is False
    assert fake.calls == []
    assert outcome.unverified_membership == 1
    assert outcome.logical_literature_count == 0
    assert repos.collection_literature_memberships.list() == ()


def test_citation_metadata_doi_is_high_confidence_membership_candidate(
    tmp_path: Path,
) -> None:
    entry = "https://collection.example/project"
    result, _ = discover(
        tmp_path,
        entry,
        {
            entry: response(
                entry,
                b"<meta name='citation_doi' content='10.1234/member'>",
            )
        },
    )

    assert result.resources[0].highest_discovery_confidence == CollectionDiscoveryConfidence.HIGH
    assert result.resources[0].membership_eligible is True
