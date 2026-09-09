from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from spc.cli import app
from spc.knowledge.acquisition import HTTPResponse, SafeHTTPFetcher
from spc.knowledge.collection import (
    CollectionImportService,
    GenericHTMLCollectionConnector,
    diff_collection_snapshots,
    make_collection_definition,
    make_collection_scope_policy,
)
from spc.models import CollectionImportStatus, CollectionResourceKind
from spc.repositories import KnowledgeRepositories, SourceEvidenceStore


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

    assert result.snapshot.completeness == CollectionImportStatus.COMPLETE
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
    assert len(diff.unchanged_resource_ids) == 1


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
    assert repos.literature_representation_selections.list() == ()
    assert repos.curations.list() == ()
    assert repos.source_claims.list() == ()
    assert repos.method_facts.list() == ()
    assert repos.model_facts.list() == ()
    assert repos.reported_results.list() == ()


def test_cli_exposes_bounded_collection_import_options() -> None:
    result = CliRunner().invoke(app, ["import-literature-collection", "--help"])

    assert result.exit_code == 0
    assert "--max-pages" in result.output
    assert "--max-resources" in result.output
    assert "--connector" in result.output
