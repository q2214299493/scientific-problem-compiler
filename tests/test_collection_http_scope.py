"""Collection request boundaries, exercised with a recording in-memory transport."""

from pathlib import Path

import pytest

from spc.knowledge.acquisition import HTTPResponse
from spc.knowledge.collection import (
    CollectionImportService,
    GenericHTMLCollectionConnector,
    make_collection_definition,
    make_collection_scope_policy,
)
from spc.models import CollectionTraversalStatus

from test_collection_import import PUBLIC_IP, fetcher, repositories, response


ENTRY = "https://collection.example/project"


def redirect(url: str, target: str) -> HTTPResponse:
    return HTTPResponse(
        url=url,
        status=302,
        headers={"location": target},
        body=b"",
        connected_ip=PUBLIC_IP,
    )


@pytest.mark.parametrize(
    "target",
    [
        "https://other.example/project/end",
        "https://collection.example/project-other/end",
    ],
)
def test_every_page_redirect_is_checked_before_transport(tmp_path: Path, target: str):
    middle = f"{ENTRY}/middle"
    safe = fetcher(
        {ENTRY: redirect(ENTRY, middle), middle: redirect(middle, target), target: response(target, b"<p>Outside</p>")}
    )
    policy = make_collection_scope_policy(ENTRY, allow_external_literature_links=True)
    result = GenericHTMLCollectionConnector().discover(
        make_collection_definition(ENTRY, "base", scope_policy=policy),
        repositories(tmp_path)[0],
        safe,
    )
    assert safe.transport.calls == [ENTRY, middle]
    assert result.snapshot.traversal_status == CollectionTraversalStatus.FAILED
    assert result.resources == ()
    # This same fetcher still works for a later, independent acquisition.
    assert safe.fetch(target, allowed_content_types=frozenset({"text/html"})).status == 200


@pytest.mark.parametrize(
    ("html", "target"),
    [
        (
            b"<meta name='citation_pdf_url' content='https://other.example/project/paper.pdf'>",
            "https://other.example/project/paper.pdf",
        ),
        (
            b"<meta name='citation_pdf_url' content='/outside/paper.pdf'>",
            "https://collection.example/outside/paper.pdf",
        ),
        (b"<meta name='citation_doi' content='10.1234/scope'>", "https://api.crossref.org/works/10.1234%2Fscope"),
    ],
)
def test_metadata_and_doi_http_requests_respect_scope(tmp_path: Path, html: bytes, target: str):
    safe = fetcher({ENTRY: response(ENTRY, html), target: response(target, b"", status=404)})
    repos, evidence = repositories(tmp_path)
    outcome = CollectionImportService(fetcher=safe).run(
        make_collection_definition(ENTRY, "base"),
        repos.collection,
        evidence,
    )
    assert safe.transport.calls == [ENTRY]
    assert outcome.failed == 1
    assert not repos.collection_literature_memberships.list()


def test_acquisition_followup_resources_keep_collection_scope(tmp_path: Path):
    article = f"{ENTRY}/article/work"
    target = "https://other.example/project/work.pdf"
    safe = fetcher(
        {
            ENTRY: response(ENTRY, f"<div class='publication'><a href='{article}'>Work</a></div>".encode()),
            article: response(article, f"<meta name='citation_pdf_url' content='{target}'>".encode()),
            target: response(target, b"", status=404),
        }
    )
    repos, evidence = repositories(tmp_path)
    CollectionImportService(fetcher=safe).run(
        make_collection_definition(ENTRY, "base"),
        repos.collection,
        evidence,
    )
    assert article in safe.transport.calls
    assert target not in safe.transport.calls


def test_allowed_page_redirect_and_explicit_external_resource_policy(tmp_path: Path):
    page = f"{ENTRY}/list"
    pdf = "https://publisher.example/project/work.pdf"
    safe = fetcher(
        {
            ENTRY: redirect(ENTRY, page),
            page: response(page, f"<div class='publication'><a href='{pdf}'>Paper</a></div>".encode()),
            pdf: response(pdf, b"", status=404),
        }
    )
    repos, evidence = repositories(tmp_path)
    policy = make_collection_scope_policy(ENTRY, allow_external_literature_links=True)
    CollectionImportService(fetcher=safe).run(
        make_collection_definition(ENTRY, "base", scope_policy=policy),
        repos.collection,
        evidence,
    )
    assert safe.transport.calls == [ENTRY, page, pdf]


def test_resource_redirect_keeps_path_limit_with_external_links_enabled(tmp_path: Path):
    pdf = f"{ENTRY}/paper.pdf"
    outside = "https://publisher.example/outside/paper.pdf"
    safe = fetcher(
        {
            ENTRY: response(ENTRY, f"<meta name='citation_pdf_url' content='{pdf}'>".encode()),
            pdf: redirect(pdf, outside),
            outside: response(outside, b"", status=404),
        }
    )
    repos, evidence = repositories(tmp_path)
    policy = make_collection_scope_policy(ENTRY, allow_external_literature_links=True)
    outcome = CollectionImportService(fetcher=safe).run(
        make_collection_definition(ENTRY, "base", scope_policy=policy),
        repos.collection,
        evidence,
    )
    assert safe.transport.calls == [ENTRY, pdf]
    assert outcome.failed == 1
    assert any("COLLECTION_URL_OUTSIDE_SCOPE" in w for w in repos.literature_acquisitions.list()[0].warnings)


def test_explicit_scope_allows_doi_metadata_request(tmp_path: Path):
    metadata = "https://api.crossref.org/works/10.1234%2Fscope"
    safe = fetcher(
        {
            ENTRY: response(ENTRY, b"<meta name='citation_doi' content='10.1234/scope'>"),
            metadata: response(metadata, b"", status=404),
        }
    )
    repos, evidence = repositories(tmp_path)
    policy = make_collection_scope_policy(
        ENTRY,
        allowed_origins=(ENTRY, "https://api.crossref.org"),
        allowed_path_prefixes=("/project", "/works"),
    )
    CollectionImportService(fetcher=safe).run(
        make_collection_definition(ENTRY, "base", scope_policy=policy),
        repos.collection,
        evidence,
    )
    assert safe.transport.calls == [ENTRY, metadata]
