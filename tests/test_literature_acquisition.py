from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spc.cli import app
from spc.knowledge.acquisition import (
    ArticleURLResolver,
    DOIResolver,
    HTMLLiteratureTextExtractor,
    HTTPResponse,
    LiteratureAcquisitionService,
    SafeHTTPError,
    SafeHTTPFetcher,
    detect_acquisition_input,
    normalize_doi,
)
from spc.models import AcquisitionInputKind, AcquisitionStatus
from spc.repositories import KnowledgeRepositories, SourceEvidenceStore


FIXTURES = Path(__file__).parent / "fixtures"
BORN_DIGITAL_PDF = FIXTURES / "generic-born-digital.pdf"
DOI = "10.1234/example"
CROSSREF_URL = "https://api.crossref.org/works/10.1234%2Fexample"
PDF_URL = "https://files.example/paper.pdf"
ARTICLE_URL = "https://journal.example/article"


def public_ips(_hostname: str) -> tuple[str, ...]:
    return ("93.184.216.34",)


class FakeHTTPTransport:
    def __init__(self, responses: dict[str, HTTPResponse]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def request(self, url: str, *, timeout: float, max_bytes: int) -> HTTPResponse:
        assert timeout > 0
        assert max_bytes > 0
        self.calls.append(url)
        try:
            return self.responses[url]
        except KeyError as error:
            raise AssertionError(f"unexpected offline HTTP request: {url}") from error


def response(
    url: str,
    body: bytes,
    media_type: str,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> HTTPResponse:
    return HTTPResponse(
        url=url,
        status=status,
        headers={"content-type": media_type, **(headers or {})},
        body=body,
    )


def crossref_body(*, include_pdf: bool = True) -> bytes:
    message: dict[str, object] = {
        "title": ["Deterministic Catalyst Study"],
        "author": [{"given": "Ada", "family": "Researcher"}],
        "published-print": {"date-parts": [[2024]]},
        "container-title": ["Journal of Offline Tests"],
        "URL": f"https://doi.org/{DOI}",
    }
    if include_pdf:
        message["link"] = [{"URL": PDF_URL, "content-type": "application/pdf"}]
    return json.dumps({"message": message}).encode()


def fetcher(responses: dict[str, HTTPResponse], *, max_bytes: int = 20_000_000) -> SafeHTTPFetcher:
    return SafeHTTPFetcher(
        FakeHTTPTransport(responses),
        dns_resolver=public_ips,
        max_bytes=max_bytes,
    )


def repositories(tmp_path: Path) -> tuple[KnowledgeRepositories, SourceEvidenceStore]:
    return KnowledgeRepositories(tmp_path / "knowledge"), SourceEvidenceStore(tmp_path / ".spc")


def request_for(value: str, tmp_path: Path):
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(fetcher({}))
    kind = detect_acquisition_input(value)
    return service._request(value, kind, "base"), repos, store


@pytest.mark.parametrize(
    "value",
    (
        DOI,
        f"doi:{DOI.upper()}",
        f"https://doi.org/{DOI.upper()}",
        f"http://dx.doi.org/{DOI}",
    ),
)
def test_doi_normalization_and_detection(value: str) -> None:
    assert normalize_doi(value) == DOI
    assert detect_acquisition_input(value) == AcquisitionInputKind.DOI


def test_doi_metadata_resolution_is_offline_and_deterministic(tmp_path: Path) -> None:
    request, _, _ = request_for(DOI, tmp_path)
    resolver = DOIResolver(
        fetcher({CROSSREF_URL: response(CROSSREF_URL, crossref_body(), "application/json")})
    )

    first = resolver.resolve(request)
    second = resolver.resolve(request)

    assert first == second
    assert first.doi == DOI
    assert first.title == "Deterministic Catalyst Study"
    assert first.authors == ("Ada Researcher",)
    assert first.fulltext_candidates[0].url == PDF_URL


def test_doi_public_pdf_is_ingested_without_auto_trust(tmp_path: Path) -> None:
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(CROSSREF_URL, crossref_body(), "application/json"),
                PDF_URL: response(PDF_URL, BORN_DIGITAL_PDF.read_bytes(), "application/pdf"),
            }
        )
    )

    outcome = service.add(DOI, "base", repos, store)

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert outcome.literature_id is not None
    assert outcome.ingestion_id is not None
    assert outcome.canonical_text_id is not None
    assert repos.literature_representation_selections.list() == ()
    assert repos.curations.list() == ()


def test_doi_metadata_only_does_not_fabricate_full_text(tmp_path: Path) -> None:
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL, crossref_body(include_pdf=False), "application/json"
                )
            }
        )
    )

    outcome = service.add(DOI, "base", repos, store)
    record = repos.literature_acquisitions.get(outcome.acquisition_id)

    assert outcome.acquisition_status == AcquisitionStatus.FULLTEXT_UNAVAILABLE
    assert outcome.literature_id is None
    assert record.selected_candidate_id is None
    assert record.resulting_ingestion_id is None
    assert repos.raw_literature_artifacts.list() == ()
    assert repos.canonical_text_artifacts.list() == ()


def test_direct_pdf_url_uses_k1b_pipeline(tmp_path: Path) -> None:
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher({PDF_URL: response(PDF_URL, BORN_DIGITAL_PDF.read_bytes(), "application/pdf")})
    )

    outcome = service.add(PDF_URL, "base", repos, store)

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert len(repos.raw_literature_artifacts.list()) == 1
    assert len(repos.literature_ingestions.list()) == 1


def test_html_article_has_deterministic_canonical_text(tmp_path: Path) -> None:
    html = b"""<!doctype html><html><head>
    <meta name="citation_title" content="HTML Catalyst Article">
    <meta name="citation_author" content="Grace Scientist">
    <meta name="citation_publication_date" content="2025-04-03">
    </head><body><article><h1>Result</h1><p>CO binds at the bridge site.</p></article></body></html>"""
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher({ARTICLE_URL: response(ARTICLE_URL, html, "text/html; charset=utf-8")})
    )

    outcome = service.add(ARTICLE_URL, "base", repos, store)
    first = HTMLLiteratureTextExtractor().extract(ARTICLE_URL, html)
    second = HTMLLiteratureTextExtractor().extract(ARTICLE_URL, html)

    assert outcome.acquisition_status == AcquisitionStatus.FULLTEXT_FOUND
    assert outcome.ingestion_id is None
    assert first == second
    assert first.canonical_text == "Result\n\nCO binds at the bridge site."
    assert len(first.blocks) == 2
    assert first.artifact_sha256 in outcome.warnings[0]
    assert first.text_sha256 in outcome.warnings[0]


def test_article_landing_page_discovers_and_ingests_pdf(tmp_path: Path) -> None:
    html = f"""<html><head>
    <meta name="citation_title" content="Landing Page Study">
    <meta name="citation_author" content="Lin Author">
    <meta name="citation_publication_date" content="2023">
    <meta name="citation_pdf_url" content="{PDF_URL}">
    </head><body>Landing page</body></html>""".encode()
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                ARTICLE_URL: response(ARTICLE_URL, html, "text/html"),
                PDF_URL: response(PDF_URL, BORN_DIGITAL_PDF.read_bytes(), "application/pdf"),
            }
        )
    )

    outcome = service.add(ARTICLE_URL, "base", repos, store)

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    resource = repos.resolved_literature_resources.list()[0]
    assert resource.fulltext_candidates[0].url == PDF_URL


def test_local_pdf_and_same_bytes_are_idempotent_across_filenames(tmp_path: Path) -> None:
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "renamed.pdf"
    first_path.write_bytes(BORN_DIGITAL_PDF.read_bytes())
    second_path.write_bytes(BORN_DIGITAL_PDF.read_bytes())
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(fetcher({}))

    first = service.add(str(first_path), "base", repos, store)
    second = service.add(str(second_path), "base", repos, store)

    assert first.literature_id == second.literature_id
    assert first.ingestion_id == second.ingestion_id
    assert len(repos.raw_literature_artifacts.list()) == 1


def test_identical_doi_is_idempotent(tmp_path: Path) -> None:
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(CROSSREF_URL, crossref_body(), "application/json"),
                PDF_URL: response(PDF_URL, BORN_DIGITAL_PDF.read_bytes(), "application/pdf"),
            }
        )
    )

    first = service.add(DOI, "base", repos, store)
    second = service.add(f"https://doi.org/{DOI.upper()}", "base", repos, store)

    assert first.literature_id == second.literature_id
    assert first.ingestion_id == second.ingestion_id
    assert first.acquisition_id != second.acquisition_id


def test_acquisition_provenance_chain_is_complete(tmp_path: Path) -> None:
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(CROSSREF_URL, crossref_body(), "application/json"),
                PDF_URL: response(PDF_URL, BORN_DIGITAL_PDF.read_bytes(), "application/pdf"),
            }
        )
    )

    outcome = service.add(DOI, "base", repos, store)
    record = repos.literature_acquisitions.get(outcome.acquisition_id)
    request = repos.acquisition_requests.get(record.request_id)
    resource = repos.resolved_literature_resources.get(record.resolved_resource_id)
    ingestion = repos.literature_ingestions.get(record.resulting_ingestion_id or "")
    selected = next(
        item for item in resource.fulltext_candidates if item.candidate_id == record.selected_candidate_id
    )

    assert record.request_hash == request.content_hash
    assert record.resolved_resource_hash == resource.content_hash
    assert record.selected_candidate_hash == selected.content_hash
    assert record.resulting_literature_id == ingestion.literature_id
    assert record.resulting_canonical_text_id == ingestion.canonical_text_id
    source = store.source_records.get(
        f"{ingestion.source_id}--{ingestion.source_version}"
    )
    assert store.verify_source_integrity(source) == source


def test_safe_fetcher_rejects_local_and_private_addresses() -> None:
    safe = fetcher({})

    with pytest.raises(SafeHTTPError, match="LOCALHOST_REJECTED"):
        safe.fetch("http://localhost/paper.pdf", allowed_content_types=frozenset({"application/pdf"}))
    with pytest.raises(SafeHTTPError, match="PRIVATE_ADDRESS_REJECTED"):
        safe.fetch("http://10.0.0.1/paper.pdf", allowed_content_types=frozenset({"application/pdf"}))


def test_safe_fetcher_revalidates_redirect_target() -> None:
    start = "https://public.example/paper"
    safe = fetcher(
        {
            start: response(
                start,
                b"",
                "text/plain",
                status=302,
                headers={"location": "http://127.0.0.1/private.pdf"},
            )
        }
    )

    with pytest.raises(SafeHTTPError, match="PRIVATE_ADDRESS_REJECTED"):
        safe.fetch(start, allowed_content_types=frozenset({"application/pdf"}))


def test_safe_fetcher_rejects_oversized_and_wrong_content_type() -> None:
    oversized_url = "https://public.example/large.pdf"
    wrong_url = "https://public.example/not-pdf"
    safe = fetcher(
        {
            oversized_url: response(oversized_url, b"12345", "application/pdf"),
            wrong_url: response(wrong_url, b"bad", "text/plain"),
        },
        max_bytes=4,
    )

    with pytest.raises(SafeHTTPError, match="RESOURCE_TOO_LARGE"):
        safe.fetch(oversized_url, allowed_content_types=frozenset({"application/pdf"}))
    with pytest.raises(SafeHTTPError, match="UNSUPPORTED_CONTENT_TYPE"):
        safe.fetch(wrong_url, allowed_content_types=frozenset({"application/pdf"}))


def test_article_resolver_classifies_authentication_and_unsupported_media(tmp_path: Path) -> None:
    request, repos, store = request_for(ARTICLE_URL, tmp_path)
    auth_fetcher = fetcher(
        {ARTICLE_URL: response(ARTICLE_URL, b"", "text/html", status=403)}
    )
    with pytest.raises(SafeHTTPError, match="REQUIRES_AUTHENTICATION"):
        ArticleURLResolver(auth_fetcher).resolve(request)

    unsupported_service = LiteratureAcquisitionService(
        fetcher({ARTICLE_URL: response(ARTICLE_URL, b"data", "image/png")})
    )
    outcome = unsupported_service.add(ARTICLE_URL, "base", repos, store)
    assert outcome.acquisition_status == AcquisitionStatus.UNSUPPORTED_MEDIA


def test_add_literature_cli_reports_local_pdf_without_selecting_it(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "add-literature",
            str(BORN_DIGITAL_PDF),
            "--domain",
            "base",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--state-dir",
            str(tmp_path / ".spc"),
        ],
    )

    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["input_kind"] == "local_file"
    assert output["acquisition_status"] == "ingested"
    assert output["literature_id"]
    assert KnowledgeRepositories(tmp_path / "knowledge").literature_representation_selections.list() == ()


def test_local_input_kind_requires_existing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(ValueError, match="input is not"):
        detect_acquisition_input(str(missing))


def test_candidate_content_hash_binds_downloaded_pdf() -> None:
    request_body = BORN_DIGITAL_PDF.read_bytes()
    request = LiteratureAcquisitionService(fetcher({}))._request(PDF_URL, AcquisitionInputKind.URL, "base")
    resolver = ArticleURLResolver(fetcher({PDF_URL: response(PDF_URL, request_body, "application/pdf")}))

    resource = resolver.resolve(request)

    assert resource.fulltext_candidates[0].content_sha256 == hashlib.sha256(request_body).hexdigest()
