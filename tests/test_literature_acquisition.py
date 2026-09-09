from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import pytest
from pypdf import PdfReader, PdfWriter
from typer.testing import CliRunner

import spc.knowledge.acquisition as acquisition_module
from spc.cli import app
from spc.domains import DomainPackLoader
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
    validate_literature_representation,
)
from spc.knowledge.ingestion import (
    LiteratureRepresentationSelector,
    create_evidence_span_from_canonical_text,
)
from spc.models import (
    AcquisitionInputKind,
    AcquisitionStatus,
    CurationStatus,
    HistoricalLiteratureEvidenceAuthorization,
    KnowledgeCurationRecord,
    MetadataValueOrigin,
)
from spc.repositories import (
    KnowledgeEvidenceStore,
    KnowledgeRepositories,
    SourceEvidenceStore,
)
from spc.serialization import content_hash


FIXTURES = Path(__file__).parent / "fixtures"
BORN_DIGITAL_PDF = FIXTURES / "generic-born-digital.pdf"
DOI = "10.1234/example"
CROSSREF_URL = "https://api.crossref.org/works/10.1234%2Fexample"
PDF_URL = "https://files.example/paper.pdf"
ARTICLE_URL = "https://journal.example/article"


def public_ips(_hostname: str) -> tuple[str, ...]:
    return ("93.184.216.34",)


class FakeHTTPTransport:
    def __init__(
        self, responses: dict[str, HTTPResponse | list[HTTPResponse]]
    ) -> None:
        self.responses = responses
        self.calls: list[str] = []
        self.validated_ip_sets: list[tuple[str, ...]] = []

    def request(
        self,
        url: str,
        *,
        validated_ips: tuple[str, ...],
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse:
        assert validated_ips
        assert timeout > 0
        assert max_bytes > 0
        self.calls.append(url)
        self.validated_ip_sets.append(validated_ips)
        try:
            configured = self.responses[url]
        except KeyError as error:
            raise AssertionError(f"unexpected offline HTTP request: {url}") from error
        if isinstance(configured, list):
            if not configured:
                raise AssertionError(f"no offline responses remain for: {url}")
            return configured.pop(0)
        return configured


def response(
    url: str,
    body: bytes,
    media_type: str,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    connected_ip: str = "93.184.216.34",
) -> HTTPResponse:
    return HTTPResponse(
        url=url,
        status=status,
        headers={"content-type": media_type, **(headers or {})},
        body=body,
        connected_ip=connected_ip,
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


def fetcher(
    responses: dict[str, HTTPResponse | list[HTTPResponse]],
    *,
    max_bytes: int = 20_000_000,
    max_redirects: int = 5,
) -> SafeHTTPFetcher:
    return SafeHTTPFetcher(
        FakeHTTPTransport(responses),
        dns_resolver=public_ips,
        max_bytes=max_bytes,
        max_redirects=max_redirects,
    )


def repositories(tmp_path: Path) -> tuple[KnowledgeRepositories, SourceEvidenceStore]:
    return KnowledgeRepositories(tmp_path / "knowledge"), SourceEvidenceStore(tmp_path / ".spc")


def accept_record(
    repos: KnowledgeRepositories,
    target_type: str,
    target_id: str,
    target_hash: str,
    *,
    evidence_refs: tuple[str, ...] = (),
    supersedes: str | None = None,
) -> KnowledgeCurationRecord:
    identity = {
        "target_type": target_type,
        "target_id": target_id,
        "target_hash": target_hash,
        "status": CurationStatus.ACCEPTED,
        "curator_id": "k1c2-test-curator",
        "rationale": "Accept the explicitly selected representation.",
        "evidence_refs": evidence_refs,
        "supersedes_curation_id": supersedes,
    }
    identity = {key: value for key, value in identity.items() if value is not None}
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    record = KnowledgeCurationRecord(
        **payload, content_hash=content_hash(payload)
    )
    repos.curations.put(record.curation_id, record)
    return record


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
                ),
                f"https://doi.org/{DOI}": response(
                    f"https://doi.org/{DOI}", b"missing", "text/plain", status=404
                ),
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

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert outcome.literature_id is not None
    assert outcome.ingestion_id is not None
    assert outcome.canonical_text_id is not None
    assert outcome.representation_id is not None
    assert first == second
    assert first.canonical_text == "Result\n\nCO binds at the bridge site."
    assert len(first.blocks) == 2
    raw = repos.raw_html_literature_artifacts.list()[0]
    canonical = repos.canonical_html_text_artifacts.list()[0]
    assert raw.sha256 == first.artifact_sha256
    assert canonical.text_sha256 == first.text_sha256
    assert repos.canonical_html_text_artifacts.read_text(
        canonical.canonical_text_id
    ) == first.canonical_text


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
    assert KnowledgeEvidenceStore(tmp_path / "knowledge").list_sources()
    assert not (tmp_path / ".spc" / "sources").exists()


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


def crossref_body_with_candidates(
    candidates: tuple[tuple[str, str], ...],
) -> bytes:
    payload = json.loads(crossref_body())
    payload["message"]["link"] = [
        {"URL": url, "content-type": media_type}
        for url, media_type in candidates
    ]
    return json.dumps(payload).encode()


def article_html(*, pdf_url: str | None = None) -> bytes:
    pdf_meta = (
        f'<meta name="citation_pdf_url" content="{pdf_url}">'
        if pdf_url is not None
        else ""
    )
    return f"""<html><head>
    <meta name="citation_title" content="Fallback Article">
    <meta name="citation_author" content="Fallback Author">
    <meta name="citation_publication_date" content="2024">
    {pdf_meta}</head><body><article><h1>Finding</h1>
    <p>Deterministic HTML full text.</p></article></body></html>""".encode()


def test_transport_connection_must_match_prevalidated_ip() -> None:
    url = "https://public.example/paper.pdf"
    safe = fetcher(
        {
            url: response(
                url,
                BORN_DIGITAL_PDF.read_bytes(),
                "application/pdf",
                connected_ip="8.8.8.8",
            )
        }
    )

    with pytest.raises(SafeHTTPError, match="CONNECTION_IP_MISMATCH"):
        safe.fetch(url, allowed_content_types=frozenset({"application/pdf"}))


def test_https_connection_uses_pinned_ip_and_original_hostname_for_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection_targets: list[tuple[str, int]] = []
    server_names: list[str] = []
    raw_socket = object()
    wrapped_socket = object()

    def create_connection(address, timeout, source_address):
        del timeout, source_address
        connection_targets.append(address)
        return raw_socket

    class TLSContext:
        def wrap_socket(self, sock, *, server_hostname):
            assert sock is raw_socket
            server_names.append(server_hostname)
            return wrapped_socket

    monkeypatch.setattr(acquisition_module.socket, "create_connection", create_connection)
    connection = acquisition_module._PinnedHTTPSConnection(
        "journal.example", 443, "93.184.216.34", 5.0, TLSContext()
    )

    connection.connect()

    assert connection_targets == [("93.184.216.34", 443)]
    assert server_names == ["journal.example"]
    assert connection.sock is wrapped_socket


def test_redirect_dns_is_revalidated_with_a_new_pinned_set() -> None:
    start = "https://first.example/article"
    target = "https://second.example/paper.pdf"
    transport = FakeHTTPTransport(
        {
            start: response(
                start,
                b"",
                "text/plain",
                status=302,
                headers={"location": target},
                connected_ip="93.184.216.34",
            ),
            target: response(
                target,
                BORN_DIGITAL_PDF.read_bytes(),
                "application/pdf",
                connected_ip="8.8.8.8",
            ),
        }
    )
    dns_calls: list[str] = []

    def resolve(hostname: str) -> tuple[str, ...]:
        dns_calls.append(hostname)
        return {
            "first.example": ("93.184.216.34",),
            "second.example": ("8.8.8.8",),
        }[hostname]

    safe = SafeHTTPFetcher(transport, dns_resolver=resolve)
    result = safe.fetch(
        start, allowed_content_types=frozenset({"application/pdf"})
    )

    assert result.url == target
    assert dns_calls == ["first.example", "second.example"]
    assert transport.validated_ip_sets == [
        ("93.184.216.34",),
        ("8.8.8.8",),
    ]


@pytest.mark.parametrize("address", ("::1", "fc00::1", "fe80::1"))
def test_ipv6_non_global_addresses_are_rejected(address: str) -> None:
    safe = fetcher({})

    with pytest.raises(SafeHTTPError, match="PRIVATE_ADDRESS_REJECTED"):
        safe.fetch(
            f"http://[{address}]/paper.pdf",
            allowed_content_types=frozenset({"application/pdf"}),
        )


def test_url_credentials_and_unsafe_schemes_are_rejected() -> None:
    safe = fetcher({})

    with pytest.raises(SafeHTTPError, match="URL_CREDENTIALS_REJECTED"):
        safe.fetch(
            "https://user:password@public.example/paper.pdf",
            allowed_content_types=frozenset({"application/pdf"}),
        )
    with pytest.raises(SafeHTTPError, match="UNSAFE_URL_SCHEME"):
        safe.fetch(
            "file:///etc/passwd",
            allowed_content_types=frozenset({"application/pdf"}),
        )


def test_redirect_limit_is_enforced() -> None:
    first = "https://public.example/one"
    second = "https://public.example/two"
    safe = fetcher(
        {
            first: response(
                first,
                b"",
                "text/plain",
                status=302,
                headers={"location": second},
            )
        },
        max_redirects=0,
    )

    with pytest.raises(SafeHTTPError, match="TOO_MANY_REDIRECTS"):
        safe.fetch(first, allowed_content_types=frozenset({"application/pdf"}))


def test_first_pdf_403_then_second_pdf_succeeds(tmp_path: Path) -> None:
    first_pdf = "https://files.example/blocked.pdf"
    second_pdf = "https://files.example/public.pdf"
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL,
                    crossref_body_with_candidates(
                        (
                            (first_pdf, "application/pdf"),
                            (second_pdf, "application/pdf"),
                        )
                    ),
                    "application/json",
                ),
                first_pdf: response(first_pdf, b"", "application/pdf", status=403),
                second_pdf: response(
                    second_pdf, BORN_DIGITAL_PDF.read_bytes(), "application/pdf"
                ),
            }
        )
    )

    outcome = service.add(DOI, "base", repos, store)
    attempts = tuple(
        repos.acquisition_attempts.get(attempt_id)
        for attempt_id in outcome.attempt_ids
    )

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert [item.failure_code for item in attempts] == [
        "REQUIRES_AUTHENTICATION",
        None,
    ]
    assert [item.attempt_index for item in attempts] == [0, 1]


def test_invalid_pdf_candidate_falls_back_to_html(tmp_path: Path) -> None:
    invalid_pdf = "https://files.example/invalid.pdf"
    html_url = "https://files.example/article.html"
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL,
                    crossref_body_with_candidates(
                        (
                            (invalid_pdf, "application/pdf"),
                            (html_url, "text/html"),
                        )
                    ),
                    "application/json",
                ),
                invalid_pdf: response(invalid_pdf, b"not-pdf", "application/pdf"),
                html_url: response(html_url, article_html(), "text/html"),
            }
        )
    )

    outcome = service.add(DOI, "base", repos, store)
    attempts = tuple(
        repos.acquisition_attempts.get(attempt_id)
        for attempt_id in outcome.attempt_ids
    )

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert attempts[0].failure_code == "INVALID_PDF_SIGNATURE"
    assert attempts[1].failure_code is None
    assert len(repos.raw_html_literature_artifacts.list()) == 1


def test_unavailable_landing_pdf_falls_back_to_html_article(tmp_path: Path) -> None:
    landing_pdf = "https://journal.example/unavailable.pdf"
    html = article_html(pdf_url=landing_pdf)
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                ARTICLE_URL: response(ARTICLE_URL, html, "text/html"),
                landing_pdf: response(
                    landing_pdf, b"", "application/pdf", status=403
                ),
            }
        )
    )

    outcome = service.add(ARTICLE_URL, "base", repos, store)

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert len(outcome.attempt_ids) == 2
    assert outcome.representation_id is not None


def test_all_candidates_unavailable_and_attempt_history_is_deterministic(
    tmp_path: Path,
) -> None:
    first_pdf = "https://files.example/one.pdf"
    second_pdf = "https://files.example/two.pdf"

    def run(root: Path):
        repos, store = repositories(root)
        outcome = LiteratureAcquisitionService(
            fetcher(
                {
                    CROSSREF_URL: response(
                        CROSSREF_URL,
                        crossref_body_with_candidates(
                            (
                                (first_pdf, "application/pdf"),
                                (second_pdf, "application/pdf"),
                            )
                        ),
                        "application/json",
                    ),
                    first_pdf: response(
                        first_pdf, b"", "application/pdf", status=403
                    ),
                    second_pdf: response(
                        second_pdf, b"", "application/pdf", status=404
                    ),
                    f"https://doi.org/{DOI}": response(
                        f"https://doi.org/{DOI}",
                        b"missing",
                        "text/plain",
                        status=404,
                    ),
                }
            )
        ).add(DOI, "base", repos, store)
        return outcome, repos.acquisition_attempts.list()

    first, first_attempts = run(tmp_path / "first")
    second, second_attempts = run(tmp_path / "second")

    assert first.acquisition_status == AcquisitionStatus.FULLTEXT_UNAVAILABLE
    assert first.literature_id is None
    assert first_attempts == second_attempts
    assert first.attempt_ids == second.attempt_ids


def test_html_artifacts_source_and_blocks_are_persisted_exactly(tmp_path: Path) -> None:
    html = article_html()
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher({ARTICLE_URL: response(ARTICLE_URL, html, "text/html")})
    ).add(ARTICLE_URL, "base", repos, store)
    raw = repos.raw_html_literature_artifacts.list()[0]
    canonical = repos.canonical_html_text_artifacts.list()[0]
    canonical_text = repos.canonical_html_text_artifacts.read_text(
        canonical.canonical_text_id
    )
    representation = repos.literature_representation_refs.get(
        outcome.representation_id or ""
    )

    raw_path = repos.root.joinpath(*PurePosixPath(raw.stored_path).parts)
    assert raw_path.read_bytes() == html
    assert raw.sha256 == hashlib.sha256(html).hexdigest()
    for block in canonical.blocks:
        recovered = canonical_text[block.start_offset : block.end_offset]
        assert hashlib.sha256(recovered.encode()).hexdigest() == block.text_hash
    source = store.source_records.get(
        f"{representation.source_id}--{representation.source_version}"
    )
    assert store.verify_source_integrity(source) == source
    assert source.content_sha256 == canonical.text_sha256
    assert outcome.literature_id is not None
    assert outcome.ingestion_id is not None
    assert repos.curations.list() == ()
    assert repos.literature_representation_selections.list() == ()
    assert repos.source_claims.list() == ()
    assert repos.method_facts.list() == ()
    assert repos.model_facts.list() == ()
    assert repos.reported_results.list() == ()
    assert repos.relations.list() == ()
    assert repos.expert_opinions.list() == ()


@pytest.mark.parametrize("target", ("raw", "canonical"))
def test_html_artifact_tampering_fails_closed(tmp_path: Path, target: str) -> None:
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher({ARTICLE_URL: response(ARTICLE_URL, article_html(), "text/html")})
    ).add(ARTICLE_URL, "base", repos, store)
    representation = repos.literature_representation_refs.get(
        outcome.representation_id or ""
    )
    if target == "raw":
        artifact = repos.raw_html_literature_artifacts.get(
            representation.raw_artifact_id
        )
        path = repos.root.joinpath(*PurePosixPath(artifact.stored_path).parts)
    else:
        canonical = repos.canonical_html_text_artifacts.get(
            representation.canonical_text_id or ""
        )
        path = repos.root.joinpath(*PurePosixPath(canonical.stored_path).parts)
    os.chmod(path, 0o666)
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="changed|hash|UTF-8"):
        validate_literature_representation(representation, repos, store)


def test_repeated_html_acquisition_is_idempotent(tmp_path: Path) -> None:
    html = article_html()
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher({ARTICLE_URL: response(ARTICLE_URL, html, "text/html")})
    )

    first = service.add(ARTICLE_URL, "base", repos, store)
    second = service.add(ARTICLE_URL, "base", repos, store)

    assert first.literature_id == second.literature_id
    assert first.ingestion_id == second.ingestion_id
    assert first.representation_id == second.representation_id
    assert len(repos.raw_html_literature_artifacts.list()) == 1
    assert len(repos.canonical_html_text_artifacts.list()) == 1


def test_crossref_without_links_uses_landing_page_pdf(tmp_path: Path) -> None:
    landing = f"https://doi.org/{DOI}"
    html = article_html(pdf_url=PDF_URL)
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL,
                    crossref_body(include_pdf=False),
                    "application/json",
                ),
                landing: response(landing, html, "text/html"),
                PDF_URL: response(
                    PDF_URL, BORN_DIGITAL_PDF.read_bytes(), "application/pdf"
                ),
            }
        )
    )

    outcome = service.add(DOI, "base", repos, store)
    acquisition = repos.literature_acquisitions.get(outcome.acquisition_id)
    resource = repos.resolved_literature_resources.get(
        acquisition.resolved_resource_id
    )

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert resource.fulltext_candidates[0].url == PDF_URL
    assert len(resource.metadata_retrieval_refs) == 2
    assert {item.discovered_by for item in resource.fulltext_candidates} == {
        "article-url-resolver"
    }


def test_crossref_without_links_uses_fulltext_html_landing(tmp_path: Path) -> None:
    landing = f"https://doi.org/{DOI}"
    html = article_html()
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL,
                    crossref_body(include_pdf=False),
                    "application/json",
                ),
                landing: response(landing, html, "text/html"),
            }
        )
    ).add(DOI, "base", repos, store)

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert len(repos.raw_html_literature_artifacts.list()) == 1


def test_crossref_metadata_response_payload_is_auditable(tmp_path: Path) -> None:
    body = crossref_body()
    repos, store = repositories(tmp_path)
    LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(CROSSREF_URL, body, "application/json"),
                PDF_URL: response(
                    PDF_URL, BORN_DIGITAL_PDF.read_bytes(), "application/pdf"
                ),
            }
        )
    ).add(DOI, "base", repos, store)

    retrieval = next(
        item
        for item in repos.metadata_retrievals.list()
        if item.resolver_id == "crossref-doi-resolver"
    )
    assert retrieval.source_url == CROSSREF_URL
    assert retrieval.response_sha256 == hashlib.sha256(body).hexdigest()
    assert retrieval.response_payload_utf8 == body.decode()


def test_explicit_metadata_completes_pdf_with_missing_embedded_fields(
    tmp_path: Path,
) -> None:
    source_reader = PdfReader(BORN_DIGITAL_PDF)
    writer = PdfWriter()
    for page in source_reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": "", "/Author": ""})
    pdf_path = tmp_path / "missing-metadata.pdf"
    with pdf_path.open("wb") as stream:
        writer.write(stream)
    repos, store = repositories(tmp_path)

    outcome = LiteratureAcquisitionService(fetcher({})).add(
        str(pdf_path),
        "base",
        repos,
        store,
        explicit_metadata={
            "title": "Explicit Metadata Study",
            "authors": ["Explicit Author"],
            "year": 2026,
        },
    )

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    document = repos.literature_documents.get(outcome.literature_id or "")
    assert document.title == "Explicit Metadata Study"
    assert document.authors == ("Explicit Author",)
    assert document.year == 2026


def test_complete_embedded_pdf_metadata_is_used(tmp_path: Path) -> None:
    repos, store = repositories(tmp_path)

    outcome = LiteratureAcquisitionService(fetcher({})).add(
        str(BORN_DIGITAL_PDF), "base", repos, store
    )

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    document = repos.literature_documents.get(outcome.literature_id or "")
    assert document.title == "Generic Non-FT Literature Fixture"
    assert document.authors == ("anonymous",)
    assert document.year == 2000


def test_missing_pdf_metadata_is_not_fabricated_from_filename(
    tmp_path: Path,
) -> None:
    source_reader = PdfReader(BORN_DIGITAL_PDF)
    writer = PdfWriter()
    for page in source_reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": "", "/Author": ""})
    pdf_path = tmp_path / "misleading-title.pdf"
    with pdf_path.open("wb") as stream:
        writer.write(stream)
    repos, store = repositories(tmp_path)

    outcome = LiteratureAcquisitionService(fetcher({})).add(
        str(pdf_path), "base", repos, store
    )

    assert outcome.acquisition_status == AcquisitionStatus.METADATA_ONLY
    assert outcome.title is None
    assert repos.literature_documents.list() == ()


def test_add_literature_cli_accepts_metadata_override(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(
        "title: CLI Metadata Study\nauthors:\n  - CLI Curator\nyear: 2026\n",
        encoding="utf-8",
    )
    knowledge_dir = tmp_path / "knowledge"

    result = CliRunner().invoke(
        app,
        [
            "add-literature",
            str(BORN_DIGITAL_PDF),
            "--domain",
            "base",
            "--knowledge-dir",
            str(knowledge_dir),
            "--state-dir",
            str(tmp_path / ".spc"),
            "--metadata",
            str(metadata_path),
        ],
    )

    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    document = KnowledgeRepositories(knowledge_dir).literature_documents.get(
        output["literature_id"]
    )
    assert document.title == "CLI Metadata Study"
    assert document.authors == ("CLI Curator",)
    assert document.year == 2026


def test_explicit_metadata_conflict_is_recorded(tmp_path: Path) -> None:
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(fetcher({})).add(
        str(BORN_DIGITAL_PDF),
        "base",
        repos,
        store,
        explicit_metadata={"title": "Curator Title"},
    )
    acquisition = repos.literature_acquisitions.get(outcome.acquisition_id)
    manifest = repos.metadata_merge_manifests.get(
        acquisition.metadata_merge_manifest_id or ""
    )
    title = next(item for item in manifest.decisions if item.field_name == "title")

    assert title.selected_origin == MetadataValueOrigin.EXPLICIT
    assert title.selected_value == "Curator Title"
    assert {
        alternative.origin for alternative in title.alternatives
    } == {MetadataValueOrigin.RESOLVED, MetadataValueOrigin.EMBEDDED}


def same_literature_html(
    body: str,
    *,
    pdf_url: str | None = None,
    include_article: bool = True,
) -> bytes:
    pdf_meta = (
        f'<meta name="citation_pdf_url" content="{pdf_url}">'
        if pdf_url is not None
        else ""
    )
    article = f"<article><p>{body}</p></article>" if include_article else body
    return f"""<html><head>
    <meta name="citation_doi" content="{DOI}">
    <meta name="citation_title" content="Deterministic Catalyst Study">
    <meta name="citation_author" content="Ada Researcher">
    <meta name="citation_publication_date" content="2024">
    <meta name="citation_journal_title" content="Journal of Offline Tests">
    {pdf_meta}</head><body>{article}</body></html>""".encode()


def accept_literature_selection(
    repos: KnowledgeRepositories,
    literature_id: str,
    selection,
) -> None:
    document = repos.literature_documents.get(literature_id)
    accept_record(
        repos,
        "literature_document",
        document.literature_id,
        document.content_hash,
    )
    accept_record(
        repos,
        "literature_representation_selection",
        selection.selection_id,
        selection.content_hash,
    )


def test_html_representation_uses_generic_selector_and_trusted_snapshot(
    tmp_path: Path,
) -> None:
    html = same_literature_html("Selected HTML representation.")
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher({ARTICLE_URL: response(ARTICLE_URL, html, "text/html")})
    ).add(ARTICLE_URL, "base", repos, store)

    selected = LiteratureRepresentationSelector().select(
        outcome.literature_id or "",
        outcome.representation_id or "",
        "html-selector",
        "Select the reviewed HTML representation.",
        repos,
        store,
    )
    accept_literature_selection(
        repos, outcome.literature_id or "", selected.selection
    )
    canonical_text = repos.canonical_html_text_artifacts.read_text(
        outcome.canonical_text_id or ""
    )
    phrase = "Selected HTML representation."
    start = canonical_text.index(phrase)
    evidence = create_evidence_span_from_canonical_text(
        outcome.canonical_text_id or "",
        start,
        start + len(phrase),
        repos,
        store,
    )
    selection_curation = next(
        item
        for item in repos.curations.list()
        if item.target_id == selected.selection.selection_id
    )
    accept_record(
        repos,
        "literature_representation_selection",
        selected.selection.selection_id,
        selected.selection.content_hash,
        evidence_refs=(evidence.evidence_id,),
        supersedes=selection_curation.curation_id,
    )
    snapshot = repos.create_snapshot(
        store, DomainPackLoader().load("base").profile
    )

    assert selected.total_pages is None
    assert outcome.representation_id in snapshot.literature_representation_hashes
    assert evidence.evidence_id in snapshot.evidence_span_hashes
    assert outcome.ingestion_id in {
        key.split(":", 1)[1]
        for key in snapshot.trusted_record_hashes
        if key.startswith("html_literature_ingestion:")
    }


def test_selector_cli_accepts_common_representation_id(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    state_dir = tmp_path / ".spc"
    repos = KnowledgeRepositories(knowledge_dir)
    store = SourceEvidenceStore(state_dir)
    outcome = LiteratureAcquisitionService(fetcher({})).add(
        str(BORN_DIGITAL_PDF), "base", repos, store
    )

    result = CliRunner().invoke(
        app,
        [
            "select-literature-representation",
            "--literature-id",
            outcome.literature_id or "",
            "--representation-id",
            outcome.representation_id or "",
            "--selected-by",
            "cli-selector",
            "--rationale",
            "Select through the common representation contract.",
            "--knowledge-dir",
            str(knowledge_dir),
            "--state-dir",
            str(state_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    selected = json.loads(result.output)["selection"]
    assert selected["representation_id"] == outcome.representation_id
    assert selected["ingestion_id"] is None


@pytest.mark.parametrize("target", ("raw", "canonical"))
def test_selected_html_tamper_breaks_trusted_snapshot(
    tmp_path: Path, target: str
) -> None:
    html = same_literature_html("Trusted HTML representation.")
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher({ARTICLE_URL: response(ARTICLE_URL, html, "text/html")})
    ).add(ARTICLE_URL, "base", repos, store)
    selection = LiteratureRepresentationSelector().select(
        outcome.literature_id or "",
        outcome.representation_id or "",
        "html-selector",
        "Select trusted HTML.",
        repos,
        store,
    ).selection
    accept_literature_selection(repos, outcome.literature_id or "", selection)
    representation = repos.literature_representation_refs.get(
        outcome.representation_id or ""
    )
    record = (
        repos.raw_html_literature_artifacts.get(representation.raw_artifact_id)
        if target == "raw"
        else repos.canonical_html_text_artifacts.get(
            representation.canonical_text_id or ""
        )
    )
    path = repos.root.joinpath(*PurePosixPath(record.stored_path).parts)
    os.chmod(path, 0o666)
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="changed|hash|UTF-8"):
        repos.create_snapshot(store, DomainPackLoader().load("base").profile)


def test_unselected_html_tamper_does_not_affect_trusted_snapshot(
    tmp_path: Path,
) -> None:
    first_html = same_literature_html("Current HTML representation.")
    second_html = same_literature_html("Historical HTML representation.")
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                ARTICLE_URL: [
                    response(ARTICLE_URL, first_html, "text/html"),
                    response(ARTICLE_URL, first_html, "text/html"),
                    response(ARTICLE_URL, second_html, "text/html"),
                    response(ARTICLE_URL, second_html, "text/html"),
                ]
            }
        )
    )
    selected = service.add(ARTICLE_URL, "base", repos, store)
    unselected = service.add(ARTICLE_URL, "base", repos, store)
    selection = LiteratureRepresentationSelector().select(
        selected.literature_id or "",
        selected.representation_id or "",
        "html-selector",
        "Keep the first representation current.",
        repos,
        store,
    ).selection
    accept_literature_selection(repos, selected.literature_id or "", selection)
    unselected_representation = repos.literature_representation_refs.get(
        unselected.representation_id or ""
    )
    unselected_raw = repos.raw_html_literature_artifacts.get(
        unselected_representation.raw_artifact_id
    )
    path = repos.root.joinpath(*PurePosixPath(unselected_raw.stored_path).parts)
    os.chmod(path, 0o666)
    path.write_bytes(path.read_bytes() + b"tamper")

    snapshot = repos.create_snapshot(
        store, DomainPackLoader().load("base").profile
    )

    assert selected.representation_id in snapshot.literature_representation_hashes
    assert unselected.representation_id not in snapshot.literature_representation_hashes


def test_switching_pdf_and_html_current_representation_changes_snapshot(
    tmp_path: Path,
) -> None:
    html = same_literature_html("Equivalent HTML full text.")
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL, crossref_body(), "application/json"
                ),
                PDF_URL: response(
                    PDF_URL, BORN_DIGITAL_PDF.read_bytes(), "application/pdf"
                ),
                ARTICLE_URL: response(ARTICLE_URL, html, "text/html"),
            }
        )
    )
    pdf = service.add(DOI, "base", repos, store)
    html_outcome = service.add(ARTICLE_URL, "base", repos, store)
    assert pdf.literature_id == html_outcome.literature_id

    pdf_selection = LiteratureRepresentationSelector().select(
        pdf.literature_id or "",
        pdf.representation_id or "",
        "selector",
        "Select PDF.",
        repos,
        store,
    ).selection
    accept_literature_selection(repos, pdf.literature_id or "", pdf_selection)
    pdf_snapshot = repos.create_snapshot(
        store, DomainPackLoader().load("base").profile
    )

    html_selection = LiteratureRepresentationSelector().select(
        html_outcome.literature_id or "",
        html_outcome.representation_id or "",
        "selector",
        "Switch to HTML.",
        repos,
        store,
    ).selection
    accept_record(
        repos,
        "literature_representation_selection",
        html_selection.selection_id,
        html_selection.content_hash,
    )
    html_snapshot = repos.create_snapshot(
        store, DomainPackLoader().load("base").profile
    )

    pdf_again = LiteratureRepresentationSelector().select(
        pdf.literature_id or "",
        pdf.representation_id or "",
        "selector",
        "Switch back to PDF.",
        repos,
        store,
    ).selection
    accept_record(
        repos,
        "literature_representation_selection",
        pdf_again.selection_id,
        pdf_again.content_hash,
    )
    final_snapshot = repos.create_snapshot(
        store, DomainPackLoader().load("base").profile
    )

    assert len({pdf_snapshot.snapshot_id, html_snapshot.snapshot_id, final_snapshot.snapshot_id}) == 3
    assert set(pdf_snapshot.literature_representation_hashes) == {
        pdf.representation_id
    }
    assert set(html_snapshot.literature_representation_hashes) == {
        html_outcome.representation_id
    }
    assert set(final_snapshot.literature_representation_hashes) == {
        pdf.representation_id
    }


def test_historical_html_evidence_requires_explicit_authorization(
    tmp_path: Path,
) -> None:
    first_html = same_literature_html("Historical sentence for evidence.")
    second_html = same_literature_html("Current sentence for evidence.")
    repos, store = repositories(tmp_path)
    service = LiteratureAcquisitionService(
        fetcher(
            {
                ARTICLE_URL: [
                    response(ARTICLE_URL, first_html, "text/html"),
                    response(ARTICLE_URL, first_html, "text/html"),
                    response(ARTICLE_URL, second_html, "text/html"),
                    response(ARTICLE_URL, second_html, "text/html"),
                ]
            }
        )
    )
    historical = service.add(ARTICLE_URL, "base", repos, store)
    current = service.add(ARTICLE_URL, "base", repos, store)
    selection = LiteratureRepresentationSelector().select(
        current.literature_id or "",
        current.representation_id or "",
        "selector",
        "Select current HTML.",
        repos,
        store,
    ).selection
    accept_literature_selection(repos, current.literature_id or "", selection)
    canonical = repos.canonical_html_text_artifacts.read_text(
        historical.canonical_text_id or ""
    )
    phrase = "Historical sentence for evidence."
    start = canonical.index(phrase)
    evidence = create_evidence_span_from_canonical_text(
        historical.canonical_text_id or "",
        start,
        start + len(phrase),
        repos,
        store,
        historical_ingestion=True,
    )
    selection_curation = next(
        item
        for item in repos.curations.list()
        if item.target_id == selection.selection_id
    )
    accept_record(
        repos,
        "literature_representation_selection",
        selection.selection_id,
        selection.content_hash,
        evidence_refs=(evidence.evidence_id,),
        supersedes=selection_curation.curation_id,
    )
    with pytest.raises(ValueError, match="historical literature evidence"):
        repos.create_snapshot(store, DomainPackLoader().load("base").profile)

    historical_representation = repos.literature_representation_refs.get(
        historical.representation_id or ""
    )
    identity = {
        "literature_id": historical_representation.literature_id,
        "representation_id": historical_representation.representation_id,
        "representation_hash": historical_representation.content_hash,
        "source_id": historical_representation.source_id,
        "source_version": historical_representation.source_version,
        "evidence_refs": (evidence.evidence_id,),
        "authorized_by": "historical-curator",
        "rationale": "Authorize this exact historical HTML evidence.",
    }
    authorization_id = (
        f"historical-evidence-authorization-{content_hash(identity)[:24]}"
    )
    payload = {"authorization_id": authorization_id, **identity}
    authorization = HistoricalLiteratureEvidenceAuthorization(
        **payload, content_hash=content_hash(payload)
    )
    repos.historical_evidence_authorizations.put(
        authorization.authorization_id, authorization
    )

    snapshot = repos.create_snapshot(
        store, DomainPackLoader().load("base").profile
    )
    assert evidence.evidence_id in snapshot.evidence_span_hashes


def test_crossref_403_then_secondary_landing_html_succeeds(
    tmp_path: Path,
) -> None:
    landing = f"https://doi.org/{DOI}"
    html = same_literature_html("Landing HTML fallback.")
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL, crossref_body(), "application/json"
                ),
                PDF_URL: response(PDF_URL, b"", "application/pdf", status=403),
                landing: response(landing, html, "text/html"),
            }
        )
    ).add(DOI, "base", repos, store)
    acquisition = repos.literature_acquisitions.get(outcome.acquisition_id)
    resource = repos.resolved_literature_resources.get(
        acquisition.resolved_resource_id
    )

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert len(outcome.attempt_ids) == 2
    assert len(resource.metadata_retrieval_refs) == 2


def test_stale_crossref_pdf_then_landing_alternate_pdf_succeeds(
    tmp_path: Path,
) -> None:
    landing = f"https://doi.org/{DOI}"
    alternate = "https://files.example/alternate.pdf"
    landing_html = same_literature_html(
        "Landing discovery.", pdf_url=alternate, include_article=False
    )
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL, crossref_body(), "application/json"
                ),
                PDF_URL: response(PDF_URL, b"stale", "application/pdf"),
                landing: response(landing, landing_html, "text/html"),
                alternate: response(
                    alternate,
                    BORN_DIGITAL_PDF.read_bytes(),
                    "application/pdf",
                ),
            }
        )
    ).add(DOI, "base", repos, store)
    attempts = [repos.acquisition_attempts.get(item) for item in outcome.attempt_ids]

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert [item.failure_code for item in attempts] == [
        "INVALID_PDF_SIGNATURE",
        None,
    ]


def test_secondary_discovery_deduplicates_already_attempted_candidate(
    tmp_path: Path,
) -> None:
    landing = f"https://doi.org/{DOI}"
    landing_html = same_literature_html(
        "HTML fallback after duplicate PDF.", pdf_url=PDF_URL
    )
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL, crossref_body(), "application/json"
                ),
                PDF_URL: response(PDF_URL, b"", "application/pdf", status=403),
                landing: response(landing, landing_html, "text/html"),
            }
        )
    ).add(DOI, "base", repos, store)
    attempts = [repos.acquisition_attempts.get(item) for item in outcome.attempt_ids]

    assert outcome.acquisition_status == AcquisitionStatus.INGESTED
    assert len(attempts) == 2
    assert [item.attempt_index for item in attempts] == [0, 1]
    assert sum(item.candidate_id == attempts[0].candidate_id for item in attempts) == 1


def test_all_doi_candidates_requiring_auth_returns_auth_status(
    tmp_path: Path,
) -> None:
    landing = f"https://doi.org/{DOI}"
    second_pdf = "https://files.example/auth-two.pdf"
    landing_html = same_literature_html(
        "Landing page only.", pdf_url=second_pdf, include_article=False
    )
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL, crossref_body(), "application/json"
                ),
                PDF_URL: response(PDF_URL, b"", "application/pdf", status=403),
                landing: response(landing, landing_html, "text/html"),
                second_pdf: response(
                    second_pdf, b"", "application/pdf", status=403
                ),
            }
        )
    ).add(DOI, "base", repos, store)

    assert outcome.acquisition_status == AcquisitionStatus.REQUIRES_AUTHENTICATION
    assert len(outcome.attempt_ids) == 2


def test_doi_with_no_candidates_in_either_stage_is_fulltext_unavailable(
    tmp_path: Path,
) -> None:
    landing = f"https://doi.org/{DOI}"
    landing_html = same_literature_html("Abstract only.", include_article=False)
    repos, store = repositories(tmp_path)
    outcome = LiteratureAcquisitionService(
        fetcher(
            {
                CROSSREF_URL: response(
                    CROSSREF_URL,
                    crossref_body(include_pdf=False),
                    "application/json",
                ),
                landing: response(landing, landing_html, "text/html"),
            }
        )
    ).add(DOI, "base", repos, store)

    assert outcome.acquisition_status == AcquisitionStatus.FULLTEXT_UNAVAILABLE
    assert outcome.attempt_ids == ()
