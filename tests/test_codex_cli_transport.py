from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest
from typer.testing import CliRunner

from spc.cli import app
from spc.knowledge.acquisition import HTTPResponse, LiteratureAcquisitionService, SafeHTTPFetcher
from spc.knowledge.ingestion import LiteratureRepresentationSelector
from spc.knowledge.literature_knowledge import (
    CodexCLIExecutionError,
    CodexCLILLMTransport,
    CodexCLIUnavailableError,
    LiteratureClaimProposal,
    LiteratureKnowledgeCompiler,
    LiteratureKnowledgeLLMResponse,
    LiteratureQuoteProposal,
    StructuredLLMLiteratureKnowledgeProvider,
    build_literature_knowledge_chunks,
    resolve_literature_knowledge_input,
)
from spc.knowledge.structure import DocumentStructureService
from spc.models import CurationStatus, EpistemicStatus, KnowledgeCurationRecord
from spc.repositories import KnowledgeEvidenceStore, KnowledgeRepositories
from spc.serialization import content_hash


ARTICLE_URL = "https://journal.example/codex-transport-paper"
HTML = b"""<html><head>
<meta name="citation_title" content="Codex Transport Evidence Paper">
<meta name="citation_author" content="A. Author">
<meta name="citation_publication_date" content="2026">
<meta name="citation_doi" content="10.0000/codex.transport">
</head><body><article><h1>Results</h1><p>Sentence A.</p></article></body></html>"""


class StaticHTMLTransport:
    def request(self, url: str, *, validated_ips, timeout: float, max_bytes: int) -> HTTPResponse:
        assert validated_ips and timeout > 0 and max_bytes > len(HTML)
        return HTTPResponse(
            url=url,
            status=200,
            headers={"content-type": "text/html"},
            body=HTML,
            connected_ip="93.184.216.34",
        )


def _curate(repositories: KnowledgeRepositories, target_type: str, target_id: str, target_hash: str) -> None:
    identity = {
        "target_type": target_type,
        "target_id": target_id,
        "target_hash": target_hash,
        "status": CurationStatus.ACCEPTED,
        "curator_id": "codex-transport-test-curator",
        "rationale": "Accept deterministic test authority.",
        "evidence_refs": (),
    }
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    repositories.curations.put(
        curation_id,
        KnowledgeCurationRecord(**payload, content_hash=content_hash(payload)),
    )


def _setup_literature(tmp_path: Path):
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    store = KnowledgeEvidenceStore(repositories.root)
    fetcher = SafeHTTPFetcher(
        StaticHTMLTransport(),
        dns_resolver=lambda _host: ("93.184.216.34",),
    )
    outcome = LiteratureAcquisitionService(fetcher).add(
        ARTICLE_URL,
        "base",
        repositories,
        store,
    )
    LiteratureRepresentationSelector().select(
        outcome.literature_id or "",
        outcome.representation_id or "",
        "codex-transport-test",
        "Select deterministic test representation.",
        repositories,
        store,
    )
    DocumentStructureService().extract(
        outcome.literature_id or "",
        outcome.representation_id or "",
        repositories,
        store,
    )
    document = repositories.literature_documents.get(outcome.literature_id or "")
    selection = repositories.literature_representation_selections.resolve_current(document.literature_id)
    _curate(repositories, "literature_document", document.literature_id, document.content_hash)
    _curate(
        repositories,
        "literature_representation_selection",
        selection.selection_id,
        selection.content_hash,
    )
    return repositories, store, outcome


def _fake_codex_command(tmp_path: Path) -> tuple[str, str]:
    executable = tmp_path / "fake_codex.py"
    executable.write_text(
        """from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

arguments = sys.argv[1:]
if arguments == ["--version"]:
    print("codex-cli 99.1.2-test")
    raise SystemExit(0)
if arguments == ["login", "status"]:
    print(os.environ.get("FAKE_CODEX_LOGIN_STATUS", "Logged in using ChatGPT"))
    raise SystemExit(int(os.environ.get("FAKE_CODEX_LOGIN_EXIT", "0")))
if not arguments or arguments[0] != "exec":
    raise SystemExit(64)
time.sleep(float(os.environ.get("FAKE_CODEX_SLEEP_SECONDS", "0")))
prompt = sys.stdin.read()
schema_path = Path(arguments[arguments.index("--output-schema") + 1])
output_path = Path(arguments[arguments.index("--output-last-message") + 1])
capture_path = Path(os.environ["FAKE_CODEX_CAPTURE"])
capture_path.write_text(json.dumps({
    "arguments": arguments,
    "prompt": prompt,
    "schema": json.loads(schema_path.read_text(encoding="utf-8")),
    "api_key_visible": "OPENAI_API_KEY" in os.environ,
    "codex_token_visible": "CODEX_ACCESS_TOKEN" in os.environ,
}), encoding="utf-8")
output_path.write_text(os.environ["FAKE_CODEX_RESPONSE"], encoding="utf-8")
item_type = os.environ.get("FAKE_CODEX_ITEM_TYPE", "agent_message")
print(json.dumps({"type": "item.completed", "item": {"type": item_type}}))
""",
        encoding="utf-8",
    )
    return sys.executable, str(executable)


def _codex_environment(tmp_path: Path, response: LiteratureKnowledgeLLMResponse) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "FAKE_CODEX_CAPTURE": str(tmp_path / "codex-capture.json"),
            "FAKE_CODEX_RESPONSE": response.model_dump_json(),
            "OPENAI_API_KEY": "must-never-reach-codex",
            "CODEX_ACCESS_TOKEN": "must-never-reach-codex",
        }
    )
    return environment


def test_codex_cli_transport_isolated_proposal_is_audited_and_materialized(
    tmp_path: Path,
) -> None:
    repositories, store, outcome = _setup_literature(tmp_path)
    compilation_input = resolve_literature_knowledge_input(
        outcome.literature_id or "",
        repositories,
        store,
    )
    chunks = build_literature_knowledge_chunks(compilation_input, repositories, store)
    chunk = next(item for item in chunks if "Sentence A." in item.text)
    response = LiteratureKnowledgeLLMResponse(
        quote_proposals=(
            LiteratureQuoteProposal(
                quote_key="codex-quote-1",
                chunk_id=chunk.chunk_id,
                block_id=chunk.block_refs[0],
                exact_text="Sentence A.",
            ),
        ),
        claim_proposals=(
            LiteratureClaimProposal(
                claim_key="codex-claim-1",
                text="Sentence A.",
                claim_type="source_statement",
                quote_keys=("codex-quote-1",),
                claim_strength="source-reported",
                epistemic_status=EpistemicStatus.SOURCE_REPORTED,
            ),
        ),
    )
    transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path),
        model="test-codex-model",
        environment=_codex_environment(tmp_path, response),
    )
    compiled = LiteratureKnowledgeCompiler().compile(
        outcome.literature_id or "",
        StructuredLLMLiteratureKnowledgeProvider(transport, max_attempts=1),
        repositories,
        store,
    )

    invocation = compiled.proposal_set.provider_invocation
    assert invocation is not None
    assert invocation.runtime_version == "codex-cli 99.1.2-test"
    assert invocation.selected_model == "test-codex-model"
    assert len(invocation.invocation_config_hash) == 64
    assert len(invocation.input_hash) == 64
    assert len(invocation.output_hash) == 64
    assert (
        repositories.literature_knowledge_proposals.get(compiled.proposal_set.proposal_set_id).provider_invocation
        == invocation
    )
    assert len(compiled.materialized.source_quotes) == 1
    assert len(compiled.materialized.source_claims) == 1
    assert len(compiled.materialized.groundings) == 1
    grounding = compiled.materialized.groundings[0]
    evidence = store.get_evidence(grounding.evidence_id)
    store.verify_evidence_integrity(evidence)
    locator = repositories.structured_evidence_locators.get(grounding.locator_id)
    assert locator.evidence_id == evidence.evidence_id
    claim = compiled.materialized.source_claims[0]
    assert any(
        item.target_id == claim.claim_id and item.status == CurationStatus.MACHINE_EXTRACTED
        for item in repositories.curations.list()
    )

    capture = json.loads((tmp_path / "codex-capture.json").read_text(encoding="utf-8"))
    arguments = capture["arguments"]
    assert arguments[:5] == [
        "exec",
        "--sandbox",
        "read-only",
        "--ask-for-approval",
        "never",
    ]
    assert "--ephemeral" in arguments
    assert "--ignore-user-config" in arguments
    assert "--ignore-rules" in arguments
    assert "--search" not in arguments
    assert capture["api_key_visible"] is False
    assert capture["codex_token_visible"] is False
    assert "LiteratureKnowledgeLLMResponse" in json.dumps(capture["schema"])
    prompt = json.loads(capture["prompt"].split("\n", 1)[1])
    assert prompt["scientific_input"]["compilation_input"]["literature_id"] == (outcome.literature_id)
    assert len(prompt["scientific_input"]["chunks"]) == len(chunks)
    assert str(repositories.root) not in capture["prompt"]


def test_codex_cli_transport_rejects_non_chatgpt_authentication(tmp_path: Path) -> None:
    environment = _codex_environment(tmp_path, LiteratureKnowledgeLLMResponse())
    environment["FAKE_CODEX_LOGIN_STATUS"] = "Logged in using an API key"
    transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path),
        environment=environment,
    )
    with pytest.raises(CodexCLIUnavailableError, match="codex login"):
        transport.inspect_runtime()


def test_codex_cli_transport_reports_missing_executable(tmp_path: Path) -> None:
    transport = CodexCLILLMTransport(tmp_path / "missing-codex")
    with pytest.raises(CodexCLIUnavailableError, match="not installed"):
        transport.inspect_runtime()


def test_codex_cli_transport_rejects_tool_activity(tmp_path: Path) -> None:
    environment = _codex_environment(tmp_path, LiteratureKnowledgeLLMResponse())
    environment["FAKE_CODEX_ITEM_TYPE"] = "command_execution"
    transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path),
        environment=environment,
    )
    with pytest.raises(CodexCLIExecutionError, match="disallowed extraction action"):
        transport.generate_structured(
            system_prompt="Return JSON only.",
            input_payload={"chunks": []},
            response_schema=LiteratureKnowledgeLLMResponse.model_json_schema(),
            temperature=0.0,
        )


def test_codex_cli_transport_rejects_oversized_output(tmp_path: Path) -> None:
    response = LiteratureKnowledgeLLMResponse(
        claim_proposals=tuple(
            LiteratureClaimProposal(
                claim_key=f"claim-{index}",
                text="oversized output text",
                claim_type="source_statement",
                quote_keys=("missing-quote",),
                claim_strength="source-reported",
                epistemic_status=EpistemicStatus.SOURCE_REPORTED,
            )
            for index in range(20)
        )
    )
    transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path),
        max_output_bytes=512,
        environment=_codex_environment(tmp_path, response),
    )
    with pytest.raises(CodexCLIExecutionError, match="structured output exceeded"):
        transport.generate_structured(
            system_prompt="Return JSON only.",
            input_payload={"chunks": []},
            response_schema=LiteratureKnowledgeLLMResponse.model_json_schema(),
            temperature=0.0,
        )


def test_codex_cli_transport_enforces_timeout(tmp_path: Path) -> None:
    environment = _codex_environment(tmp_path, LiteratureKnowledgeLLMResponse())
    environment["FAKE_CODEX_SLEEP_SECONDS"] = "2"
    transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path),
        timeout_seconds=1,
        environment=environment,
    )
    with pytest.raises(CodexCLIExecutionError, match="timed out after 1 seconds"):
        transport.generate_structured(
            system_prompt="Return JSON only.",
            input_payload={"chunks": []},
            response_schema=LiteratureKnowledgeLLMResponse.model_json_schema(),
            temperature=0.0,
        )


def test_cli_codex_provider_uses_fake_cli_without_real_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repositories, _store, outcome = _setup_literature(tmp_path)
    fake_transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path),
        environment=_codex_environment(tmp_path, LiteratureKnowledgeLLMResponse()),
    )
    monkeypatch.setattr(
        "spc.cli.CodexCLILLMTransport",
        lambda *_args, **_kwargs: fake_transport,
    )
    result = CliRunner().invoke(
        app,
        [
            "extract-literature-knowledge",
            "--literature-id",
            outcome.literature_id or "",
            "--knowledge-dir",
            str(repositories.root),
            "--provider",
            "codex",
            "--max-attempts",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["provider_id"] == "structured-llm-literature-knowledge"
    assert payload["provider_invocation"]["transport_id"] == "codex-cli"
    assert "untrusted" in payload["provider_notice"]
    assert "must-never-reach-codex" not in result.output
