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


def _fake_codex_command(
    tmp_path: Path,
    *,
    response: LiteratureKnowledgeLLMResponse | None = None,
    login_status: str = "Logged in using ChatGPT",
    login_exit: int = 0,
    item_type: str = "agent_message",
    sleep_seconds: float = 0,
) -> tuple[str, str]:
    executable = tmp_path / "fake_codex.py"
    calls_path = tmp_path / "codex-calls.jsonl"
    capture_path = tmp_path / "codex-capture.json"
    response_json = (response or LiteratureKnowledgeLLMResponse()).model_dump_json()
    executable.write_text(
        f"""from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

CALLS_PATH = Path({str(calls_path)!r})
CAPTURE_PATH = Path({str(capture_path)!r})
RESPONSE = {response_json!r}
LOGIN_STATUS = {login_status!r}
LOGIN_EXIT = {login_exit!r}
ITEM_TYPE = {item_type!r}
SLEEP_SECONDS = {sleep_seconds!r}
arguments = sys.argv[1:]
with CALLS_PATH.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{
        "arguments": arguments,
        "environment_keys": sorted(os.environ),
    }}) + "\\n")
if arguments == ["--version"]:
    print("codex-cli 99.1.2-test")
    raise SystemExit(0)
if arguments == ["login", "status"]:
    print(LOGIN_STATUS)
    raise SystemExit(LOGIN_EXIT)
if arguments[-2:] == ["features", "list"]:
    overrides = {{
        arguments[index + 1]
        for index, value in enumerate(arguments[:-1])
        if value == "-c"
    }}
    for feature in ("shell_tool", "shell_snapshot", "standalone_web_search"):
        state = "false" if f"features.{{feature}}=false" in overrides else "true"
        print(f"{{feature}} stable {{state}}")
    raise SystemExit(0)
if arguments and arguments[0] == "exec" and "--help" in arguments:
    print("Usage: codex exec [OPTIONS] [PROMPT]")
    raise SystemExit(0)
if not arguments or arguments[0] != "exec":
    raise SystemExit(64)
time.sleep(SLEEP_SECONDS)
prompt = sys.stdin.read()
schema_path = Path(arguments[arguments.index("--output-schema") + 1])
output_path = Path(arguments[arguments.index("--output-last-message") + 1])
CAPTURE_PATH.write_text(json.dumps({{
    "arguments": arguments,
    "prompt": prompt,
    "schema": json.loads(schema_path.read_text(encoding="utf-8")),
    "api_key_visible": "OPENAI_API_KEY" in os.environ,
    "codex_token_visible": "CODEX_ACCESS_TOKEN" in os.environ,
    "environment_keys": sorted(os.environ),
}}), encoding="utf-8")
output_path.write_text(RESPONSE, encoding="utf-8")
print(json.dumps({{"type": "item.completed", "item": {{"type": ITEM_TYPE}}}}))
""",
        encoding="utf-8",
    )
    return sys.executable, str(executable)


def _source_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OPENAI_API_KEY": "must-never-reach-codex",
            "CODEX_ACCESS_TOKEN": "must-never-reach-codex",
            "AWS_SECRET_ACCESS_KEY": "must-never-reach-codex",
            "DATABASE_PASSWORD": "must-never-reach-codex",
            "UNRELATED_TOKEN": "must-never-reach-codex",
        }
    )
    return environment


def _recorded_calls(tmp_path: Path) -> tuple[dict, ...]:
    return tuple(json.loads(line) for line in (tmp_path / "codex-calls.jsonl").read_text(encoding="utf-8").splitlines())


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
        _fake_codex_command(tmp_path, response=response),
        model="test-codex-model",
        environment=_source_environment(),
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
    assert invocation.transport_version == "1.1.0"
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
    assert arguments[0] == "exec"
    assert ["-c", "features.shell_tool=false"] == arguments[1:3]
    assert ["-c", "features.shell_snapshot=false"] == arguments[3:5]
    assert ["-c", "features.standalone_web_search=false"] == arguments[5:7]
    assert arguments[7] == "--strict-config"
    assert ["-c", 'approval_policy="never"'] == arguments[8:10]
    assert ["--sandbox", "read-only"] == arguments[10:12]
    assert "--ephemeral" in arguments
    assert "--ignore-user-config" in arguments
    assert "--ignore-rules" in arguments
    assert "--search" not in arguments
    assert capture["api_key_visible"] is False
    assert capture["codex_token_visible"] is False
    assert not {
        "AWS_SECRET_ACCESS_KEY",
        "DATABASE_PASSWORD",
        "UNRELATED_TOKEN",
    } & set(capture["environment_keys"])
    assert "LiteratureKnowledgeLLMResponse" in json.dumps(capture["schema"])
    prompt = json.loads(capture["prompt"].split("\n", 1)[1])
    assert prompt["scientific_input"]["compilation_input"]["literature_id"] == (outcome.literature_id)
    assert len(prompt["scientific_input"]["chunks"]) == len(chunks)
    assert str(repositories.root) not in capture["prompt"]


def test_codex_cli_transport_rejects_non_chatgpt_authentication(tmp_path: Path) -> None:
    transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path, login_status="Logged in using an API key"),
        model="test-codex-model",
        environment=_source_environment(),
    )
    with pytest.raises(CodexCLIUnavailableError, match="codex login"):
        transport.inspect_runtime()


def test_codex_cli_preflight_accepts_tool_disable_config_without_model_inference(
    tmp_path: Path,
) -> None:
    transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path),
        model="test-codex-model",
        environment=_source_environment(),
    )
    runtime = transport.inspect_runtime()
    calls = _recorded_calls(tmp_path)

    assert runtime.authentication_status == "chatgpt"
    assert runtime.selected_model == "test-codex-model"
    assert runtime.disabled_features == (
        "shell_tool",
        "shell_snapshot",
        "standalone_web_search",
    )
    assert any(call["arguments"][-2:] == ["features", "list"] for call in calls)
    assert any(call["arguments"][0] == "exec" and "--help" in call["arguments"] for call in calls)
    assert not any(call["arguments"][0] == "exec" and "--help" not in call["arguments"] for call in calls)
    for call in calls:
        assert not {
            "OPENAI_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "DATABASE_PASSWORD",
            "UNRELATED_TOKEN",
        } & set(call["environment_keys"])


def test_codex_cli_transport_reports_missing_executable(tmp_path: Path) -> None:
    transport = CodexCLILLMTransport(
        tmp_path / "missing-codex",
        model="test-codex-model",
    )
    with pytest.raises(CodexCLIUnavailableError, match="not installed"):
        transport.inspect_runtime()


def test_codex_cli_transport_rejects_tool_activity(tmp_path: Path) -> None:
    transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path, item_type="command_execution"),
        model="test-codex-model",
        environment=_source_environment(),
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
        _fake_codex_command(tmp_path, response=response),
        model="test-codex-model",
        max_output_bytes=512,
        environment=_source_environment(),
    )
    with pytest.raises(CodexCLIExecutionError, match="structured output exceeded"):
        transport.generate_structured(
            system_prompt="Return JSON only.",
            input_payload={"chunks": []},
            response_schema=LiteratureKnowledgeLLMResponse.model_json_schema(),
            temperature=0.0,
        )


def test_codex_cli_transport_enforces_timeout(tmp_path: Path) -> None:
    transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path, sleep_seconds=2),
        model="test-codex-model",
        timeout_seconds=1,
        environment=_source_environment(),
    )
    with pytest.raises(CodexCLIExecutionError, match="timed out after 1 seconds"):
        transport.generate_structured(
            system_prompt="Return JSON only.",
            input_payload={"chunks": []},
            response_schema=LiteratureKnowledgeLLMResponse.model_json_schema(),
            temperature=0.0,
        )


def test_cli_codex_provider_requires_explicit_model() -> None:
    result = CliRunner().invoke(
        app,
        [
            "extract-literature-knowledge",
            "--literature-id",
            "literature-unused",
            "--provider",
            "codex",
        ],
    )
    assert result.exit_code != 0
    assert "requires an explicit --codex-model" in result.output


def test_cli_codex_preflight_diagnostic_never_runs_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path),
        model="test-codex-model",
        environment=_source_environment(),
    )
    monkeypatch.setattr(
        "spc.cli.CodexCLILLMTransport",
        lambda *_args, **_kwargs: fake_transport,
    )
    result = CliRunner().invoke(
        app,
        ["check-codex-provider", "--codex-model", "test-codex-model"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model_inference_performed"] is False
    assert payload["required_flags_accepted"] is True
    assert payload["disabled_features"] == [
        "shell_tool",
        "shell_snapshot",
        "standalone_web_search",
    ]
    assert not any(
        call["arguments"][0] == "exec" and "--help" not in call["arguments"] for call in _recorded_calls(tmp_path)
    )


def test_cli_codex_provider_uses_fake_cli_without_real_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repositories, _store, outcome = _setup_literature(tmp_path)
    fake_transport = CodexCLILLMTransport(
        _fake_codex_command(tmp_path),
        model="test-codex-model",
        environment=_source_environment(),
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
            "--codex-model",
            "test-codex-model",
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
