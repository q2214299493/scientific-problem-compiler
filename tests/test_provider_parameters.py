"""Public provider parameter contracts at their original validation boundaries."""

import pytest
from typer.testing import CliRunner

from spc.approval import StructuredLLMApprovalProvider
from spc.cli import app
from spc.knowledge.expert_knowledge import (
    StructuredLLMExpertKnowledgeProvider,
    build_expert_knowledge_chunks,
    resolve_expert_knowledge_input,
)
from spc.knowledge.literature_knowledge import (
    CodexCLILLMTransport,
    StructuredLLMLiteratureKnowledgeProvider,
    partition_literature_knowledge_chunks,
)
from spc.planning import FakeLLMTransport, HTTPJSONLLMTransport, StructuredLLMPlanningProvider
from spc.workflow.providers import bind_provider

from test_codex_cli_transport import _captured_cli_text
from test_expert_knowledge import _accept_authority, _prepare_source


PLANNING_ERROR = "temperature must be a finite non-negative number"
KNOWLEDGE_ERROR = "temperature must be finite and non-negative"
ATTEMPTS_ERROR = "max_attempts must be between one and five"

PROVIDERS = (
    (StructuredLLMPlanningProvider, PLANNING_ERROR),
    (StructuredLLMApprovalProvider, PLANNING_ERROR),
    (StructuredLLMLiteratureKnowledgeProvider, KNOWLEDGE_ERROR),
    (StructuredLLMExpertKnowledgeProvider, KNOWLEDGE_ERROR),
)


@pytest.mark.parametrize(("provider", "temperature_error"), PROVIDERS)
def test_python_constructor_errors_and_precedence(provider, temperature_error):
    transport = FakeLLMTransport(())
    with pytest.raises(TypeError):
        provider()
    for temperature in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError) as failure:
            provider(transport, temperature=temperature, max_attempts=0)
        assert str(failure.value) == temperature_error
    for attempts in (0, 6):
        with pytest.raises(ValueError) as failure:
            provider(transport, max_attempts=attempts)
        assert str(failure.value) == ATTEMPTS_ERROR
    for field in ("temperature", "max_attempts"):
        for value in (None, ""):
            with pytest.raises(TypeError):
                provider(transport, **{field: value})
    for attempts in (1, 5, True, 1.5):
        configured = provider(transport, temperature=0, max_attempts=attempts)
        assert configured.max_attempts == attempts
        assert type(configured.max_attempts) is type(attempts)
        assert type(configured.temperature) is int
    assert transport.call_count == 0


@pytest.mark.parametrize(("provider", "provider_id", "version", "attempts", "config_hash"), [
    (StructuredLLMPlanningProvider, "structured-llm-planning", "structured-llm-planning-1.1.1", 2,
     "a3a4178d13463ef1844edad64e641e4285a53d9b14722e0bfe89e1efc0e3dc94"),
    (StructuredLLMApprovalProvider, "structured-llm-approval", "structured-llm-approval-1.0.0", 2,
     "ab5ca6dcb3144a47447354dd971b40a4a3724307e30b6ddd5d5fc75d479d9b9a"),
    (StructuredLLMLiteratureKnowledgeProvider, "structured-llm-literature-knowledge",
     "structured-llm-literature-knowledge-1.2.0", 1,
     "3379bdc22eb1b6806338d375ec5201a7081b1514ee5752d711a182c256a4f26c"),
    (StructuredLLMExpertKnowledgeProvider, "structured-llm-expert-knowledge", "1.0.0", 1,
     "c9a134b80a008ee52a183e0ac8497733316ba1a594f503d5c9c5f60db04d97a2"),
])
def test_default_identity_matches_baseline(provider, provider_id, version, attempts, config_hash):
    configured = provider(FakeLLMTransport((), model_id="test-codex-model"))
    assert (configured.provider_id, configured.provider_version) == (provider_id, version)
    assert (configured.temperature, configured.max_attempts) == (0.0, attempts)
    if provider in (StructuredLLMPlanningProvider, StructuredLLMApprovalProvider):
        stage = "planning" if provider is StructuredLLMPlanningProvider else "approval"
        assert bind_provider(stage, configured).configuration_hash == config_hash
    else:
        assert configured.provider_config_hash == config_hash


def test_knowledge_provider_batch_error_order_and_timing(tmp_path):
    transport = FakeLLMTransport(())
    limits = {"max_chunks_per_batch": 0, "max_batch_text_characters": 0}
    for construct in (lambda **kw: StructuredLLMLiteratureKnowledgeProvider(transport, **kw),
                      lambda **kw: partition_literature_knowledge_chunks((), **kw)):
        with pytest.raises(ValueError, match="^max_chunks_per_batch must be positive$"):
            construct(**limits)
        with pytest.raises(ValueError, match="^max_batch_text_characters must be positive$"):
            construct(max_batch_text_characters=0)
    with pytest.raises(ValueError, match="^max_batches must be positive when provided$"):
        StructuredLLMLiteratureKnowledgeProvider(transport, max_batches=0, batch_failure_policy="")
    with pytest.raises(ValueError, match="^batch_failure_policy must be 'stop' or 'continue'$"):
        StructuredLLMLiteratureKnowledgeProvider(transport, batch_failure_policy="")
    with pytest.raises(ValueError, match="^max_batches must be positive$"):
        StructuredLLMExpertKnowledgeProvider(transport, **limits, max_batches=0)
    expert = StructuredLLMExpertKnowledgeProvider(transport, **limits)
    repositories, _, source = _prepare_source(tmp_path)
    _accept_authority(repositories, source)
    compilation_input = resolve_expert_knowledge_input(source.expert_source_id, repositories)
    chunks = build_expert_knowledge_chunks(compilation_input, repositories)
    with pytest.raises(ValueError, match="^expert provider batch limits must be positive$"):
        expert.propose(compilation_input, chunks)
    assert transport.call_count == 0


def test_timeout_limits_remain_transport_specific():
    for timeout in (1, 1800):
        assert CodexCLILLMTransport(model="m", timeout_seconds=timeout).timeout_seconds == timeout
    for timeout in (0, 1801, float("nan")):
        with pytest.raises(ValueError, match="^Codex CLI timeout must be between 1 and 1800 seconds$"):
            CodexCLILLMTransport(model="m", timeout_seconds=timeout)
        assert HTTPJSONLLMTransport("https://example.invalid", "m", timeout_seconds=timeout).timeout_seconds is timeout


@pytest.mark.parametrize("options", [[], ["--llm-endpoint", "https://example.invalid"], ["--llm-model", "m"]])
def test_literature_cli_requires_both_llm_options(options):
    result = CliRunner().invoke(app, ["extract-literature-knowledge", "--literature-id", "paper",
                                     "--provider", "llm", *options], terminal_width=160)
    assert result.exit_code == 2
    assert "--provider llm requires --llm-endpoint and --llm-model" in _captured_cli_text(result)
