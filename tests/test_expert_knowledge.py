from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spc.cli import app
from spc.knowledge.expert_knowledge import (
    ExpertKnowledgeCompiler,
    ExpertKnowledgeLLMResponse,
    ExpertKnowledgeViewBuilder,
    ExpertOpinionProposal,
    ExpertQuoteProposal,
    ExpertSourceIngestionService,
    MockExpertKnowledgeProvider,
    StructuredLLMExpertKnowledgeProvider,
    expert_knowledge_llm_wire_schema,
    make_expert_profile,
    resolve_expert_knowledge_input,
)
from spc.knowledge.acquisition import HTTPResponse, SafeHTTPFetcher
from spc.knowledge.expert_knowledge.provider import (
    build_expert_knowledge_proposal_set,
)
from spc.knowledge.graph import KnowledgeGraphBuilder
from spc.knowledge.literature_knowledge.validation import curate_knowledge_record
from spc.knowledge.trust import TrustedKnowledgeError, TrustedKnowledgeValidator
from spc.models import CurationStatus, KnowledgeViewMode
from spc.repositories import KnowledgeRepositories


NOTE = (
    "The mechanism is not sufficiently convincing. "
    "Compare a discriminating observable against an explicit baseline."
)


def _put_profile(repositories: KnowledgeRepositories):
    profile = make_expert_profile(
        "Dr Example",
        organization="Example Institute",
        role="Reviewer",
    )
    repositories.expert_profiles.put(profile.expert_id, profile)
    return profile


def _prepare_source(
    tmp_path: Path,
    *,
    text: str = NOTE,
    attribution_status: str = "confirmed",
):
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    profile = _put_profile(repositories)
    source_path = tmp_path / "review.md"
    source_path.write_text(text, encoding="utf-8")
    source = ExpertSourceIngestionService().ingest(
        profile.expert_id,
        source_path,
        "reviewer_comments",
        repositories,
        domain="base",
        source_relationship="reviewer of the submitted work",
        attribution_basis="explicit user attribution",
        attribution_status=attribution_status,
    )
    return repositories, profile, source


def _accept_authority(repositories: KnowledgeRepositories, source) -> None:
    for target_type, target_id in (
        ("expert_source", source.expert_source_id),
        ("expert_attribution", source.attribution_id),
    ):
        curate_knowledge_record(
            repositories,
            target_type=target_type,
            target_id=target_id,
            status=CurationStatus.ACCEPTED,
            curator_id="tester",
            rationale="Verified source identity and expert attribution.",
        )


def _compile(tmp_path: Path):
    repositories, profile, source = _prepare_source(tmp_path)
    _accept_authority(repositories, source)
    outcome = ExpertKnowledgeCompiler().compile(
        source.expert_source_id,
        MockExpertKnowledgeProvider(),
        repositories,
    )
    return repositories, profile, source, outcome


def test_local_text_source_exact_grounding_and_machine_trust_boundary(tmp_path: Path) -> None:
    repositories, profile, _source, outcome = _compile(tmp_path)
    opinion = outcome.materialized.opinions[0]
    evidence = repositories.evidence_store.get_evidence(opinion.evidence_refs[0])

    assert evidence.text == "The mechanism is not sufficiently convincing. Compare a discriminating observable against an explicit baseline."
    assert repositories.evidence_store.verify_evidence_integrity(evidence)
    audit = ExpertKnowledgeViewBuilder().build(profile.expert_id, repositories)
    trusted = ExpertKnowledgeViewBuilder().build(
        profile.expert_id, repositories, view_mode="trusted"
    )
    assert audit["expert_opinions"][0]["curation_status"] == "machine_extracted"
    assert audit["expert_cases"][0]["wrong_formulations"]
    assert trusted["expert_opinions"] == []
    assert trusted["expert_cases"] == []


def test_pdf_expert_source_reuses_born_digital_extractor(tmp_path: Path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    profile = _put_profile(repositories)
    source = ExpertSourceIngestionService().ingest(
        profile.expert_id,
        Path("tests/fixtures/generic-born-digital.pdf"),
        "pdf",
        repositories,
        domain="base",
        source_relationship="authored expert note",
        attribution_basis="explicit user attribution",
    )

    assert source.input_kind == "local_pdf"
    assert source.page_ranges
    stored = repositories.evidence_store.get_source(source.source_id, source.source_version)
    assert stored.media_type == "text/plain"


def test_public_webpage_uses_safe_http_and_existing_html_extractor(tmp_path: Path) -> None:
    html = b"<html><body><main><p>Use a matched baseline.</p></main></body></html>"

    class StaticTransport:
        def request(self, url, *, validated_ips, timeout, max_bytes):
            assert validated_ips == ("93.184.216.34",)
            assert timeout > 0 and max_bytes > len(html)
            return HTTPResponse(
                url=url,
                status=200,
                headers={"content-type": "text/html"},
                body=html,
                connected_ip=validated_ips[0],
            )

    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    profile = _put_profile(repositories)
    source = ExpertSourceIngestionService(
        fetcher=SafeHTTPFetcher(
            StaticTransport(),
            dns_resolver=lambda _host: ("93.184.216.34",),
        )
    ).ingest(
        profile.expert_id,
        "https://example.org/expert-note",
        "expert_webpage",
        repositories,
        domain="base",
        source_relationship="public expert guidance",
        attribution_basis="explicit user attribution",
    )

    assert source.input_kind == "public_webpage"
    stored = repositories.evidence_store.get_source(source.source_id, source.source_version)
    assert stored.source_type.value == "expert_webpage"


def test_unresolved_attribution_cannot_be_extracted(tmp_path: Path) -> None:
    repositories, _profile, source = _prepare_source(
        tmp_path, attribution_status="unresolved"
    )
    _accept_authority(repositories, source)

    with pytest.raises(ValueError, match="attribution is unresolved"):
        resolve_expert_knowledge_input(source.expert_source_id, repositories)


def test_fabricated_quote_is_rejected(tmp_path: Path) -> None:
    repositories, _profile, source = _prepare_source(tmp_path)
    _accept_authority(repositories, source)
    compilation_input = resolve_expert_knowledge_input(source.expert_source_id, repositories)
    chunks = repositories.expert_knowledge_chunks.list()
    if not chunks:
        from spc.knowledge.expert_knowledge import build_expert_knowledge_chunks

        chunks = build_expert_knowledge_chunks(compilation_input, repositories)
    response = ExpertKnowledgeLLMResponse(
        quote_proposals=(
            ExpertQuoteProposal(
                quote_key="q",
                chunk_id=chunks[0].chunk_id,
                exact_text="fabricated source wording",
            ),
        ),
        opinion_proposals=(
            ExpertOpinionProposal(
                opinion_key="o",
                quote_keys=("q",),
                normalized_text="Fabricated opinion.",
                opinion_category="criticism",
                authority_status="attributed_expert_opinion",
                expert_id=compilation_input.expert_id,
                topic="mechanism",
                scope="generic",
                rationale="proposal only",
                conditions=(),
            ),
        ),
    )
    proposal_set = build_expert_knowledge_proposal_set(
        compilation_input,
        provider_id="fake",
        provider_version="1",
        provider_config_hash="0" * 64,
        response=response,
    )
    from spc.knowledge.expert_knowledge.materializer import ExpertKnowledgeMaterializer

    result = ExpertKnowledgeMaterializer().materialize(
        compilation_input, chunks, proposal_set, repositories
    )
    assert result.opinions == ()
    assert {item.rejection_code for item in result.record.rejected_proposals} == {
        "QUOTE_NOT_EXACT",
        "OPINION_QUOTE_REJECTED",
    }


def test_accepted_opinion_and_case_enter_trusted_view_and_graph(tmp_path: Path) -> None:
    repositories, profile, _source, outcome = _compile(tmp_path)
    opinion = outcome.materialized.opinions[0]
    case = outcome.materialized.cases[0]
    for target_type, target_id in (
        ("expert_opinion", opinion.opinion_id),
        ("expert_case", case.case_id),
    ):
        curate_knowledge_record(
            repositories,
            target_type=target_type,
            target_id=target_id,
            status=CurationStatus.ACCEPTED,
            curator_id="reviewer",
            rationale="Verified attributed wording and reusable transformation.",
        )
    trusted = ExpertKnowledgeViewBuilder().build(
        profile.expert_id, repositories, view_mode="trusted"
    )
    graph = KnowledgeGraphBuilder().build(
        repositories,
        repositories.evidence_store,
        view_mode=KnowledgeViewMode.TRUSTED,
    )

    assert [item["opinion_id"] for item in trusted["expert_opinions"]] == [opinion.opinion_id]
    assert [item["case_id"] for item in trusted["expert_cases"]] == [case.case_id]
    graph_keys = {(item.record_type, item.record_id) for item in graph.nodes}
    assert ("expert_profile", profile.expert_id) in graph_keys
    assert ("expert_opinion", opinion.opinion_id) in graph_keys
    assert ("expert_case", case.case_id) in graph_keys


def test_accepted_case_with_machine_opinion_fails_closed(tmp_path: Path) -> None:
    repositories, _profile, _source, outcome = _compile(tmp_path)
    case = outcome.materialized.cases[0]
    curate_knowledge_record(
        repositories,
        target_type="expert_case",
        target_id=case.case_id,
        status=CurationStatus.ACCEPTED,
        curator_id="reviewer",
        rationale="Case reviewed before its opinion dependency.",
    )

    with pytest.raises(TrustedKnowledgeError, match="unaccepted expert_opinion"):
        TrustedKnowledgeValidator(repositories, repositories.evidence_store).validate()


def test_rejected_opinion_is_excluded(tmp_path: Path) -> None:
    repositories, profile, _source, outcome = _compile(tmp_path)
    opinion = outcome.materialized.opinions[0]
    curate_knowledge_record(
        repositories,
        target_type="expert_opinion",
        target_id=opinion.opinion_id,
        status=CurationStatus.REJECTED,
        curator_id="reviewer",
        rationale="The normalization was not faithful.",
    )
    trusted = ExpertKnowledgeViewBuilder().build(
        profile.expert_id, repositories, view_mode="trusted"
    )
    assert trusted["expert_opinions"] == []


def test_rerun_is_idempotent_and_does_not_downgrade_accepted_curation(tmp_path: Path) -> None:
    repositories, _profile, source, first = _compile(tmp_path)
    opinion = first.materialized.opinions[0]
    accepted = curate_knowledge_record(
        repositories,
        target_type="expert_opinion",
        target_id=opinion.opinion_id,
        status=CurationStatus.ACCEPTED,
        curator_id="reviewer",
        rationale="Verified attributed wording.",
    )
    second = ExpertKnowledgeCompiler().compile(
        source.expert_source_id,
        MockExpertKnowledgeProvider(),
        repositories,
    )
    current = TrustedKnowledgeValidator(
        repositories, repositories.evidence_store
    ).resolve_current_curations()

    assert second.materialized.record == first.materialized.record
    assert current[("expert_opinion", opinion.opinion_id)] == accepted


def test_expert_compilation_never_creates_literature_fact_records(tmp_path: Path) -> None:
    repositories, _profile, _source, outcome = _compile(tmp_path)

    assert outcome.materialized.opinions
    assert repositories.source_claims.list() == ()
    assert repositories.method_facts.list() == ()
    assert repositories.model_facts.list() == ()
    assert repositories.reported_results.list() == ()


def test_source_and_attribution_tamper_fail_trusted_validation(tmp_path: Path) -> None:
    repositories, _profile, _source, outcome = _compile(tmp_path)
    opinion = outcome.materialized.opinions[0]
    curate_knowledge_record(
        repositories,
        target_type="expert_opinion",
        target_id=opinion.opinion_id,
        status=CurationStatus.ACCEPTED,
        curator_id="reviewer",
        rationale="Verified attributed wording.",
    )
    source = repositories.evidence_store.get_source(
        outcome.compilation_input.source_id,
        outcome.compilation_input.source_version,
    )
    source_path = repositories.evidence_store.state_root.joinpath(*source.stored_path.split("/"))
    os.chmod(source_path, 0o666)
    source_path.write_text("tampered", encoding="utf-8")

    with pytest.raises((TrustedKnowledgeError, ValueError)):
        TrustedKnowledgeValidator(repositories, repositories.evidence_store).validate()


def test_attribution_record_tamper_fails_closed(tmp_path: Path) -> None:
    repositories, _profile, source, outcome = _compile(tmp_path)
    opinion = outcome.materialized.opinions[0]
    curate_knowledge_record(
        repositories,
        target_type="expert_opinion",
        target_id=opinion.opinion_id,
        status=CurationStatus.ACCEPTED,
        curator_id="reviewer",
        rationale="Verified attributed wording.",
    )
    record_path = repositories.expert_attributions.root / f"{source.attribution_id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["attribution_basis"] = "tampered attribution"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TrustedKnowledgeError, ValueError)):
        TrustedKnowledgeValidator(repositories, repositories.evidence_store).validate()


def test_wire_schema_is_strict_and_fake_transport_materializes_internal_contract(
    tmp_path: Path,
) -> None:
    schema = expert_knowledge_llm_wire_schema()

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for item in node.values():
                visit(item)

    visit(schema)
    repositories, _profile, source = _prepare_source(tmp_path)
    _accept_authority(repositories, source)
    compilation_input = resolve_expert_knowledge_input(source.expert_source_id, repositories)
    from spc.knowledge.expert_knowledge import build_expert_knowledge_chunks

    chunks = build_expert_knowledge_chunks(compilation_input, repositories)

    class FakeCodexTransport:
        model_id = "fake-codex"
        calls = 0

        def generate_structured(self, **_kwargs):
            self.calls += 1
            return json.dumps(
                {
                    "quote_proposals": [
                        {
                            "quote_key": "q",
                            "chunk_id": chunks[0].chunk_id,
                            "exact_text": chunks[0].text,
                        }
                    ],
                    "opinion_proposals": [
                        {
                            "opinion_key": "o",
                            "quote_keys": ["q"],
                            "normalized_text": "The mechanism needs discriminating evidence.",
                            "opinion_category": "missing_evidence",
                            "authority_status": "attributed_expert_opinion",
                            "expert_id": compilation_input.expert_id,
                            "topic": "mechanism",
                            "scope": "stated pathway",
                            "rationale": "Exact expert wording is interpreted as a concern.",
                            "conditions": [],
                        }
                    ],
                    "case_proposals": [
                        {
                            "case_key": "c",
                            "opinion_keys": ["o"],
                            "original_wording": chunks[0].text,
                            "latent_concern": "The mechanism is underdetermined.",
                            "atomic_questions": ["Which observable discriminates the pathways?"],
                            "good_question_formulations": ["Does observable X distinguish A from B?"],
                            "wrong_formulations": ["Prove the preferred mechanism."],
                            "answerability_conditions": ["A and B predict different X values."],
                            "required_evidence_types": ["discriminating observable"],
                            "baseline_guidance": ["Compare under matched conditions."],
                            "common_misinterpretations": ["Opinion is not fact."],
                            "resolution_pattern": "Compare a discriminating observable.",
                            "applicability": ["mechanism criticism"],
                        }
                    ],
                }
            )

    transport = FakeCodexTransport()
    provider = StructuredLLMExpertKnowledgeProvider(transport, max_batches=1)
    result = ExpertKnowledgeCompiler().compile(
        source.expert_source_id, provider, repositories
    )

    assert transport.calls == 1
    assert result.materialized.opinions
    assert result.materialized.cases[0].atomic_questions
    assert result.materialized.opinions[0].__class__.__name__ == "ExpertOpinion"


def test_cli_expert_workflow_is_explicit_and_inspectable(tmp_path: Path) -> None:
    runner = CliRunner()
    knowledge = tmp_path / "knowledge"
    note = tmp_path / "meeting.txt"
    note.write_text("The comparison lacks a matched baseline.", encoding="utf-8")
    profile_result = runner.invoke(
        app,
        [
            "add-expert",
            "--name",
            "CLI Expert",
            "--organization",
            "Example Lab",
            "--role",
            "Reviewer",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert profile_result.exit_code == 0
    expert_id = json.loads(profile_result.output)["expert_id"]
    source_result = runner.invoke(
        app,
        [
            "add-expert-source",
            "--expert-id",
            expert_id,
            "--source",
            str(note),
            "--source-type",
            "meeting_notes",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert source_result.exit_code == 0
    source = json.loads(source_result.output)
    for target_type, target_id in (
        ("expert_source", source["expert_source_id"]),
        ("expert_attribution", source["attribution_id"]),
    ):
        result = runner.invoke(
            app,
            [
                "curate-knowledge",
                "--target-type",
                target_type,
                "--target-id",
                target_id,
                "--status",
                "accepted",
                "--curator-id",
                "cli-reviewer",
                "--rationale",
                "Verified source authority.",
                "--knowledge-dir",
                str(knowledge),
            ],
        )
        assert result.exit_code == 0
    repeated_source = runner.invoke(
        app,
        [
            "add-expert-source",
            "--expert-id",
            expert_id,
            "--source",
            str(note),
            "--source-type",
            "meeting_notes",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert repeated_source.exit_code == 0
    assert json.loads(repeated_source.output)["expert_source_id"] == source["expert_source_id"]
    extract_result = runner.invoke(
        app,
        [
            "extract-expert-knowledge",
            "--expert-source-id",
            source["expert_source_id"],
            "--provider",
            "mock",
            "--max-batches",
            "1",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert extract_result.exit_code == 0
    assert json.loads(extract_result.output)["expert_opinion_ids"]
    inspect_result = runner.invoke(
        app,
        [
            "inspect-expert-knowledge",
            "--expert-id",
            expert_id,
            "--view",
            "audit",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert inspect_result.exit_code == 0
    assert json.loads(inspect_result.output)["expert_cases"]
