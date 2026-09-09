from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spc.cli import app
from spc.knowledge.evidence_migration import migrate_knowledge_evidence
from spc.knowledge.ingestion import (
    LiteratureIngestionService,
    LiteratureRepresentationSelector,
)
from spc.models import EvidenceSpan
from spc.repositories import (
    CompositeEvidenceStore,
    FilesystemEvidenceStore,
    KnowledgeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
)
from spc.retrieval import ScientificContextBuilder
from spc.serialization import dump_yaml


FIXTURES = Path(__file__).parent / "fixtures"
PDF = FIXTURES / "generic-born-digital.pdf"


def metadata() -> dict[str, object]:
    return {
        "title": "Shared Evidence Store Fixture",
        "authors": ["A. Researcher"],
        "year": 2026,
        "journal": "Journal of Persistent Evidence",
        "doi": "10.0000/example.k1e0",
        "url": "https://example.org/k1e0",
        "domain": "base",
        "topics": ["shared evidence"],
        "keywords": ["provenance"],
        "citation_refs": [],
    }


def add_text_evidence(
    store: FilesystemEvidenceStore,
    root: Path,
    *,
    source_id: str,
    evidence_id: str,
    text: str,
    title: str = "Evidence",
) -> tuple[object, EvidenceSpan]:
    source_path = root / f"{source_id}-{evidence_id}.txt"
    source_path.write_text(text, encoding="utf-8")
    source = store.ingest_source(source_path, source_id, "v1", title)
    evidence = EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        source_version="v1",
        content_sha256=source.content_sha256,
        start_offset=0,
        end_offset=len(text),
        text=text,
    )
    store.add_evidence(evidence)
    return source, evidence


def test_filesystem_store_is_root_agnostic_and_creates_no_project_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "standalone-evidence"
    store = FilesystemEvidenceStore(root)

    assert store.list_sources() == ()
    assert store.list_evidence() == ()
    assert (root / "sources").is_dir()
    assert (root / "evidence").is_dir()
    assert not (root / "project.yaml").exists()
    assert not (root / "artifacts.jsonl").exists()


def test_shared_knowledge_and_current_project_are_composed_without_leakage(
    tmp_path: Path,
) -> None:
    knowledge = KnowledgeEvidenceStore(tmp_path / "knowledge")
    project_a = ProjectEvidenceStore(tmp_path / "project-a" / ".spc")
    project_b = ProjectEvidenceStore(tmp_path / "project-b" / ".spc")
    add_text_evidence(
        knowledge,
        tmp_path,
        source_id="literature-source",
        evidence_id="ev-literature",
        text="shared catalytic pathway evidence",
    )
    add_text_evidence(
        project_a,
        tmp_path,
        source_id="reviewer-source",
        evidence_id="ev-reviewer-a",
        text="project A reviewer comment",
    )
    view_a = CompositeEvidenceStore(knowledge, project_a)
    view_b = CompositeEvidenceStore(knowledge, project_b)

    assert {item.evidence_id for item in view_a.list_evidence()} == {
        "ev-literature",
        "ev-reviewer-a",
    }
    assert view_b.get_evidence("ev-literature").text.startswith("shared")
    with pytest.raises(FileNotFoundError):
        view_b.get_evidence("ev-reviewer-a")
    assert project_b.list_sources() == ()


def test_same_source_identity_and_bytes_are_not_a_false_collision(
    tmp_path: Path,
) -> None:
    first = FilesystemEvidenceStore(tmp_path / "first")
    second = FilesystemEvidenceStore(tmp_path / "second")
    source_path = tmp_path / "same.txt"
    source_path.write_text("same bytes", encoding="utf-8")
    source_one = first.ingest_source(source_path, "same-source", "v1", "Same title")
    source_two = second.ingest_source(source_path, "same-source", "v1", "Same title")
    evidence = EvidenceSpan(
        evidence_id="ev-same",
        source_id="same-source",
        source_version="v1",
        content_sha256=source_one.content_sha256,
        start_offset=0,
        end_offset=10,
        text="same bytes",
    )
    first.add_evidence(evidence)
    second.add_evidence(evidence)

    view = CompositeEvidenceStore(first, second)
    source = view.get_source("same-source", "v1")

    assert source_one.content_sha256 == source_two.content_sha256
    assert source.content_sha256 == source_one.content_sha256
    assert view.get_evidence("ev-same") == evidence


def test_conflicting_source_identity_fails_closed(tmp_path: Path) -> None:
    first = FilesystemEvidenceStore(tmp_path / "first")
    second = FilesystemEvidenceStore(tmp_path / "second")
    one = tmp_path / "one.txt"
    two = tmp_path / "two.txt"
    one.write_text("one", encoding="utf-8")
    two.write_text("two", encoding="utf-8")
    first.ingest_source(one, "collision", "v1")
    second.ingest_source(two, "collision", "v1")

    with pytest.raises(ValueError, match="conflicting evidence-store identity"):
        CompositeEvidenceStore(first, second).get_source("collision", "v1")


def test_conflicting_evidence_identity_fails_closed(tmp_path: Path) -> None:
    first = FilesystemEvidenceStore(tmp_path / "first")
    second = FilesystemEvidenceStore(tmp_path / "second")
    add_text_evidence(
        first,
        tmp_path,
        source_id="source-one",
        evidence_id="ev-collision",
        text="first evidence",
    )
    add_text_evidence(
        second,
        tmp_path,
        source_id="source-two",
        evidence_id="ev-collision",
        text="second evidence",
    )

    with pytest.raises(ValueError, match="conflicting evidence-store identity"):
        CompositeEvidenceStore(first, second).get_evidence("ev-collision")


def test_composite_writes_require_an_explicit_target(tmp_path: Path) -> None:
    view = CompositeEvidenceStore(
        KnowledgeEvidenceStore(tmp_path / "knowledge"),
        ProjectEvidenceStore(tmp_path / ".spc"),
    )

    with pytest.raises(TypeError, match="explicit target store"):
        view.ingest_source(tmp_path / "unused", "source", "v1")


def test_composite_fails_closed_on_source_and_evidence_tampering(
    tmp_path: Path,
) -> None:
    knowledge = KnowledgeEvidenceStore(tmp_path / "knowledge")
    source, evidence = add_text_evidence(
        knowledge,
        tmp_path,
        source_id="tamper-source",
        evidence_id="ev-tamper",
        text="immutable evidence",
    )
    view = CompositeEvidenceStore(knowledge, ProjectEvidenceStore(tmp_path / ".spc"))
    content_path = knowledge.source_content_path(source)
    content_path.chmod(0o644)
    content_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        view.get_source(source.source_id, source.version)

    content_path.write_text("immutable evidence", encoding="utf-8")
    evidence_path = knowledge.evidence_records.root / f"{evidence.evidence_id}.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["text"] = "different evidence"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="EvidenceSpan integrity failed"):
        view.get_evidence(evidence.evidence_id)


def test_literature_source_is_shared_and_reusable_from_project_b(
    tmp_path: Path,
) -> None:
    knowledge_root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(knowledge_root)
    knowledge = KnowledgeEvidenceStore(knowledge_root)
    project_a = tmp_path / "project-a" / ".spc"
    project_b = ProjectEvidenceStore(tmp_path / "project-b" / ".spc")
    outcome = LiteratureIngestionService().ingest(
        PDF, metadata(), repositories, knowledge
    )
    selection = LiteratureRepresentationSelector().select(
        outcome.literature_id,
        outcome.ingestion_id,
        "k1e0-test",
        "Select the shared representation.",
        repositories,
        knowledge,
    ).selection
    project_b_view = CompositeEvidenceStore(knowledge, project_b)

    resolved = LiteratureRepresentationSelector.resolve_selection(
        selection, repositories, project_b_view
    )
    assert resolved.source_id is not None
    assert (knowledge_root / "evidence_store" / "sources").is_dir()
    assert not (project_a / "sources").exists()
    assert project_b.list_sources() == ()


def test_project_b_retrieval_reads_shared_knowledge_evidence(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge"
    knowledge = KnowledgeEvidenceStore(knowledge_root)
    add_text_evidence(
        knowledge,
        tmp_path,
        source_id="shared-retrieval",
        evidence_id="ev-shared-retrieval",
        text="surface pathway observable",
    )

    packet = ScientificContextBuilder().build(
        "surface pathway observable",
        "base",
        state_dir=tmp_path / "project-b" / ".spc",
        knowledge_dir=knowledge_root,
    )

    assert {hit.record_id for hit in packet.evidence_hits} == {
        "ev-shared-retrieval"
    }


def test_literature_cli_defaults_to_knowledge_evidence_store(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.yaml"
    dump_yaml(metadata_path, metadata())
    knowledge_root = tmp_path / "knowledge"
    state_root = tmp_path / ".spc"

    result = CliRunner().invoke(
        app,
        [
            "ingest-literature",
            str(PDF),
            "--metadata",
            str(metadata_path),
            "--knowledge-dir",
            str(knowledge_root),
            "--state-dir",
            str(state_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert KnowledgeEvidenceStore(knowledge_root).list_sources()
    assert not (state_root / "sources").exists()


def test_legacy_literature_evidence_migration_is_idempotent(tmp_path: Path) -> None:
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    legacy = ProjectEvidenceStore(tmp_path / ".spc")
    knowledge = KnowledgeEvidenceStore(tmp_path / "knowledge")
    outcome = LiteratureIngestionService().ingest(
        PDF, metadata(), repositories, legacy
    )
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    source = legacy.get_source(
        ingestion.source_id or "", ingestion.source_version or ""
    )
    text = repositories.canonical_text_artifacts.read_text(
        outcome.canonical_text_id or ""
    )
    evidence = EvidenceSpan(
        evidence_id="ev-legacy-literature",
        source_id=source.source_id,
        source_version=source.version,
        content_sha256=source.content_sha256,
        start_offset=0,
        end_offset=min(20, len(text)),
        text=text[:20],
    )
    legacy.add_evidence(evidence)

    first = migrate_knowledge_evidence(repositories, legacy, knowledge)
    second = migrate_knowledge_evidence(repositories, legacy, knowledge)

    assert first.copied_source_count == 1
    assert first.copied_evidence_count == 1
    assert second.copied_source_count == 0
    assert second.copied_evidence_count == 0
    assert knowledge.get_source(source.source_id, source.version) == source
    assert legacy.get_evidence(evidence.evidence_id) == evidence
    assert knowledge.get_evidence(evidence.evidence_id) == evidence

    cli_result = CliRunner().invoke(
        app,
        [
            "migrate-knowledge-evidence",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--state-dir",
            str(tmp_path / ".spc"),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert json.loads(cli_result.output)["copied_source_count"] == 0
