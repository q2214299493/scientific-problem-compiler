from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spc.cli import app
from spc.domains import DomainPackLoader
from spc.knowledge.evidence_migration import migrate_knowledge_evidence
from spc.knowledge.ingestion import (
    LiteratureIngestionService,
    LiteratureRepresentationSelector,
)
from spc.models import CurationStatus, EvidenceSpan, KnowledgeCurationRecord
from spc.repositories import (
    CompositeEvidenceStore,
    FilesystemEvidenceStore,
    KnowledgeEvidenceStore,
    KnowledgeRepositories,
    ProjectEvidenceStore,
)
from spc.retrieval import ScientificContextBuilder
from spc.serialization import content_hash, dump_yaml


FIXTURES = Path(__file__).parent / "fixtures"
PDF = FIXTURES / "generic-born-digital.pdf"


def metadata(suffix: str = "k1e0") -> dict[str, object]:
    return {
        "title": f"Shared Evidence Store Fixture {suffix}",
        "authors": ["A. Researcher"],
        "year": 2026,
        "journal": "Journal of Persistent Evidence",
        "doi": f"10.0000/example.{suffix}",
        "url": f"https://example.org/{suffix}",
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


def accept_record(
    repositories: KnowledgeRepositories,
    *,
    target_type: str,
    target_id: str,
    target_hash: str,
) -> None:
    identity = {
        "target_type": target_type,
        "target_id": target_id,
        "target_hash": target_hash,
        "status": CurationStatus.ACCEPTED,
        "curator_id": "k1e0-integrity-test",
        "rationale": "Accepted after explicit integrity review.",
        "evidence_refs": (),
    }
    curation_id = f"knowledge-curation-{content_hash(identity)[:24]}"
    payload = {"curation_id": curation_id, **identity}
    curation = KnowledgeCurationRecord(
        **payload,
        content_hash=content_hash(payload),
    )
    repositories.curations.put(curation.curation_id, curation)


def ingest_and_trust_shared_literature(
    tmp_path: Path,
) -> tuple[KnowledgeRepositories, KnowledgeEvidenceStore, object]:
    knowledge_root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(knowledge_root)
    knowledge = KnowledgeEvidenceStore(knowledge_root)
    outcome = LiteratureIngestionService().ingest(
        PDF, metadata("trusted"), repositories, knowledge
    )
    selection = LiteratureRepresentationSelector().select(
        outcome.literature_id,
        outcome.ingestion_id,
        "k1e0-integrity-test",
        "Select the verified shared representation.",
        repositories,
        knowledge,
    ).selection
    document = repositories.literature_documents.get(outcome.literature_id)
    accept_record(
        repositories,
        target_type="literature_document",
        target_id=document.literature_id,
        target_hash=document.content_hash,
    )
    accept_record(
        repositories,
        target_type="literature_representation_selection",
        target_id=selection.selection_id,
        target_hash=selection.content_hash,
    )
    return repositories, knowledge, outcome


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


def test_get_source_rejects_repository_key_identity_mismatch(tmp_path: Path) -> None:
    store = FilesystemEvidenceStore(tmp_path / "evidence")
    source, _ = add_text_evidence(
        store,
        tmp_path,
        source_id="actual-source",
        evidence_id="ev-actual",
        text="identity-bound source",
    )
    alias_path = store.source_records.root / "alias-source--v1.json"
    alias_path.write_text(source.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match repository key"):
        store.get_source("alias-source", "v1")
    with pytest.raises(ValueError, match="does not match record identity"):
        store.list_sources()


def test_get_evidence_rejects_repository_key_identity_mismatch(tmp_path: Path) -> None:
    store = FilesystemEvidenceStore(tmp_path / "evidence")
    _, evidence = add_text_evidence(
        store,
        tmp_path,
        source_id="identity-source",
        evidence_id="ev-actual",
        text="identity-bound evidence",
    )
    alias_path = store.evidence_records.root / "ev-alias.json"
    alias_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="key does not match"):
        store.get_evidence("ev-alias")
    with pytest.raises(ValueError, match="key does not match"):
        store.list_evidence()


def test_list_sources_fails_on_tampered_source_bytes(tmp_path: Path) -> None:
    store = FilesystemEvidenceStore(tmp_path / "evidence")
    source, _ = add_text_evidence(
        store,
        tmp_path,
        source_id="tampered-list-source",
        evidence_id="ev-tampered-list-source",
        text="original source bytes",
    )
    content_path = store.root / source.stored_path
    content_path.chmod(0o644)
    content_path.write_text("tampered source bytes", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        store.list_sources()


def test_list_evidence_fails_on_tampered_evidence_record(tmp_path: Path) -> None:
    store = FilesystemEvidenceStore(tmp_path / "evidence")
    _, evidence = add_text_evidence(
        store,
        tmp_path,
        source_id="tampered-evidence-source",
        evidence_id="ev-tampered-list",
        text="original evidence text",
    )
    record_path = store.evidence_records.root / f"{evidence.evidence_id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["text"] = "tampered evidence text"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="offsets"):
        store.list_evidence()


@pytest.mark.parametrize("record_kind", ("source", "evidence"))
def test_list_rejects_symlinked_repository_record(
    tmp_path: Path, record_kind: str
) -> None:
    store = FilesystemEvidenceStore(tmp_path / "evidence")
    source, evidence = add_text_evidence(
        store,
        tmp_path,
        source_id="symlink-source",
        evidence_id="ev-symlink",
        text="symlink-safe evidence",
    )
    if record_kind == "source":
        target = store.source_records.root / f"{source.source_id}--{source.version}.json"
        link = store.source_records.root / "alias--v1.json"
        listing = store.list_sources
    else:
        target = store.evidence_records.root / f"{evidence.evidence_id}.json"
        link = store.evidence_records.root / "ev-alias.json"
        listing = store.list_evidence
    try:
        link.symlink_to(target.name)
    except OSError:
        pytest.skip("repository-record symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="cannot be a symlink"):
        listing()


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


def test_composite_list_cannot_hide_tampered_duplicate_source(
    tmp_path: Path,
) -> None:
    clean = FilesystemEvidenceStore(tmp_path / "clean")
    tampered = FilesystemEvidenceStore(tmp_path / "tampered")
    source_path = tmp_path / "duplicate.txt"
    source_path.write_text("duplicate source", encoding="utf-8")
    clean.ingest_source(source_path, "duplicate-source", "v1", "Duplicate")
    broken = tampered.ingest_source(
        source_path, "duplicate-source", "v1", "Duplicate"
    )
    broken_path = tampered.root / broken.stored_path
    broken_path.chmod(0o644)
    broken_path.write_text("tampered duplicate", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        CompositeEvidenceStore(clean, tampered).list_sources()


def test_composite_list_cannot_hide_tampered_duplicate_evidence(
    tmp_path: Path,
) -> None:
    clean = FilesystemEvidenceStore(tmp_path / "clean")
    tampered = FilesystemEvidenceStore(tmp_path / "tampered")
    for store in (clean, tampered):
        add_text_evidence(
            store,
            tmp_path,
            source_id="duplicate-evidence-source",
            evidence_id="ev-duplicate",
            text="duplicate evidence",
        )
    record_path = tampered.evidence_records.root / "ev-duplicate.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["locator"] = "tampered locator"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting evidence-store identity"):
        CompositeEvidenceStore(clean, tampered).list_evidence()


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


def test_trusted_shared_pdf_snapshot_is_project_independent(tmp_path: Path) -> None:
    repositories, knowledge, outcome = ingest_and_trust_shared_literature(tmp_path)
    profile = DomainPackLoader().load("base").profile
    before = repositories.create_snapshot(knowledge, profile)
    project_a = ProjectEvidenceStore(tmp_path / "project-a" / ".spc")
    project_b = ProjectEvidenceStore(tmp_path / "project-b" / ".spc")
    add_text_evidence(
        project_a,
        tmp_path,
        source_id="project-a-reviewer",
        evidence_id="ev-project-a-reviewer",
        text="project A only",
    )
    add_text_evidence(
        project_b,
        tmp_path,
        source_id="project-b-reviewer",
        evidence_id="ev-project-b-reviewer",
        text="project B only",
    )
    after = repositories.create_snapshot(knowledge, profile)

    assert before.snapshot_id == after.snapshot_id
    assert before.model_dump(exclude={"created_at"}) == after.model_dump(
        exclude={"created_at"}
    )
    assert outcome.literature_id in before.literature_document_hashes
    assert all("project-a" not in key for key in before.evidence_source_versions)
    assert all("project-b" not in key for key in before.evidence_source_versions)


def test_trusted_shared_pdf_snapshot_fails_on_source_tampering(tmp_path: Path) -> None:
    repositories, knowledge, outcome = ingest_and_trust_shared_literature(tmp_path)
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    source = knowledge.get_source(
        ingestion.source_id or "", ingestion.source_version or ""
    )
    content_path = knowledge.root / source.stored_path
    content_path.chmod(0o644)
    content_path.write_text("tampered canonical source", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        repositories.create_snapshot(
            knowledge, DomainPackLoader().load("base").profile
        )


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


def test_already_shared_only_literature_migration_succeeds(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(knowledge_root)
    legacy = ProjectEvidenceStore(tmp_path / ".spc")
    knowledge = KnowledgeEvidenceStore(knowledge_root)
    LiteratureIngestionService().ingest(
        PDF, metadata("shared-only"), repositories, knowledge
    )

    first = migrate_knowledge_evidence(repositories, legacy, knowledge)
    LiteratureIngestionService().ingest(
        PDF, metadata("shared-added-later"), repositories, knowledge
    )
    second = migrate_knowledge_evidence(repositories, legacy, knowledge)

    assert first.referenced_source_count == 1
    assert first.legacy_source_count == 0
    assert first.already_shared_source_count == 1
    assert first.copied_source_count == 0
    assert second.referenced_source_count == 2
    assert second.already_shared_source_count == 2
    assert second.copied_source_count == 0


def test_mixed_legacy_and_shared_literature_migration(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(knowledge_root)
    legacy = ProjectEvidenceStore(tmp_path / ".spc")
    knowledge = KnowledgeEvidenceStore(knowledge_root)
    legacy_outcome = LiteratureIngestionService().ingest(
        PDF, metadata("legacy-a"), repositories, legacy
    )
    shared_outcome = LiteratureIngestionService().ingest(
        PDF, metadata("shared-b"), repositories, knowledge
    )
    add_text_evidence(
        legacy,
        tmp_path,
        source_id="unrelated-reviewer-source",
        evidence_id="ev-unrelated-reviewer",
        text="project-only reviewer evidence",
    )

    result = migrate_knowledge_evidence(repositories, legacy, knowledge)
    legacy_ingestion = repositories.literature_ingestions.get(
        legacy_outcome.ingestion_id
    )
    shared_ingestion = repositories.literature_ingestions.get(
        shared_outcome.ingestion_id
    )

    assert result.referenced_source_count == 2
    assert result.legacy_source_count == 1
    assert result.already_shared_source_count == 1
    assert result.copied_source_count == 1
    assert knowledge.get_source(
        legacy_ingestion.source_id or "", legacy_ingestion.source_version or ""
    )
    assert knowledge.get_source(
        shared_ingestion.source_id or "", shared_ingestion.source_version or ""
    )
    with pytest.raises(FileNotFoundError):
        knowledge.get_evidence("ev-unrelated-reviewer")
    assert legacy.get_evidence("ev-unrelated-reviewer")


def test_migration_fails_when_referenced_source_is_missing_everywhere(
    tmp_path: Path,
) -> None:
    knowledge_root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(knowledge_root)
    orphan_store = FilesystemEvidenceStore(tmp_path / "orphan")
    LiteratureIngestionService().ingest(
        PDF, metadata("missing"), repositories, orphan_store
    )

    with pytest.raises(FileNotFoundError, match="missing from both"):
        migrate_knowledge_evidence(
            repositories,
            ProjectEvidenceStore(tmp_path / ".spc"),
            KnowledgeEvidenceStore(knowledge_root),
        )


def test_migration_rejects_conflicting_shared_source(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(knowledge_root)
    legacy = ProjectEvidenceStore(tmp_path / ".spc")
    knowledge = KnowledgeEvidenceStore(knowledge_root)
    outcome = LiteratureIngestionService().ingest(
        PDF, metadata("conflict"), repositories, legacy
    )
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    conflicting_path = tmp_path / "conflicting-source.txt"
    conflicting_path.write_text("different valid source bytes", encoding="utf-8")
    knowledge.ingest_source(
        conflicting_path,
        ingestion.source_id or "",
        ingestion.source_version or "",
        "Conflicting shared source",
        source_role="literature_author",
        source_type="literature_article",
    )

    with pytest.raises(ValueError, match="conflicting knowledge source identity"):
        migrate_knowledge_evidence(repositories, legacy, knowledge)


def test_migration_rejects_conflicting_shared_evidence(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(knowledge_root)
    legacy = ProjectEvidenceStore(tmp_path / ".spc")
    knowledge = KnowledgeEvidenceStore(knowledge_root)
    outcome = LiteratureIngestionService().ingest(
        PDF, metadata("evidence-conflict"), repositories, legacy
    )
    ingestion = repositories.literature_ingestions.get(outcome.ingestion_id)
    source = legacy.get_source(
        ingestion.source_id or "", ingestion.source_version or ""
    )
    text = repositories.canonical_text_artifacts.read_text(
        outcome.canonical_text_id or ""
    )
    legacy_evidence = EvidenceSpan(
        evidence_id="ev-migration-conflict",
        source_id=source.source_id,
        source_version=source.version,
        content_sha256=source.content_sha256,
        start_offset=0,
        end_offset=10,
        text=text[:10],
    )
    legacy.add_evidence(legacy_evidence)
    knowledge.import_source_record(source, legacy.read_source_bytes(source))
    knowledge.add_evidence(
        legacy_evidence.model_copy(
            update={
                "start_offset": 1,
                "end_offset": 11,
                "text": text[1:11],
            }
        )
    )

    with pytest.raises(
        ValueError, match="conflicting knowledge EvidenceSpan identity"
    ):
        migrate_knowledge_evidence(repositories, legacy, knowledge)
