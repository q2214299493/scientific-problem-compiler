from pathlib import Path
import json

import pytest

from spc.knowledge.trust import TrustedKnowledgeValidator
from spc.models import EvidenceSpan
from spc.repositories import CompositeEvidenceStore, FilesystemEvidenceStore, KnowledgeRepositories
from test_evidence_stores import _shared_source_evidence


@pytest.mark.parametrize("composite", [False, True], ids=["filesystem", "composite"])
def test_evidence_offsets_reuse_the_exact_source_bytes_already_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, composite: bool
):
    store, source, first, _second = _shared_source_evidence(tmp_path)
    reader = CompositeEvidenceStore(store, FilesystemEvidenceStore(tmp_path / "empty")) if composite else store
    paths = {
        store.root / source.stored_path: "content",
        store.source_records.root / f"{source.source_id}--{source.version}.json": "metadata",
    }
    reads = {"metadata": 0, "content": 0}
    original = Path.open

    def count_open(path, *args, **kwargs):
        if path in paths:
            reads[paths[path]] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_open)
    assert reader.verify_evidence_integrity(first) == source
    print({"composite": composite, "reads": reads})
    assert reads["metadata"] == (4 if composite else 2)
    assert reads["content"] == (2 if composite else 1)


@pytest.mark.parametrize("composite", [False, True], ids=["filesystem", "composite"])
def test_closing_and_later_trusted_reads_reopen_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, composite: bool
):
    store, source, first, _second = _shared_source_evidence(tmp_path)
    reader = CompositeEvidenceStore(store, FilesystemEvidenceStore(tmp_path / "empty")) if composite else store
    repositories = KnowledgeRepositories(tmp_path / "knowledge")
    path = store.root / source.stored_path
    original = Path.open
    reads = 0

    def count_open(current, *args, **kwargs):
        nonlocal reads
        if current == path:
            reads += 1
        return original(current, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_open)
    validator = TrustedKnowledgeValidator(repositories, reader)
    assert validator._verify_evidence(first.evidence_id) == (first, source)
    first_reads = 3 if composite else 1
    assert reads == first_reads
    assert validator._verify_evidence(first.evidence_id) == (first, source)
    assert reads == first_reads
    validator._revalidate_cached_evidence()
    assert reads == first_reads * 2
    fresh = TrustedKnowledgeValidator(repositories, reader)
    assert fresh._verify_evidence(first.evidence_id) == (first, source)
    assert reads == first_reads * 3


@pytest.mark.parametrize("scope", ["instance", "class"])
@pytest.mark.parametrize("reject", [False, True], ids=["delegate", "reject"])
def test_evidence_verification_keeps_custom_source_integrity_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str, reject: bool
):
    store, source, first, _second = _shared_source_evidence(tmp_path)
    owner = store if scope == "instance" else type(store)
    original_check = getattr(owner, "verify_source_integrity")
    original_open = Path.open
    source_path = store.root / source.stored_path
    calls = 0
    reads = 0

    def custom_check(*args, **kwargs):
        nonlocal calls
        calls += 1
        if reject:
            raise ValueError("custom source gate")
        return original_check(*args, **kwargs)

    def count_open(path, *args, **kwargs):
        nonlocal reads
        if path == source_path:
            reads += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(owner, "verify_source_integrity", custom_check)
    monkeypatch.setattr(Path, "open", count_open)
    if reject:
        with pytest.raises(ValueError, match="custom source gate"):
            store.verify_evidence_integrity(first)
        assert reads == 0
    else:
        assert store.verify_evidence_integrity(first) == source
        assert reads == 2
    assert calls == 1


@pytest.mark.parametrize("composite", [False, True], ids=["filesystem", "composite"])
@pytest.mark.parametrize("target", ["content", "metadata", "evidence"])
def test_trusted_closing_rejects_changes_after_evidence_verification(
    tmp_path: Path, composite: bool, target: str
):
    store, source, first, _second = _shared_source_evidence(tmp_path)
    reader = CompositeEvidenceStore(store, FilesystemEvidenceStore(tmp_path / "empty")) if composite else store
    validator = TrustedKnowledgeValidator(KnowledgeRepositories(tmp_path / "knowledge"), reader)
    validator._verify_evidence(first.evidence_id)
    if target == "content":
        path = store.root / source.stored_path
        path.chmod(0o600)
        path.write_bytes(b"tampered after verification")
    else:
        path = (
            store.source_records.root / f"{source.source_id}--{source.version}.json"
            if target == "metadata"
            else store.evidence_records.root / f"{first.evidence_id}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["title" if target == "metadata" else "locator"] = "changed after verification"
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        validator._revalidate_cached_evidence()


def test_closing_rejects_body_changed_after_the_verified_buffer_was_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store, source, first, _second = _shared_source_evidence(tmp_path)
    validator = TrustedKnowledgeValidator(KnowledgeRepositories(tmp_path / "knowledge"), store)
    path = store.root / source.stored_path
    original = store._verified_source_bytes
    changed = False

    def change_after_read(record):
        nonlocal changed
        content = original(record)
        if not changed:
            path.chmod(0o600)
            path.write_bytes(b"changed after SHA256 verification")
            changed = True
        return content

    monkeypatch.setattr(store, "_verified_source_bytes", change_after_read)
    assert validator._verify_evidence(first.evidence_id) == (first, source)
    with pytest.raises(ValueError, match="content hash"):
        validator._revalidate_cached_evidence()


def test_second_metadata_read_still_rejects_a_changed_source_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store, _source, first, _second = _shared_source_evidence(tmp_path)
    original = store.get_source
    calls = 0

    def changed_record(*args, **kwargs):
        nonlocal calls
        calls += 1
        record = original(*args, **kwargs)
        return record.model_copy(update={"title": "changed source record"}) if calls == 2 else record

    monkeypatch.setattr(store, "get_source", changed_record)
    with pytest.raises(ValueError, match="differs from its repository record"):
        store.verify_evidence_integrity(first)
    assert calls == 2


@pytest.mark.parametrize("target", ["hash", "text", "offset"])
def test_reused_buffer_preserves_every_evidence_binding_check(tmp_path: Path, target: str):
    store, _source, first, _second = _shared_source_evidence(tmp_path)
    update, error = {
        "hash": ({"content_sha256": "0" * 64}, "content hash"),
        "text": ({"text": "x" + first.text[1:]}, "text does not match"),
        "offset": ({"end_offset": first.end_offset + 1}, "exceeds stored source"),
    }[target]
    invalid = first.model_copy(update=update)
    with pytest.raises(ValueError, match=error):
        store._verify_evidence_against_source(invalid)


def test_verified_body_still_requires_utf8_for_evidence_offsets(tmp_path: Path):
    store = FilesystemEvidenceStore(tmp_path / "store")
    path = tmp_path / "binary.txt"
    path.write_bytes(b"\xff")
    source = store.ingest_source(path, "invalid-utf8", "v1")
    evidence = EvidenceSpan(
        evidence_id="ev-invalid-utf8", source_id=source.source_id, source_version=source.version,
        content_sha256=source.content_sha256, start_offset=0, end_offset=1, text="x",
    )
    store.evidence_records.put(evidence.evidence_id, evidence)
    with pytest.raises(ValueError, match="UTF-8"):
        store.verify_evidence_integrity(evidence)


def test_composite_cannot_hide_corruption_in_another_store(tmp_path: Path):
    first_store, source, first, _second = _shared_source_evidence(tmp_path)
    second_store = FilesystemEvidenceStore(tmp_path / "second-store")
    second_store.import_source_record(source, (first_store.root / source.stored_path).read_bytes())
    second_store.add_evidence(first)
    reader = CompositeEvidenceStore(first_store, second_store)
    assert reader.verify_evidence_integrity(first) == source
    path = second_store.root / source.stored_path
    path.chmod(0o600)
    path.write_bytes(b"tampered duplicate source")
    with pytest.raises(ValueError):
        reader.verify_evidence_integrity(first)
