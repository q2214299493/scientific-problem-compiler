"""Behavior shared by the four knowledge-domain immutable write entry points."""

from hashlib import sha256
from json import JSONDecodeError
from pathlib import Path
from unittest.mock import Mock, call, patch, sentinel
from zipfile import ZipFile

import pytest

from spc.interpretation.claim_extractor import source_quote_id
from spc.knowledge.expert_knowledge import context, source
from spc.knowledge.expert_knowledge import materializer as expert_materializer
from spc.knowledge.literature_knowledge import materializer as literature_materializer
from spc.models import SourceQuote
from spc.repositories import KnowledgeRepositories
from spc.serialization import file_sha256


KEY = "record-7"


@pytest.fixture(params=[
    pytest.param((source._put_reuse, "conflicting immutable expert record"), id="expert-source"),
    pytest.param((context._put_reuse, "conflicting immutable expert compilation record"), id="expert-context"),
    pytest.param((expert_materializer._put_reuse, "conflicting materialized expert record"), id="expert-materializer"),
    pytest.param((literature_materializer._put_reuse, "conflicting materialized record"), id="literature-materializer"),
])
def entry(request):
    return request.param


class ComparableRecord:
    def __init__(self, compare):
        self.compare = compare

    def __ne__(self, other):
        return self.compare(other)


def test_probe_missing_record(entry):
    write, _ = entry
    repo = Mock()
    repo.get.side_effect = FileNotFoundError(KEY)
    repo.put.return_value = sentinel.path

    assert write(repo, KEY, sentinel.record) is None
    assert repo.mock_calls == [call.get(KEY), call.put(KEY, sentinel.record)]


@pytest.mark.parametrize("different", [False, True], ids=["equal", "conflict"])
def test_probe_existing_record(entry, different):
    write, prefix = entry
    repo = Mock()
    incoming = ComparableRecord(Mock(side_effect=AssertionError("incoming record compared first")))
    repo.get.return_value = ComparableRecord(repo.compare)
    repo.compare.return_value = different

    if different:
        with pytest.raises(FileExistsError) as failure:
            write(repo, KEY, incoming)
        assert type(failure.value) is FileExistsError
        assert str(failure.value) == f"{prefix}: {KEY}"
    else:
        assert write(repo, KEY, incoming) is None
    assert repo.mock_calls == [call.get(KEY), call.compare(incoming)]


@pytest.mark.parametrize("error_type", [PermissionError, ValueError], ids=["permission", "binding"])
def test_probe_read_failure(entry, error_type):
    write, _ = entry
    repo = Mock()
    error = error_type("read failed")
    repo.get.side_effect = error
    with pytest.raises(error_type) as failure:
        write(repo, KEY, sentinel.record)
    assert failure.value is error
    assert str(failure.value) == "read failed"
    assert repo.mock_calls == [call.get(KEY)]


def test_probe_comparison_failure(entry):
    write, _ = entry
    repo = Mock()
    repo.get.return_value = ComparableRecord(repo.compare)
    error = FileNotFoundError("comparison failed")
    repo.compare.side_effect = error
    with pytest.raises(FileNotFoundError) as failure:
        write(repo, KEY, sentinel.record)
    assert failure.value is error
    assert str(failure.value) == "comparison failed"
    assert repo.mock_calls == [call.get(KEY), call.compare(sentinel.record)]


@pytest.mark.parametrize("error_type", [FileExistsError, FileNotFoundError], ids=["late-conflict", "write-missing"])
def test_probe_write_failure(entry, error_type):
    write, _ = entry
    repo = Mock()
    repo.get.side_effect = FileNotFoundError(KEY)
    error = error_type("write failed")
    repo.put.side_effect = error
    with pytest.raises(error_type) as failure:
        write(repo, KEY, sentinel.record)
    assert failure.value is error
    assert str(failure.value) == "write failed"
    assert repo.mock_calls == [call.get(KEY), call.put(KEY, sentinel.record)]


def test_missing_inside_custom_reader_still_writes(entry):
    write, _ = entry
    repo = Mock()
    repo.get.side_effect = repo.read_artifact
    repo.read_artifact.side_effect = FileNotFoundError("missing inside custom reader")
    assert write(repo, KEY, sentinel.record) is None
    assert repo.mock_calls == [call.get(KEY), call.read_artifact(KEY), call.put(KEY, sentinel.record)]


def make_quote(ref):
    return SourceQuote(
        quote_id=source_quote_id(ref, 0, 1, "x"), evidence_ref=ref, relative_start_offset=0,
        relative_end_offset=1, text="x", source_id="src", source_version="1",
        source_role="author", source_type="manuscript",
    )


def test_quote_read_errors_leave_files_alone(tmp_path):
    repo = KnowledgeRepositories(tmp_path, initialize=False).source_quotes
    quote = make_quote("ev")
    path = repo.put(quote.quote_id, quote)
    alias = repo.root / "alias.json"
    alias.write_bytes(path.read_bytes())
    bad = repo.root / "bad.json"
    bad.write_bytes(b"{")
    before = {p: file_sha256(p) for p in repo.root.iterdir()}

    with patch.object(repo, "put", wraps=repo.put) as put:
        for key, message in (
            ("../escape", "repository key is not a safe path component: '../escape'"),
            ("alias", "stored record ID does not match repository key"),
        ):
            with pytest.raises(ValueError) as failure:
                literature_materializer._put_reuse(repo, key, quote)
            assert type(failure.value) is ValueError
            assert str(failure.value) == message
        with pytest.raises(JSONDecodeError):
            literature_materializer._put_reuse(repo, "bad", quote)
        put.assert_not_called()
    assert {p: file_sha256(p) for p in repo.root.iterdir()} == before


def test_quote_key_mismatch_respects_read_before_put(tmp_path):
    repo = KnowledgeRepositories(tmp_path, initialize=False).source_quotes
    stored = make_quote("first")
    incoming = make_quote("other")
    path = repo.put(stored.quote_id, stored)
    digest = file_sha256(path)
    with patch.object(repo, "put", wraps=repo.put) as put:
        with pytest.raises(FileExistsError) as conflict:
            literature_materializer._put_reuse(repo, stored.quote_id, incoming)
        assert str(conflict.value) == f"conflicting materialized record: {stored.quote_id}"
        put.assert_not_called()
        with pytest.raises(ValueError) as invalid:
            literature_materializer._put_reuse(repo, "missing", incoming)
        assert str(invalid.value) == "repository key must equal record quote_id"
        put.assert_called_once_with("missing", incoming)
    assert file_sha256(path) == digest
    assert not (repo.root / "missing.json").exists()


@pytest.mark.parametrize("different", [False, True], ids=["equal-on-write", "conflict-on-write"])
def test_real_immutable_put_after_a_read_miss(tmp_path, different):
    repo = KnowledgeRepositories(tmp_path, initialize=False).source_quotes
    stored = make_quote("ev")
    path = repo.put(stored.quote_id, stored)
    digest = file_sha256(path)
    # source_version is part of the record, but not of the quote ID.
    incoming = stored.model_copy(update={"source_version": "2"}) if different else stored
    with patch.object(repo, "get", side_effect=FileNotFoundError(stored.quote_id)) as get:
        with patch.object(repo, "put", wraps=repo.put) as put:
            if different:
                with pytest.raises(FileExistsError) as failure:
                    literature_materializer._put_reuse(repo, stored.quote_id, incoming)
                assert str(failure.value) == f"refusing to overwrite different record: {path}"
            else:
                assert literature_materializer._put_reuse(repo, stored.quote_id, incoming) is None
            get.assert_called_once_with(stored.quote_id)
            put.assert_called_once_with(stored.quote_id, incoming)
    assert file_sha256(path) == digest


def test_reuse_preserves_archived_expert_chunk_bytes(tmp_path):
    repo = KnowledgeRepositories(tmp_path, initialize=False).expert_knowledge_chunks
    repo.root.mkdir(parents=True)
    with ZipFile(Path(__file__).parent / "fixtures" / "knowledge-chunks-b414df2.zip") as archive:
        members = [name for name in archive.namelist() if name.startswith("expert-pdf/")]
        assert len(members) == 2
        for member in members:
            path = repo.root / Path(member).name
            raw = archive.read(member)
            path.write_bytes(raw)
            chunk = repo.get(path.stem)
            with patch.object(repo, "put", wraps=repo.put) as put:
                assert context._put_reuse(repo, chunk.chunk_id, chunk) is None
                put.assert_not_called()
            assert file_sha256(path) == sha256(raw).hexdigest()
