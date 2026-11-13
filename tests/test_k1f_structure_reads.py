from pathlib import Path

import pytest

from spc.knowledge import structure as structure_module
from spc.knowledge.structure import validate_document_structure
from spc.models import LiteratureRepresentationKind
from test_literature_knowledge import (
    _fingerprinted_structure_index,
    setup_html_literature,
    setup_pdf_literature,
)


@pytest.fixture(params=[setup_html_literature, setup_pdf_literature], ids=["html", "pdf"])
def structure_read_case(request, tmp_path):
    values = request.param(tmp_path)
    repositories, store = values[:2]
    artifact = values[-1].artifact
    canonical_repository = (
        repositories.canonical_text_artifacts
        if artifact.representation_kind == LiteratureRepresentationKind.PDF
        else repositories.canonical_html_text_artifacts
    )
    return repositories, store, artifact, canonical_repository


def test_trusted_structure_reads_canonical_record_and_text_once(
    structure_read_case, monkeypatch: pytest.MonkeyPatch
):
    repositories, store, artifact, canonical_repository = structure_read_case
    directory = canonical_repository.root / artifact.canonical_text_id
    paths = {directory / "metadata.json": "metadata", directory / "content.txt": "text"}
    reads = {"metadata": 0, "text": 0}
    original_open = Path.open

    def counted_open(path, *args, **kwargs):
        if path in paths:
            reads[paths[path]] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    ordinary = validate_document_structure(artifact.structure_id, repositories, store)
    ordinary_reads = dict(reads)
    reads.update(metadata=0, text=0)
    trusted = validate_document_structure(artifact.structure_id, repositories, store, _file_hashes={})
    print({"kind": artifact.representation_kind.value, "ordinary": ordinary_reads, "trusted": reads})
    assert trusted == ordinary == artifact
    # Representation/source verification remains; only the later redundant
    # canonical get/read_text/get/open sequence is removed.
    assert ordinary_reads["metadata"] - reads["metadata"] == 1
    assert ordinary_reads["text"] - reads["text"] == 2


def test_trusted_structure_text_is_read_afresh_on_later_validation(structure_read_case):
    repositories, store, artifact, canonical_repository = structure_read_case
    validate_document_structure(artifact.structure_id, repositories, store, _file_hashes={})
    path = canonical_repository.root / artifact.canonical_text_id / "content.txt"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="canonical text"):
        validate_document_structure(artifact.structure_id, repositories, store, _file_hashes={})


@pytest.mark.parametrize("scope", ["instance", "class"])
@pytest.mark.parametrize("method", ["get", "read_text"])
def test_trusted_structure_keeps_custom_canonical_readers(
    structure_read_case, monkeypatch: pytest.MonkeyPatch, scope: str, method: str
):
    repositories, store, artifact, canonical_repository = structure_read_case
    owner = canonical_repository if scope == "instance" else type(canonical_repository)
    original = getattr(owner, method)
    calls = 0

    def custom_reader(*args, **kwargs):
        nonlocal calls
        calls += 1
        if method == "read_text":
            raise ValueError("custom canonical text gate")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, method, custom_reader)
    if method == "read_text":
        with pytest.raises(ValueError, match="custom canonical text gate"):
            validate_document_structure(artifact.structure_id, repositories, store, _file_hashes={})
        assert calls == 1
    else:
        assert validate_document_structure(
            artifact.structure_id, repositories, store, _file_hashes={}
        ) == artifact
        assert calls == 3


def test_trusted_closing_check_rejects_text_changed_after_structure_validation(
    structure_read_case, monkeypatch: pytest.MonkeyPatch
):
    repositories, store, artifact, canonical_repository = structure_read_case
    selection = repositories.document_structure_selections.resolve_current(artifact.representation_id)
    index = _fingerprinted_structure_index(repositories, store)
    path = canonical_repository.root / artifact.canonical_text_id / "content.txt"
    original = structure_module._validate_structure_records

    def change_after_checks(*args, **kwargs):
        result = original(*args, **kwargs)
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"changed after checks")
        return result

    monkeypatch.setattr(structure_module, "_validate_structure_records", change_after_checks)
    index._validate_structure_once(selection, repositories, store)
    with pytest.raises(ValueError, match="canonical text"):
        index.revalidate_structures()
