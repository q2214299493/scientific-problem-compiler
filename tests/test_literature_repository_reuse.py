"""Operation-scoped storage bindings for literature compilation."""

from collections import Counter
from unittest.mock import Mock

import pytest

from spc.knowledge import literature_knowledge
from spc.knowledge.literature_knowledge import (
    LiteratureKnowledgeCompiler,
    MockLiteratureKnowledgeProvider,
    validate_literature_knowledge_chunk,
)
from spc.knowledge.literature_knowledge import repositories as stores
from spc.repositories import IdentityBoundRepository, KnowledgeRepositories
from spc.serialization import content_hash, file_sha256

from test_knowledge_chunking import FIXTURES, extract_fixture_store
from test_literature_knowledge import _curate


REPOSITORY_TYPES = (
    stores.LiteratureKnowledgeCompilationInputRepository,
    stores.LiteratureKnowledgeChunkRepository,
    stores.LiteratureKnowledgeProposalSetRepository,
    stores.LiteratureKnowledgeGroundingRepository,
    stores.LiteratureKnowledgeCompilationRepository,
)


def historical_literature(tmp_path):
    repos = extract_fixture_store(FIXTURES / "knowledge-01a-3f53101.zip", "html", tmp_path)
    (document,) = repos.literature_documents.list()
    selection = repos.literature_representation_selections.resolve_current(document.literature_id)
    _curate(repos, "literature_document", document)
    _curate(repos, "literature_representation_selection", selection)
    return repos, document.literature_id


def test_compile_reuses_only_its_assembled_handles(tmp_path, monkeypatch):
    constructed = []
    original_init = IdentityBoundRepository.__init__

    def counted(handle, root, model_type, identity_field):
        original_init(handle, root, model_type, identity_field)
        if type(handle) in REPOSITORY_TYPES:
            constructed.append(handle)

    monkeypatch.setattr(IdentityBoundRepository, "__init__", counted)
    first, literature_id = historical_literature(tmp_path)
    compiler = LiteratureKnowledgeCompiler()
    provider = MockLiteratureKnowledgeProvider()

    def compile_once(repos):
        handles = tuple(constructed)
        assert Counter(map(type, handles)) == Counter(REPOSITORY_TYPES)
        for handle in handles:
            assert getattr(literature_knowledge, type(handle).__name__) is type(handle)
            assert handle is getattr(repos, handle.root.name)
        result = compiler.compile(literature_id, provider, repos)
        assert tuple(constructed) == handles
        assert content_hash(list(result.chunks)) == "754a705e04acd2fd036b4c2a1eef8ad31ca18a5fe84c34914d54655ac3fba3db"
        assert result.materialized.record.content_hash == "a776329d547251e34fa2ba8fb00eae8e92dd3cbeb7653ff5dbf719b3750a4e11"
        files = {
            path.relative_to(repos.root): file_sha256(path)
            for handle in handles for path in handle.root.glob("*.json")
        }
        assert len(files) == 13
        return handles, files

    first_handles, written = compile_once(first)
    constructed.clear()
    second_handles, reused = compile_once(KnowledgeRepositories(first.root, initialize=False))
    assert set(first_handles).isdisjoint(second_handles)
    assert reused == written


@pytest.mark.parametrize("difference", [
    "root", "root_case", "model", "id_field", "writer", "subclass", "other_owner",
])
def test_compiler_uses_native_chunk_storage_for_non_equivalent_handles(tmp_path, difference):
    repos, literature_id = historical_literature(tmp_path)
    native = repos.literature_knowledge_chunks
    chunk_type = stores.LiteratureKnowledgeChunkRepository
    other_root = tmp_path / "other"
    if difference == "root":
        native.root = other_root / "literature_knowledge_chunks"
    elif difference == "root_case":
        native.root = native.root.with_name(native.root.name.upper())
    elif difference == "model":
        native.model_type = literature_knowledge.LiteratureKnowledgeCompilationInput
    elif difference == "id_field":
        native.identity_field = "literature_id"
    elif difference == "writer":
        native.put = Mock(side_effect=AssertionError("custom writer reached"))
    elif difference == "subclass":
        class AlternateChunkRepository(chunk_type):
            def put(self, key, model):
                raise AssertionError("alternate writer reached")

        native = AlternateChunkRepository(repos.root)
        repos.literature_knowledge_chunks = repos.literature.literature_knowledge_chunks = native
    else:
        native = KnowledgeRepositories(repos.root, initialize=False).literature_knowledge_chunks
        repos.literature_knowledge_chunks = native

    assert chunk_type.for_operation(repos) is not native
    result = LiteratureKnowledgeCompiler().compile(literature_id, MockLiteratureKnowledgeProvider(), repos)
    stored = chunk_type(repos.root)
    assert tuple(stored.get(chunk.chunk_id) for chunk in result.chunks) == result.chunks
    assert not other_root.exists()


def test_shadow_reader_cannot_satisfy_the_compilers_disk_check(tmp_path):
    repos, literature_id = historical_literature(tmp_path)
    result = LiteratureKnowledgeCompiler().compile(literature_id, MockLiteratureKnowledgeProvider(), repos)
    chunk = result.chunks[0]
    empty = KnowledgeRepositories(tmp_path / "empty", initialize=False)
    reader = Mock(return_value=chunk)
    empty.literature_knowledge_chunks.get = reader
    with pytest.raises(FileNotFoundError) as error:
        validate_literature_knowledge_chunk(chunk, result.compilation_input, empty)
    assert str(empty.root / "literature_knowledge_chunks" / f"{chunk.chunk_id}.json") == error.value.filename
    reader.assert_not_called()
    assert not empty.root.exists()
