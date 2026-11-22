"""Golden records captured with the unmodified builders at b414df2."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from spc.knowledge.expert_knowledge import (
    build_expert_knowledge_chunks,
    resolve_expert_knowledge_input,
)
from spc.knowledge.expert_knowledge.repositories import ExpertKnowledgeChunkRepository
from spc.knowledge.literature_knowledge import (
    build_literature_knowledge_chunks,
    resolve_literature_knowledge_input,
)
from spc.knowledge.literature_knowledge.repositories import LiteratureKnowledgeChunkRepository
from spc.knowledge.text_chunking import bounded_text_ranges
from spc.repositories import KnowledgeRepositories

from test_literature_knowledge import HTML, _curate, setup_html_literature


FIXTURES = Path(__file__).parent / "fixtures"
LONG_TEXT = "中e\u0301🙂A" * 1620
CHUNK_ARCHIVE = FIXTURES / "knowledge-chunks-b414df2.zip"


@pytest.mark.parametrize(("text", "start", "end", "offsets"), [
    ("", 0, 0, []),
    ("\t\u2003\n", 0, 3, []),
    ("  一段🙂 ", 0, 6, [(2, 5)]),
    ("! \tA\n\nB  ?", 1, 9, [(3, 7)]),
    ("abcdefgh", 0, 8, [(0, 8)]),
    ("abcdefghi", 0, 9, [(0, 8), (8, 9)]),
    ("abcd efghi", 0, 10, [(0, 8), (8, 10)]),  # A space exactly halfway is ineligible.
    ("abcde fghi", 0, 10, [(0, 5), (6, 10)]),
    ("abcde x\nyz", 0, 10, [(0, 7), (8, 10)]),
    ("abcde\u2003fghi", 0, 10, [(0, 8), (8, 10)]),
    ("abcdefge\u0301中甲", 0, 11, [(0, 8), (8, 11)]),
    ("        XY", 0, 10, [(9, 10)]),  # Preserve advancement after an all-whitespace window.
])
def test_bounded_text_ranges_match_baseline(text, start, end, offsets):
    assert list(bounded_text_ranges(text, start, end, max_characters=8)) == offsets


def test_range_limit_must_be_positive():
    with pytest.raises(ValueError, match="max_characters must be positive"):
        list(bounded_text_ranges("x", 0, 1, max_characters=0))


def extract_fixture_store(archive_path, directory, tmp_path):
    with ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if member.startswith(directory + "/"):
                archive.extract(member, tmp_path)
    return KnowledgeRepositories(tmp_path / directory)


def expert_chunks(tmp_path, source):
    # The original ingestion timestamp participates in the expert source hash.
    repositories = extract_fixture_store(CHUNK_ARCHIVE, f"inputs/expert-{source}", tmp_path)
    (source_record,) = repositories.expert_sources.list()
    compilation = resolve_expert_knowledge_input(source_record.expert_source_id, repositories)
    assert (compilation.chunk_policy_id, compilation.chunk_policy_version) == (
        "spc-expert-source-offset-chunks", "1.0.0"
    )
    return build_expert_knowledge_chunks(compilation, repositories)


def literature_chunks(tmp_path, source):
    if source == "long":
        html = (
            HTML.split(b"<body>", 1)[0] + b"<body><article><p>"
            + LONG_TEXT.encode("utf-8") + b"</p></article></body></html>"
        )
        repositories, _, outcome, _ = setup_html_literature(tmp_path, body=html)
        literature_id = outcome.literature_id
    else:
        repositories = extract_fixture_store(FIXTURES / "knowledge-01a-3f53101.zip", source, tmp_path)
        (document,) = repositories.literature_documents.list()
        literature_id = document.literature_id
        selection = repositories.literature_representation_selections.resolve_current(literature_id)
        _curate(repositories, "literature_document", document)
        _curate(repositories, "literature_representation_selection", selection)
    compilation = resolve_literature_knowledge_input(literature_id, repositories)
    assert (compilation.chunk_policy_id, compilation.chunk_policy_version) == (
        "spc-structure-block-bounded-chunks", "1.0.0"
    )
    return build_literature_knowledge_chunks(compilation, repositories)


@pytest.mark.parametrize(("domain", "source", "count"), [
    ("expert", "pdf", 2),
    ("expert", "long", 5),
    ("literature", "html", 9),
    ("literature", "pdf", 3),
    ("literature", "long", 2),
])
def test_rebuild_matches_ordered_historical_chunks(tmp_path, domain, source, count):
    if domain == "expert":
        store = ExpertKnowledgeChunkRepository(tmp_path / "baseline")
        rebuilt = expert_chunks(tmp_path, source)
    else:
        store = LiteratureKnowledgeChunkRepository(tmp_path / "baseline")
        rebuilt = literature_chunks(tmp_path, source)

    # ZIP member order records the old builder's return order; read its original
    # JSON bytes through the repository independently of the current builder.
    expected = []
    store.root.mkdir(parents=True, exist_ok=True)
    with ZipFile(CHUNK_ARCHIVE) as archive:
        for member in archive.namelist():
            if member.startswith(f"{domain}-{source}/"):
                path = store.root / Path(member).name
                path.write_bytes(archive.read(member))
                expected.append(store.get(path.stem))
    assert len(expected) == count
    assert rebuilt == tuple(expected)
