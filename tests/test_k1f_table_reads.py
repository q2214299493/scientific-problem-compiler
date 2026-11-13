from pathlib import Path
from types import SimpleNamespace

import pytest

from spc.knowledge.literature_knowledge import (
    LiteratureClaimProposal,
    LiteratureKnowledgeCompiler,
    LiteratureKnowledgeLLMResponse,
    LiteratureQuoteProposal,
    curate_knowledge_record,
)
from spc.knowledge.literature_knowledge import context as context_module
from spc.knowledge.k1f_authority import K1FCompilationAuthorityIndex
from spc.knowledge.trust import TrustedKnowledgeValidator
from spc.models import CurationStatus, EpistemicStatus
from spc.repositories import ModelRepository
from test_literature_knowledge import StaticProvider, _fingerprinted_structure_index, setup_html_literature


@pytest.fixture
def table_compilation(tmp_path):
    repositories, store, outcome, structure = setup_html_literature(tmp_path)

    def response(chunks):
        cells = [chunk for chunk in chunks if chunk.table_context is not None
                 and chunk.table_context.get("cell_id") and not chunk.table_context.get("is_header")]
        assert len(cells) >= 2
        quotes = tuple(LiteratureQuoteProposal(
            quote_key=f"cell-{i}", chunk_id=chunk.chunk_id,
            block_id=chunk.block_refs[0], exact_text=chunk.text,
        ) for i, chunk in enumerate(cells[:2]))
        return LiteratureKnowledgeLLMResponse(
            quote_proposals=quotes,
            claim_proposals=(LiteratureClaimProposal(
                claim_key="table-cells", text="Two separately located table cells.",
                claim_type="source_statement", quote_keys=tuple(q.quote_key for q in quotes),
                claim_strength="source-reported", epistemic_status=EpistemicStatus.SOURCE_REPORTED,
            ),),
        )

    compiled = LiteratureKnowledgeCompiler().compile(
        outcome.literature_id, StaticProvider(response), repositories, store
    ).materialized
    assert len(compiled.groundings) == 2
    assert len(compiled.source_claims) == 1
    return repositories, store, structure, compiled


def test_trusted_inventory_reads_table_ownership_records_once(table_compilation, monkeypatch):
    repositories, store, structure, compiled = table_compilation
    original_open = Path.open
    reads = {"table": 0, "cell": 0}

    def count_open(path, *args, **kwargs):
        if path.parent == repositories.table_structures.root:
            reads["table"] += 1
        elif path.parent == repositories.table_cell_structures.root:
            reads["cell"] += 1
        return original_open(path, *args, **kwargs)

    index = _fingerprinted_structure_index(repositories, store)
    monkeypatch.setattr(Path, "open", count_open)
    assert index.classify("source_claim", compiled.source_claims[0].claim_id).value == "current"
    cells = len(structure.table_cells)
    print({"cells": cells, "authority_reads": dict(reads)})
    # Initial full validation, two independent locator checks, one ownership scan.
    assert reads == {"table": 4, "cell": cells * 2 + 2}
    initial = dict(reads)
    index.revalidate_structures()
    # Closing manifest construction and raw-byte checks still reopen all files.
    assert reads == {"table": initial["table"] + 2, "cell": initial["cell"] + cells}


def test_table_records_do_not_cross_trusted_reads(table_compilation, monkeypatch):
    repositories, store, _structure, compiled = table_compilation
    original = context_module.resolve_table_ownership
    seen = []

    def observe(*args, **kwargs):
        seen.append(kwargs.get("_table_records"))
        return original(*args, **kwargs)

    monkeypatch.setattr(context_module, "resolve_table_ownership", observe)
    for _ in range(2):
        index = _fingerprinted_structure_index(repositories, store)
        assert index.classify("source_claim", compiled.source_claims[0].claim_id).value == "current"
        index.revalidate_structures()
    assert len(seen) == 4
    assert seen[0] is seen[1] and seen[2] is seen[3]
    assert seen[0] is not None and seen[2] is not None and seen[0] is not seen[2]


def test_index_without_closing_fingerprints_keeps_fresh_ownership_reads(table_compilation, monkeypatch):
    repositories, store, structure, compiled = table_compilation
    current = TrustedKnowledgeValidator(repositories, store).resolve_current_curations()
    index = K1FCompilationAuthorityIndex(repositories, store, current)
    reads = {"table": 0, "cell": 0}
    original = Path.open

    def counted(path, *args, **kwargs):
        if path.parent == repositories.table_structures.root:
            reads["table"] += 1
        elif path.parent == repositories.table_cell_structures.root:
            reads["cell"] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    assert index.classify("source_claim", compiled.source_claims[0].claim_id).value == "current"
    assert reads == {"table": 5, "cell": len(structure.table_cells) * 3 + 2}


@pytest.mark.parametrize("target", ["table", "uncited_cell"])
def test_trusted_return_rejects_structure_changed_after_ownership_reuse(
    table_compilation, monkeypatch, target
):
    repositories, store, structure, compiled = table_compilation
    claim = compiled.source_claims[0]
    curate_knowledge_record(
        repositories, target_type="source_claim", target_id=claim.claim_id,
        status=CurationStatus.ACCEPTED, curator_id="test-reviewer",
        rationale="Accept the exact table fixture claim.",
    )
    table = structure.tables[0]
    cited = {repositories.structured_evidence_locators.get(g.locator_id).table_cell_id
             for g in compiled.groundings}
    if target == "table":
        path = repositories.table_structures.root / f"{table.table_id}.json"
    else:
        uncited = next(cell_id for cell_id in table.cell_refs if cell_id not in cited)
        path = repositories.table_cell_structures.root / f"{uncited}.json"
    original = context_module.resolve_table_ownership
    changed = False

    def mutate_after_first_scan(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if kwargs.get("_table_records") is not None and not changed:
            # Valid JSON/identical parsed content must not reset the bound byte hash.
            path.write_bytes(path.read_bytes() + b"\n")
            changed = True
        return result

    monkeypatch.setattr(context_module, "resolve_table_ownership", mutate_after_first_scan)
    with pytest.raises(ValueError, match="structure changed during trusted read"):
        TrustedKnowledgeValidator(repositories, store).validate()
    assert changed


@pytest.mark.parametrize("kind", ["table", "cell"])
@pytest.mark.parametrize("scope", ["instance", "class"])
def test_custom_structure_getters_keep_the_original_ownership_path(
    table_compilation, monkeypatch, kind, scope
):
    repositories, store, structure, compiled = table_compilation
    repository = repositories.table_structures if kind == "table" else repositories.table_cell_structures
    owner = repository if scope == "instance" else type(repository)
    original = getattr(owner, "get")
    calls = 0

    def custom_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    index = _fingerprinted_structure_index(repositories, store)
    monkeypatch.setattr(owner, "get", custom_get)
    assert index.classify("source_claim", compiled.source_claims[0].claim_id).value == "current"
    assert calls == (5 if kind == "table" else len(structure.table_cells) * 3 + 2)


def _owned_block_and_cache(table_compilation):
    repositories, _store, structure, compiled = table_compilation
    block = repositories.document_structure_blocks.get(compiled.groundings[0].block_id)
    cache = {}
    assert context_module.resolve_table_ownership(
        block, structure.artifact, repositories, _table_records=cache
    ) is not None
    return block, cache


@pytest.mark.parametrize("field", ["start_offset", "text_hash"])
def test_cached_records_still_match_each_blocks_exact_text_region(table_compilation, field):
    repositories, _store, structure, _compiled = table_compilation
    block, cache = _owned_block_and_cache(table_compilation)
    fields = block.model_dump(mode="python")
    fields[field] = fields[field] + 1 if field == "start_offset" else "0" * 64
    assert context_module.resolve_table_ownership(
        SimpleNamespace(**fields), structure.artifact, repositories, _table_records=cache
    ) is None


def test_cached_records_keep_ambiguous_ownership_rejection(table_compilation):
    repositories, _store, structure, _compiled = table_compilation
    block, cache = _owned_block_and_cache(table_compilation)
    fields = block.model_dump(mode="python")
    fields["block_id"] = structure.tables[0].caption_block_ref
    with pytest.raises(ValueError, match="ambiguous table ownership"):
        context_module.resolve_table_ownership(
            SimpleNamespace(**fields), structure.artifact, repositories, _table_records=cache
        )


def test_cached_records_keep_selected_structure_binding(table_compilation):
    repositories, _store, structure, _compiled = table_compilation
    block, cache = _owned_block_and_cache(table_compilation)
    fields = structure.artifact.model_dump(mode="python")
    fields["structure_id"] = "another-structure"
    with pytest.raises(ValueError, match="not bound to the selected structure"):
        context_module.resolve_table_ownership(
            block, SimpleNamespace(**fields), repositories, _table_records=cache
        )


def test_cached_records_do_not_skip_canonical_integrity_checks(table_compilation):
    repositories, _store, structure, _compiled = table_compilation
    block, cache = _owned_block_and_cache(table_compilation)
    path = repositories.canonical_html_text_artifacts.root / structure.artifact.canonical_text_id / "content.txt"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"changed canonical bytes")
    with pytest.raises(ValueError, match="canonical text"):
        context_module.resolve_table_ownership(block, structure.artifact, repositories, _table_records=cache)


@pytest.mark.parametrize("target", ["model_get", "identity_check", "identity_field"])
def test_cached_records_do_not_bypass_changed_repository_checks(
    table_compilation, monkeypatch, target
):
    repositories, _store, structure, _compiled = table_compilation
    block, cache = _owned_block_and_cache(table_compilation)

    def reject(*args, **kwargs):
        raise ValueError("custom structure record gate")

    if target == "model_get":
        monkeypatch.setattr(ModelRepository, "get", reject)
    elif target == "identity_check":
        monkeypatch.setattr(repositories.table_structures, "_record_id", reject)
    else:
        monkeypatch.setattr(repositories.table_structures, "identity_field", "unknown_identity")
    with pytest.raises(ValueError):
        context_module.resolve_table_ownership(block, structure.artifact, repositories, _table_records=cache)
