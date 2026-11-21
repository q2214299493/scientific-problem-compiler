from __future__ import annotations

import csv
import json
from importlib import import_module
from pathlib import Path
from zipfile import ZipFile

import pytest

from spc import repositories as legacy
from spc.serialization import content_hash


def test_repository_exports_point_to_owning_module():
    inventory = Path(__file__).parents[1] / "docs" / "repository-inventory-ac25bd4.csv"
    with inventory.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            cls = getattr(legacy, row["class"])
            owner = import_module(f"spc.repository_components.{row['domain']}")
            assert cls is getattr(owner, row["class"])
            for method in filter(None, row["declared_methods"].split("; ")):
                assert hasattr(cls, method), (cls.__name__, method)


def test_knowledge_composition_shares_stores_with_scoped_callers(tmp_path):
    root = tmp_path / "knowledge"
    repos = legacy.KnowledgeRepositories(root, initialize=False)

    assert repos.literature.evidence_store is repos.expert.evidence_store is repos.evidence_store
    assert repos.collection.raw_literature_artifacts is repos.literature_artifacts.raw_artifacts
    assert repos.literature.canonical_text_artifacts is repos.literature_artifacts.canonical_texts
    assert repos.collection.collection_page_artifacts is repos.collection_page_artifacts
    assert repos.expert.expert_profiles is repos.expert_profiles
    assert repos.scientific_records.reported_results is repos.reported_results
    assert not hasattr(repos.literature, "expert_sources")
    assert not hasattr(repos.literature, "collection_imports")
    assert not hasattr(repos.literature, "reported_results")
    assert not hasattr(repos.collection, "expert_profiles")
    assert not hasattr(repos.expert, "literature_documents")
    assert not root.exists()


def test_baseline_repository_tree_is_readable_and_reusable(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "repositories-ac25bd4.zip"
    root = tmp_path / "knowledge"
    with ZipFile(fixture) as archive:
        expected = json.loads(archive.read("expected-hashes.json"))
        files = {name: archive.read(name) for name in archive.namelist() if name != "expected-hashes.json"}
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    repos = legacy.KnowledgeRepositories(root, initialize=False)
    for member, hashes in expected.items():
        assert [content_hash(record) for record in getattr(repos, member).list()] == hashes, member

    for page in repos.collection_pages.list():
        repos.collection_page_artifacts.verify_page_record(page)
    for resource in repos.collection_resources.list():
        assert repos.collection_resources.get(resource.discovered_resource_id, resource.content_hash) == resource
        repos.collection_resources.put(resource.discovered_resource_id, resource)
    profile, = repos.expert_profiles.list()
    repos.expert_profiles.put(profile.expert_id, profile)
    raw, = repos.raw_literature_artifacts.list()
    assert repos.raw_literature_artifacts.put(root / raw.stored_path, raw.literature_id) == raw
    text, = repos.canonical_text_artifacts.list()
    assert repos.canonical_text_artifacts.read_text(text.canonical_text_id)

    assert {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()} == files
    with pytest.raises(ValueError):
        repos.collection_resources.get("../outside", resource.content_hash)
