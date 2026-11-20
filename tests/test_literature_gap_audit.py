from __future__ import annotations

import json
from hashlib import sha256

import pytest

from scripts.literature_gap_audit import (
    build_report,
    load_roles,
    local_supplement_manifest,
    normalize_doi,
    render_markdown,
)
from spc.models import LiteratureDocument, literature_identity_id
from spc.repositories import KnowledgeRepositories
from spc.serialization import content_hash


@pytest.mark.parametrize(
    ("raw", "normalized"),
    (
        ("10.1021/ABC", "10.1021/abc"),
        ("https://doi.org/10.1038/TEST", "10.1038/test"),
        ("doi:10.1000/Example", "10.1000/example"),
    ),
)
def test_normalize_doi_for_inventory_dedup(raw: str, normalized: str) -> None:
    assert normalize_doi(raw) == normalized


def test_role_inventory_requires_exact_ids_and_supported_roles(tmp_path) -> None:
    path = tmp_path / "roles.json"
    path.write_text(json.dumps({"paper-a": "fe_ft_experimental"}), encoding="utf-8")
    assert load_roles(path, {"paper-a"}) == {"paper-a": "fe_ft_experimental"}
    with pytest.raises(ValueError, match="exactly"):
        load_roles(path, {"paper-a", "paper-b"})
    path.write_text(json.dumps({"paper-a": "scientifically_approved"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_roles(path, {"paper-a"})


def test_local_supplement_manifest_binds_original_bytes(tmp_path) -> None:
    path = tmp_path / "10.1000_test_si.pdf"
    path.write_bytes(b"%PDF-audit-only")
    assert local_supplement_manifest(tmp_path) == {path.name: sha256(b"%PDF-audit-only").hexdigest()}
    path.write_bytes(b"%PDF-changed")
    assert local_supplement_manifest(tmp_path)[path.name] == sha256(b"%PDF-changed").hexdigest()


def test_local_supplement_manifest_requires_existing_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        local_supplement_manifest(tmp_path / "missing")


def test_unselected_literature_is_reported_as_gap(tmp_path) -> None:
    root = tmp_path / "knowledge"
    repositories = KnowledgeRepositories(root)
    title = "Unselected iron catalyst study"
    authors = ("A. Researcher",)
    doi = "10.1000/unselected"
    identity = {
        "literature_id": literature_identity_id(title=title, authors=authors, year=2025, doi=doi),
        "title": title,
        "authors": authors,
        "year": 2025,
        "journal": "Catalysis Journal",
        "doi": doi,
        "url": "https://doi.org/10.1000/unselected",
        "domain": "fischer_tropsch",
        "topics": (),
        "keywords": (),
        "raw_artifact_ref": "artifacts/unselected.pdf",
        "canonical_text_ref": "texts/unselected.txt",
        "source_id": "source-unselected",
        "source_version": "v1",
        "citation_refs": (),
    }
    document = LiteratureDocument(**identity, content_hash=content_hash(identity))
    repositories.literature_documents.put(document.literature_id, document)

    report = build_report(root, target_count=1)

    row = report["papers"][0]
    assert row["representation_kind"] is None
    assert row["representation_selection_id"] is None
    assert row["representation_curation"] == "missing"
    assert "REPRESENTATION_SELECTION_MISSING" in row["gap_flags"]
    assert report["summary"]["pending_curations_by_type"]["literature_representation_selection"] == 0


def test_markdown_does_not_present_role_as_trust_decision() -> None:
    report = {
        "inventory_basis_hash": "a" * 64,
        "summary": {
            "literature_documents": 1,
            "target_count": 120,
            "source_trusted": 0,
            "source_authority_pending": 1,
            "current_k1f_compiled_papers": 0,
            "papers_with_accepted_science": 0,
            "papers_with_local_supplement_match": 0,
            "trusted_knowledge_relations": 0,
            "knowledge_relations": 0,
        },
        "papers": [
            {
                "doi": "10.1000/test",
                "title": "Fe <sub>3</sub> C | test",
                "year": 2025,
                "role_proposal": "fe_ft_experimental",
                "source_trusted": False,
                "current_k1f_compilations": 0,
                "scientific_records": {"source_claim": {"trusted": 0}},
                "local_supplement_file_matches": [],
                "gap_flags": ["SOURCE_AUTHORITY_PENDING"],
            }
        ],
    }
    markdown = render_markdown(report)
    assert "not a curation decision" in markdown
    assert "Fe 3 C \\| test" in markdown
    assert "SOURCE_AUTHORITY_PENDING" in markdown
