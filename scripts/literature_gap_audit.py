"""Read-only, provenance-aware coverage inventory for a local SPC literature store.

This report is an audit queue, never a curation or scientific authority record.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from spc.knowledge.k1f_authority import (
    K1FScientificAuthorityStatus,
    validate_k1f_compilation_authority,
)
from spc.knowledge.trust import TrustedKnowledgeValidator
from spc.repositories import KnowledgeRepositories
from spc.serialization import content_hash


SCIENTIFIC_FIELDS = {
    "source_claim": "source_claim_ids",
    "method_fact": "method_fact_ids",
    "model_fact": "model_fact_ids",
    "reported_result": "reported_result_ids",
}
CORE_BIBLIOGRAPHIC_FIELDS = ("title", "authors", "year", "journal", "doi")
ALLOWED_ROLES = {
    "fe_ft_experimental",
    "fe_ft_model_or_computational",
    "fe_ft_special_process",
    "non_fe_method_comparator",
    "adjacent_reaction",
    "general_methodology",
    "excluded_by_user",
    "needs_role_review",
}


def normalize_doi(value: str) -> str:
    doi = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi


def load_roles(path: Path | None, literature_ids: set[str]) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != literature_ids:
        raise ValueError("role inventory must contain exactly the current literature IDs")
    if any(role not in ALLOWED_ROLES for role in payload.values()):
        raise ValueError("role inventory contains an unsupported role")
    return payload


def _status(curations: dict, record_type: str, record_id: str) -> str:
    curation = curations.get((record_type, record_id))
    return curation.status.value if curation is not None else "uncurated"


def local_supplement_manifest(directory: Path | None) -> dict[str, str]:
    """Hash local SI bytes for audit only; this is not evidence-store ingestion."""
    if directory is None:
        return {}
    if not directory.is_dir():
        raise ValueError(f"supplement directory does not exist: {directory}")
    manifest = {}
    for path in sorted(directory.glob("*.pdf")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe supplement file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest[path.name] = digest.hexdigest()
    return manifest


def build_report(
    knowledge_dir: Path,
    *,
    target_count: int,
    role_path: Path | None = None,
    supplement_dir: Path | None = None,
) -> dict:
    if target_count < 1:
        raise ValueError("target_count must be positive")
    repositories = KnowledgeRepositories(knowledge_dir.resolve())
    evidence_store = repositories.evidence_store
    validator = TrustedKnowledgeValidator(repositories, evidence_store)
    curations = dict(validator.resolve_current_curations())
    trusted = validator.validate().trusted_records
    documents = sorted(repositories.literature_documents.list(), key=lambda item: item.literature_id)
    roles = load_roles(role_path, {item.literature_id for item in documents})
    inputs = {item.compilation_input_id: item for item in repositories.literature_knowledge_inputs.list()}
    compilations_by_literature: dict[str, list] = defaultdict(list)
    for compilation in repositories.literature_knowledge_compilations.list():
        compilation_input = inputs.get(compilation.compilation_input_id)
        if compilation_input is None:
            raise ValueError(f"missing K1F compilation input: {compilation.compilation_input_id}")
        if compilation.compilation_input_hash != compilation_input.content_hash:
            raise ValueError(f"K1F compilation input hash mismatch: {compilation.compilation_id}")
        compilations_by_literature[compilation_input.literature_id].append(compilation)

    doi_ids: dict[str, str] = {}
    rows: list[dict] = []
    supplement_manifest = local_supplement_manifest(supplement_dir)
    for document in documents:
        doi = normalize_doi(document.doi or "")
        if doi:
            prior = doi_ids.setdefault(doi, document.literature_id)
            if prior != document.literature_id:
                raise ValueError(f"duplicate DOI in literature store: {doi}")
        try:
            selection = repositories.literature_representation_selections.resolve_current(document.literature_id)
        except FileNotFoundError:
            selection = None
        representation = None
        structure = None
        structure_status = "missing"
        if selection is not None:
            representation = repositories.literature_representation_refs.get(selection.representation_id)
            if (
                representation.literature_id != document.literature_id
                or representation.content_hash != selection.representation_hash
            ):
                raise ValueError(f"representation binding mismatch: {document.literature_id}")
            try:
                structure = repositories.document_structure_selections.resolve_current(representation.representation_id)
                structure_artifact = repositories.document_structure_artifacts.get(structure.structure_id)
                if (
                    structure_artifact.content_hash != structure.structure_hash
                    or structure_artifact.literature_id != document.literature_id
                    or structure_artifact.representation_id != representation.representation_id
                    or structure.representation_hash != representation.content_hash
                ):
                    raise ValueError(f"structure selection binding mismatch: {document.literature_id}")
                structure_status = "selected_metadata_bound"
            except FileNotFoundError:
                pass

        all_compilations = sorted(
            compilations_by_literature.get(document.literature_id, ()),
            key=lambda item: item.compilation_id,
        )
        current_compilations = []
        for compilation in all_compilations:
            authority = validate_k1f_compilation_authority(compilation, repositories, evidence_store, curations)
            if authority == K1FScientificAuthorityStatus.CURRENT:
                current_compilations.append(compilation)
        scientific_ids: dict[str, set[str]] = {kind: set() for kind in SCIENTIFIC_FIELDS}
        rejection_codes: Counter[str] = Counter()
        for compilation in current_compilations:
            for kind, field in SCIENTIFIC_FIELDS.items():
                scientific_ids[kind].update(getattr(compilation, field))
            rejection_codes.update(item.rejection_code for item in compilation.rejected_proposals)
        scientific_counts = {
            kind: {
                "present": len(ids),
                "trusted": sum((kind, record_id) in trusted for record_id in ids),
                "pending_review": sum(
                    _status(curations, kind, record_id) in {"uncurated", "machine_extracted", "human_reviewed"}
                    for record_id in ids
                ),
            }
            for kind, ids in scientific_ids.items()
        }
        core_missing = [field for field in CORE_BIBLIOGRAPHIC_FIELDS if not getattr(document, field)]
        supplement_slug = doi.replace("/", "_")
        matched_supplements = [name for name in supplement_manifest if name.startswith(supplement_slug + "_")]
        source_trusted = ("literature_document", document.literature_id) in trusted
        flags = []
        if not source_trusted:
            flags.append("SOURCE_AUTHORITY_PENDING")
        if selection is None:
            flags.append("REPRESENTATION_SELECTION_MISSING")
        if structure is None:
            flags.append("STRUCTURE_SELECTION_MISSING")
        if not current_compilations:
            flags.append("CURRENT_K1F_EXTRACTION_MISSING")
        if not any(value["trusted"] for value in scientific_counts.values()):
            flags.append("NO_ACCEPTED_SCIENTIFIC_RECORD")
        if not document.topics or not document.keywords:
            flags.append("OPTIONAL_TOPIC_METADATA_EMPTY")
        if representation is not None and representation.representation_kind.value == "html":
            flags.append("HTML_FULLTEXT_SCOPE_REVIEW_REQUIRED")
        if roles.get(document.literature_id) == "excluded_by_user":
            flags.append("EXCLUDED_BY_USER")
        rows.append(
            {
                "literature_id": document.literature_id,
                "doi": doi,
                "title": document.title,
                "authors": list(document.authors),
                "year": document.year,
                "journal": document.journal,
                "role_proposal": roles.get(document.literature_id, "needs_role_review"),
                "core_bibliography_missing": core_missing,
                "topics_present": bool(document.topics),
                "keywords_present": bool(document.keywords),
                "representation_kind": representation.representation_kind.value if representation else None,
                "representation_selection_id": selection.selection_id if selection else None,
                "literature_curation": _status(curations, "literature_document", document.literature_id),
                "representation_curation": _status(
                    curations, "literature_representation_selection", selection.selection_id
                ) if selection else "missing",
                "source_trusted": source_trusted,
                "structure_selection": structure_status,
                "structure_selection_id": structure.selection_id if structure else None,
                "current_k1f_compilations": len(current_compilations),
                "historical_k1f_compilations": len(all_compilations) - len(current_compilations),
                "scientific_records": scientific_counts,
                "rejected_proposals": dict(sorted(rejection_codes.items())),
                "local_supplement_file_matches": matched_supplements,
                "local_supplement_sha256": {name: supplement_manifest[name] for name in matched_supplements},
                "gap_flags": flags,
            }
        )
    inventory_basis = {
        "documents": {item.literature_id: item.content_hash for item in documents},
        "selections": {row["literature_id"]: row["representation_selection_id"] for row in rows},
        "structure_selections": {
            item.selection_id: item.content_hash for item in repositories.document_structure_selections.list()
        },
        "curations": {item.curation_id: item.content_hash for item in repositories.curations.list()},
        "compilations": {
            item.compilation_id: item.content_hash for items in compilations_by_literature.values() for item in items
        },
        "roles": roles,
        "local_supplements": supplement_manifest,
        "relations": {item.relation_id: item.content_hash for item in repositories.relations.list()},
        "workflow_patterns": {item.pattern_id: content_hash(item) for item in repositories.workflow_patterns.list()},
    }
    relation_records = repositories.relations.list()
    pending_states = {"uncurated", "machine_extracted", "human_reviewed"}
    summary = {
        "literature_documents": len(rows),
        "target_count": target_count,
        "additional_documents_to_target": max(0, target_count - len(rows)),
        "source_trusted": sum(row["source_trusted"] for row in rows),
        "source_authority_pending": sum(not row["source_trusted"] for row in rows),
        "current_k1f_compiled_papers": sum(bool(row["current_k1f_compilations"]) for row in rows),
        "papers_with_accepted_science": sum(
            any(item["trusted"] for item in row["scientific_records"].values()) for row in rows
        ),
        "papers_with_local_supplement_match": sum(bool(row["local_supplement_file_matches"]) for row in rows),
        "missing_optional_topics": sum(not row["topics_present"] for row in rows),
        "missing_optional_keywords": sum(not row["keywords_present"] for row in rows),
        "knowledge_relations": len(relation_records),
        "trusted_knowledge_relations": sum(
            ("knowledge_relation", item.relation_id) in trusted for item in relation_records
        ),
        "pending_knowledge_relations": sum(
            _status(curations, "knowledge_relation", item.relation_id) in pending_states for item in relation_records
        ),
        "workflow_patterns": len(repositories.workflow_patterns.list()),
        "pending_curations_by_type": {
            "literature_document": sum(
                _status(curations, "literature_document", item.literature_id) in pending_states for item in documents
            ),
            "literature_representation_selection": sum(
                row["representation_selection_id"] is not None
                and _status(curations, "literature_representation_selection", row["representation_selection_id"])
                in pending_states
                for row in rows
            ),
            "source_claim": sum(row["scientific_records"]["source_claim"]["pending_review"] for row in rows),
            "method_fact": sum(row["scientific_records"]["method_fact"]["pending_review"] for row in rows),
            "knowledge_relation": sum(
                _status(curations, "knowledge_relation", item.relation_id) in pending_states
                for item in relation_records
            ),
        },
    }
    return {
        "report_version": "literature-gap-audit-1",
        "inventory_basis_hash": content_hash(inventory_basis),
        "summary": summary,
        "papers": rows,
        "source_authority_note": (
            "Role proposals and local supplement filename matches are audit annotations, "
            "not SPC curation or scientific facts."
        ),
    }


def _display_title(title: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", title))).strip().replace("|", "\\|")


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# SPC literature coverage audit",
        "",
        "This is an audit queue, not a curation decision or a claim of scientific reproducibility.",
        f"Inventory basis SHA-256: `{report['inventory_basis_hash']}`.",
        "",
        f"- Literature records: {summary['literature_documents']} / target {summary['target_count']}.",
        f"- Source-trusted: {summary['source_trusted']}; pending source authority: {summary['source_authority_pending']}.",
        f"- Papers with current K1F compilation: {summary['current_k1f_compiled_papers']}.",
        f"- Papers with accepted scientific records: {summary['papers_with_accepted_science']}.",
        f"- Papers with matching local SI PDF filenames: {summary['papers_with_local_supplement_match']} "
        "(not trusted SI ingestion).",
        f"- Trusted cross-paper relations: {summary['trusted_knowledge_relations']} "
        f"of {summary['knowledge_relations']} stored relations.",
        "",
        "| DOI | Year | Proposed role | Source | K1F | Trusted science | Local SI | Gaps |",
        "| --- | ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in report["papers"]:
        accepted = sum(item["trusted"] for item in row["scientific_records"].values())
        lines.append(
            f"| [{row['doi']}](https://doi.org/{row['doi']}) "
            f"— {_display_title(row['title'])} | {row['year']} | {row['role_proposal']} "
            f"| {'trusted' if row['source_trusted'] else 'pending'} "
            f"| {row['current_k1f_compilations']} | {accepted} "
            f"| {len(row['local_supplement_file_matches'])} "
            f"| {', '.join(row['gap_flags']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion contains exact IDs, curation states, per-type scientific counts, "
            "rejected-proposal codes, and matched local SI filenames. An unlisted SI file is unknown, "
            "not proof that the paper has no supplementary information.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=120)
    parser.add_argument("--role-annotations", type=Path)
    parser.add_argument("--supplement-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_report(
        arguments.knowledge_dir,
        target_count=arguments.target_count,
        role_path=arguments.role_annotations,
        supplement_dir=arguments.supplement_dir,
    )
    stem = f"literature-gap-{report['inventory_basis_hash'][:12]}"
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        output_dir / f"{stem}.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        output_dir / f"{stem}.md": render_markdown(report),
    }
    for path, content in outputs.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite different audit report: {path}")
    for path, content in outputs.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
