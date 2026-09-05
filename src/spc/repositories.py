from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from pathlib import PurePosixPath
from typing import Generic, Iterable, TypeVar

from pydantic import BaseModel

from .models import (
    EvidenceSpan,
    ExpertCase,
    LiteratureWorkflowPattern,
    ScientificCapability,
    SourceDocument,
)
from .serialization import dump_json, dump_yaml, load_data, load_model, require_safe_path_component

ModelT = TypeVar("ModelT", bound=BaseModel)


STATE_DIRECTORIES = (
    "sources",
    "evidence",
    "fingerprints",
    "candidates",
    "tasks",
    "approvals",
    "comparisons",
)


def initialize_state(state_root: Path, *, domain: str | None = None) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    for name in STATE_DIRECTORIES:
        (state_root / name).mkdir(exist_ok=True)
    project_file = state_root / "project.yaml"
    if not project_file.exists():
        dump_yaml(
            project_file,
            {"schema_version": "1.0.0", "domain": domain or "unselected", "phase": "planning_only"},
        )
    elif domain is not None:
        project = load_data(project_file)
        current_domain = project.get("domain")
        if current_domain not in {"unselected", domain}:
            raise ValueError(f"state domain is {current_domain!r}, not {domain!r}")
        if current_domain == "unselected":
            dump_yaml(project_file, {**project, "domain": domain})
    for name in ("artifacts.jsonl", "decisions.jsonl", "events.jsonl"):
        path = state_root / name
        if not path.exists():
            path.write_text("", encoding="utf-8")


class ModelRepository(Generic[ModelT]):
    def __init__(self, root: Path, model_type: type[ModelT]) -> None:
        self.root = root
        self.model_type = model_type

    def put(self, key: str, model: ModelT) -> Path:
        require_safe_path_component(key, field="repository key")
        path = self.root / f"{key}.json"
        if path.exists():
            existing = load_model(path, self.model_type)
            if existing != model:
                raise FileExistsError(f"refusing to overwrite different record: {path}")
            return path
        dump_json(path, model)
        return path

    def get(self, key: str) -> ModelT:
        require_safe_path_component(key, field="repository key")
        return load_model(self.root / f"{key}.json", self.model_type)

    def list(self) -> tuple[ModelT, ...]:
        if not self.root.exists():
            return ()
        return tuple(load_model(path, self.model_type) for path in sorted(self.root.glob("*.json")))


class SourceEvidenceStore:
    def __init__(self, state_root: Path) -> None:
        initialize_state(state_root)
        self.state_root = state_root
        self.source_records = ModelRepository(state_root / "sources", SourceDocument)
        self.evidence_records = ModelRepository(state_root / "evidence", EvidenceSpan)

    def get(self, key: str) -> EvidenceSpan:
        return self.evidence_records.get(key)

    def _verify_evidence_against_source(self, evidence: EvidenceSpan) -> SourceDocument:
        require_safe_path_component(evidence.source_id, field="evidence source_id")
        require_safe_path_component(evidence.source_version, field="evidence source_version")
        source = self.source_records.get(f"{evidence.source_id}--{evidence.source_version}")
        if (source.source_id, source.version) != (
            evidence.source_id,
            evidence.source_version,
        ):
            raise ValueError("SourceDocument source_id/version does not match EvidenceSpan")
        if not source.read_only:
            raise ValueError("SourceDocument must be marked read_only")
        if source.content_sha256 != evidence.content_sha256:
            raise ValueError("EvidenceSpan content hash does not match SourceDocument")
        stored = PurePosixPath(source.stored_path)
        expected = PurePosixPath("sources") / source.source_id / source.version / "content"
        if stored != expected or stored.is_absolute() or ".." in stored.parts:
            raise ValueError("SourceDocument stored_path is not the canonical source path")
        content_path = self.state_root.joinpath(*stored.parts)
        resolved_root = self.state_root.resolve()
        resolved_content = content_path.resolve()
        if not resolved_content.is_relative_to(resolved_root) or content_path.is_symlink():
            raise ValueError("SourceDocument content path escapes the evidence store or is a symlink")
        if not content_path.is_file():
            raise FileNotFoundError(f"stored source content is missing: {source.stored_path}")
        content = content_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != source.content_sha256:
            raise ValueError("stored source content hash does not match SourceDocument")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("EvidenceSpan offsets require UTF-8 source content") from error
        if evidence.end_offset > len(text):
            raise ValueError("EvidenceSpan exceeds stored source content")
        if text[evidence.start_offset : evidence.end_offset] != evidence.text:
            raise ValueError("EvidenceSpan text does not match stored source offsets")
        return source

    def verify_evidence_integrity(self, evidence: EvidenceSpan) -> SourceDocument:
        stored_evidence = self.evidence_records.get(evidence.evidence_id)
        if stored_evidence != evidence:
            raise ValueError("EvidenceSpan differs from its repository record")
        return self._verify_evidence_against_source(stored_evidence)

    def ingest(self, source_path: Path, source_id: str, version: str, title: str | None = None) -> SourceDocument:
        require_safe_path_component(source_id, field="source_id")
        require_safe_path_component(version, field="version")
        content = source_path.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        record_key = f"{source_id}--{version}"
        destination = self.state_root / "sources" / source_id / version / "content"
        record_path = self.state_root / "sources" / f"{record_key}.json"
        if record_path.exists():
            existing = self.source_records.get(record_key)
            if (
                existing.content_sha256 == sha256
                and existing.title == (title or source_path.name)
                and existing.stored_path == destination.relative_to(self.state_root).as_posix()
            ):
                return existing
            raise FileExistsError(f"source version already exists with different metadata: {record_path}")
        if destination.exists() and destination.read_bytes() != content:
            raise FileExistsError(f"source version already exists with different content: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source_path, destination)
            os.chmod(destination, 0o444)
        record = SourceDocument(
            source_id=source_id,
            version=version,
            title=title or source_path.name,
            content_sha256=sha256,
            stored_path=destination.relative_to(self.state_root).as_posix(),
        )
        self.source_records.put(record_key, record)
        return record

    def add_evidence(self, evidence: EvidenceSpan) -> Path:
        self._verify_evidence_against_source(evidence)
        return self.evidence_records.put(evidence.evidence_id, evidence)


class KnowledgeRepositories:
    def __init__(self, root: Path) -> None:
        self.expert_cases = ModelRepository(root / "expert_cases", ExpertCase)
        self.workflow_patterns = ModelRepository(root / "workflow_patterns", LiteratureWorkflowPattern)
        self.capabilities = ModelRepository(root / "capabilities", ScientificCapability)

    def load_capabilities(self, records: Iterable[ScientificCapability]) -> None:
        for record in records:
            self.capabilities.put(record.capability_id, record)
