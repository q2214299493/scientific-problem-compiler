from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
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
        source = self.source_records.get(f"{evidence.source_id}--{evidence.source_version}")
        if source.content_sha256 != evidence.content_sha256:
            raise ValueError("evidence content hash does not match source version")
        content_path = self.state_root / source.stored_path
        text = content_path.read_text(encoding="utf-8")
        if text[evidence.start_offset : evidence.end_offset] != evidence.text:
            raise ValueError("evidence span text does not match stored source offsets")
        return self.evidence_records.put(evidence.evidence_id, evidence)


class KnowledgeRepositories:
    def __init__(self, root: Path) -> None:
        self.expert_cases = ModelRepository(root / "expert_cases", ExpertCase)
        self.workflow_patterns = ModelRepository(root / "workflow_patterns", LiteratureWorkflowPattern)
        self.capabilities = ModelRepository(root / "capabilities", ScientificCapability)

    def load_capabilities(self, records: Iterable[ScientificCapability]) -> None:
        for record in records:
            self.capabilities.put(record.capability_id, record)
