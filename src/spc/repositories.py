from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from pathlib import PurePosixPath
from typing import Generic, Iterable, TypeVar

from pydantic import BaseModel

from .models import (
    DomainProfile,
    EvidenceSpan,
    ExpertCase,
    KnowledgeSnapshot,
    LiteratureWorkflowPattern,
    ScientificCapability,
    SourceDocument,
)
from .serialization import (
    content_hash,
    dump_json,
    dump_yaml,
    load_data,
    load_model,
    require_safe_path_component,
)

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


class ExpertCaseRepository(ModelRepository[ExpertCase]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root / "expert_cases", ExpertCase)

    def put(self, key: str, model: ExpertCase) -> Path:
        if key != model.case_id:
            raise ValueError("expert case repository key must equal case_id")
        return super().put(key, model)

    def get(self, key: str) -> ExpertCase:
        record = super().get(key)
        if record.case_id != key:
            raise ValueError("stored expert case ID does not match repository key")
        return record

    def list(self) -> tuple[ExpertCase, ...]:
        records = super().list()
        if len({record.case_id for record in records}) != len(records):
            raise ValueError("expert case repository contains duplicate record IDs")
        for record in records:
            self.get(record.case_id)
        return records


class LiteratureWorkflowRepository(ModelRepository[LiteratureWorkflowPattern]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root / "workflow_patterns", LiteratureWorkflowPattern)

    def put(self, key: str, model: LiteratureWorkflowPattern) -> Path:
        if key != model.pattern_id:
            raise ValueError("workflow repository key must equal pattern_id")
        return super().put(key, model)

    def get(self, key: str) -> LiteratureWorkflowPattern:
        record = super().get(key)
        if record.pattern_id != key:
            raise ValueError("stored workflow pattern ID does not match repository key")
        return record

    def list(self) -> tuple[LiteratureWorkflowPattern, ...]:
        records = super().list()
        if len({record.pattern_id for record in records}) != len(records):
            raise ValueError("workflow repository contains duplicate record IDs")
        for record in records:
            self.get(record.pattern_id)
        return records


class ScientificCapabilityRepository(ModelRepository[ScientificCapability]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root / "capabilities", ScientificCapability)

    def put(self, key: str, model: ScientificCapability) -> Path:
        if key != model.capability_id:
            raise ValueError("capability repository key must equal capability_id")
        return super().put(key, model)

    def get(self, key: str) -> ScientificCapability:
        record = super().get(key)
        if record.capability_id != key:
            raise ValueError("stored capability ID does not match repository key")
        return record

    def list(self) -> tuple[ScientificCapability, ...]:
        records = super().list()
        if len({record.capability_id for record in records}) != len(records):
            raise ValueError("capability repository contains duplicate record IDs")
        for record in records:
            self.get(record.capability_id)
        return records


class KnowledgeRepositories:
    def __init__(self, root: Path) -> None:
        self.expert_cases = ExpertCaseRepository(root)
        self.workflow_patterns = LiteratureWorkflowRepository(root)
        self.capabilities = ScientificCapabilityRepository(root)

    def load_expert_cases(self, records: Iterable[ExpertCase]) -> None:
        for record in records:
            self.expert_cases.put(record.case_id, record)

    def load_workflow_patterns(self, records: Iterable[LiteratureWorkflowPattern]) -> None:
        for record in records:
            self.workflow_patterns.put(record.pattern_id, record)

    def load_capabilities(self, records: Iterable[ScientificCapability]) -> None:
        for record in records:
            self.capabilities.put(record.capability_id, record)

    def create_snapshot(
        self,
        evidence_store: SourceEvidenceStore,
        domain_profile: DomainProfile,
    ) -> KnowledgeSnapshot:
        payload = {
            "domain_profile_hash": content_hash(domain_profile),
            "expert_case_hashes": {
                item.case_id: content_hash(item) for item in self.expert_cases.list()
            },
            "workflow_pattern_hashes": {
                item.pattern_id: content_hash(item) for item in self.workflow_patterns.list()
            },
            "capability_hashes": {
                item.capability_id: content_hash(item) for item in self.capabilities.list()
            },
            "evidence_span_hashes": {
                item.evidence_id: content_hash(item) for item in evidence_store.evidence_records.list()
            },
            "evidence_source_versions": {
                f"{item.source_id}@{item.version}": item.content_sha256
                for item in evidence_store.source_records.list()
            },
        }
        return KnowledgeSnapshot(
            snapshot_id=f"snapshot-{content_hash(payload)[:24]}",
            **payload,
        )
