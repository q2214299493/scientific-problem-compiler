from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Generic, Iterable, TypeVar

from pydantic import BaseModel

from .models import (
    AcquisitionAttemptRecord,
    CanonicalTextArtifact,
    CanonicalTextBlock,
    CanonicalHTMLTextArtifact,
    CollectionAcquisitionLink,
    CollectionDefinition,
    CollectionDiff,
    CollectionImportRecord,
    CollectionPageRecord,
    CollectionResourceOccurrence,
    CollectionSnapshot,
    DiscoveredCollectionResource,
    DomainProfile,
    EvidenceSpan,
    ExpertAttributionRecord,
    ExpertCase,
    ExpertOpinion,
    ExpertProfile,
    HistoricalLiteratureEvidenceAuthorization,
    HTMLLiteratureIngestionRecord,
    KnowledgeSnapshot,
    KnowledgeCurationRecord,
    KnowledgeRelation,
    LiteratureAcquisitionRecord,
    LiteratureAcquisitionRequest,
    LiteratureDocument,
    LiteratureIngestionRecord,
    LiteratureRepresentationSelection,
    LiteratureRepresentationReference,
    LiteratureWorkflowPattern,
    MethodFact,
    MetadataMergeManifest,
    MetadataRetrievalRecord,
    ModelFact,
    RawLiteratureArtifact,
    RawHTMLLiteratureArtifact,
    ReportedResult,
    ResolvedLiteratureResource,
    ScientificCapability,
    SourceClaim,
    SourceDocument,
    SourceQuote,
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


class IdentityBoundRepository(ModelRepository[ModelT]):
    def __init__(
        self,
        root: Path,
        model_type: type[ModelT],
        identity_field: str,
    ) -> None:
        super().__init__(root, model_type)
        self.identity_field = identity_field

    def _record_id(self, model: ModelT) -> str:
        value = getattr(model, self.identity_field, None)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"record has no valid {self.identity_field!r} identity field"
            )
        return value

    def put(self, key: str, model: ModelT) -> Path:
        if key != self._record_id(model):
            raise ValueError(
                f"repository key must equal record {self.identity_field}"
            )
        return super().put(key, model)

    def get(self, key: str) -> ModelT:
        record = super().get(key)
        if self._record_id(record) != key:
            raise ValueError("stored record ID does not match repository key")
        return record

    def list(self) -> tuple[ModelT, ...]:
        records = super().list()
        record_ids = [self._record_id(record) for record in records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("repository contains duplicate record IDs")
        for record_id in record_ids:
            self.get(record_id)
        return records


class SourceEvidenceStore:
    def __init__(self, state_root: Path) -> None:
        initialize_state(state_root)
        self.state_root = state_root
        self.source_records = ModelRepository(state_root / "sources", SourceDocument)
        self.evidence_records = ModelRepository(state_root / "evidence", EvidenceSpan)

    def get(self, key: str) -> EvidenceSpan:
        return self.evidence_records.get(key)

    def verify_source_integrity(self, source: SourceDocument) -> SourceDocument:
        require_safe_path_component(source.source_id, field="source_id")
        require_safe_path_component(source.version, field="source version")
        stored_source = self.source_records.get(f"{source.source_id}--{source.version}")
        if stored_source != source:
            raise ValueError("SourceDocument differs from its repository record")
        if not source.read_only:
            raise ValueError("SourceDocument must be marked read_only")
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
        return source

    def _verify_evidence_against_source(self, evidence: EvidenceSpan) -> SourceDocument:
        require_safe_path_component(evidence.source_id, field="evidence source_id")
        require_safe_path_component(evidence.source_version, field="evidence source_version")
        source = self.source_records.get(f"{evidence.source_id}--{evidence.source_version}")
        if (source.source_id, source.version) != (
            evidence.source_id,
            evidence.source_version,
        ):
            raise ValueError("SourceDocument source_id/version does not match EvidenceSpan")
        self.verify_source_integrity(source)
        if source.content_sha256 != evidence.content_sha256:
            raise ValueError("EvidenceSpan content hash does not match SourceDocument")
        content_path = self.state_root.joinpath(*PurePosixPath(source.stored_path).parts)
        content = content_path.read_bytes()
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

    def ingest(
        self,
        source_path: Path,
        source_id: str,
        version: str,
        title: str | None = None,
        *,
        source_role: str = "unspecified",
        source_type: str = "unspecified",
    ) -> SourceDocument:
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
                and existing.source_role == source_role
                and existing.source_type == source_type
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
            source_role=source_role,
            source_type=source_type,
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


class LiteratureDocumentRepository(IdentityBoundRepository[LiteratureDocument]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_documents",
            LiteratureDocument,
            "literature_id",
        )


class ExpertProfileRepository(IdentityBoundRepository[ExpertProfile]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "expert_profiles",
            ExpertProfile,
            "expert_id",
        )


class ExpertOpinionRepository(IdentityBoundRepository[ExpertOpinion]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "expert_opinions",
            ExpertOpinion,
            "opinion_id",
        )


class ExpertAttributionRepository(
    IdentityBoundRepository[ExpertAttributionRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "expert_attributions",
            ExpertAttributionRecord,
            "attribution_id",
        )


class KnowledgeRelationRepository(IdentityBoundRepository[KnowledgeRelation]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "relations",
            KnowledgeRelation,
            "relation_id",
        )

    def put(self, key: str, model: KnowledgeRelation) -> Path:
        require_safe_path_component(key, field="repository key")
        if (self.root / f"{key}.json").exists():
            raise FileExistsError(f"duplicate knowledge relation ID: {key}")
        return super().put(key, model)


class KnowledgeCurationRepository(
    IdentityBoundRepository[KnowledgeCurationRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "curations",
            KnowledgeCurationRecord,
            "curation_id",
        )


class RawLiteratureArtifactRepository:
    def __init__(self, knowledge_root: Path) -> None:
        self.knowledge_root = knowledge_root
        self.root = knowledge_root / "literature_artifacts"

    def put(self, source_path: Path, literature_id: str) -> RawLiteratureArtifact:
        require_safe_path_component(literature_id, field="literature_id")
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("raw literature source must be a regular non-symlink file")
        content = source_path.read_bytes()
        if not content.startswith(b"%PDF-"):
            raise ValueError("raw literature artifact is not a PDF")
        digest = hashlib.sha256(content).hexdigest()
        identity = {"literature_id": literature_id, "sha256": digest}
        artifact_id = f"literature-artifact-{content_hash(identity)[:24]}"
        destination = self.root / artifact_id
        if destination.exists():
            existing = self.get(artifact_id)
            if (
                existing.literature_id != literature_id
                or existing.sha256 != digest
                or existing.byte_size != len(content)
            ):
                raise FileExistsError(
                    f"refusing to reuse conflicting raw artifact: {artifact_id}"
                )
            return existing
        payload = {
            "artifact_id": artifact_id,
            "literature_id": literature_id,
            "original_filename": source_path.name,
            "media_type": "application/pdf",
            "byte_size": len(content),
            "sha256": digest,
            "stored_path": f"literature_artifacts/{artifact_id}/artifact.pdf",
        }
        record = RawLiteratureArtifact(
            **payload,
            content_hash=content_hash(payload),
        )
        self._write_directory(destination, "artifact.pdf", content, record)
        return self.get(artifact_id)

    def get(self, artifact_id: str) -> RawLiteratureArtifact:
        require_safe_path_component(artifact_id, field="artifact_id")
        directory = self.root / artifact_id
        content_path = directory / "artifact.pdf"
        metadata_path = directory / "metadata.json"
        self._validate_paths(directory, content_path, metadata_path)
        record = load_model(metadata_path, RawLiteratureArtifact)
        if record.artifact_id != artifact_id:
            raise ValueError("raw artifact metadata ID does not match its directory")
        content = content_path.read_bytes()
        if len(content) != record.byte_size:
            raise ValueError("raw literature artifact byte size changed")
        if hashlib.sha256(content).hexdigest() != record.sha256:
            raise ValueError("raw literature artifact hash changed")
        if not content.startswith(b"%PDF-"):
            raise ValueError("stored raw literature artifact is not a PDF")
        return record

    def list(self) -> tuple[RawLiteratureArtifact, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            self.get(path.name)
            for path in sorted(self.root.iterdir())
            if path.is_dir() or path.is_symlink()
        )

    def list_metadata(self) -> tuple[RawLiteratureArtifact, ...]:
        if not self.root.exists():
            return ()
        records: list[RawLiteratureArtifact] = []
        for directory in sorted(self.root.iterdir()):
            if not (directory.is_dir() or directory.is_symlink()):
                continue
            content_path = directory / "artifact.pdf"
            metadata_path = directory / "metadata.json"
            self._validate_paths(directory, content_path, metadata_path)
            record = load_model(metadata_path, RawLiteratureArtifact)
            if record.artifact_id != directory.name:
                raise ValueError("raw artifact metadata ID does not match its directory")
            records.append(record)
        return tuple(records)

    def _write_directory(
        self,
        destination: Path,
        content_name: str,
        content: bytes,
        record: BaseModel,
    ) -> None:
        if self.root.exists() and self.root.is_symlink():
            raise ValueError("literature artifact store root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=".staging-", dir=self.root)
        )
        try:
            content_path = staging / content_name
            content_path.write_bytes(content)
            dump_json(staging / "metadata.json", record)
            os.chmod(content_path, 0o444)
            os.chmod(staging / "metadata.json", 0o444)
            os.replace(staging, destination)
        finally:
            if staging.exists():
                resolved_staging = staging.resolve()
                resolved_root = self.root.resolve()
                if not resolved_staging.is_relative_to(resolved_root):
                    raise ValueError("artifact staging path escaped its store")
                shutil.rmtree(staging)

    def _validate_paths(
        self,
        directory: Path,
        content_path: Path,
        metadata_path: Path,
    ) -> None:
        resolved_root = self.root.resolve()
        if self.root.is_symlink() or directory.is_symlink():
            raise ValueError("literature artifact store paths cannot be symlinks")
        for path in (content_path, metadata_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError("literature artifact file is missing or is a symlink")
            if not path.resolve().is_relative_to(resolved_root):
                raise ValueError("literature artifact path escapes its store")


class CanonicalTextArtifactRepository(RawLiteratureArtifactRepository):
    def __init__(self, knowledge_root: Path) -> None:
        self.knowledge_root = knowledge_root
        self.root = knowledge_root / "canonical_text"

    def put(
        self,
        text: str,
        *,
        literature_id: str,
        raw_artifact: RawLiteratureArtifact,
        parser_id: str,
        parser_version: str,
        parser_config_hash: str,
        page_count: int,
        blocks: tuple[CanonicalTextBlock, ...],
    ) -> CanonicalTextArtifact:
        require_safe_path_component(literature_id, field="literature_id")
        require_safe_path_component(parser_id, field="parser_id")
        require_safe_path_component(parser_version, field="parser_version")
        encoded = text.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        identity = {
            "literature_id": literature_id,
            "raw_artifact_id": raw_artifact.artifact_id,
            "raw_artifact_hash": raw_artifact.content_hash,
            "parser_id": parser_id,
            "parser_version": parser_version,
            "parser_config_hash": parser_config_hash,
            "text_sha256": digest,
            "character_count": len(text),
            "page_count": page_count,
            "blocks": blocks,
        }
        canonical_text_id = f"canonical-text-{content_hash(identity)[:24]}"
        payload = {
            "canonical_text_id": canonical_text_id,
            **identity,
            "stored_path": f"canonical_text/{canonical_text_id}/content.txt",
        }
        record = CanonicalTextArtifact(
            **payload,
            content_hash=content_hash(payload),
        )
        self._verify_blocks(record, text)
        destination = self.root / canonical_text_id
        if destination.exists():
            existing = self.get(canonical_text_id)
            if existing != record:
                raise FileExistsError(
                    f"refusing to overwrite different canonical text: {canonical_text_id}"
                )
            return existing
        self._write_directory(destination, "content.txt", encoded, record)
        return self.get(canonical_text_id)

    def get(self, canonical_text_id: str) -> CanonicalTextArtifact:
        require_safe_path_component(canonical_text_id, field="canonical_text_id")
        directory = self.root / canonical_text_id
        content_path = directory / "content.txt"
        metadata_path = directory / "metadata.json"
        self._validate_paths(directory, content_path, metadata_path)
        record = load_model(metadata_path, CanonicalTextArtifact)
        if record.canonical_text_id != canonical_text_id:
            raise ValueError("canonical text metadata ID does not match its directory")
        try:
            text = content_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("canonical text is not valid UTF-8") from error
        if len(text) != record.character_count:
            raise ValueError("canonical text character count changed")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != record.text_sha256:
            raise ValueError("canonical text hash changed")
        self._verify_blocks(record, text)
        return record

    def read_text(self, canonical_text_id: str) -> str:
        record = self.get(canonical_text_id)
        return self.knowledge_root.joinpath(
            *PurePosixPath(record.stored_path).parts
        ).read_text(encoding="utf-8")

    def list(self) -> tuple[CanonicalTextArtifact, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            self.get(path.name)
            for path in sorted(self.root.iterdir())
            if path.is_dir() or path.is_symlink()
        )

    def list_metadata(self) -> tuple[CanonicalTextArtifact, ...]:
        if not self.root.exists():
            return ()
        records: list[CanonicalTextArtifact] = []
        for directory in sorted(self.root.iterdir()):
            if not (directory.is_dir() or directory.is_symlink()):
                continue
            content_path = directory / "content.txt"
            metadata_path = directory / "metadata.json"
            self._validate_paths(directory, content_path, metadata_path)
            record = load_model(metadata_path, CanonicalTextArtifact)
            if record.canonical_text_id != directory.name:
                raise ValueError(
                    "canonical text metadata ID does not match its directory"
                )
            records.append(record)
        return tuple(records)

    @staticmethod
    def _verify_blocks(record: CanonicalTextArtifact, text: str) -> None:
        for block in record.blocks:
            recovered = text[block.start_offset : block.end_offset]
            if hashlib.sha256(recovered.encode("utf-8")).hexdigest() != block.text_hash:
                raise ValueError(
                    f"canonical text block does not recover its text hash: {block.block_id}"
                )


class RawHTMLLiteratureArtifactRepository(RawLiteratureArtifactRepository):
    def __init__(self, knowledge_root: Path) -> None:
        self.knowledge_root = knowledge_root
        self.root = knowledge_root / "html_literature_artifacts"

    def put(
        self,
        content: bytes,
        *,
        source_url: str,
        literature_id: str,
        media_type: str,
    ) -> RawHTMLLiteratureArtifact:
        require_safe_path_component(literature_id, field="literature_id")
        if media_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("HTML artifact requires an HTML media type")
        digest = hashlib.sha256(content).hexdigest()
        identity = {
            "source_url": source_url,
            "literature_id": literature_id,
            "sha256": digest,
        }
        artifact_id = f"html-artifact-{content_hash(identity)[:24]}"
        payload = {
            "artifact_id": artifact_id,
            **identity,
            "media_type": media_type,
            "byte_size": len(content),
            "stored_path": (
                f"html_literature_artifacts/{artifact_id}/artifact.html"
            ),
        }
        record = RawHTMLLiteratureArtifact(
            **payload, content_hash=content_hash(payload)
        )
        destination = self.root / artifact_id
        if destination.exists():
            existing = self.get(artifact_id)
            if existing != record:
                raise FileExistsError(
                    f"refusing to overwrite different HTML artifact: {artifact_id}"
                )
            return existing
        self._write_directory(destination, "artifact.html", content, record)
        return self.get(artifact_id)

    def get(self, artifact_id: str) -> RawHTMLLiteratureArtifact:
        require_safe_path_component(artifact_id, field="artifact_id")
        directory = self.root / artifact_id
        content_path = directory / "artifact.html"
        metadata_path = directory / "metadata.json"
        self._validate_paths(directory, content_path, metadata_path)
        record = load_model(metadata_path, RawHTMLLiteratureArtifact)
        if record.artifact_id != artifact_id:
            raise ValueError("HTML artifact metadata ID does not match its directory")
        content = content_path.read_bytes()
        if len(content) != record.byte_size:
            raise ValueError("raw HTML artifact byte size changed")
        if hashlib.sha256(content).hexdigest() != record.sha256:
            raise ValueError("raw HTML artifact hash changed")
        return record

    def list(self) -> tuple[RawHTMLLiteratureArtifact, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            self.get(path.name)
            for path in sorted(self.root.iterdir())
            if path.is_dir() or path.is_symlink()
        )

    def list_metadata(self) -> tuple[RawHTMLLiteratureArtifact, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            load_model(path / "metadata.json", RawHTMLLiteratureArtifact)
            for path in sorted(self.root.iterdir())
            if path.is_dir() or path.is_symlink()
        )


class CanonicalHTMLTextArtifactRepository(RawLiteratureArtifactRepository):
    def __init__(self, knowledge_root: Path) -> None:
        self.knowledge_root = knowledge_root
        self.root = knowledge_root / "canonical_html_text"

    def put(
        self,
        text: str,
        *,
        raw_artifact: RawHTMLLiteratureArtifact,
        extractor_id: str,
        extractor_version: str,
        extractor_config_hash: str,
        blocks: tuple[CanonicalTextBlock, ...],
    ) -> CanonicalHTMLTextArtifact:
        encoded = text.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        identity = {
            "literature_id": raw_artifact.literature_id,
            "raw_artifact_id": raw_artifact.artifact_id,
            "raw_artifact_hash": raw_artifact.content_hash,
            "source_url": raw_artifact.source_url,
            "extractor_id": extractor_id,
            "extractor_version": extractor_version,
            "extractor_config_hash": extractor_config_hash,
            "text_sha256": digest,
            "character_count": len(text),
            "blocks": blocks,
        }
        canonical_text_id = f"canonical-html-text-{content_hash(identity)[:24]}"
        payload = {
            "canonical_text_id": canonical_text_id,
            **identity,
            "stored_path": (
                f"canonical_html_text/{canonical_text_id}/content.txt"
            ),
        }
        record = CanonicalHTMLTextArtifact(
            **payload, content_hash=content_hash(payload)
        )
        self._verify_blocks(record, text)
        destination = self.root / canonical_text_id
        if destination.exists():
            existing = self.get(canonical_text_id)
            if existing != record:
                raise FileExistsError(
                    f"refusing to overwrite different HTML text: {canonical_text_id}"
                )
            return existing
        self._write_directory(destination, "content.txt", encoded, record)
        return self.get(canonical_text_id)

    def get(self, canonical_text_id: str) -> CanonicalHTMLTextArtifact:
        require_safe_path_component(canonical_text_id, field="canonical_text_id")
        directory = self.root / canonical_text_id
        content_path = directory / "content.txt"
        metadata_path = directory / "metadata.json"
        self._validate_paths(directory, content_path, metadata_path)
        record = load_model(metadata_path, CanonicalHTMLTextArtifact)
        if record.canonical_text_id != canonical_text_id:
            raise ValueError("HTML canonical metadata ID does not match directory")
        try:
            text = content_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("HTML canonical text is not valid UTF-8") from error
        if len(text) != record.character_count:
            raise ValueError("HTML canonical text character count changed")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != record.text_sha256:
            raise ValueError("HTML canonical text hash changed")
        self._verify_blocks(record, text)
        return record

    def read_text(self, canonical_text_id: str) -> str:
        record = self.get(canonical_text_id)
        return self.knowledge_root.joinpath(
            *PurePosixPath(record.stored_path).parts
        ).read_text(encoding="utf-8")

    def list(self) -> tuple[CanonicalHTMLTextArtifact, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            self.get(path.name)
            for path in sorted(self.root.iterdir())
            if path.is_dir() or path.is_symlink()
        )

    def list_metadata(self) -> tuple[CanonicalHTMLTextArtifact, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            load_model(path / "metadata.json", CanonicalHTMLTextArtifact)
            for path in sorted(self.root.iterdir())
            if path.is_dir() or path.is_symlink()
        )

    @staticmethod
    def _verify_blocks(record: CanonicalHTMLTextArtifact, text: str) -> None:
        for block in record.blocks:
            recovered = text[block.start_offset : block.end_offset]
            if hashlib.sha256(recovered.encode("utf-8")).hexdigest() != block.text_hash:
                raise ValueError(
                    f"HTML block does not recover exact text: {block.block_id}"
                )


class LiteratureArtifactStore:
    def __init__(self, knowledge_root: Path) -> None:
        self.raw_artifacts = RawLiteratureArtifactRepository(knowledge_root)
        self.canonical_texts = CanonicalTextArtifactRepository(knowledge_root)


class LiteratureIngestionRepository(
    IdentityBoundRepository[LiteratureIngestionRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_ingestions",
            LiteratureIngestionRecord,
            "ingestion_id",
        )


class LiteratureRepresentationSelectionRepository(
    IdentityBoundRepository[LiteratureRepresentationSelection]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_representation_selections",
            LiteratureRepresentationSelection,
            "selection_id",
        )

    def resolve_current(
        self, literature_id: str
    ) -> LiteratureRepresentationSelection:
        require_safe_path_component(literature_id, field="literature_id")
        all_records = {item.selection_id: item for item in self.list()}
        records = tuple(
            item for item in all_records.values() if item.literature_id == literature_id
        )
        if not records:
            raise FileNotFoundError(
                f"no representation selection for literature: {literature_id}"
            )
        successors: dict[str, str] = {}
        for record in records:
            predecessor_id = record.supersedes_selection_id
            if predecessor_id is None:
                continue
            predecessor = all_records.get(predecessor_id)
            if predecessor is None:
                raise ValueError(
                    f"missing superseded representation selection: {predecessor_id}"
                )
            if predecessor.literature_id != literature_id:
                raise ValueError(
                    "representation selection history must retain literature_id"
                )
            if predecessor_id in successors:
                raise ValueError(
                    f"representation selection has multiple successors: {predecessor_id}"
                )
            successors[predecessor_id] = record.selection_id
        heads = [record for record in records if record.selection_id not in successors]
        if len(heads) != 1:
            raise ValueError(
                f"expected exactly one current representation selection: {literature_id}"
            )
        seen: set[str] = set()
        current = heads[0]
        cursor: LiteratureRepresentationSelection | None = current
        while cursor is not None:
            if cursor.selection_id in seen:
                raise ValueError(
                    f"cyclic representation selection history: {literature_id}"
                )
            seen.add(cursor.selection_id)
            predecessor_id = cursor.supersedes_selection_id
            cursor = all_records.get(predecessor_id) if predecessor_id else None
        if len(seen) != len(records):
            raise ValueError(
                f"cyclic or disconnected representation selection history: {literature_id}"
            )
        return current


class HistoricalLiteratureEvidenceAuthorizationRepository(
    IdentityBoundRepository[HistoricalLiteratureEvidenceAuthorization]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "historical_evidence_authorizations",
            HistoricalLiteratureEvidenceAuthorization,
            "authorization_id",
        )


class LiteratureAcquisitionRequestRepository(
    IdentityBoundRepository[LiteratureAcquisitionRequest]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_acquisition_requests",
            LiteratureAcquisitionRequest,
            "request_id",
        )


class ResolvedLiteratureResourceRepository(
    IdentityBoundRepository[ResolvedLiteratureResource]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "resolved_literature_resources",
            ResolvedLiteratureResource,
            "resource_id",
        )


class LiteratureAcquisitionRepository(
    IdentityBoundRepository[LiteratureAcquisitionRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_acquisitions",
            LiteratureAcquisitionRecord,
            "acquisition_id",
        )


class AcquisitionAttemptRepository(
    IdentityBoundRepository[AcquisitionAttemptRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "acquisition_attempts",
            AcquisitionAttemptRecord,
            "attempt_id",
        )


class MetadataRetrievalRepository(
    IdentityBoundRepository[MetadataRetrievalRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "metadata_retrievals",
            MetadataRetrievalRecord,
            "retrieval_id",
        )


class MetadataMergeManifestRepository(
    IdentityBoundRepository[MetadataMergeManifest]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "metadata_merge_manifests",
            MetadataMergeManifest,
            "manifest_id",
        )


class HTMLLiteratureIngestionRepository(
    IdentityBoundRepository[HTMLLiteratureIngestionRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "html_literature_ingestions",
            HTMLLiteratureIngestionRecord,
            "ingestion_id",
        )


class LiteratureRepresentationReferenceRepository(
    IdentityBoundRepository[LiteratureRepresentationReference]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_representation_refs",
            LiteratureRepresentationReference,
            "representation_id",
        )


class CollectionDefinitionRepository(IdentityBoundRepository[CollectionDefinition]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_definitions",
            CollectionDefinition,
            "collection_id",
        )


class CollectionPageRepository(IdentityBoundRepository[CollectionPageRecord]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_pages",
            CollectionPageRecord,
            "page_id",
        )


class DiscoveredCollectionResourceRepository(
    IdentityBoundRepository[DiscoveredCollectionResource]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_resources",
            DiscoveredCollectionResource,
            "discovered_resource_id",
        )


class CollectionResourceOccurrenceRepository(
    IdentityBoundRepository[CollectionResourceOccurrence]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_occurrences",
            CollectionResourceOccurrence,
            "occurrence_id",
        )


class CollectionSnapshotRepository(IdentityBoundRepository[CollectionSnapshot]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_snapshots",
            CollectionSnapshot,
            "snapshot_id",
        )


class CollectionAcquisitionLinkRepository(
    IdentityBoundRepository[CollectionAcquisitionLink]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_acquisition_links",
            CollectionAcquisitionLink,
            "link_id",
        )


class CollectionImportRepository(IdentityBoundRepository[CollectionImportRecord]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_imports",
            CollectionImportRecord,
            "import_id",
        )


class CollectionDiffRepository(IdentityBoundRepository[CollectionDiff]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "collection_diffs",
            CollectionDiff,
            "diff_id",
        )


class KnowledgeRepositories:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.expert_cases = ExpertCaseRepository(root)
        self.workflow_patterns = LiteratureWorkflowRepository(root)
        self.capabilities = ScientificCapabilityRepository(root)
        self.literature_documents = LiteratureDocumentRepository(root)
        self.literature_artifacts = LiteratureArtifactStore(root)
        self.raw_literature_artifacts = self.literature_artifacts.raw_artifacts
        self.canonical_text_artifacts = self.literature_artifacts.canonical_texts
        self.raw_html_literature_artifacts = (
            RawHTMLLiteratureArtifactRepository(root)
        )
        self.canonical_html_text_artifacts = (
            CanonicalHTMLTextArtifactRepository(root)
        )
        self.literature_ingestions = LiteratureIngestionRepository(root)
        self.html_literature_ingestions = HTMLLiteratureIngestionRepository(root)
        self.literature_representation_selections = (
            LiteratureRepresentationSelectionRepository(root)
        )
        self.historical_evidence_authorizations = (
            HistoricalLiteratureEvidenceAuthorizationRepository(root)
        )
        self.acquisition_requests = LiteratureAcquisitionRequestRepository(root)
        self.resolved_literature_resources = (
            ResolvedLiteratureResourceRepository(root)
        )
        self.literature_acquisitions = LiteratureAcquisitionRepository(root)
        self.acquisition_attempts = AcquisitionAttemptRepository(root)
        self.metadata_retrievals = MetadataRetrievalRepository(root)
        self.metadata_merge_manifests = MetadataMergeManifestRepository(root)
        self.literature_representation_refs = (
            LiteratureRepresentationReferenceRepository(root)
        )
        self.collection_definitions = CollectionDefinitionRepository(root)
        self.collection_pages = CollectionPageRepository(root)
        self.collection_resources = DiscoveredCollectionResourceRepository(root)
        self.collection_occurrences = CollectionResourceOccurrenceRepository(root)
        self.collection_snapshots = CollectionSnapshotRepository(root)
        self.collection_acquisition_links = CollectionAcquisitionLinkRepository(root)
        self.collection_imports = CollectionImportRepository(root)
        self.collection_diffs = CollectionDiffRepository(root)
        self.expert_profiles = ExpertProfileRepository(root)
        self.expert_opinions = ExpertOpinionRepository(root)
        self.expert_attributions = ExpertAttributionRepository(root)
        self.relations = KnowledgeRelationRepository(root)
        self.curations = KnowledgeCurationRepository(root)
        self.source_quotes = IdentityBoundRepository(
            root / "source_quotes", SourceQuote, "quote_id"
        )
        self.source_claims = IdentityBoundRepository(
            root / "source_claims", SourceClaim, "claim_id"
        )
        self.reported_results = IdentityBoundRepository(
            root / "reported_results", ReportedResult, "result_id"
        )
        self.method_facts = IdentityBoundRepository(
            root / "method_facts", MethodFact, "fact_id"
        )
        self.model_facts = IdentityBoundRepository(
            root / "model_facts", ModelFact, "fact_id"
        )

    def load_expert_cases(self, records: Iterable[ExpertCase]) -> None:
        for record in records:
            self.expert_cases.put(record.case_id, record)

    def load_workflow_patterns(self, records: Iterable[LiteratureWorkflowPattern]) -> None:
        for record in records:
            self.workflow_patterns.put(record.pattern_id, record)

    def load_capabilities(self, records: Iterable[ScientificCapability]) -> None:
        for record in records:
            self.capabilities.put(record.capability_id, record)

    def load_literature_documents(
        self, records: Iterable[LiteratureDocument]
    ) -> None:
        for record in records:
            self.literature_documents.put(record.literature_id, record)

    def load_literature_ingestions(
        self, records: Iterable[LiteratureIngestionRecord]
    ) -> None:
        for record in records:
            self.literature_ingestions.put(record.ingestion_id, record)

    def load_literature_representation_selections(
        self, records: Iterable[LiteratureRepresentationSelection]
    ) -> None:
        for record in records:
            self.literature_representation_selections.put(
                record.selection_id, record
            )

    def load_historical_evidence_authorizations(
        self, records: Iterable[HistoricalLiteratureEvidenceAuthorization]
    ) -> None:
        for record in records:
            self.historical_evidence_authorizations.put(
                record.authorization_id, record
            )

    def load_acquisition_requests(
        self, records: Iterable[LiteratureAcquisitionRequest]
    ) -> None:
        for record in records:
            self.acquisition_requests.put(record.request_id, record)

    def load_resolved_literature_resources(
        self, records: Iterable[ResolvedLiteratureResource]
    ) -> None:
        for record in records:
            self.resolved_literature_resources.put(record.resource_id, record)

    def load_literature_acquisitions(
        self, records: Iterable[LiteratureAcquisitionRecord]
    ) -> None:
        for record in records:
            self.literature_acquisitions.put(record.acquisition_id, record)

    def load_expert_profiles(self, records: Iterable[ExpertProfile]) -> None:
        for record in records:
            self.expert_profiles.put(record.expert_id, record)

    def load_expert_opinions(self, records: Iterable[ExpertOpinion]) -> None:
        for record in records:
            self.expert_opinions.put(record.opinion_id, record)

    def load_expert_attributions(
        self, records: Iterable[ExpertAttributionRecord]
    ) -> None:
        for record in records:
            self.expert_attributions.put(record.attribution_id, record)

    def load_relations(self, records: Iterable[KnowledgeRelation]) -> None:
        for record in records:
            self.relations.put(record.relation_id, record)

    def load_curations(self, records: Iterable[KnowledgeCurationRecord]) -> None:
        for record in records:
            self.curations.put(record.curation_id, record)

    def create_snapshot(
        self,
        evidence_store: SourceEvidenceStore,
        domain_profile: DomainProfile,
    ) -> KnowledgeSnapshot:
        from .knowledge.trust import TrustedKnowledgeValidator

        trusted = TrustedKnowledgeValidator(self, evidence_store).validate()
        literature_source_keys = {
            (item.source_id, item.source_version)
            for item in self.literature_ingestions.list()
            if item.source_id is not None and item.source_version is not None
        }
        literature_source_keys.update(
            (item.source_id, item.source_version)
            for item in self.html_literature_ingestions.list()
        )
        trusted_evidence_ids = {
            record_id
            for (record_type, record_id) in trusted.trusted_records
            if record_type == "evidence_span"
        }
        trusted_source_keys = {
            (record.source_id, record.version)
            for (record_type, _), record in trusted.trusted_records.items()
            if record_type == "source_document"
            and isinstance(record, SourceDocument)
        }
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
                item.evidence_id: content_hash(item)
                for item in evidence_store.evidence_records.list()
                if (item.source_id, item.source_version) not in literature_source_keys
                or item.evidence_id in trusted_evidence_ids
            },
            "evidence_source_versions": {
                f"{item.source_id}@{item.version}": item.content_sha256
                for item in evidence_store.source_records.list()
                if (item.source_id, item.version) not in literature_source_keys
                or (item.source_id, item.version) in trusted_source_keys
            },
            "literature_document_hashes": {
                item.literature_id: item.content_hash
                for item in trusted.literature_documents
            },
            "raw_literature_artifact_hashes": {
                record_id: record.content_hash
                for (record_type, record_id), record in trusted.trusted_records.items()
                if record_type == "raw_literature_artifact"
            },
            "canonical_text_artifact_hashes": {
                record_id: record.content_hash
                for (record_type, record_id), record in trusted.trusted_records.items()
                if record_type == "canonical_text_artifact"
            },
            "literature_ingestion_hashes": {
                record_id: record.content_hash
                for (record_type, record_id), record in trusted.trusted_records.items()
                if record_type == "literature_ingestion"
            },
            "literature_representation_selection_hashes": {
                record_id: record.content_hash
                for (record_type, record_id), record in trusted.trusted_records.items()
                if record_type == "literature_representation_selection"
            },
            "literature_representation_hashes": {
                record_id: record.content_hash
                for (record_type, record_id), record in trusted.trusted_records.items()
                if record_type == "literature_representation_ref"
            },
            "historical_evidence_authorization_hashes": {
                record_id: record.content_hash
                for (record_type, record_id), record in trusted.trusted_records.items()
                if record_type == "historical_evidence_authorization"
            },
            "expert_profile_hashes": {
                item.expert_id: item.content_hash
                for item in trusted.expert_profiles
            },
            "expert_opinion_hashes": {
                item.opinion_id: item.content_hash
                for item in trusted.expert_opinions
            },
            "expert_attribution_hashes": {
                item.attribution_id: item.content_hash
                for item in trusted.expert_attributions
            },
            "knowledge_relation_hashes": {
                item.relation_id: item.content_hash
                for item in trusted.relations
            },
            "curation_record_hashes": {
                item.curation_id: item.content_hash for item in trusted.curations
            },
            "trusted_record_hashes": {
                f"{record_type}:{record_id}": (
                    getattr(record, "content_hash", None) or content_hash(record)
                )
                for (record_type, record_id), record in trusted.trusted_records.items()
            },
        }
        snapshot_identity = {
            key: value
            for key, value in payload.items()
            if value
            or key
            not in {
                "literature_document_hashes",
                "raw_literature_artifact_hashes",
                "canonical_text_artifact_hashes",
                "literature_ingestion_hashes",
                "literature_representation_selection_hashes",
                "literature_representation_hashes",
                "historical_evidence_authorization_hashes",
                "expert_profile_hashes",
                "expert_opinion_hashes",
                "expert_attribution_hashes",
                "knowledge_relation_hashes",
                "curation_record_hashes",
                "trusted_record_hashes",
            }
        }
        return KnowledgeSnapshot(
            snapshot_id=f"snapshot-{content_hash(snapshot_identity)[:24]}",
            **payload,
        )
