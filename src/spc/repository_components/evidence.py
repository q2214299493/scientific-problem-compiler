from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from pathlib import PurePosixPath
from typing import Protocol, cast

from pydantic import BaseModel

from ..models import EvidenceSpan, SourceDocument
from ..serialization import load_model, require_safe_path_component

from .base import ModelRepository, initialize_state


_MAX_LIST_SOURCE_BYTES = 8 * 1024 * 1024


def source_documents_equivalent(
    left: SourceDocument, right: SourceDocument
) -> bool:
    """Compare source identity/content metadata, excluding storage event time."""
    return left.model_dump(exclude={"ingested_at"}) == right.model_dump(
        exclude={"ingested_at"}
    )


class EvidenceStore(Protocol):
    project_state_root: Path | None

    def get_evidence(self, evidence_id: str) -> EvidenceSpan: ...

    def get_source(self, source_id: str, source_version: str) -> SourceDocument: ...

    def list_evidence(self) -> tuple[EvidenceSpan, ...]: ...

    def list_sources(self) -> tuple[SourceDocument, ...]: ...

    def ingest_source(
        self,
        source_path: Path,
        source_id: str,
        version: str,
        title: str | None = None,
        *,
        source_role: str = "unspecified",
        source_type: str = "unspecified",
    ) -> SourceDocument: ...

    def add_evidence(self, evidence: EvidenceSpan) -> Path: ...

    def verify_source_integrity(self, source: SourceDocument) -> SourceDocument: ...

    def verify_evidence_integrity(self, evidence: EvidenceSpan) -> SourceDocument: ...


class FilesystemEvidenceStore:
    """Root-agnostic immutable filesystem storage for sources and evidence."""

    project_state_root: Path | None = None

    def __init__(self, root: Path, *, initialize: bool = True) -> None:
        if initialize:
            root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve()
        for directory in (root / "sources", root / "evidence"):
            if initialize:
                directory.mkdir(exist_ok=True)
            if directory.is_symlink() or not directory.resolve().is_relative_to(
                resolved_root
            ):
                raise ValueError("evidence repository directory escapes its root")
        self.root = root
        self.state_root = root
        self.source_records = ModelRepository(root / "sources", SourceDocument)
        self.evidence_records = ModelRepository(root / "evidence", EvidenceSpan)

    def get(self, key: str) -> EvidenceSpan:
        return self.get_evidence(key)

    def get_evidence(self, evidence_id: str) -> EvidenceSpan:
        require_safe_path_component(evidence_id, field="evidence_id")
        record_path = self.evidence_records.root / f"{evidence_id}.json"
        if record_path.is_symlink():
            raise ValueError("EvidenceSpan repository record cannot be a symlink")
        evidence = self.evidence_records.get(evidence_id)
        if evidence.evidence_id != evidence_id:
            raise ValueError("EvidenceSpan repository key does not match record identity")
        return evidence

    def get_source(self, source_id: str, source_version: str) -> SourceDocument:
        require_safe_path_component(source_id, field="source_id")
        require_safe_path_component(source_version, field="source version")
        record_key = f"{source_id}--{source_version}"
        record_path = self.source_records.root / f"{record_key}.json"
        if record_path.is_symlink():
            raise ValueError("SourceDocument repository record cannot be a symlink")
        source = self.source_records.get(record_key)
        if (source.source_id, source.version) != (source_id, source_version):
            raise ValueError(
                "SourceDocument source_id/version does not match repository key identity"
            )
        return source

    def _content_destination(self, source_id: str, version: str) -> Path:
        destination = self.root / "sources" / source_id / version / "content"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            destination.is_symlink()
            or not destination.parent.resolve().is_relative_to(self.root.resolve())
        ):
            raise ValueError("source content destination escapes the evidence store")
        return destination

    def list_evidence(self) -> tuple[EvidenceSpan, ...]:
        records: list[EvidenceSpan] = []
        if not self.evidence_records.root.exists():
            return ()
        # Preserve custom verification hooks. The standard path can share
        # verified source text inside this call and close every binding below.
        reuse_sources = all(
            getattr(getattr(self, name), "__func__", None) is getattr(FilesystemEvidenceStore, name)
            for name in ("get_source", "get_evidence", "verify_source_integrity", "verify_evidence_integrity", "_verify_evidence_against_source")
        ) and (
            getattr(self.evidence_records, "model_type", None) is EvidenceSpan
            and getattr(self.evidence_records.get, "__func__", None) is ModelRepository.get
            and getattr(self.source_records, "model_type", None) is SourceDocument
            and getattr(self.source_records.get, "__func__", None) is ModelRepository.get
        )
        sources: dict[tuple[str, str], tuple[SourceDocument, str | None]] = {}
        cached_source_bytes = 0
        file_hashes: dict[Path, str] = {}
        for path in sorted(self.evidence_records.root.glob("*.json")):
            if path.is_symlink():
                raise ValueError("EvidenceSpan repository record cannot be a symlink")
            evidence = load_model(path, EvidenceSpan, _file_hashes=file_hashes) if reuse_sources else load_model(path, EvidenceSpan)
            if path.stem != evidence.evidence_id:
                raise ValueError(
                    "EvidenceSpan repository key does not match record identity"
                )
            try:
                if reuse_sources:
                    require_safe_path_component(evidence.evidence_id, field="evidence_id")
                    key = (evidence.source_id, evidence.source_version)
                    verified = sources.get(key)
                    if verified is None:
                        source = self.get_source(*key)
                        content = self._verified_source_bytes(source)
                        text = None
                        if cached_source_bytes + len(content) <= _MAX_LIST_SOURCE_BYTES:
                            try:
                                text = content.decode("utf-8")
                            except UnicodeDecodeError as error:
                                raise ValueError("EvidenceSpan offsets require UTF-8 source content") from error
                            cached_source_bytes += len(content)
                        del content
                        verified = sources[key] = (source, text)
                    if verified[1] is None:
                        self.verify_evidence_integrity(evidence)
                    else:
                        self._verify_evidence_against_source(evidence, _verified_source=verified)
                else:
                    self.verify_evidence_integrity(evidence)
            except (FileNotFoundError, OSError, ValueError) as error:
                raise ValueError(
                    f"EvidenceSpan integrity failed for {evidence.evidence_id}: {error}"
                ) from error
            records.append(evidence)
        for source, _text in sources.values():
            try:
                self.verify_source_integrity(source)
            except (FileNotFoundError, OSError, ValueError) as error:
                raise ValueError(f"SourceDocument integrity failed during evidence read: {error}") from error
        # Detect changes during enumeration, including record changes made
        # while the shared sources were being reverified.
        for path, expected_hash in file_hashes.items():
            if path.is_symlink():
                raise ValueError("EvidenceSpan repository record cannot be a symlink")
            try:
                current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                raise ValueError(f"EvidenceSpan repository record is unavailable: {path.stem}") from error
            if current_hash != expected_hash:
                raise ValueError(f"EvidenceSpan repository record changed during list read: {path.stem}")
        return tuple(records)

    def list_sources(self) -> tuple[SourceDocument, ...]:
        records: list[SourceDocument] = []
        if not self.source_records.root.exists():
            return ()
        for path in sorted(self.source_records.root.glob("*.json")):
            if path.is_symlink():
                raise ValueError("SourceDocument repository record cannot be a symlink")
            source = load_model(path, SourceDocument)
            expected_key = f"{source.source_id}--{source.version}"
            if path.stem != expected_key:
                raise ValueError(
                    "SourceDocument repository key does not match record identity"
                )
            try:
                self.verify_source_integrity(source)
            except (FileNotFoundError, OSError, ValueError) as error:
                raise ValueError(
                    "SourceDocument integrity failed for "
                    f"{source.source_id}--{source.version}: {error}"
                ) from error
            records.append(source)
        return tuple(records)

    def verify_source_integrity(self, source: SourceDocument) -> SourceDocument:
        self._verified_source_bytes(source)
        return source

    def _verified_source_bytes(self, source: SourceDocument) -> bytes:
        """Return the exact bytes whose complete source binding was verified."""
        require_safe_path_component(source.source_id, field="source_id")
        require_safe_path_component(source.version, field="source version")
        stored_source = self.get_source(source.source_id, source.version)
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
        return content

    def source_content_path(self, source: SourceDocument) -> Path:
        self.verify_source_integrity(source)
        return self.root.joinpath(*PurePosixPath(source.stored_path).parts)

    def read_source_bytes(self, source: SourceDocument) -> bytes:
        return self.source_content_path(source).read_bytes()

    def import_source_record(
        self, source: SourceDocument, content: bytes
    ) -> SourceDocument:
        require_safe_path_component(source.source_id, field="source_id")
        require_safe_path_component(source.version, field="source version")
        if hashlib.sha256(content).hexdigest() != source.content_sha256:
            raise ValueError("imported source bytes do not match SourceDocument")
        expected_path = (
            PurePosixPath("sources")
            / source.source_id
            / source.version
            / "content"
        )
        if PurePosixPath(source.stored_path) != expected_path:
            raise ValueError("imported SourceDocument path is not canonical")
        destination = self._content_destination(source.source_id, source.version)
        if destination.exists() and destination.read_bytes() != content:
            raise FileExistsError(
                f"source version already exists with different content: {destination}"
            )
        if not destination.exists():
            destination.write_bytes(content)
            os.chmod(destination, 0o444)
        record_key = f"{source.source_id}--{source.version}"
        record_path = self.source_records.root / f"{record_key}.json"
        if record_path.is_symlink():
            raise ValueError("SourceDocument repository record cannot be a symlink")
        self.source_records.put(record_key, source)
        return self.verify_source_integrity(source)

    def _verify_evidence_against_source(
        self, evidence: EvidenceSpan, *, _verified_source: tuple[SourceDocument, str] | None = None
    ) -> SourceDocument:
        require_safe_path_component(evidence.source_id, field="evidence source_id")
        require_safe_path_component(evidence.source_version, field="evidence source_version")
        source = self.get_source(evidence.source_id, evidence.source_version) if _verified_source is None else _verified_source[0]
        if (source.source_id, source.version) != (
            evidence.source_id,
            evidence.source_version,
        ):
            raise ValueError("SourceDocument source_id/version does not match EvidenceSpan")
        content = None
        if _verified_source is None:
            if getattr(self.verify_source_integrity, "__func__", None) is _STANDARD_SOURCE_INTEGRITY_CHECK:
                # Keep the exact bytes that passed the full source check for
                # this evidence verification only. Closing checks read afresh.
                content = self._verified_source_bytes(source)
            else:
                self.verify_source_integrity(source)
        if source.content_sha256 != evidence.content_sha256:
            raise ValueError("EvidenceSpan content hash does not match SourceDocument")
        if _verified_source is None:
            if content is None:
                content = self.state_root.joinpath(*PurePosixPath(source.stored_path).parts).read_bytes()
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("EvidenceSpan offsets require UTF-8 source content") from error
        else:
            text = _verified_source[1]
        if evidence.end_offset > len(text):
            raise ValueError("EvidenceSpan exceeds stored source content")
        if text[evidence.start_offset : evidence.end_offset] != evidence.text:
            raise ValueError("EvidenceSpan text does not match stored source offsets")
        return source

    def verify_evidence_integrity(self, evidence: EvidenceSpan) -> SourceDocument:
        stored_evidence = self.get_evidence(evidence.evidence_id)
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
        return self.ingest_source(
            source_path,
            source_id,
            version,
            title,
            source_role=source_role,
            source_type=source_type,
        )

    def ingest_source(
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
        destination = self._content_destination(source_id, version)
        record_path = self.state_root / "sources" / f"{record_key}.json"
        if record_path.is_symlink():
            raise ValueError("SourceDocument repository record cannot be a symlink")
        if record_path.exists():
            existing = self.source_records.get(record_key)
            if (
                existing.content_sha256 == sha256
                and existing.title == (title or source_path.name)
                and existing.stored_path == destination.relative_to(self.state_root).as_posix()
                and existing.source_role == source_role
                and existing.source_type == source_type
            ):
                return self.verify_source_integrity(existing)
            raise FileExistsError(f"source version already exists with different metadata: {record_path}")
        if destination.exists() and destination.read_bytes() != content:
            raise FileExistsError(f"source version already exists with different content: {destination}")
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
        return self.verify_source_integrity(record)

    def add_evidence(self, evidence: EvidenceSpan) -> Path:
        self._verify_evidence_against_source(evidence)
        record_path = self.evidence_records.root / f"{evidence.evidence_id}.json"
        if record_path.is_symlink():
            raise ValueError("EvidenceSpan repository record cannot be a symlink")
        return self.evidence_records.put(evidence.evidence_id, evidence)


# A function identity, not a source cache: overridden checks retain their path.
_STANDARD_SOURCE_INTEGRITY_CHECK = FilesystemEvidenceStore.verify_source_integrity


class ProjectEvidenceStore(FilesystemEvidenceStore):
    def __init__(self, state_root: Path, *, initialize: bool = True) -> None:
        if initialize:
            initialize_state(state_root)
        super().__init__(state_root, initialize=initialize)
        self.project_state_root = state_root


class SourceEvidenceStore(ProjectEvidenceStore):
    """Backward-compatible name for the project-local evidence store."""


class KnowledgeEvidenceStore(FilesystemEvidenceStore):
    def __init__(self, knowledge_root: Path, *, initialize: bool = True) -> None:
        self.knowledge_root = knowledge_root
        super().__init__(knowledge_root / "evidence_store", initialize=initialize)


class CompositeEvidenceStore:
    """Conflict-detecting read view over shared and project evidence stores."""

    def __init__(self, *stores: EvidenceStore) -> None:
        if not stores:
            raise ValueError("CompositeEvidenceStore requires at least one store")
        self.stores = stores
        project_roots = {
            store.project_state_root
            for store in stores
            if store.project_state_root is not None
        }
        if len(project_roots) > 1:
            raise ValueError("composite cannot contain multiple project evidence roots")
        self.project_state_root = next(iter(project_roots), None)

    @staticmethod
    def _require_unambiguous(records: list[BaseModel], identity: str) -> BaseModel:
        if not records:
            raise FileNotFoundError(identity)
        first = records[0]
        if any(record != first for record in records[1:]):
            raise ValueError(f"conflicting evidence-store identity: {identity}")
        return first

    @staticmethod
    def _require_unambiguous_sources(
        records: list[SourceDocument], identity: str
    ) -> SourceDocument:
        if not records:
            raise FileNotFoundError(identity)
        first = min(records, key=lambda item: item.ingested_at)
        if any(not source_documents_equivalent(record, first) for record in records):
            raise ValueError(f"conflicting evidence-store identity: {identity}")
        return first

    def get_evidence(self, evidence_id: str) -> EvidenceSpan:
        matches: list[EvidenceSpan] = []
        for store in self.stores:
            try:
                evidence = store.get_evidence(evidence_id)
            except FileNotFoundError:
                continue
            try:
                store.verify_evidence_integrity(evidence)
            except (FileNotFoundError, OSError, ValueError) as error:
                raise ValueError(
                    f"EvidenceSpan integrity failed for {evidence_id}: {error}"
                ) from error
            matches.append(evidence)
        return cast(EvidenceSpan, self._require_unambiguous(matches, evidence_id))

    def get(self, key: str) -> EvidenceSpan:
        return self.get_evidence(key)

    def get_source(self, source_id: str, source_version: str) -> SourceDocument:
        identity = f"{source_id}--{source_version}"
        matches: list[SourceDocument] = []
        for store in self.stores:
            try:
                source = store.get_source(source_id, source_version)
            except FileNotFoundError:
                continue
            store.verify_source_integrity(source)
            matches.append(source)
        return self._require_unambiguous_sources(matches, identity)

    def list_evidence(self) -> tuple[EvidenceSpan, ...]:
        grouped: dict[str, list[EvidenceSpan]] = {}
        for store in self.stores:
            for evidence in store.list_evidence():
                grouped.setdefault(evidence.evidence_id, []).append(evidence)
        return tuple(
            cast(
                EvidenceSpan,
                self._require_unambiguous(grouped[evidence_id], evidence_id),
            )
            for evidence_id in sorted(grouped)
        )

    def list_sources(self) -> tuple[SourceDocument, ...]:
        grouped: dict[tuple[str, str], list[SourceDocument]] = {}
        for store in self.stores:
            for source in store.list_sources():
                grouped.setdefault((source.source_id, source.version), []).append(
                    source
                )
        return tuple(
            self._require_unambiguous_sources(grouped[key], "--".join(key))
            for key in sorted(grouped)
        )

    def verify_source_integrity(self, source: SourceDocument) -> SourceDocument:
        stored = self.get_source(source.source_id, source.version)
        if not source_documents_equivalent(stored, source):
            raise ValueError("SourceDocument differs from composite repository record")
        return stored

    def verify_evidence_integrity(self, evidence: EvidenceSpan) -> SourceDocument:
        stored = self.get_evidence(evidence.evidence_id)
        if stored != evidence:
            raise ValueError("EvidenceSpan differs from composite repository record")
        return self.get_source(evidence.source_id, evidence.source_version)

    @staticmethod
    def _writes_forbidden() -> None:
        raise TypeError("writes through CompositeEvidenceStore require an explicit target store")

    def ingest_source(self, *args, **kwargs) -> SourceDocument:
        self._writes_forbidden()

    def ingest(self, *args, **kwargs) -> SourceDocument:
        self._writes_forbidden()

    def add_evidence(self, evidence: EvidenceSpan) -> Path:
        self._writes_forbidden()
