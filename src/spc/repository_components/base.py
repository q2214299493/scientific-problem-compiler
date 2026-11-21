from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generic, Iterable, TypeVar

from pydantic import BaseModel

from ..serialization import (
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


def put_immutable_model(path: Path, model: ModelT, model_type: type[ModelT]) -> Path:
    """Reuse an equal JSON record and reject a conflicting write."""
    if path.exists():
        existing = load_model(path, model_type)
        if existing != model:
            raise FileExistsError(f"refusing to overwrite different record: {path}")
        return path
    dump_json(path, model)
    return path


class ModelRepository(Generic[ModelT]):
    def __init__(self, root: Path, model_type: type[ModelT]) -> None:
        self.root = root
        self.model_type = model_type

    def _record_path(self, key: str) -> Path:
        require_safe_path_component(key, field="repository key")
        return self.root / f"{key}.json"

    def put(self, key: str, model: ModelT) -> Path:
        return put_immutable_model(self._record_path(key), model, self.model_type)

    def get(self, key: str, *, _file_hashes: dict[Path, str] | None = None) -> ModelT:
        path = self._record_path(key)
        if _file_hashes is None:
            return load_model(path, self.model_type)
        return load_model(path, self.model_type, _file_hashes=_file_hashes)

    def list(self) -> tuple[ModelT, ...]:
        if not self.root.exists():
            return ()
        return tuple(load_model(path, self.model_type) for path in sorted(self.root.glob("*.json")))


class IdentityBoundRepository(ModelRepository[ModelT]):
    _key_error = "repository key must equal record {field}"
    _stored_id_error = "stored record ID does not match repository key"

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
                self._key_error.format(field=self.identity_field)
            )
        return super().put(key, model)

    def get(self, key: str, *, _file_hashes: dict[Path, str] | None = None) -> ModelT:
        record = super().get(key, _file_hashes=_file_hashes)
        if self._record_id(record) != key:
            raise ValueError(self._stored_id_error)
        return record

    def list(self) -> tuple[ModelT, ...]:
        if not self.root.exists():
            return ()
        # get() checks the filename/key against the same validated record we
        # return. Loading first and then reopening by record ID duplicates I/O
        # and can validate a different version than the one being returned.
        records = tuple(self.get(path.stem) for path in sorted(self.root.glob("*.json")))
        record_ids = [self._record_id(record) for record in records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("repository contains duplicate record IDs")
        return records


class CatalogRepository(IdentityBoundRepository[ModelT]):
    """Legacy catalogs check duplicate IDs before resolving their canonical keys."""

    _duplicate_error: str

    def _record_id(self, model: ModelT) -> str:
        return getattr(model, self.identity_field)

    def list(self) -> tuple[ModelT, ...]:
        records = ModelRepository.list(self)
        ids = [self._record_id(record) for record in records]
        if len(set(ids)) != len(records):
            raise ValueError(self._duplicate_error)
        for key in ids:
            self.get(key)
        return records


def verify_artifact_bytes(content: bytes, size: int, digest: str, label: str) -> None:
    if len(content) != size:
        raise ValueError(f"{label} byte size changed")
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"{label} hash changed")


class ArtifactDirectory(Generic[ModelT]):
    """Shared mechanics for immutable directories containing metadata and payload bytes."""

    def __init__(self, knowledge_root: Path, directory: str) -> None:
        self.knowledge_root = knowledge_root
        self.root = knowledge_root / directory

    def _directories(self) -> Iterable[Path]:
        if self.root.exists():
            for path in sorted(self.root.iterdir()):
                if path.is_dir() or path.is_symlink():
                    yield path

    def get(self, key: str) -> ModelT:
        raise NotImplementedError

    def list(self) -> tuple[ModelT, ...]:
        return tuple(self.get(path.name) for path in self._directories())

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
