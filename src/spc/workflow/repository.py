from __future__ import annotations

from contextlib import contextmanager
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterator

import yaml

from ..models import ScientificProblemRun, ScientificRunArtifactBinding
from ..serialization import (
    canonical_json_bytes,
    content_hash,
    dump_yaml,
    load_data,
    load_model,
    require_safe_path_component,
    to_primitive,
)


class ScientificProblemRunRepository:
    """Small immutable-history store for current unified-workflow state."""

    def __init__(self, state_dir: Path) -> None:
        self.root = state_dir / "scientific_runs"

    def run_dir(self, run_id: str) -> Path:
        require_safe_path_component(run_id, field="run_id")
        return self.root / run_id

    def get(self, run_id: str) -> ScientificProblemRun:
        return load_model(self.run_dir(run_id) / "run.json", ScientificProblemRun)

    def list(self) -> tuple[ScientificProblemRun, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            load_model(path, ScientificProblemRun)
            for path in sorted(self.root.glob("*/run.json"))
        )

    def save(self, run: ScientificProblemRun) -> Path:
        destination = self.run_dir(run.run_id) / "run.json"
        history = destination.parent / "history" / f"{run.content_hash}.json"
        payload = canonical_json_bytes(run) + b"\n"
        if history.exists() and history.read_bytes() != payload:
            raise FileExistsError(f"conflicting scientific run history: {history}")
        history.parent.mkdir(parents=True, exist_ok=True)
        if not history.exists():
            history.write_bytes(payload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".run-",
            suffix=".json",
            dir=destination.parent,
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        return destination

    def write_artifact(
        self,
        run_id: str,
        relative_path: str,
        artifact_type: str,
        artifact_id: str,
        value: object,
    ) -> ScientificRunArtifactBinding:
        path = self.resolve_artifact_path(run_id, relative_path)
        dump_yaml(path, value)  # deterministic, derived workflow artifact
        return ScientificRunArtifactBinding(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            artifact_hash=content_hash(value),
            relative_path=relative_path,
        )

    def write_immutable_artifact(
        self,
        run_id: str,
        relative_path: str,
        artifact_type: str,
        artifact_id: str,
        value: object,
    ) -> ScientificRunArtifactBinding:
        path = self.resolve_artifact_path(run_id, relative_path)
        expected_hash = content_hash(value)
        if path.exists():
            if path.is_symlink() or content_hash(load_data(path)) != expected_hash:
                raise FileExistsError(f"conflicting immutable workflow artifact: {path}")
        else:
            dump_yaml(path, value)
        return ScientificRunArtifactBinding(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            artifact_hash=expected_hash,
            relative_path=relative_path,
        )

    def write_exclusive_artifact(
        self,
        run_id: str,
        relative_path: str,
        artifact_type: str,
        artifact_id: str,
        value: object,
    ) -> ScientificRunArtifactBinding:
        path = self.resolve_artifact_path(run_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_dump(
            to_primitive(value), sort_keys=True, allow_unicode=True
        ).encode("utf-8")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return ScientificRunArtifactBinding(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            artifact_hash=content_hash(value),
            relative_path=relative_path,
        )

    @contextmanager
    def acquire_run_lock(self, run_id: str) -> Iterator[None]:
        run_directory = self.run_dir(run_id)
        run_directory.mkdir(parents=True, exist_ok=True)
        lock_path = run_directory / ".resume.lock"
        try:
            os.mkdir(lock_path)
        except FileExistsError as error:
            raise RuntimeError(
                "scientific run is already being resumed or has an unreconciled stale lock"
            ) from error
        try:
            yield
        finally:
            os.rmdir(lock_path)

    def resolve_artifact_path(self, run_id: str, relative_path: str) -> Path:
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("workflow artifact path must be safe and relative")
        path = self.run_dir(run_id).joinpath(*relative.parts)
        if not path.resolve().is_relative_to(self.run_dir(run_id).resolve()):
            raise ValueError("workflow artifact path escapes the run directory")
        return path

    def load_artifact(self, run_id: str, relative_path: str, model_type):
        return load_model(self.resolve_artifact_path(run_id, relative_path), model_type)
