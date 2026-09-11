from __future__ import annotations

import os
import tempfile
from pathlib import Path, PurePosixPath

from ..models import ScientificProblemRun, ScientificRunArtifactBinding
from ..serialization import (
    canonical_json_bytes,
    content_hash,
    dump_yaml,
    load_model,
    require_safe_path_component,
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
