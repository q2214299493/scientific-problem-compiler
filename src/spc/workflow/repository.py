from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Iterator

import yaml

from ..models import (
    ApprovalReviewInput,
    PlanningEvidenceCycleRecord,
    PlanRevisionAttemptOutcome,
    PlanRevisionAttemptStart,
    PlanRevisionChain,
    RevisionApprovalReviewInput,
    ScientificProblemRun,
    ScientificRunArtifactBinding,
    parse_approval_review_input,
)
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
        path.parent.mkdir(parents=True, exist_ok=True)
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
            path.parent.mkdir(parents=True, exist_ok=True)
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

    @staticmethod
    def revision_chain_path() -> str:
        return "plan-revisions/revision-chain.yaml"

    def load_revision_chain(self, run_id: str) -> PlanRevisionChain | None:
        try:
            return self.load_artifact(
                run_id, self.revision_chain_path(), PlanRevisionChain
            )
        except FileNotFoundError:
            return None

    def save_revision_chain(self, chain: PlanRevisionChain) -> None:
        self.write_immutable_artifact(
            chain.run_id,
            f"plan-revisions/history/{chain.content_hash}.yaml",
            "plan_revision_chain",
            chain.chain_id,
            chain,
        )
        self.write_artifact(
            chain.run_id,
            self.revision_chain_path(),
            "plan_revision_chain",
            chain.chain_id,
            chain,
        )

    @staticmethod
    def evidence_cycle_prefix(cycle_index: int) -> str:
        return f"planning-evidence/cycle-{cycle_index}"

    def load_evidence_cycles(
        self, run_id: str
    ) -> tuple[PlanningEvidenceCycleRecord, ...]:
        root = self.resolve_artifact_path(run_id, "planning-evidence")
        if not root.exists():
            return ()
        records: list[PlanningEvidenceCycleRecord] = []
        directories = sorted(
            root.glob("cycle-*"),
            key=lambda path: int(path.name.removeprefix("cycle-")),
        )
        for expected_index, directory in enumerate(directories, start=1):
            if directory.is_symlink() or directory.name != f"cycle-{expected_index}":
                raise ValueError("planning evidence cycle indices are not contiguous")
            cycle_path = directory / "cycle.yaml"
            if cycle_path.exists():
                record = self.load_artifact(
                    run_id,
                    f"{self.evidence_cycle_prefix(expected_index)}/cycle.yaml",
                    PlanningEvidenceCycleRecord,
                )
                if record.cycle_index != expected_index:
                    raise ValueError("planning evidence cycle index is invalid")
                records.append(record)
        return tuple(records)

    def evidence_attempt_count(self, run_id: str) -> int:
        root = self.resolve_artifact_path(run_id, "planning-evidence")
        if not root.exists():
            return 0
        return sum(
            (directory / "request-attempt.yaml").exists()
            or (directory / "request-set.yaml").exists()
            for directory in root.glob("cycle-*")
            if directory.is_dir() and not directory.is_symlink()
        )

    def revision_artifact_root(self, run_id: str) -> str:
        cycles = self.load_evidence_cycles(run_id)
        if cycles and cycles[-1].child_context_id is not None:
            return f"evidence-replan-{cycles[-1].cycle_index}"
        return "plan-revisions"

    def hierarchical_root(self, run_id: str) -> str:
        cycles = self.load_evidence_cycles(run_id)
        if cycles and cycles[-1].child_context_id is not None:
            return f"evidence-hierarchy-{cycles[-1].cycle_index}"
        return "hierarchical-planning"

    def round_prefix(self, run_id: str, round_index: int) -> str:
        root = self.revision_artifact_root(run_id)
        name = f"round-{round_index}" if root == "plan-revisions" else f"r-{round_index}"
        return f"{root}/{name}"

    def approval_prefix(self, run_id: str) -> str:
        chain = self.load_revision_chain(run_id)
        if chain is None:
            return "approval"
        return f"{self.round_prefix(run_id, chain.rounds[-1].round_index)}/approval"

    def load_bound_artifact(self, run_id: str, binding, model_type):
        value = self.load_artifact(run_id, binding.relative_path, model_type)
        if content_hash(value) != binding.artifact_hash:
            raise ValueError(
                f"workflow artifact hash mismatch: {binding.relative_path}"
            )
        return value

    def load_approval_input_binding(
        self,
        run_id: str,
        binding: ScientificRunArtifactBinding,
    ) -> ApprovalReviewInput | RevisionApprovalReviewInput:
        value = parse_approval_review_input(
            load_data(self.resolve_artifact_path(run_id, binding.relative_path))
        )
        if content_hash(value) != binding.artifact_hash:
            raise ValueError(
                f"workflow artifact hash mismatch: {binding.relative_path}"
            )
        return value

    def attempt_prefix(self, run_id: str, attempt_index: int) -> str:
        root = self.revision_artifact_root(run_id)
        if root == "plan-revisions":
            return f"{root}/attempts/attempt-{attempt_index}"
        return f"{root}/a-{attempt_index}"

    def load_revision_attempts(
        self, run_id: str
    ) -> tuple[tuple[PlanRevisionAttemptStart, PlanRevisionAttemptOutcome | None], ...]:
        run_root = self.run_dir(run_id)
        locations = [(run_root / "plan-revisions" / "attempts", "attempt-")]
        locations.extend(
            (path, "a-")
            for path in sorted(
                run_root.glob("evidence-replan-*"),
                key=lambda item: int(item.name.removeprefix("evidence-replan-")),
            )
            if path.is_dir() and not path.is_symlink()
        )
        attempts: list[
            tuple[PlanRevisionAttemptStart, PlanRevisionAttemptOutcome | None]
        ] = []
        for attempts_root, name_prefix in locations:
            if not attempts_root.exists():
                continue
            for directory in sorted(
                attempts_root.glob(f"{name_prefix}*"),
                key=lambda path: int(path.name.removeprefix(name_prefix)),
            ):
                if directory.is_symlink():
                    raise ValueError("revision attempt directory cannot be a symlink")
                try:
                    path_index = int(directory.name.removeprefix(name_prefix))
                except ValueError as error:
                    raise ValueError("revision attempt index is invalid") from error
                relative = directory.relative_to(run_root).as_posix()
                start = self.load_artifact(
                    run_id,
                    f"{relative}/start.yaml",
                    PlanRevisionAttemptStart,
                )
                if start.attempt_index != path_index:
                    raise ValueError("revision attempt path does not match its index")
                outcome_path = directory / "outcome.yaml"
                outcome = (
                    self.load_artifact(
                        run_id,
                        f"{relative}/outcome.yaml",
                        PlanRevisionAttemptOutcome,
                    )
                    if outcome_path.exists()
                    else None
                )
                if outcome is not None and (
                    outcome.attempt_id != start.attempt_id
                    or outcome.attempt_start_hash != start.content_hash
                ):
                    raise ValueError("revision attempt outcome does not bind its start")
                attempts.append((start, outcome))
        attempts.sort(key=lambda item: item[0].attempt_index)
        if tuple(item[0].attempt_index for item in attempts) != tuple(
            range(1, len(attempts) + 1)
        ):
            raise ValueError(
                "revision attempt indices must be globally unique and contiguous"
            )
        return tuple(attempts)
