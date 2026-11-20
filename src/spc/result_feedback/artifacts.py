from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import tempfile

from ..models import (
    KnowledgeCurationRecord,
    ResultEvidenceAssessment,
    ResultEvidenceMaterializationReceipt,
    ResultEvidenceSubmission,
    ResultIntakeIssue,
    ResultArtifactManifestEntry,
)
from ..serialization import file_sha256
from ..workflow.repository import ScientificProblemRunRepository
from .submission import ResultFeedbackError


def _safe_artifact_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ResultFeedbackError(
            "UNSAFE_RESULT_ARTIFACT_PATH",
            f"result artifact path is unsafe: {relative_path}",
        )
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ResultFeedbackError(
            "UNSAFE_RESULT_ARTIFACT_PATH",
            f"result artifact escapes its declared root: {relative_path}",
        )
    return path


def _manifest_mismatch(path: Path, artifact: ResultArtifactManifestEntry) -> str | None:
    if path.stat().st_size != artifact.size_bytes:
        return "artifact byte size does not match manifest"
    if file_sha256(path) != artifact.sha256:
        return "artifact checksum does not match manifest"
    return None


class ResultArtifacts:
    def __init__(self, runs: ScientificProblemRunRepository) -> None:
        self.runs = runs

    @staticmethod
    def submission_prefix(submission_id: str) -> str:
        return f"result-evidence/submissions/{submission_id}"

    @staticmethod
    def accepted_prefix(submission_id: str) -> str:
        return f"result-evidence/accepted/s-{submission_id[-12:]}"

    def verify_and_archive_artifacts(
        self,
        submission: ResultEvidenceSubmission,
        artifact_root: Path,
    ) -> tuple[ResultIntakeIssue, ...]:
        issues: list[ResultIntakeIssue] = []
        root = artifact_root.resolve()
        for artifact in submission.source_artifact_manifest:
            try:
                path = _safe_artifact_path(root, artifact.relative_path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                mismatch = _manifest_mismatch(path, artifact)
                if mismatch:
                    raise ValueError(mismatch)
                destination = self.runs.resolve_artifact_path(
                    submission.parent_run_id,
                    f"{self.submission_prefix(submission.submission_id)}/raw/"
                    f"{artifact.relative_path}",
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = path.read_bytes()
                if destination.exists():
                    if destination.is_symlink() or destination.read_bytes() != content:
                        raise ValueError("archived result artifact conflicts with existing bytes")
                else:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=".result-artifact-",
                        dir=destination.parent,
                        delete=False,
                    ) as handle:
                        handle.write(content)
                        temporary = Path(handle.name)
                    os.replace(temporary, destination)
                    os.chmod(destination, 0o444)
                if _manifest_mismatch(destination, artifact):
                    raise ValueError("archived result artifact failed checksum verification")
            except (FileNotFoundError, OSError, ValueError, ResultFeedbackError) as error:
                issues.append(
                    ResultIntakeIssue(
                        code="RESULT_ARTIFACT_INTEGRITY_FAILURE",
                        message=f"{artifact.relative_path}: {error}",
                    )
                )
        return tuple(issues)

    def load_submission(self, run_id: str, submission_id: str) -> ResultEvidenceSubmission:
        return self.runs.load_artifact(
            run_id,
            f"{self.submission_prefix(submission_id)}/submission.yaml",
            ResultEvidenceSubmission,
        )

    def load_assessment(self, run_id: str, submission_id: str) -> ResultEvidenceAssessment:
        return self.runs.load_artifact(
            run_id,
            f"{self.submission_prefix(submission_id)}/assessment.yaml",
            ResultEvidenceAssessment,
        )

    def verify_archived_artifacts(
        self, submission: ResultEvidenceSubmission
    ) -> None:
        for artifact in submission.source_artifact_manifest:
            archived = self.runs.resolve_artifact_path(
                submission.parent_run_id,
                f"{self.submission_prefix(submission.submission_id)}/raw/"
                f"{artifact.relative_path}",
            )
            if (
                archived.is_symlink()
                or not archived.is_file()
                or _manifest_mismatch(archived, artifact)
            ):
                raise ResultFeedbackError(
                    "TAMPERED_ACCEPTED_RESULT_ARTIFACT",
                    f"archived result changed: {artifact.relative_path}",
                )

    def current_curation(
        self, run_id: str, submission_id: str
    ) -> KnowledgeCurationRecord | None:
        root = self.runs.resolve_artifact_path(
            run_id, f"result-evidence/curations/{submission_id}"
        )
        if not root.exists():
            return None
        records = tuple(
            self.runs.load_artifact(
                run_id,
                f"result-evidence/curations/{submission_id}/{path.name}",
                KnowledgeCurationRecord,
            )
            for path in sorted(root.glob("*.yaml"))
        )
        superseded = {
            item.supersedes_curation_id
            for item in records
            if item.supersedes_curation_id is not None
        }
        current = tuple(item for item in records if item.curation_id not in superseded)
        if len(current) != 1:
            raise ResultFeedbackError(
                "AMBIGUOUS_RESULT_CURATION",
                "result curation history must have exactly one current record",
            )
        return current[0]

    def load_receipt(
        self, run_id: str, submission_id: str
    ) -> ResultEvidenceMaterializationReceipt:
        return self.runs.load_artifact(
            run_id,
            f"{self.accepted_prefix(submission_id)}/receipt.yaml",
            ResultEvidenceMaterializationReceipt,
        )
