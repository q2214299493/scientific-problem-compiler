from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..models import (
    KnowledgeSnapshot,
    ScientificProblemRun,
    ScientificProblemRunStatus,
    ScientificRunFailure,
)
from ..serialization import content_hash
from .repository import ScientificProblemRunRepository

_SECRET = re.compile(r"(?i)(authorization|cookie|token|password|secret|api[_-]?key)\s*[:=]\s*\S+")


@dataclass(frozen=True)
class WorkflowResult:
    run: ScientificProblemRun
    dry_run_report: dict[str, Any] | None = None


def snapshot_hash(snapshot: KnowledgeSnapshot) -> str:
    return content_hash(snapshot.model_dump(mode="json", exclude={"created_at"}))


def make_run(payload: dict[str, Any]) -> ScientificProblemRun:
    draft = ScientificProblemRun.model_construct(**payload, content_hash="0" * 64)
    normalized = draft.model_dump(mode="json", exclude={"content_hash"})
    return ScientificProblemRun(**normalized, content_hash=content_hash(normalized))


def update_run(run: ScientificProblemRun, **updates: Any) -> ScientificProblemRun:
    payload = {
        field_name: getattr(run, field_name)
        for field_name in ScientificProblemRun.model_fields
        if field_name != "content_hash"
    }
    payload.update(updates)
    return make_run(payload)


def sanitize_failure(error: Exception) -> str:
    return _SECRET.sub(lambda match: f"{match.group(1)}=<redacted>", str(error))[:2000]


def fail_run(
    runs: ScientificProblemRunRepository, run, stage: str, error: Exception, *, retryable: bool
) -> WorkflowResult:
    failed = update_run(
        run,
        status=ScientificProblemRunStatus.FAILED,
        last_successful_status=run.status,
        failure=ScientificRunFailure(
            stage=stage,
            category=type(error).__name__,
            message=sanitize_failure(error),
            retryable=retryable,
        ),
    )
    runs.save(failed)
    return WorkflowResult(failed)
