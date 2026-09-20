from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..serialization import content_hash


def derive_candidate_task_id(
    candidate_key: str,
    task_payload: Mapping[str, Any],
) -> str:
    """Return the stable task identity used by planning materialization."""
    return f"task-{content_hash({'candidate_key': candidate_key, **task_payload})[:24]}"
