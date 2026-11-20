from __future__ import annotations

from typing import Any

from ..models import ResultEvidenceSubmission
from ..serialization import content_hash


RESULT_FEEDBACK_VERSION = "result-feedback-1.0.0"


class ResultFeedbackError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def build_content_bound(model_type, prefix: str, payload: dict[str, Any]):
    draft = model_type.model_construct(
        **payload,
        **{
            next(
                name
                for name in model_type.model_fields
                if name.endswith("_id") and name not in payload
            ): "pending",
            "content_hash": "0" * 64,
        },
    )
    normalized = draft.model_dump(
        mode="json",
        exclude={"content_hash", *(
            name
            for name in model_type.model_fields
            if name.endswith("_id") and name not in payload
        )},
        exclude_none=True,
    )
    id_field = next(
        name
        for name in model_type.model_fields
        if name.endswith("_id") and name not in payload
    )
    identifier = f"{prefix}-{content_hash(normalized)[:24]}"
    value = {id_field: identifier, **normalized}
    return model_type(**value, content_hash=content_hash(value))


def build_result_evidence_submission(**values: Any) -> ResultEvidenceSubmission:
    """Build the immutable external-declaration envelope without trusting it."""

    payload = ResultEvidenceSubmission.model_construct(
        submission_id="pending",
        content_hash="0" * 64,
        **values,
    ).model_dump(
        mode="json",
        exclude={"submission_id", "content_hash"},
        exclude_none=True,
    )
    submission_id = f"result-submission-{content_hash(payload)[:24]}"
    return ResultEvidenceSubmission(
        submission_id=submission_id,
        **payload,
        content_hash=content_hash({"submission_id": submission_id, **payload}),
    )
