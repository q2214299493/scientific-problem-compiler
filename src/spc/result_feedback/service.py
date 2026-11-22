"""Compatibility exports for the original result feedback service module."""

from .intake import ResultEvidenceIntakeService
from .status import result_feedback_status
from .submission import (
    RESULT_FEEDBACK_VERSION,
    ResultFeedbackError,
    build_result_evidence_submission,
)
from .successor import SuccessorPlanningService


__all__ = [
    "RESULT_FEEDBACK_VERSION",
    "ResultEvidenceIntakeService",
    "ResultFeedbackError",
    "SuccessorPlanningService",
    "build_result_evidence_submission",
    "result_feedback_status",
]
