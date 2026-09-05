from __future__ import annotations

from typing import Protocol

from ..models import ApprovalLLMResponse, ApprovalReviewInput


class ApprovalProvider(Protocol):
    provider_id: str
    provider_version: str

    def review(self, review_input: ApprovalReviewInput) -> ApprovalLLMResponse: ...
