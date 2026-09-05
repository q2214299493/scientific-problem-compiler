from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    ApprovalReviewInput,
    ApprovalReviewRecord,
    ApprovalScores,
    ApprovalVerdict,
)
from ..serialization import content_hash, to_primitive
from .materializer import ScientificPlanApprover
from .policy import ApprovalPolicy
from .provider import ApprovalProvider
from .validators import ApprovalResponseError, validate_approval_response


@dataclass(frozen=True)
class ApprovalResult:
    review: ApprovalReviewRecord
    verdict: ApprovalVerdict


class IndependentApprovalService:
    def __init__(
        self,
        provider: ApprovalProvider,
        *,
        approver_id: str,
        policy: ApprovalPolicy | None = None,
    ) -> None:
        if not approver_id.strip():
            raise ValueError("approver_id must not be blank")
        self.provider = provider
        self.approver_id = approver_id
        self.policy = policy or ApprovalPolicy()

    def review(self, review_input: ApprovalReviewInput) -> ApprovalResult:
        candidate_before = review_input.candidate_plan
        candidate_hash_before = content_hash(candidate_before)
        response = self.provider.review(review_input)
        if (
            review_input.candidate_plan != candidate_before
            or content_hash(review_input.candidate_plan) != candidate_hash_before
        ):
            raise RuntimeError("ApprovalProvider attempted to mutate the candidate")
        response_report = validate_approval_response(response, review_input)
        if not response_report.valid:
            raise ApprovalResponseError(response_report)
        policy_result = self.policy.apply(review_input, response)
        provider_config = dict(getattr(self.provider, "provider_config", {}))
        identity = to_primitive({
            "review_input_id": review_input.review_input_id,
            "review_input_hash": review_input.content_hash,
            "provider_id": self.provider.provider_id,
            "provider_version": self.provider.provider_version,
            "provider_config": provider_config,
            "response": response,
            "policy_decision": policy_result.decision,
            "policy_reasons": policy_result.reasons,
        })
        review_id = f"approval-review-{content_hash(identity)[:24]}"
        payload = {"review_id": review_id, **identity}
        review = ApprovalReviewRecord(**payload, content_hash=content_hash(payload))
        detailed_scores = response.scores
        scores = ApprovalScores(
            **{
                name: getattr(detailed_scores, name).score
                for name in ApprovalScores.model_fields
            }
        )
        verdict_id = (
            "approval-verdict-"
            + content_hash(
                {
                    "review_hash": review.content_hash,
                    "candidate_hash": candidate_hash_before,
                    "approver_id": self.approver_id,
                }
            )[:24]
        )
        verdict = ScientificPlanApprover(self.approver_id).bind_verdict(
            candidate_before,
            verdict_id=verdict_id,
            scores=scores,
            decision=policy_result.decision,
            hard_red_flags=tuple(flag.code for flag in response.hard_red_flags),
            required_fixes=response.required_fixes,
            human_decisions_required=tuple(
                decision.decision_id
                for decision in candidate_before.required_human_decisions
            ),
        )
        return ApprovalResult(review=review, verdict=verdict)
