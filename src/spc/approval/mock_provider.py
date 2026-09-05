from __future__ import annotations

from collections.abc import Mapping
import re

from ..models import (
    ApprovalDecision,
    ApprovalDimensionScore,
    ApprovalHardRedFlag,
    ApprovalLLMResponse,
    ApprovalRedFlagSeverity,
    ApprovalReviewInput,
    ApprovalReviewScores,
    EpistemicStatus,
    EvidenceClassification,
    SourceRole,
)

MOCK_APPROVAL_PROVIDER_VERSION = "mock-approval-1.0.0"
_STOPWORDS = {
    "additional",
    "compare",
    "comparison",
    "determine",
    "evidence",
    "plan",
    "request",
    "scientific",
    "which",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _flag(
    code: str,
    description: str,
    *,
    plan_path: str | None = None,
    evidence_refs: tuple[str, ...] = (),
    claim_refs: tuple[str, ...] = (),
) -> ApprovalHardRedFlag:
    return ApprovalHardRedFlag(
        code=code,
        severity=ApprovalRedFlagSeverity.BLOCKING,
        description=description,
        plan_path=plan_path,
        evidence_refs=evidence_refs,
        claim_refs=claim_refs,
    )


def _text_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            text for child in value.values() for text in _text_values(child)
        )
    if isinstance(value, (list, tuple)):
        return tuple(text for child in value for text in _text_values(child))
    return ()


class MockApprovalProvider:
    """Offline deterministic reviewer used for tests and local review workflows."""

    provider_id = "mock-approval"
    provider_version = MOCK_APPROVAL_PROVIDER_VERSION
    provider_config = {"mode": "deterministic", "network": False}

    def review(self, review_input: ApprovalReviewInput) -> ApprovalLLMResponse:
        plan = review_input.candidate_plan
        flags: list[ApprovalHardRedFlag] = []
        plan_claim_ids = tuple(
            entry.removeprefix("claim:")
            for entry in plan.source_query_manifest
            if entry.startswith("claim:")
        )
        claims_by_id = {claim.claim_id: claim for claim in review_input.source_claims}
        candidate_claims = tuple(
            claims_by_id[claim_id]
            for claim_id in plan_claim_ids
            if claim_id in claims_by_id
        )
        plan_evidence = tuple(
            reference.evidence_id
            for reference in plan.evidence_refs
            if reference.evidence_id in review_input.allowed_evidence_ids
        )

        request_tokens = _tokens(review_input.original_request)
        candidate_tokens = _tokens(
            " ".join(
                (
                    plan.original_question,
                    plan.latent_concern,
                    *(question.text for question in plan.atomic_questions),
                )
            )
        )
        if request_tokens and request_tokens.isdisjoint(candidate_tokens):
            flags.append(
                _flag(
                    "ADJACENT_OR_WRONG_SCIENTIFIC_QUESTION",
                    "candidate does not preserve the original request's scientific subject",
                    plan_path="atomic_questions",
                    evidence_refs=plan_evidence,
                    claim_refs=plan_claim_ids,
                )
            )

        promoted_statements = tuple(
            statement
            for statement in (
                plan.hypothesis.primary,
                plan.hypothesis.null,
                plan.model.description,
                *(item.description for item in plan.observables),
                *(item.description for item in plan.comparison_baselines),
            )
            if statement.classification == EvidenceClassification.EVIDENCE
        )
        if promoted_statements:
            promoted_evidence = tuple(
                dict.fromkeys(
                    evidence_id
                    for statement in promoted_statements
                    for evidence_id in statement.evidence_refs
                )
            )
            flags.append(
                _flag(
                    "UNSUPPORTED_FACTUAL_PROMOTION",
                    "compiler-authored scientific statements are presented as established evidence",
                    plan_path="hypothesis",
                    evidence_refs=promoted_evidence,
                    claim_refs=plan_claim_ids,
                )
            )
            hypothesis_claims = tuple(
                claim.claim_id
                for claim in candidate_claims
                if claim.epistemic_status == EpistemicStatus.SOURCE_HYPOTHESIS
            )
            if hypothesis_claims:
                flags.append(
                    _flag(
                        "SOURCE_HYPOTHESIS_TREATED_AS_FACT",
                        "a source hypothesis is promoted to an established plan fact",
                        plan_path="hypothesis.primary",
                        evidence_refs=promoted_evidence,
                        claim_refs=hypothesis_claims,
                    )
                )
            reviewer_claims = tuple(
                claim.claim_id
                for claim in candidate_claims
                if claim.source_role == SourceRole.REVIEWER
            )
            if reviewer_claims:
                flags.append(
                    _flag(
                        "REVIEWER_STATEMENT_TREATED_AS_FACT",
                        "a reviewer statement is promoted to an established plan fact",
                        plan_path="hypothesis.primary",
                        evidence_refs=promoted_evidence,
                        claim_refs=reviewer_claims,
                    )
                )

        if any(
            phrase in item.description.text.casefold()
            for item in plan.observables
            for phrase in ("missing observable", "no observable", "undefined observable")
        ):
            flags.append(
                _flag(
                    "MISSING_OR_WEAK_OBSERVABLE",
                    "candidate does not define a usable discriminating observable",
                    plan_path="observables",
                )
            )
        if any(
            phrase in item.description.text.casefold()
            for item in plan.comparison_baselines
            for phrase in ("missing baseline", "no baseline", "undefined baseline")
        ):
            flags.append(
                _flag(
                    "MISSING_BASELINE_OR_CONTROL",
                    "candidate does not define a usable baseline or control",
                    plan_path="comparison_baselines",
                )
            )
        if any(not difference.disclosed_deviation for difference in plan.fingerprint_differences):
            flags.append(
                _flag(
                    "UNDISCLOSED_METHOD_DEVIATION",
                    "candidate contains a fingerprint difference that is not disclosed",
                    plan_path="fingerprint_differences",
                )
            )

        declared_comparison_fields = {
            *plan.system_fingerprint.attributes.keys(),
            *plan.method_fingerprint.attributes.keys(),
            *(deviation.statement for deviation in plan.proposed_deviations),
            *(
                criterion
                for task in plan.tasks
                for criterion in task.success_criteria
            ),
        }
        for constraint in review_input.comparison_constraints:
            missing_fields = tuple(
                field
                for field in constraint.must_match_fields
                if not any(
                    field in str(declaration)
                    for declaration in declared_comparison_fields
                )
            )
            if missing_fields:
                flags.append(
                    _flag(
                        "INCOMPATIBLE_SYSTEMS_COMPARED_DIRECTLY",
                        "comparison compatibility fields are not enforced or disclosed: "
                        + ", ".join(missing_fields),
                        plan_path="comparison_baselines",
                        evidence_refs=constraint.evidence_refs,
                    )
                )

        retained_claims = set(plan_claim_ids)
        for conflict in review_input.conflict_sets:
            if (
                conflict.resolution_status == "unresolved"
                and not set(conflict.claim_refs).issubset(retained_claims)
            ):
                flags.append(
                    _flag(
                        "UNRESOLVED_CONFLICT_SILENTLY_RESOLVED",
                        f"unresolved conflict {conflict.conflict_id} is not retained",
                        plan_path="source_query_manifest",
                        claim_refs=conflict.claim_refs,
                    )
                )

        addressed_gap_ids = {
            gap_id
            for task in plan.tasks
            for gap_id in dict(task.inputs).get("evidence_gap_ids", ())
        }
        limitation_text = " ".join(plan.limitations)
        decision_text = " ".join(
            decision.question for decision in plan.required_human_decisions
        )
        for gap in review_input.evidence_gaps:
            if (
                gap.blocking
                and gap.gap_id not in addressed_gap_ids
                and gap.gap_id not in limitation_text
                and gap.gap_id not in decision_text
            ):
                flags.append(
                    _flag(
                        "BLOCKING_EVIDENCE_GAP_DROPPED",
                        f"blocking evidence gap {gap.gap_id} is not addressed or propagated",
                        plan_path="tasks",
                        evidence_refs=gap.evidence_refs,
                    )
                )

        axis = (plan.distinguishing_axis or "").casefold()
        if any(
            term in axis
            for term in ("encut", "k-point", "kpoint", "threshold", "cutoff")
        ):
            flags.append(
                _flag(
                    "PSEUDO_DIVERSE_CANDIDATE",
                    "candidate distinction is only numerical parameter tuning",
                    plan_path="distinguishing_axis",
                )
            )
        validation_codes = set(review_input.plan_validation_record.issue_codes)
        if validation_codes & {
            "MISSING_EVIDENCE_REF",
            "EVIDENCE_NOT_FOUND",
            "SOURCE_INTEGRITY_FAILURE",
            "UNKNOWN_EVIDENCE_REF",
        }:
            flags.append(
                _flag(
                    "MISSING_CRITICAL_EVIDENCE",
                    "a critical plan statement lacks valid source evidence",
                    plan_path="evidence_refs",
                    evidence_refs=plan_evidence,
                    claim_refs=plan_claim_ids,
                )
            )
        if validation_codes & {"DAG_CYCLE", "UNKNOWN_DEPENDENCY", "DUPLICATE_TASK_ID"}:
            flags.append(
                _flag(
                    "IMPOSSIBLE_OR_INCOMPLETE_DAG",
                    "deterministic validation found an invalid task graph",
                    plan_path="tasks",
                )
            )
        if "PHASE1_RUNNABLE_TASK" in validation_codes or any(
            task.runnable for task in plan.tasks
        ):
            flags.append(
                _flag(
                    "UNAUTHORIZED_EXECUTION_OR_RUNNABLE_TASK",
                    "planning-only approval cannot authorize runnable tasks",
                    plan_path="tasks",
                )
            )
        if "PHASE1_EXECUTION_PAYLOAD" in validation_codes:
            flags.append(
                _flag(
                    "PROMPT_INJECTION_EXECUTION_ATTEMPT",
                    "candidate contains an instruction-like execution payload",
                    plan_path="tasks",
                )
            )
        task_input_text = " ".join(
            text.casefold()
            for task in plan.tasks
            for text in _text_values(task.inputs)
        )
        if any(
            phrase in task_input_text
            for phrase in (
                "ignore previous instructions",
                "run shell",
                "execute command",
            )
        ) and not any(
            flag.code == "PROMPT_INJECTION_EXECUTION_ATTEMPT" for flag in flags
        ):
            flags.append(
                _flag(
                    "PROMPT_INJECTION_EXECUTION_ATTEMPT",
                    "candidate task data attempts to override review policy or execute actions",
                    plan_path="tasks",
                )
            )
        falsification_text = " ".join(
            criterion.statement.casefold() for criterion in plan.falsification_criteria
        )
        if any(
            phrase in falsification_text
            for phrase in ("not falsifiable", "cannot be falsified", "always succeeds")
        ) or not any(
            token in falsification_text
            for token in ("null", "not", "fail", "reject", "falsif", "retain")
        ):
            flags.append(
                _flag(
                    "NON_FALSIFIABLE_PLAN",
                    "falsification criteria do not define a rejecting outcome",
                    plan_path="falsification_criteria",
                )
            )

        scientific_statement_text = " ".join(
            (
                plan.hypothesis.primary.text,
                plan.hypothesis.null.text,
                plan.model.description.text,
            )
        ).casefold()
        reported_result_tokens = {
            (result.value, result.unit.casefold())
            for result in review_input.reported_results
        }
        numerical_claims = set(
            (float(match.group("value")), match.group("unit").casefold())
            for match in re.finditer(
                r"\b(?P<value>\d+(?:\.\d+)?)\s*"
                r"(?P<unit>ev|kj/mol|kcal/mol|bar|k)\b",
                scientific_statement_text,
            )
        )
        if numerical_claims and not numerical_claims.issubset(reported_result_tokens):
            flags.append(
                _flag(
                    "FABRICATED_PRECOMPUTATION_RESULT",
                    "candidate asserts a numerical scientific result absent from reported evidence",
                    plan_path="hypothesis",
                    evidence_refs=plan_evidence,
                )
            )
        if any(
            phrase in scientific_statement_text
            for phrase in (
                "proves all real systems",
                "valid for every catalyst",
                "universally valid",
                "all surfaces behave identically",
            )
        ):
            flags.append(
                _flag(
                    "IDEAL_MODEL_OVERGENERALIZATION",
                    "candidate overgeneralizes an idealized model beyond its evidence scope",
                    plan_path="model",
                    evidence_refs=plan_evidence,
                )
            )

        score_value = 1 if flags else 5
        score = ApprovalDimensionScore(
            score=score_value,
            rationale=(
                "Blocking review findings require revision."
                if flags
                else "No deterministic or mock-review defect was identified."
            ),
            evidence_refs=plan_evidence,
            claim_refs=plan_claim_ids,
        )
        scores = ApprovalReviewScores(
            intent_fidelity=score,
            evidence_grounding=score,
            model_observable_alignment=score,
            method_consistency=score,
            dag_executability=score,
            falsifiability=score,
            scientific_scope_adequacy=score,
        )
        unresolved_decisions = tuple(
            decision.decision_id for decision in plan.required_human_decisions
        )
        recommendation = (
            ApprovalDecision.REQUEST_REVISION
            if flags
            else (
                ApprovalDecision.NEEDS_HUMAN_CHOICE
                if unresolved_decisions
                else ApprovalDecision.APPROVE
            )
        )
        return ApprovalLLMResponse(
            scores=scores,
            decision_recommendation=recommendation,
            summary=(
                "Independent review identified blocking scientific issues."
                if flags
                else "Independent review found the candidate suitable for approval consideration."
            ),
            evidence_basis=tuple(dict.fromkeys((*plan_evidence, *plan_claim_ids))),
            hard_red_flags=tuple(flags),
            unresolved_human_decisions=unresolved_decisions,
        )
