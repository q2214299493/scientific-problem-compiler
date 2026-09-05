from __future__ import annotations

from itertools import combinations

from ..models import ComparisonConstraint, ReportedResult
from ..serialization import content_hash


def _flatten(prefix: str, values: object) -> dict[str, object]:
    return {f"{prefix}.{key}": value for key, value in dict(values).items()}


def analyze_comparisons(results: tuple[ReportedResult, ...]) -> tuple[ComparisonConstraint, ...]:
    constraints: list[ComparisonConstraint] = []
    for left, right in combinations(results, 2):
        if left.quantity != right.quantity:
            continue
        left_fields = {
            **_flatten("system_context", left.system_context),
            **_flatten("method_context", left.method_context),
        }
        right_fields = {
            **_flatten("system_context", right.system_context),
            **_flatten("method_context", right.method_context),
        }
        mismatched_fields = {
            field
            for field in set(left_fields) | set(right_fields)
            if left_fields.get(field) != right_fields.get(field)
            or "not specified" in str(left_fields.get(field, "")).casefold()
            or "not specified" in str(right_fields.get(field, "")).casefold()
        }
        if left.unit != right.unit:
            mismatched_fields.add("unit")
        mismatches = tuple(sorted(mismatched_fields))
        if not mismatches:
            continue
        target = f"{left.result_id} vs {right.result_id}"
        identity = {"target": target, "mismatches": mismatches}
        constraints.append(
            ComparisonConstraint(
                constraint_id=f"constraint-{content_hash(identity)[:24]}",
                comparison_target=target,
                must_match_fields=mismatches,
                disclosure_required_fields=mismatches,
                rationale="Direct comparison is blocked until mismatched system or method fields match or are explicitly disclosed.",
                evidence_refs=tuple(dict.fromkeys((*left.evidence_refs, *right.evidence_refs))),
            )
        )
    return tuple(constraints)
