from __future__ import annotations

import re

from ..immutable import FrozenDict
from ..models import MethodFact, ModelFact, ReportedResult, ResultContext, ResultStatus, SourceClaim
from ..serialization import content_hash

_VALUE_UNIT = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>meV|eV|kJ\s*/\s*mol|kcal\s*/\s*mol|K|bar|Pa|atm|%)\b",
    re.IGNORECASE,
)
_FACET = re.compile(r"\b([A-Z][a-z]?)\s*\(\s*(\d{3})\s*\)")


_QUANTITY_LABELS = (
    (
        re.compile(r"\b(?:activation barriers?|activation energ(?:y|ies)|barriers?)\b", re.I),
        "activation_barrier",
    ),
    (re.compile(r"\breaction energ(?:y|ies)\b", re.I), "reaction_energy"),
    (re.compile(r"\badsorption energ(?:y|ies)\b", re.I), "adsorption_energy"),
)


def _quantity(text: str, unit: str, value_start: int) -> str:
    normalized_unit = unit.casefold()
    if normalized_unit == "k":
        return "temperature"
    if normalized_unit in {"bar", "pa", "atm"}:
        return "pressure"
    if unit == "%":
        return "percentage"
    labels = [
        (match.start(), match.end(), quantity)
        for pattern, quantity in _QUANTITY_LABELS
        for match in pattern.finditer(text)
    ]
    preceding = [item for item in labels if item[1] <= value_start]
    if preceding:
        return max(preceding, key=lambda item: item[1])[2]
    following = [item for item in labels if item[0] > value_start]
    if following and min(following, key=lambda item: item[0])[0] - value_start <= 60:
        return min(following, key=lambda item: item[0])[2]
    return "reported_quantity"


def _system_context(text: str) -> FrozenDict:
    facet = _FACET.search(text)
    if facet:
        return FrozenDict({"facet": f"{facet.group(1)}({facet.group(2)})"})
    return FrozenDict({"system": "not specified by source excerpt"})


def _method_context_and_status(text: str) -> tuple[FrozenDict, ResultStatus]:
    lowered = text.casefold()
    if "game" in lowered or "bep" in lowered or "predict" in lowered:
        models = [name for name in ("GAME", "BEP") if name.casefold() in lowered]
        return (
            FrozenDict(
                {
                    "method_family": "model_prediction",
                    "model": "+".join(models) or "unspecified predictive model",
                }
            ),
            ResultStatus.PREDICTED_REPORTED,
        )
    if "experiment" in lowered or "measured" in lowered:
        return FrozenDict({"method_family": "experiment"}), ResultStatus.EXPERIMENTAL_REPORTED
    if any(term in lowered for term in ("dft", "pbe", "vasp", "computed", "calculated")):
        context = {"method_family": "DFT"}
        if "pbe" in lowered:
            context["functional"] = "PBE"
        return FrozenDict(context), ResultStatus.COMPUTED_REPORTED
    if "literature" in lowered or "reported" in lowered:
        return FrozenDict({"method_family": "not specified"}), ResultStatus.LITERATURE_REPORTED
    return FrozenDict({"method_family": "not specified"}), ResultStatus.UNKNOWN_ORIGIN


def extract_reported_results(
    claims: tuple[SourceClaim, ...],
    method_facts: tuple[MethodFact, ...] = (),
    model_facts: tuple[ModelFact, ...] = (),
) -> tuple[ReportedResult, ...]:
    results: list[ReportedResult] = []
    for claim in claims:
        for index, match in enumerate(_VALUE_UNIT.finditer(claim.text), start=1):
            unit = re.sub(r"\s+", "", match.group("unit"))
            method_context, status = _method_context_and_status(claim.text)
            system_context = _system_context(claim.text)
            method_fact_refs = tuple(
                fact.fact_id
                for fact in method_facts
                if set(fact.evidence_refs) & set(claim.evidence_refs)
            )
            model_fact_refs = tuple(
                fact.fact_id
                for fact in model_facts
                if set(fact.evidence_refs) & set(claim.evidence_refs)
            )
            result_context_identity = {
                "system_context": dict(system_context),
                "method_context": dict(method_context),
                "method_fact_refs": method_fact_refs,
                "model_fact_refs": model_fact_refs,
            }
            result_context = ResultContext(
                context_id=f"result-context-{content_hash(result_context_identity)[:24]}",
                **result_context_identity,
            )
            identity = {
                "claim_id": claim.claim_id,
                "index": index,
                "value": match.group("value"),
                "unit": unit,
            }
            results.append(
                ReportedResult(
                    result_id=f"result-{content_hash(identity)[:24]}",
                    quantity=_quantity(claim.text, unit, match.start()),
                    value=float(match.group("value")),
                    unit=unit,
                    system_context=system_context,
                    method_context=method_context,
                    result_context=result_context,
                    evidence_refs=claim.evidence_refs,
                    result_status=status,
                )
            )
    return tuple(results)
