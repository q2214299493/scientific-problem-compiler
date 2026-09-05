from __future__ import annotations

import re

from ..immutable import FrozenDict
from ..models import ReportedResult, ResultStatus, SourceClaim
from ..serialization import content_hash

_VALUE_UNIT = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>meV|eV|kJ\s*/\s*mol|kcal\s*/\s*mol|K|bar|Pa|atm|%)\b",
    re.IGNORECASE,
)
_FACET = re.compile(r"\b([A-Z][a-z]?)\s*\(\s*(\d{3})\s*\)")


def _quantity(text: str, unit: str) -> str:
    lowered = text.casefold()
    if "barrier" in lowered or "activation energ" in lowered:
        return "activation_barrier"
    if "reaction energ" in lowered:
        return "reaction_energy"
    if "adsorption energ" in lowered:
        return "adsorption_energy"
    if unit.casefold() == "k":
        return "temperature"
    if unit.casefold() in {"bar", "pa", "atm"}:
        return "pressure"
    if unit == "%":
        return "percentage"
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


def extract_reported_results(claims: tuple[SourceClaim, ...]) -> tuple[ReportedResult, ...]:
    results: list[ReportedResult] = []
    for claim in claims:
        for index, match in enumerate(_VALUE_UNIT.finditer(claim.text), start=1):
            unit = re.sub(r"\s+", "", match.group("unit"))
            method_context, status = _method_context_and_status(claim.text)
            identity = {
                "claim_id": claim.claim_id,
                "index": index,
                "value": match.group("value"),
                "unit": unit,
            }
            results.append(
                ReportedResult(
                    result_id=f"result-{content_hash(identity)[:24]}",
                    quantity=_quantity(claim.text, unit),
                    value=float(match.group("value")),
                    unit=unit,
                    system_context=_system_context(claim.text),
                    method_context=method_context,
                    evidence_refs=claim.evidence_refs,
                    result_status=status,
                )
            )
    return tuple(results)
