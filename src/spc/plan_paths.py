from __future__ import annotations

from dataclasses import dataclass
import re
from pydantic import BaseModel

from .serialization import content_hash, to_primitive


_PATH = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?)*"
)
_SEGMENT = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\[(?P<index>\d+)\])?")
_GENERATED_ID_FIELDS = {
    "plan_id",
    "question_id",
    "statement_id",
    "model_id",
    "observable_id",
    "baseline_id",
    "criterion_id",
    "deviation_id",
    "assumption_id",
    "unknown_id",
    "fingerprint_id",
    "task_id",
}


@dataclass(frozen=True)
class PlanPathSegment:
    name: str
    index: int | None = None


@dataclass(frozen=True)
class ResolvedPlanPath:
    exists: bool
    value: object | None = None


def parse_plan_path(path: str) -> tuple[PlanPathSegment, ...]:
    if _PATH.fullmatch(path) is None:
        raise ValueError(f"invalid bounded plan path: {path}")
    segments: list[PlanPathSegment] = []
    for raw in path.split("."):
        match = _SEGMENT.fullmatch(raw)
        if match is None:
            raise ValueError(f"invalid bounded plan path segment: {raw}")
        index_text = match.group("index")
        segments.append(
            PlanPathSegment(
                name=match.group("name"),
                index=int(index_text) if index_text is not None else None,
            )
        )
    return tuple(segments)


def resolve_plan_path(root: object, path: str) -> ResolvedPlanPath:
    value = root
    for segment in parse_plan_path(path):
        if isinstance(value, BaseModel):
            if segment.name not in type(value).model_fields:
                return ResolvedPlanPath(False)
            value = getattr(value, segment.name)
        elif isinstance(value, dict):
            if segment.name not in value:
                return ResolvedPlanPath(False)
            value = value[segment.name]
        else:
            return ResolvedPlanPath(False)
        if segment.index is not None:
            if not isinstance(value, (list, tuple)) or segment.index >= len(value):
                return ResolvedPlanPath(False)
            value = value[segment.index]
    return ResolvedPlanPath(True, value)


def _scientific_primitive(value: object) -> object:
    primitive = (
        to_primitive(value)
        if isinstance(value, (BaseModel, dict, list))
        else list(value)
        if isinstance(value, tuple)
        else value
    )
    if isinstance(primitive, dict):
        return {
            key: _scientific_primitive(child)
            for key, child in primitive.items()
            if key not in _GENERATED_ID_FIELDS
        }
    if isinstance(primitive, list):
        return [_scientific_primitive(child) for child in primitive]
    return primitive


def scientific_value_hash(value: object) -> str:
    return content_hash({"value": _scientific_primitive(value)})


def scientific_values_equal(left: object, right: object) -> bool:
    return _scientific_primitive(left) == _scientific_primitive(right)
