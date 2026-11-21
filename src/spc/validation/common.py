"""Shared validation reports, traversal, and content binding."""

from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict

from ..serialization import content_hash


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str
    path: str | None = None


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    valid: bool
    issues: tuple[ValidationIssue, ...] = ()


def validation_report(issues: Iterable[ValidationIssue]) -> ValidationReport:
    result = tuple(issues)
    return ValidationReport(valid=not result, issues=result)


def walk_values(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from walk_values(getattr(value, name), f"{path}.{name}" if path else name)
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from walk_values(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk_values(child, f"{path}[{index}]")


def format_issue_messages(report: ValidationReport) -> str:
    return "; ".join(f"{item.code}: {item.message}" for item in report.issues)


def format_issue_codes(report: ValidationReport) -> str:
    return ", ".join(issue.code for issue in report.issues)


def content_binding_error(model: BaseModel, *, id_field: str, prefix: str) -> str | None:
    """Check the content-derived identifier, then the hash of the identified payload."""
    identity = model.model_dump(mode="json", exclude={id_field, "content_hash"}, exclude_none=True)
    expected_id = f"{prefix}-{content_hash(identity)[:24]}"
    if getattr(model, id_field) != expected_id:
        return f"{type(model).__name__} identity is not content-bound"
    if getattr(model, "content_hash") != content_hash({id_field: expected_id, **identity}):
        return f"{type(model).__name__} content_hash is invalid"
    return None


def require_content_binding(model: BaseModel, *, id_field: str, prefix: str) -> None:
    error = content_binding_error(model, id_field=id_field, prefix=prefix)
    if error is not None:
        raise ValueError(error)
