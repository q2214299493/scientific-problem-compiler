from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, TypeVar

import yaml
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)
SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def require_safe_path_component(value: str, *, field: str = "identifier") -> str:
    if not SAFE_PATH_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} is not a safe path component: {value!r}")
    return value


def to_primitive(value: BaseModel | dict[str, Any] | list[Any]) -> Any:
    if isinstance(value, BaseModel):
        return to_primitive(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {key: to_primitive(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(child) for child in value]
    return value


def canonical_json_bytes(value: BaseModel | dict[str, Any] | list[Any]) -> bytes:
    return json.dumps(
        to_primitive(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_hash(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: BaseModel | dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_primitive(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def dump_yaml(path: Path, value: BaseModel | dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(to_primitive(value), sort_keys=True, allow_unicode=True), encoding="utf-8"
    )


def append_jsonl(path: Path, values: Iterable[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json_bytes(value).decode("utf-8") + "\n")


def load_data(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(load_data(path))


def export_json_schemas(output_dir: Path, model_types: Iterable[type[BaseModel]]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model_type in model_types:
        path = output_dir / f"{model_type.__name__}.schema.json"
        dump_json(path, model_type.model_json_schema())
        paths.append(path)
    return paths
