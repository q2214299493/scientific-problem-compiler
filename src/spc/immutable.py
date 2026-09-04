from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from pydantic_core import core_schema


def deep_freeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(item) for item in value)
    return value


def deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [deep_thaw(item) for item in value]
    return value


class FrozenDict(Mapping[str, Any]):
    """Recursively immutable, JSON-serializable string-keyed mapping."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(
            self,
            "_data",
            MappingProxyType({key: deep_freeze(item) for key, item in (value or {}).items()}),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("FrozenDict is immutable")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        del source_type, handler
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.dict_schema(core_schema.str_schema(), core_schema.any_schema()),
            serialization=core_schema.plain_serializer_function_ser_schema(deep_thaw, when_used="always"),
        )
