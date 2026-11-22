from __future__ import annotations

from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StringConstraints,
)


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("text must not be blank")
    return value


NonBlankStr = Annotated[
    str,
    StringConstraints(min_length=1),
    AfterValidator(_require_non_blank),
]


Sha256Str = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


KnowledgeEntityType = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]*$"),
    AfterValidator(_require_non_blank),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> StrictModel:
        del deep
        data = self.model_dump(mode="python")
        data.update(update or {})
        return type(self).model_validate(data)
