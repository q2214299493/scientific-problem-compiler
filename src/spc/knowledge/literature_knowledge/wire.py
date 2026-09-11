from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from ...immutable import FrozenDict
from ...models import (
    EpistemicStatus,
    KnowledgePredicate,
    NonBlankStr,
    ResultStatus,
    StrictModel,
)
from .contracts import (
    LiteratureClaimProposal,
    LiteratureKnowledgeLLMResponse,
    LiteratureKnowledgeRecordType,
    LiteratureMethodFactProposal,
    LiteratureModelFactProposal,
    LiteratureQuoteProposal,
    LiteratureRelationProposal,
    LiteratureReportedResultProposal,
)


class LiteratureKnowledgeLLMWireEntry(StrictModel):
    key: NonBlankStr
    value: str


class LiteratureQuoteLLMWireProposal(StrictModel):
    quote_key: NonBlankStr
    chunk_id: NonBlankStr
    block_id: NonBlankStr
    exact_text: NonBlankStr


class LiteratureClaimLLMWireProposal(StrictModel):
    claim_key: NonBlankStr
    text: NonBlankStr
    claim_type: NonBlankStr
    quote_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    claim_strength: NonBlankStr
    epistemic_status: EpistemicStatus


class LiteratureMethodFactLLMWireProposal(StrictModel):
    fact_key: NonBlankStr
    text: NonBlankStr
    claim_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    attribute_entries: tuple[LiteratureKnowledgeLLMWireEntry, ...]

    @field_validator("attribute_entries")
    @classmethod
    def reject_duplicate_attribute_keys(
        cls,
        entries: tuple[LiteratureKnowledgeLLMWireEntry, ...],
    ) -> tuple[LiteratureKnowledgeLLMWireEntry, ...]:
        return _require_unique_entry_keys(entries)


class LiteratureModelFactLLMWireProposal(StrictModel):
    fact_key: NonBlankStr
    text: NonBlankStr
    claim_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    attribute_entries: tuple[LiteratureKnowledgeLLMWireEntry, ...]

    @field_validator("attribute_entries")
    @classmethod
    def reject_duplicate_attribute_keys(
        cls,
        entries: tuple[LiteratureKnowledgeLLMWireEntry, ...],
    ) -> tuple[LiteratureKnowledgeLLMWireEntry, ...]:
        return _require_unique_entry_keys(entries)


class LiteratureReportedResultLLMWireProposal(StrictModel):
    result_key: NonBlankStr
    claim_keys: tuple[NonBlankStr, ...] = Field(min_length=1)
    quantity: NonBlankStr
    value: float = Field(allow_inf_nan=False)
    unit: NonBlankStr | None
    system_context_entries: tuple[LiteratureKnowledgeLLMWireEntry, ...]
    method_context_entries: tuple[LiteratureKnowledgeLLMWireEntry, ...]
    method_fact_keys: tuple[NonBlankStr, ...]
    model_fact_keys: tuple[NonBlankStr, ...]
    result_status: ResultStatus

    @field_validator("system_context_entries", "method_context_entries")
    @classmethod
    def reject_duplicate_context_keys(
        cls,
        entries: tuple[LiteratureKnowledgeLLMWireEntry, ...],
    ) -> tuple[LiteratureKnowledgeLLMWireEntry, ...]:
        return _require_unique_entry_keys(entries)


class LiteratureRelationLLMWireProposal(StrictModel):
    relation_key: NonBlankStr
    subject_type: LiteratureKnowledgeRecordType
    subject_key: NonBlankStr
    predicate: KnowledgePredicate
    object_type: LiteratureKnowledgeRecordType
    object_key: NonBlankStr
    rationale: NonBlankStr


class LiteratureKnowledgeLLMWireResponse(StrictModel):
    quote_proposals: tuple[LiteratureQuoteLLMWireProposal, ...]
    claim_proposals: tuple[LiteratureClaimLLMWireProposal, ...]
    method_fact_proposals: tuple[LiteratureMethodFactLLMWireProposal, ...]
    model_fact_proposals: tuple[LiteratureModelFactLLMWireProposal, ...]
    reported_result_proposals: tuple[LiteratureReportedResultLLMWireProposal, ...]
    relation_proposals: tuple[LiteratureRelationLLMWireProposal, ...]


def _require_unique_entry_keys(
    entries: tuple[LiteratureKnowledgeLLMWireEntry, ...],
) -> tuple[LiteratureKnowledgeLLMWireEntry, ...]:
    keys = tuple(entry.key for entry in entries)
    if len(keys) != len(set(keys)):
        raise ValueError("wire entries contain a duplicate key")
    return entries


def _entries_to_frozen_dict(
    entries: tuple[LiteratureKnowledgeLLMWireEntry, ...],
    *,
    field_name: str,
) -> FrozenDict:
    keys = tuple(entry.key for entry in entries)
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate key in {field_name}")
    return FrozenDict({entry.key: entry.value for entry in entries})


def literature_knowledge_wire_to_internal(
    response: LiteratureKnowledgeLLMWireResponse,
) -> LiteratureKnowledgeLLMResponse:
    return LiteratureKnowledgeLLMResponse(
        quote_proposals=tuple(
            LiteratureQuoteProposal(**item.model_dump(mode="python"))
            for item in response.quote_proposals
        ),
        claim_proposals=tuple(
            LiteratureClaimProposal(**item.model_dump(mode="python"))
            for item in response.claim_proposals
        ),
        method_fact_proposals=tuple(
            LiteratureMethodFactProposal(
                fact_key=item.fact_key,
                text=item.text,
                claim_keys=item.claim_keys,
                attributes=_entries_to_frozen_dict(
                    item.attribute_entries,
                    field_name="method_fact.attribute_entries",
                ),
            )
            for item in response.method_fact_proposals
        ),
        model_fact_proposals=tuple(
            LiteratureModelFactProposal(
                fact_key=item.fact_key,
                text=item.text,
                claim_keys=item.claim_keys,
                attributes=_entries_to_frozen_dict(
                    item.attribute_entries,
                    field_name="model_fact.attribute_entries",
                ),
            )
            for item in response.model_fact_proposals
        ),
        reported_result_proposals=tuple(
            LiteratureReportedResultProposal(
                result_key=item.result_key,
                claim_keys=item.claim_keys,
                quantity=item.quantity,
                value=item.value,
                unit=item.unit,
                system_context=_entries_to_frozen_dict(
                    item.system_context_entries,
                    field_name="reported_result.system_context_entries",
                ),
                method_context=_entries_to_frozen_dict(
                    item.method_context_entries,
                    field_name="reported_result.method_context_entries",
                ),
                method_fact_keys=item.method_fact_keys,
                model_fact_keys=item.model_fact_keys,
                result_status=item.result_status,
            )
            for item in response.reported_result_proposals
        ),
        relation_proposals=tuple(
            LiteratureRelationProposal(**item.model_dump(mode="python"))
            for item in response.relation_proposals
        ),
    )


def validate_strict_structured_output_schema(schema: dict[str, Any]) -> None:
    def visit(node: Any, path: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return
        if not node:
            raise ValueError(f"K1F wire schema contains an unconstrained Any at {path}")
        unsupported = {"allOf", "not", "oneOf", "patternProperties"} & set(node)
        if unsupported:
            raise ValueError(
                f"K1F wire schema uses unsupported constructs at {path}: "
                + ", ".join(sorted(unsupported))
            )
        if "default" in node:
            raise ValueError(f"K1F wire schema contains an optional default at {path}")
        if "anyOf" in node:
            variants = node["anyOf"]
            if not isinstance(variants, list) or len(variants) != 2:
                raise ValueError(f"K1F wire schema has an unsupported union at {path}")
            null_variants = [item for item in variants if item.get("type") == "null"]
            if len(null_variants) != 1:
                raise ValueError(f"K1F wire schema union is not explicitly nullable at {path}")
        if node.get("type") == "object":
            properties = node.get("properties")
            if not isinstance(properties, dict):
                raise ValueError(f"K1F wire schema has an arbitrary object at {path}")
            required = node.get("required")
            if not isinstance(required, list) or set(required) != set(properties):
                raise ValueError(
                    f"K1F wire schema object fields are not all required at {path}"
                )
            if node.get("additionalProperties") is not False:
                raise ValueError(
                    f"K1F wire schema permits additional properties at {path}"
                )
        for key, value in node.items():
            visit(value, f"{path}.{key}")

    visit(schema, "$")


def literature_knowledge_llm_wire_schema() -> dict[str, Any]:
    schema = LiteratureKnowledgeLLMWireResponse.model_json_schema()
    validate_strict_structured_output_schema(schema)
    return schema


__all__ = [
    "LiteratureClaimLLMWireProposal",
    "LiteratureKnowledgeLLMWireEntry",
    "LiteratureKnowledgeLLMWireResponse",
    "LiteratureMethodFactLLMWireProposal",
    "LiteratureModelFactLLMWireProposal",
    "LiteratureQuoteLLMWireProposal",
    "LiteratureRelationLLMWireProposal",
    "LiteratureReportedResultLLMWireProposal",
    "literature_knowledge_llm_wire_schema",
    "literature_knowledge_wire_to_internal",
    "validate_strict_structured_output_schema",
]
