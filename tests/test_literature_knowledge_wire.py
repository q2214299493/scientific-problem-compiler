from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from spc.knowledge.literature_knowledge import (
    LiteratureClaimLLMWireProposal,
    LiteratureKnowledgeLLMResponse,
    LiteratureKnowledgeLLMWireEntry,
    LiteratureKnowledgeLLMWireResponse,
    LiteratureMethodFactLLMWireProposal,
    LiteratureModelFactLLMWireProposal,
    LiteratureQuoteLLMWireProposal,
    LiteratureReportedResultLLMWireProposal,
    literature_knowledge_llm_wire_schema,
    literature_knowledge_wire_to_internal,
)
from spc.models import EpistemicStatus, ResultStatus


def _empty_wire_response() -> LiteratureKnowledgeLLMWireResponse:
    return LiteratureKnowledgeLLMWireResponse(
        quote_proposals=(),
        claim_proposals=(),
        method_fact_proposals=(),
        model_fact_proposals=(),
        reported_result_proposals=(),
        relation_proposals=(),
    )


def _walk_schema(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_schema(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_schema(value)


def test_wire_schema_requires_every_field_and_forbids_additional_properties() -> None:
    schema = literature_knowledge_llm_wire_schema()
    objects = [node for node in _walk_schema(schema) if node.get("type") == "object"]

    assert objects
    assert all(set(node["required"]) == set(node["properties"]) for node in objects)
    assert all(node["additionalProperties"] is False for node in objects)
    assert not any(node == {} for node in _walk_schema(schema))


def test_wire_schema_nullable_unit_is_present_required_and_nullable() -> None:
    schema = literature_knowledge_llm_wire_schema()
    result_schema = schema["$defs"]["LiteratureReportedResultLLMWireProposal"]
    unit_schema = result_schema["properties"]["unit"]

    assert "unit" in result_schema["required"]
    assert {item.get("type") for item in unit_schema["anyOf"]} == {
        "string",
        "null",
    }


def test_wire_schema_accepts_explicit_empty_arrays() -> None:
    response = LiteratureKnowledgeLLMWireResponse.model_validate(
        {
            "quote_proposals": [],
            "claim_proposals": [],
            "method_fact_proposals": [],
            "model_fact_proposals": [],
            "reported_result_proposals": [],
            "relation_proposals": [],
        }
    )

    assert response == _empty_wire_response()


def test_wire_response_converts_deterministically_to_internal_contract() -> None:
    response = LiteratureKnowledgeLLMWireResponse(
        quote_proposals=(
            LiteratureQuoteLLMWireProposal(
                quote_key="quote-1",
                chunk_id="chunk-1",
                block_id="block-1",
                exact_text="The reported barrier was 1.25 eV.",
            ),
        ),
        claim_proposals=(
            LiteratureClaimLLMWireProposal(
                claim_key="claim-1",
                text="The source reports a 1.25 eV barrier.",
                claim_type="reported_result",
                quote_keys=("quote-1",),
                claim_strength="explicit",
                epistemic_status=EpistemicStatus.REPORTED_RESULT,
            ),
        ),
        method_fact_proposals=(
            LiteratureMethodFactLLMWireProposal(
                fact_key="method-1",
                text="The calculation used VASP.",
                claim_keys=("claim-1",),
                attribute_entries=(
                    LiteratureKnowledgeLLMWireEntry(key="software", value="VASP"),
                ),
            ),
        ),
        model_fact_proposals=(
            LiteratureModelFactLLMWireProposal(
                fact_key="model-1",
                text="The model was periodic.",
                claim_keys=("claim-1",),
                attribute_entries=(
                    LiteratureKnowledgeLLMWireEntry(
                        key="boundary",
                        value="periodic",
                    ),
                ),
            ),
        ),
        reported_result_proposals=(
            LiteratureReportedResultLLMWireProposal(
                result_key="result-1",
                claim_keys=("claim-1",),
                quantity="activation_barrier",
                value=1.25,
                unit="eV",
                system_context_entries=(
                    LiteratureKnowledgeLLMWireEntry(
                        key="catalyst",
                        value="Pd-Au",
                    ),
                ),
                method_context_entries=(
                    LiteratureKnowledgeLLMWireEntry(key="method", value="DFT"),
                ),
                method_fact_keys=("method-1",),
                model_fact_keys=("model-1",),
                result_status=ResultStatus.COMPUTED_REPORTED,
            ),
        ),
        relation_proposals=(),
    )

    internal = literature_knowledge_wire_to_internal(response)

    assert isinstance(internal, LiteratureKnowledgeLLMResponse)
    assert internal.method_fact_proposals[0].attributes == {"software": "VASP"}
    assert internal.model_fact_proposals[0].attributes == {"boundary": "periodic"}
    assert internal.reported_result_proposals[0].system_context == {
        "catalyst": "Pd-Au"
    }
    assert internal.reported_result_proposals[0].method_context == {"method": "DFT"}


def test_wire_response_rejects_duplicate_context_keys() -> None:
    entry = {"key": "method", "value": "DFT"}
    payload = _empty_wire_response().model_dump(mode="json")
    payload["reported_result_proposals"] = [
        {
            "result_key": "result-1",
            "claim_keys": ["claim-1"],
            "quantity": "barrier",
            "value": 1.25,
            "unit": "eV",
            "system_context_entries": [],
            "method_context_entries": [entry, entry],
            "method_fact_keys": [],
            "model_fact_keys": [],
            "result_status": "computed_reported",
        }
    ]

    with pytest.raises(ValidationError, match="duplicate key"):
        LiteratureKnowledgeLLMWireResponse.model_validate(payload)
