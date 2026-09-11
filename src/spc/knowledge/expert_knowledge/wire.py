from __future__ import annotations

from ...models import NonBlankStr, StrictModel
from ..literature_knowledge.wire import validate_strict_structured_output_schema
from .contracts import (
    ExpertCaseProposal,
    ExpertKnowledgeLLMResponse,
    ExpertOpinionProposal,
    ExpertQuoteProposal,
)


class ExpertQuoteLLMWireProposal(StrictModel):
    quote_key: NonBlankStr
    chunk_id: NonBlankStr
    exact_text: NonBlankStr


class ExpertOpinionLLMWireProposal(StrictModel):
    opinion_key: NonBlankStr
    quote_keys: tuple[NonBlankStr, ...]
    normalized_text: NonBlankStr
    opinion_category: NonBlankStr
    authority_status: NonBlankStr
    expert_id: NonBlankStr
    topic: NonBlankStr
    scope: NonBlankStr
    rationale: NonBlankStr
    conditions: tuple[NonBlankStr, ...]


class ExpertCaseLLMWireProposal(StrictModel):
    case_key: NonBlankStr
    opinion_keys: tuple[NonBlankStr, ...]
    original_wording: NonBlankStr
    latent_concern: NonBlankStr
    atomic_questions: tuple[NonBlankStr, ...]
    good_question_formulations: tuple[NonBlankStr, ...]
    wrong_formulations: tuple[NonBlankStr, ...]
    answerability_conditions: tuple[NonBlankStr, ...]
    required_evidence_types: tuple[NonBlankStr, ...]
    baseline_guidance: tuple[NonBlankStr, ...]
    common_misinterpretations: tuple[NonBlankStr, ...]
    resolution_pattern: NonBlankStr
    applicability: tuple[NonBlankStr, ...]


class ExpertKnowledgeLLMWireResponse(StrictModel):
    quote_proposals: tuple[ExpertQuoteLLMWireProposal, ...]
    opinion_proposals: tuple[ExpertOpinionLLMWireProposal, ...]
    case_proposals: tuple[ExpertCaseLLMWireProposal, ...]


def expert_knowledge_wire_to_internal(
    response: ExpertKnowledgeLLMWireResponse,
) -> ExpertKnowledgeLLMResponse:
    return ExpertKnowledgeLLMResponse(
        quote_proposals=tuple(
            ExpertQuoteProposal(**item.model_dump(mode="python"))
            for item in response.quote_proposals
        ),
        opinion_proposals=tuple(
            ExpertOpinionProposal(**item.model_dump(mode="python"))
            for item in response.opinion_proposals
        ),
        case_proposals=tuple(
            ExpertCaseProposal(**item.model_dump(mode="python"))
            for item in response.case_proposals
        ),
    )


def expert_knowledge_llm_wire_schema() -> dict:
    schema = ExpertKnowledgeLLMWireResponse.model_json_schema()
    validate_strict_structured_output_schema(schema)
    return schema


__all__ = [
    "ExpertCaseLLMWireProposal",
    "ExpertKnowledgeLLMWireResponse",
    "ExpertOpinionLLMWireProposal",
    "ExpertQuoteLLMWireProposal",
    "expert_knowledge_llm_wire_schema",
    "expert_knowledge_wire_to_internal",
]
