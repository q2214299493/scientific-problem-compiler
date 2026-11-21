"""Contracts shared by record-specific trust checks."""
from collections.abc import Callable, Mapping

from pydantic import BaseModel

from ...models import EvidenceSpan, KnowledgeCurationRecord, SourceDocument

RecordKey = tuple[str, str]
Curations = Mapping[RecordKey, KnowledgeCurationRecord]
VisitRecord = Callable[[RecordKey], BaseModel]
RequireRecord = Callable[[str, str], BaseModel]
VerifySource = Callable[[str, str], SourceDocument]
VerifyEvidence = Callable[[str], tuple[EvidenceSpan, SourceDocument]]
VerifyEvidenceRefs = Callable[[tuple[str, ...]], None]
RememberEvidence = Callable[[EvidenceSpan, SourceDocument], None]


CURATION_REQUIRED_TYPES = frozenset(
    {
        "expert_source",
        "literature_document",
        "literature_representation_selection",
        "literature_supplementary_source",
        "expert_opinion",
        "source_claim",
        "method_fact",
        "model_fact",
        "reported_result",
        "knowledge_relation",
    }
)


REPRESENTATION_COMPONENT_TYPES = frozenset(
    {
        "raw_literature_artifact",
        "canonical_text_artifact",
        "literature_ingestion",
        "raw_html_literature_artifact",
        "canonical_html_text_artifact",
        "html_literature_ingestion",
    }
)


class TrustedKnowledgeError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")
