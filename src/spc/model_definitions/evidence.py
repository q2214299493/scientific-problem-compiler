from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import Field, model_validator

from ..immutable import FrozenDict

from .base import NonBlankStr, Sha256Str, StrictModel


class EvidenceClassification(StrEnum):
    EVIDENCE = "evidence"
    ASSUMPTION = "assumption"
    DEFAULT = "default"
    PROPOSED_DEVIATION = "proposed_deviation"
    UNKNOWN = "unknown"


class EpistemicStatus(StrEnum):
    SOURCE_REPORTED = "source_reported"
    SOURCE_HYPOTHESIS = "source_hypothesis"
    SOURCE_INTERPRETATION = "source_interpretation"
    BACKGROUND_STATEMENT = "background_statement"
    METHOD_STATEMENT = "method_statement"
    MODEL_STATEMENT = "model_statement"
    REPORTED_RESULT = "reported_result"
    UNRESOLVED = "unresolved"


class ResultStatus(StrEnum):
    LITERATURE_REPORTED = "literature_reported"
    EXPERIMENTAL_REPORTED = "experimental_reported"
    COMPUTED_REPORTED = "computed_reported"
    PREDICTED_REPORTED = "predicted_reported"
    UNKNOWN_ORIGIN = "unknown_origin"


class EvidenceAssessmentStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"
    INCOMPARABLE = "incomparable"


class SourceRole(StrEnum):
    AUTHOR = "author"
    REVIEWER = "reviewer"
    LITERATURE_AUTHOR = "literature_author"
    INTERNAL_RESEARCHER = "internal_researcher"
    SYSTEM = "system"
    UNSPECIFIED = "unspecified"


class SourceType(StrEnum):
    MANUSCRIPT = "manuscript"
    SUPPORTING_INFORMATION = "supporting_information"
    REVIEWER_COMMENT = "reviewer_comment"
    AUTHOR_RESPONSE = "author_response"
    LITERATURE_ARTICLE = "literature_article"
    CALCULATION_ARCHIVE = "calculation_archive"
    INTERNAL_NOTE = "internal_note"
    EXPERT_WEBPAGE = "expert_webpage"
    UNSPECIFIED = "unspecified"


class SourceDocument(StrictModel):
    source_id: str
    version: str
    title: NonBlankStr
    media_type: str = "text/plain"
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stored_path: str
    source_role: SourceRole = SourceRole.UNSPECIFIED
    source_type: SourceType = SourceType.UNSPECIFIED
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    read_only: bool = True

    @model_validator(mode="after")
    def validate_provenance(self) -> SourceDocument:
        if self.source_type == SourceType.AUTHOR_RESPONSE and self.source_role != SourceRole.AUTHOR:
            raise ValueError("author_response source_type requires author source_role")
        if self.source_type == SourceType.REVIEWER_COMMENT and self.source_role != SourceRole.REVIEWER:
            raise ValueError("reviewer_comment source_type requires reviewer source_role")
        if self.source_type == SourceType.LITERATURE_ARTICLE and self.source_role == SourceRole.REVIEWER:
            raise ValueError("literature_article cannot use reviewer source_role")
        return self


class EvidenceSpan(StrictModel):
    evidence_id: str
    source_id: str
    source_version: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text: NonBlankStr
    locator: str | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> EvidenceSpan:
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class EvidenceReference(StrictModel):
    evidence_id: str
    source_id: str
    source_version: str


class GroundedStatement(StrictModel):
    statement_id: str
    text: NonBlankStr
    classification: EvidenceClassification
    evidence_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_grounding(self) -> GroundedStatement:
        if self.classification == EvidenceClassification.EVIDENCE and not self.evidence_refs:
            raise ValueError("evidence-classified statements require evidence_refs")
        return self


class SourceQuote(StrictModel):
    quote_id: NonBlankStr
    evidence_ref: NonBlankStr
    relative_start_offset: int = Field(ge=0)
    relative_end_offset: int = Field(gt=0)
    text: NonBlankStr
    source_id: NonBlankStr
    source_version: NonBlankStr
    source_role: SourceRole
    source_type: SourceType

    @model_validator(mode="after")
    def validate_quote_id(self) -> SourceQuote:
        from ..serialization import content_hash

        if self.relative_end_offset <= self.relative_start_offset:
            raise ValueError("SourceQuote end offset must be greater than start offset")
        identity = {
            "evidence_ref": self.evidence_ref,
            "relative_start_offset": self.relative_start_offset,
            "relative_end_offset": self.relative_end_offset,
            "text_hash": content_hash({"text": self.text}),
        }
        if self.quote_id != f"quote-{content_hash(identity)[:24]}":
            raise ValueError("SourceQuote quote_id is not content-bound")
        return self


class SourceClaim(StrictModel):
    claim_id: NonBlankStr
    text: NonBlankStr
    claim_type: NonBlankStr
    source_role: SourceRole
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    source_quote_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    claim_strength: NonBlankStr
    epistemic_status: EpistemicStatus

    @model_validator(mode="after")
    def validate_quote_bindings(self) -> SourceClaim:
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("SourceClaim evidence_refs must be unique")
        if len(set(self.source_quote_refs)) != len(self.source_quote_refs):
            raise ValueError("SourceClaim source_quote_refs must be unique")
        return self


class ResultContext(StrictModel):
    context_id: NonBlankStr
    system_context: FrozenDict
    method_context: FrozenDict
    method_fact_refs: tuple[NonBlankStr, ...] = ()
    model_fact_refs: tuple[NonBlankStr, ...] = ()

    @model_validator(mode="after")
    def validate_context_id(self) -> ResultContext:
        from ..serialization import content_hash

        identity = self.model_dump(mode="json", exclude={"context_id"})
        if self.context_id != f"result-context-{content_hash(identity)[:24]}":
            raise ValueError("ResultContext context_id is not content-bound")
        return self


class ReportedResult(StrictModel):
    result_id: NonBlankStr
    quantity: NonBlankStr
    value: float = Field(allow_inf_nan=False)
    unit: NonBlankStr
    system_context: FrozenDict
    method_context: FrozenDict
    result_context: ResultContext | None = None
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    result_status: ResultStatus


class ReportedObservation(StrictModel):
    """A provenance-bound qualitative observation without numeric semantics."""

    observation_id: NonBlankStr
    quantity: NonBlankStr
    qualitative_value: NonBlankStr
    system_context: FrozenDict
    method_context: FrozenDict
    result_context: ResultContext | None = None
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    result_status: ResultStatus


class MethodFact(StrictModel):
    fact_id: NonBlankStr
    text: NonBlankStr
    attributes: FrozenDict
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    epistemic_status: EpistemicStatus = EpistemicStatus.METHOD_STATEMENT

    @model_validator(mode="after")
    def validate_status(self) -> MethodFact:
        if self.epistemic_status != EpistemicStatus.METHOD_STATEMENT:
            raise ValueError("MethodFact must remain a method_statement")
        return self


class ModelFact(StrictModel):
    fact_id: NonBlankStr
    text: NonBlankStr
    attributes: FrozenDict
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)
    epistemic_status: EpistemicStatus = EpistemicStatus.MODEL_STATEMENT

    @model_validator(mode="after")
    def validate_status(self) -> ModelFact:
        if self.epistemic_status != EpistemicStatus.MODEL_STATEMENT:
            raise ValueError("ModelFact must remain a model_statement")
        return self


class EvidenceAssessment(StrictModel):
    assessment_id: NonBlankStr
    claim_ref: NonBlankStr
    supporting_evidence_refs: tuple[NonBlankStr, ...] = ()
    contradicting_evidence_refs: tuple[NonBlankStr, ...] = ()
    assessment: EvidenceAssessmentStatus
    limitations: tuple[NonBlankStr, ...] = ()
    confidence_basis: NonBlankStr


class ConflictSet(StrictModel):
    conflict_id: NonBlankStr
    topic: NonBlankStr
    claim_refs: tuple[NonBlankStr, ...] = Field(min_length=2)
    conflict_type: NonBlankStr
    possible_causes: tuple[NonBlankStr, ...] = ()
    required_discrimination: tuple[NonBlankStr, ...] = Field(min_length=1)
    resolution_status: NonBlankStr


class ComparisonConstraint(StrictModel):
    constraint_id: NonBlankStr
    comparison_target: NonBlankStr
    must_match_fields: tuple[NonBlankStr, ...] = Field(min_length=1)
    may_vary_fields: tuple[NonBlankStr, ...] = ()
    disclosure_required_fields: tuple[NonBlankStr, ...] = ()
    rationale: NonBlankStr
    evidence_refs: tuple[NonBlankStr, ...] = Field(min_length=1)


class EvidenceGap(StrictModel):
    gap_id: NonBlankStr
    scientific_question: NonBlankStr
    missing_evidence: NonBlankStr
    why_it_matters: NonBlankStr
    blocking: bool
    candidate_capabilities: tuple[NonBlankStr, ...] = ()
    evidence_refs: tuple[NonBlankStr, ...] = ()


class InterpretationProposal(StrictModel):
    proposal_id: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    provider_id: NonBlankStr
    provider_version: NonBlankStr
    source_quotes: tuple[SourceQuote, ...] = ()
    source_claims: tuple[SourceClaim, ...] = ()
    reported_results: tuple[ReportedResult, ...] = ()
    method_facts: tuple[MethodFact, ...] = ()
    model_facts: tuple[ModelFact, ...] = ()
    evidence_assessments: tuple[EvidenceAssessment, ...] = ()
    conflict_sets: tuple[ConflictSet, ...] = ()
    comparison_constraints: tuple[ComparisonConstraint, ...] = ()
    evidence_gaps: tuple[EvidenceGap, ...] = ()
    unknowns: tuple[NonBlankStr, ...] = ()
    assumption_candidates: tuple[NonBlankStr, ...] = ()
    capability_candidates: tuple[NonBlankStr, ...] = ()


class ScientificEvidencePacket(StrictModel):
    packet_id: NonBlankStr
    context_id: NonBlankStr
    context_hash: Sha256Str
    source_quotes: tuple[SourceQuote, ...] = ()
    source_claims: tuple[SourceClaim, ...] = ()
    reported_results: tuple[ReportedResult, ...] = ()
    reported_observations: tuple[ReportedObservation, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    method_facts: tuple[MethodFact, ...] = ()
    model_facts: tuple[ModelFact, ...] = ()
    evidence_assessments: tuple[EvidenceAssessment, ...] = ()
    conflict_sets: tuple[ConflictSet, ...] = ()
    comparison_constraints: tuple[ComparisonConstraint, ...] = ()
    evidence_gaps: tuple[EvidenceGap, ...] = ()
    unknowns: tuple[NonBlankStr, ...] = ()
    assumption_candidates: tuple[NonBlankStr, ...] = ()
    capability_candidates: tuple[NonBlankStr, ...] = ()
    provenance_manifest: FrozenDict
    content_hash: Sha256Str

    @model_validator(mode="after")
    def validate_identity(self) -> ScientificEvidencePacket:
        from ..serialization import content_hash

        identity = self.model_dump(mode="json", exclude={"packet_id", "content_hash"})
        legacy_identity = self.model_dump(mode="json", exclude={"packet_id", "content_hash"})
        legacy_identity.pop("source_quotes", None)
        for claim in legacy_identity["source_claims"]:
            claim.pop("source_quote_refs", None)
        for result in legacy_identity["reported_results"]:
            result.pop("result_context", None)
        valid_ids = {
            f"evidence-packet-{content_hash(identity)[:24]}",
            f"evidence-packet-{content_hash(legacy_identity)[:24]}",
        }
        if self.packet_id not in valid_ids:
            raise ValueError("ScientificEvidencePacket packet_id is not content-bound")
        expected_hash = content_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        )
        legacy_payload = {"packet_id": self.packet_id, **legacy_identity}
        if self.content_hash not in {expected_hash, content_hash(legacy_payload)}:
            raise ValueError("ScientificEvidencePacket content hash is invalid")
        return self
