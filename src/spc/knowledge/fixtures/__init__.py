from __future__ import annotations

from pathlib import Path

from ...models import (
    EvidenceSpan,
    ExpertAttributionRecord,
    ExpertOpinion,
    ExpertProfile,
    KnowledgeCurationRecord,
    KnowledgeRelation,
    LiteratureDocument,
    LiteratureWorkflowPattern,
    SourceClaim,
    SourceQuote,
)
from ...repositories import KnowledgeRepositories, SourceEvidenceStore
from ...serialization import load_data


def load_knowledge_fixture(
    path: Path,
    repositories: KnowledgeRepositories,
    evidence_store: SourceEvidenceStore | None = None,
) -> None:
    """Load an explicit offline fixture through the normal immutable repositories."""

    payload = load_data(path)
    if not isinstance(payload, dict):
        raise ValueError("knowledge fixture must be an object")
    if evidence_store is not None:
        fixture_root = path.parent.resolve()
        for item in payload.get("source_artifacts", ()):
            artifact = (fixture_root / item["path"]).resolve()
            if not artifact.is_relative_to(fixture_root) or artifact.is_symlink():
                raise ValueError("fixture source artifact path is unsafe")
            evidence_store.ingest(
                artifact,
                item["source_id"],
                item["source_version"],
                item.get("title"),
                source_role=item.get("source_role", "unspecified"),
                source_type=item.get("source_type", "unspecified"),
            )
        for item in payload.get("evidence_spans", ()):
            evidence_store.add_evidence(EvidenceSpan.model_validate(item))
    literature = tuple(
        LiteratureDocument.model_validate(item)
        for item in payload.get("literature_documents", ())
    )
    profiles = tuple(
        ExpertProfile.model_validate(item)
        for item in payload.get("expert_profiles", ())
    )
    opinions = tuple(
        ExpertOpinion.model_validate(item)
        for item in payload.get("expert_opinions", ())
    )
    attributions = tuple(
        ExpertAttributionRecord.model_validate(item)
        for item in payload.get("expert_attributions", ())
    )
    quotes = tuple(
        SourceQuote.model_validate(item) for item in payload.get("source_quotes", ())
    )
    claims = tuple(
        SourceClaim.model_validate(item) for item in payload.get("source_claims", ())
    )
    workflows = tuple(
        LiteratureWorkflowPattern.model_validate(item)
        for item in payload.get("workflow_patterns", ())
    )
    relations = tuple(
        KnowledgeRelation.model_validate(item)
        for item in payload.get("relations", ())
    )
    curations = tuple(
        KnowledgeCurationRecord.model_validate(item)
        for item in payload.get("curations", ())
    )
    repositories.load_literature_documents(literature)
    repositories.load_expert_profiles(profiles)
    repositories.load_expert_attributions(attributions)
    repositories.load_expert_opinions(opinions)
    for quote in quotes:
        repositories.source_quotes.put(quote.quote_id, quote)
    for claim in claims:
        repositories.source_claims.put(claim.claim_id, claim)
    repositories.load_workflow_patterns(workflows)
    repositories.load_relations(relations)
    repositories.load_curations(curations)
