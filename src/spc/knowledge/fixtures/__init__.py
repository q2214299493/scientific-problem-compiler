from __future__ import annotations

from pathlib import Path

from ...models import (
    ExpertOpinion,
    ExpertProfile,
    KnowledgeRelation,
    LiteratureDocument,
    LiteratureWorkflowPattern,
    SourceClaim,
)
from ...repositories import KnowledgeRepositories
from ...serialization import load_data


def load_knowledge_fixture(
    path: Path,
    repositories: KnowledgeRepositories,
) -> None:
    """Load an explicit offline fixture through the normal immutable repositories."""

    payload = load_data(path)
    if not isinstance(payload, dict):
        raise ValueError("knowledge fixture must be an object")
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
    repositories.load_literature_documents(literature)
    repositories.load_expert_profiles(profiles)
    repositories.load_expert_opinions(opinions)
    for claim in claims:
        repositories.source_claims.put(claim.claim_id, claim)
    repositories.load_workflow_patterns(workflows)
    repositories.load_relations(relations)
