from __future__ import annotations

from pathlib import Path

from ..models import (
    MethodFact,
    ModelFact,
    ReportedResult,
    ScientificCapability,
    SourceClaim,
    SourceQuote,
)

from .base import CatalogRepository, IdentityBoundRepository


class ScientificCapabilityRepository(CatalogRepository[ScientificCapability]):
    _key_error = "capability repository key must equal capability_id"
    _stored_id_error = "stored capability ID does not match repository key"
    _duplicate_error = "capability repository contains duplicate record IDs"

    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root / "capabilities", ScientificCapability, "capability_id")


class ScientificRecordRepositories:
    def __init__(self, root: Path) -> None:
        self.capabilities = ScientificCapabilityRepository(root)
        self.source_quotes = IdentityBoundRepository(
            root / "source_quotes", SourceQuote, "quote_id"
        )
        self.source_claims = IdentityBoundRepository(
            root / "source_claims", SourceClaim, "claim_id"
        )
        self.reported_results = IdentityBoundRepository(
            root / "reported_results", ReportedResult, "result_id"
        )
        self.method_facts = IdentityBoundRepository(
            root / "method_facts", MethodFact, "fact_id"
        )
        self.model_facts = IdentityBoundRepository(
            root / "model_facts", ModelFact, "fact_id"
        )
