"""A new selected structure may reuse an immutable exact source span."""

from types import SimpleNamespace

import pytest

from spc.knowledge.literature_knowledge import materializer
from spc.models import EvidenceSpan
from spc.serialization import content_hash


def _evidence(text: str) -> EvidenceSpan:
    identity = {
        "canonical_text_id": "canonical-1",
        "start_offset": 10,
        "end_offset": 10 + len(text),
        "text_hash": content_hash({"text": text}),
    }
    return EvidenceSpan(
        evidence_id=f"evidence-{content_hash(identity)[:24]}",
        source_id="source-1",
        source_version="version-1",
        content_sha256="a" * 64,
        start_offset=10,
        end_offset=10 + len(text),
        text=text,
        locator="K1F exact quote in old-structure-block",
    )


def test_existing_exact_evidence_is_reused_across_structure_versions(monkeypatch):
    existing = _evidence("163")
    source = SimpleNamespace(content_sha256="a" * 64)

    class Store:
        def get_evidence(self, evidence_id):
            assert evidence_id == existing.evidence_id
            return existing

        def verify_evidence_integrity(self, evidence):
            assert evidence is existing
            return source

    monkeypatch.setattr(
        materializer,
        "create_evidence_span_from_canonical_text",
        lambda *args, **kwargs: pytest.fail("must not rewrite an existing span"),
    )
    compilation_input = SimpleNamespace(
        canonical_text_id="canonical-1",
        source_id="source-1",
        source_version="version-1",
    )
    rebound = materializer._rebind_or_create_quote_evidence(
        compilation_input, 10, 13, "163", "new-structure-block", None, Store()
    )
    assert rebound is existing
    assert rebound.locator == "K1F exact quote in old-structure-block"


def test_conflicting_existing_span_is_rejected():
    existing = _evidence("163")

    class Store:
        def get_evidence(self, evidence_id):
            return existing

        def verify_evidence_integrity(self, evidence):
            return SimpleNamespace(content_sha256="a" * 64)

    compilation_input = SimpleNamespace(
        canonical_text_id="canonical-1",
        source_id="another-source",
        source_version="version-1",
    )
    with pytest.raises(ValueError, match="conflicts with current canonical source"):
        materializer._rebind_or_create_quote_evidence(
            compilation_input, 10, 13, "163", "new-structure-block", None, Store()
        )
