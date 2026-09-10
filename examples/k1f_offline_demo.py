"""Offline two-document K1F pipeline demonstration.

This proves deterministic plumbing and trust gating. It does not measure
scientific extraction accuracy because it intentionally uses the mock provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from spc.knowledge.acquisition import (
    HTTPResponse,
    LiteratureAcquisitionService,
    SafeHTTPFetcher,
)
from spc.knowledge.ingestion import LiteratureRepresentationSelector
from spc.knowledge.literature_knowledge import (
    LiteratureKnowledgeCompiler,
    LiteratureScientificKnowledgeViewBuilder,
    MockLiteratureKnowledgeProvider,
    curate_knowledge_record,
)
from spc.knowledge.structure import DocumentStructureService
from spc.models import CurationStatus
from spc.repositories import KnowledgeEvidenceStore, KnowledgeRepositories


def article(doi: str, title: str, value: str) -> bytes:
    return f"""<html><head>
    <meta name="citation_title" content="{title}">
    <meta name="citation_author" content="A. Demo Author">
    <meta name="citation_publication_date" content="2026">
    <meta name="citation_doi" content="{doi}">
    </head><body><article><h1>Results</h1>
    <p>The source reports one bounded offline demonstration.</p>
    <table><caption>Table 1 Barrier</caption>
    <tr><th>Method</th><th>Barrier (eV)</th></tr>
    <tr><td>DFT</td><td>{value}</td></tr>
    </table></article></body></html>""".encode()


class OfflineHTMLTransport:
    def __init__(self, resources: dict[str, bytes]) -> None:
        self.resources = resources

    def request(
        self,
        url: str,
        *,
        validated_ips: tuple[str, ...],
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse:
        body = self.resources[url]
        assert validated_ips and timeout > 0 and len(body) < max_bytes
        return HTTPResponse(
            url=url,
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            connected_ip="93.184.216.34",
        )


def run(knowledge_dir: Path) -> dict[str, object]:
    urls = (
        "https://demo.example/k1f-a",
        "https://demo.example/k1f-b",
    )
    resources = {
        urls[0]: article("10.0000/k1f.demo-a", "K1F Demo A", "1.25"),
        urls[1]: article("10.0000/k1f.demo-b", "K1F Demo B", "0.75"),
    }
    repositories = KnowledgeRepositories(knowledge_dir)
    store = KnowledgeEvidenceStore(knowledge_dir)
    acquisition = LiteratureAcquisitionService(
        SafeHTTPFetcher(
            OfflineHTMLTransport(resources),
            dns_resolver=lambda _host: ("93.184.216.34",),
        )
    )
    summaries = []
    table_ids = []
    for url in urls:
        outcome = acquisition.add(url, "base", repositories, store)
        literature_id = outcome.literature_id or ""
        representation_id = outcome.representation_id or ""
        LiteratureRepresentationSelector().select(
            literature_id,
            representation_id,
            "k1f-offline-demo",
            "Select the deterministic offline demonstration representation.",
            repositories,
            store,
        )
        structure = DocumentStructureService().extract(
            literature_id,
            representation_id,
            repositories,
            store,
        )
        document = repositories.literature_documents.get(literature_id)
        selection = repositories.literature_representation_selections.resolve_current(
            literature_id
        )
        for target_type, target_id in (
            ("literature_document", document.literature_id),
            ("literature_representation_selection", selection.selection_id),
        ):
            curate_knowledge_record(
                repositories,
                target_type=target_type,
                target_id=target_id,
                status=CurationStatus.ACCEPTED,
                curator_id="k1f-offline-demo",
                rationale="Accept deterministic source authority for the offline demo.",
            )
        compiled = LiteratureKnowledgeCompiler().compile(
            literature_id,
            MockLiteratureKnowledgeProvider(),
            repositories,
            store,
        )
        audit = LiteratureScientificKnowledgeViewBuilder().build(
            literature_id, repositories, store
        )
        claim = compiled.materialized.source_claims[0]
        curate_knowledge_record(
            repositories,
            target_type="source_claim",
            target_id=claim.claim_id,
            status=CurationStatus.ACCEPTED,
            curator_id="k1f-offline-demo-reviewer",
            rationale="Explicitly accept the exact grounded demo claim.",
        )
        trusted = LiteratureScientificKnowledgeViewBuilder().build(
            literature_id,
            repositories,
            store,
            view_mode="trusted_current",
        )
        value_chunk = next(chunk for chunk in compiled.chunks if chunk.text in {"1.25", "0.75"})
        table_ids.append(value_chunk.table_context["table_id"])
        summaries.append(
            {
                "literature_id": literature_id,
                "structure_id": structure.artifact.structure_id,
                "audit_status": audit.records[0].curation_status,
                "trusted_record_ids": [item.record_id for item in trusted.records],
                "supporting_quote": trusted.records[0].supporting_quotes[0].exact_text,
                "locator": trusted.records[0].supporting_quotes[0].locator,
                "table_id": value_chunk.table_context["table_id"],
            }
        )
    return {
        "knowledge_dir": str(knowledge_dir),
        "provider": "mock-literature-knowledge (pipeline test only)",
        "documents": summaries,
        "table_context_isolated": len(set(table_ids)) == len(table_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-dir", type=Path)
    args = parser.parse_args()
    root = args.knowledge_dir or Path(tempfile.mkdtemp(prefix="spc-k1f-demo-"))
    print(json.dumps(run(root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
