"""Protection for persisted identities created before the knowledge module split."""

from spc.knowledge.structure import (
    create_evidence_from_figure_caption,
    create_evidence_from_table_cell,
    verify_structured_evidence_locator,
)
from spc.repositories import KnowledgeRepositories

from test_document_structure import html_structure


def test_historical_html_structure_and_evidence_identities(tmp_path):
    repositories, store, outcome, structure = html_structure(tmp_path)
    assert outcome.representation_id == "literature-representation-7f7192cc0a428dada446f5aa"
    assert structure.artifact.structure_id == "document-structure-04cb69a131152b17d06d29f4"

    cell = create_evidence_from_table_cell(
        structure.artifact.structure_id, structure.tables[0].table_id,
        structure.table_cells[-1].cell_id, repositories, store,
    )
    caption = create_evidence_from_figure_caption(
        structure.artifact.structure_id, structure.figures[0].figure_id, repositories, store,
    )
    reopened = KnowledgeRepositories(repositories.root)
    for (evidence, locator), expected_span, expected_locator in (
        (cell, "evidence-4e53ef2dd414cfaeae594c11", "structured-locator-d7641c26ccf1a2ab84855972"),
        (caption, "evidence-c5ac59b2432a1185835820f1", "structured-locator-de87bfee123d73aa932ae6e4"),
    ):
        assert evidence.evidence_id == expected_span
        assert locator.locator_id == expected_locator
        assert verify_structured_evidence_locator(locator, reopened, reopened.evidence_store) == locator
