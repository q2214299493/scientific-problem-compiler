"""Protection for persisted identities created before the knowledge module split."""

import ast
from graphlib import TopologicalSorter
import hashlib
from importlib.util import resolve_name
from pathlib import Path
import subprocess
import sys

import pytest

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

def test_public_extractors_and_locators_keep_legacy_object_identity():
    import spc.knowledge as knowledge
    from spc.knowledge import acquisition, ingestion, representation, structure
    from spc.knowledge.acquisition import resolvers, transport
    from spc.knowledge.ingestion import docx, html, pdf
    from spc.knowledge.structure import evidence, validation

    assert acquisition.HTMLLiteratureTextExtractor is ingestion.HTMLLiteratureTextExtractor is html.HTMLLiteratureTextExtractor
    assert ingestion.PypdfLiteratureTextExtractor is pdf.PypdfLiteratureTextExtractor
    assert ingestion.DocxSupplementaryTextExtractor is docx.DocxSupplementaryTextExtractor
    assert acquisition.SafeHTTPFetcher is transport.SafeHTTPFetcher
    assert acquisition.PinnedHTTPTransport is transport.PinnedHTTPTransport
    assert acquisition.DOIResolver is resolvers.DOIResolver
    assert acquisition.ArticleURLResolver is resolvers.ArticleURLResolver
    assert acquisition.LocalPDFResolver is resolvers.LocalPDFResolver
    assert acquisition.validate_literature_representation is representation.validate_literature_representation
    assert ingestion.LiteratureRepresentationSelector is representation.LiteratureRepresentationSelector
    assert structure.validate_document_structure is validation.validate_document_structure
    assert structure._create_locator is structure.create_structured_evidence_locator is evidence.create_structured_evidence_locator
    assert knowledge.LiteratureAcquisitionService is acquisition.LiteratureAcquisitionService
    assert knowledge.LiteratureIngestionService is ingestion.LiteratureIngestionService
    assert knowledge.DocumentStructureService is structure.DocumentStructureService
    for name in knowledge.__all__:
        assert getattr(knowledge, name) is not None


@pytest.mark.parametrize(("module", "forbidden"), [
    ("representation", ("acquisition", "ingestion", "structure")),
    ("ingestion", ("acquisition", "structure")),
    ("acquisition", ("structure",)),
    ("structure", ("acquisition",)),
])
def test_cold_import_only_loads_required_literature_domains(module, forbidden):
    # Exercise the package initializer in a fresh interpreter for each entry point.
    script = """
import importlib
import sys

importlib.import_module("spc.knowledge." + sys.argv[1])
for domain in sys.argv[2:]:
    prefix = "spc.knowledge." + domain
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
"""
    subprocess.run([sys.executable, "-c", script, module, *forbidden], check=True)


def test_extractors_share_canonical_text_and_block_contract(tmp_path):
    from spc.knowledge.ingestion import (
        CanonicalTextResult,
        DocxSupplementaryTextExtractor,
        HTMLLiteratureTextExtractor,
        PypdfLiteratureTextExtractor,
    )
    from test_document_structure import ARTICLE_URL, PDF, structured_html
    from test_literature_supplementary import _supplement_docx

    repos = KnowledgeRepositories(tmp_path / "knowledge")
    docx = _supplement_docx(tmp_path / "supplement.docx")
    pdf_artifact = repos.raw_literature_artifacts.put(PDF, "literature-test")
    docx_artifact = repos.raw_literature_artifacts.put(docx, "literature-test", allow_docx=True)
    extractions: tuple[CanonicalTextResult, ...] = (
        PypdfLiteratureTextExtractor().extract(pdf_artifact, PDF),
        DocxSupplementaryTextExtractor().extract(docx_artifact, docx),
        HTMLLiteratureTextExtractor().extract(ARTICLE_URL, structured_html()),
    )
    for result in extractions:
        assert result.canonical_text and result.blocks
        for block in result.blocks:
            text = result.canonical_text[block.start_offset:block.end_offset]
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == block.text_hash


def test_literature_module_dependencies_are_acyclic():
    import spc.knowledge as knowledge

    root = Path(knowledge.__file__).parent
    files = [root / name for name in ("literature_text.py", "representation.py", "structure_selection.py")]
    for package in ("acquisition", "ingestion", "structure"):
        files.extend((root / package).glob("*.py"))
    modules = {}
    for path in files:
        name = ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
        modules["spc.knowledge." + name] = path

    dependencies = {}
    for module, path in modules.items():
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        imports = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                target = resolve_name("." * node.level + (node.module or ""), package)
                imports.add(target)
                imports.update(target + "." + alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
        dependencies[module] = imports.intersection(modules)
    assert set(TopologicalSorter(dependencies).static_order()) == set(modules)


@pytest.mark.parametrize(("domain", "expected_files"), [("pdf", 17), ("html", 39), ("docx", 20)])
def test_reads_persisted_baseline_without_rewriting_records(tmp_path, domain, expected_files):
    from zipfile import ZipFile

    from spc.knowledge.ingestion import validate_literature_supplementary_source
    from spc.knowledge.representation import validate_literature_representation
    from spc.knowledge.structure import validate_document_structure
    from spc.serialization import file_sha256

    archive = Path(__file__).parent / "fixtures" / "knowledge-01a-3f53101.zip"
    with ZipFile(archive) as fixture:
        for item in fixture.infolist():
            if item.filename.startswith(domain + "/"):
                fixture.extract(item, tmp_path)
    root = tmp_path / domain

    def file_hashes():
        return {p.relative_to(root): file_sha256(p) for p in root.rglob("*") if p.is_file()}

    before = file_hashes()
    assert len(before) == expected_files
    repositories = KnowledgeRepositories(root)
    store = repositories.evidence_store
    (representation,) = repositories.literature_representation_refs.list()
    assert validate_literature_representation(representation, repositories, store) == representation
    if domain == "docx":
        (supplement,) = repositories.literature_supplementary_sources.list()
        _, canonical, source = validate_literature_supplementary_source(supplement, repositories, store)
        assert source.content_sha256 == canonical.text_sha256
    else:
        (structure,) = repositories.document_structure_artifacts.list()
        assert validate_document_structure(structure.structure_id, repositories, store) == structure
    if domain == "html":
        from spc.knowledge.ingestion import create_evidence_span_from_canonical_text
        from spc.models import literature_identity_id
        from spc.serialization import content_hash

        (document,) = repositories.literature_documents.list()
        assert literature_identity_id(
            title=document.title, authors=document.authors, year=document.year, doi=document.doi
        ) == document.literature_id
        assert document.content_hash == "f82eeb8f171f3b00ac68ed0f270393c22031426bff28df2ab52aea3129f6b616"
        expected_spans = {
            "evidence-4e53ef2dd414cfaeae594c11": "ab99205b9c69af2b3645a78ef661114a9aa90deadea71ab742532bb9cd8c53d8",
            "evidence-c5ac59b2432a1185835820f1": "e3084046d7f77481b39130d6ef22eaa3a8de7691cdc3fb3a50f84f86487a31d2",
        }
        spans = store.list_evidence()
        assert {span.evidence_id: content_hash(span) for span in spans} == expected_spans
        for span in spans:
            rebuilt = create_evidence_span_from_canonical_text(
                representation.canonical_text_id, span.start_offset, span.end_offset,
                repositories, store, locator=span.locator,
            )
            assert rebuilt == span
        locators = repositories.structured_evidence_locators.list()
        assert len(locators) == 2
        for locator in locators:
            assert verify_structured_evidence_locator(locator, repositories, store) == locator
    assert file_hashes() == before
