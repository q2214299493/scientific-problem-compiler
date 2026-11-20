from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import unicodedata
from typing import Any, Mapping, Protocol
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pypdf
from pydantic import Field

from ..models import (
    CanonicalTextArtifact,
    CanonicalTextBlock,
    EvidenceSpan,
    LiteratureDocument,
    LiteratureIngestionOutcome,
    LiteratureIngestionRecord,
    LiteratureIngestionStatus,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    LiteratureRepresentationSelection,
    LiteratureRepresentationSelectionOutcome,
    LiteratureSupplementarySource,
    NonBlankStr,
    RawLiteratureArtifact,
    SourceRole,
    SourceDocument,
    SourceType,
    StrictModel,
    literature_identity_id,
)
from ..repositories import EvidenceStore, KnowledgeEvidenceStore, KnowledgeRepositories
from ..serialization import content_hash


PAGE_SEPARATOR = "\n\n\f\n\n"
CANONICALIZATION_POLICY = {
    "page_order": "pdf_page_tree_order",
    "line_breaks": "normalize_to_lf",
    "unicode": "NFKC",
    "page_separator": "two_lf_form_feed_two_lf",
    "whitespace": "collapse_horizontal_and_repeated_blank_lines",
    "hyphenation": "preserve",
}


class LiteratureMetadata(StrictModel):
    title: NonBlankStr
    authors: tuple[NonBlankStr, ...] = Field(min_length=1)
    year: int = Field(ge=1000, le=9999)
    journal: NonBlankStr | None = None
    doi: NonBlankStr | None = None
    url: NonBlankStr | None = None
    domain: NonBlankStr
    topics: tuple[NonBlankStr, ...] = ()
    keywords: tuple[NonBlankStr, ...] = ()
    citation_refs: tuple[NonBlankStr, ...] = ()


@dataclass(frozen=True)
class LiteratureTextExtraction:
    text: str | None
    page_texts: tuple[str, ...]
    blocks: tuple[CanonicalTextBlock, ...]
    warnings: tuple[str, ...]
    requires_ocr: bool

    @property
    def total_pages(self) -> int:
        return len(self.page_texts)

    @property
    def pages_with_text(self) -> int:
        return sum(
            1
            for text in self.page_texts
            if PypdfLiteratureTextExtractor._has_usable_text(text)
        )

    @property
    def pages_without_text(self) -> int:
        return self.total_pages - self.pages_with_text

    @property
    def text_coverage_ratio(self) -> float:
        return self.pages_with_text / self.total_pages if self.total_pages else 0.0


class LiteratureExtractionError(ValueError):
    """Expected, safely reportable failure while extracting a PDF."""


@dataclass(frozen=True)
class ValidatedLiteratureIngestionChain:
    ingestion: LiteratureIngestionRecord
    raw_artifact: RawLiteratureArtifact
    canonical_text: CanonicalTextArtifact
    source: SourceDocument


class LiteratureTextExtractor(Protocol):
    parser_id: str
    parser_version: str
    parser_config_hash: str

    def extract(
        self,
        artifact: RawLiteratureArtifact,
        artifact_path: Path,
    ) -> LiteratureTextExtraction:
        ...


class PypdfLiteratureTextExtractor:
    parser_id = "pypdf-born-digital"

    def __init__(self, *, parser_version: str = "1.0.0") -> None:
        self.parser_version = parser_version
        self.parser_config_hash = content_hash(
            {
                "canonicalization": CANONICALIZATION_POLICY,
                "backend": "pypdf",
                "backend_version": pypdf.__version__,
            }
        )

    def extract(
        self,
        artifact: RawLiteratureArtifact,
        artifact_path: Path,
    ) -> LiteratureTextExtraction:
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError("raw PDF path must be a regular non-symlink file")
        if artifact_path.stat().st_size != artifact.byte_size:
            raise ValueError("raw PDF changed before extraction")
        try:
            reader = pypdf.PdfReader(artifact_path, strict=True)
            page_texts = tuple(
                self._canonicalize_page(page.extract_text() or "")
                for page in reader.pages
            )
        except pypdf.errors.PdfReadError as error:
            raise LiteratureExtractionError("PDF extraction failed") from error
        warnings = tuple(
            f"page {page_number} has no extractable text"
            for page_number, text in enumerate(page_texts, start=1)
            if not self._has_usable_text(text)
        )
        if not page_texts or not any(self._has_usable_text(text) for text in page_texts):
            return LiteratureTextExtraction(
                text=None,
                page_texts=page_texts,
                blocks=(),
                warnings=(*warnings, "PDF has no usable born-digital text; OCR is required"),
                requires_ocr=True,
            )
        text = PAGE_SEPARATOR.join(page_texts)
        blocks: list[CanonicalTextBlock] = []
        cursor = 0
        for page_number, page_text in enumerate(page_texts, start=1):
            if page_text:
                end = cursor + len(page_text)
                block_identity = {
                    "page_number": page_number,
                    "block_type": "page_text",
                    "section_path": (),
                    "start_offset": cursor,
                    "end_offset": end,
                    "text_hash": hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
                }
                blocks.append(
                    CanonicalTextBlock(
                        block_id=(
                            f"canonical-block-{content_hash(block_identity)[:24]}"
                        ),
                        **block_identity,
                    )
                )
            cursor += len(page_text)
            if page_number < len(page_texts):
                cursor += len(PAGE_SEPARATOR)
        return LiteratureTextExtraction(
            text=text,
            page_texts=page_texts,
            blocks=tuple(blocks),
            warnings=warnings,
            requires_ocr=False,
        )

    @staticmethod
    def _canonicalize_page(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[\t ]+", " ", line).strip() for line in normalized.split("\n")]
        compact: list[str] = []
        for line in lines:
            if line or (compact and compact[-1]):
                compact.append(line)
        while compact and not compact[-1]:
            compact.pop()
        return "\n".join(compact)

    @staticmethod
    def _has_usable_text(text: str) -> bool:
        return any(character.isalnum() for character in text)


class DocxSupplementaryTextExtractor:
    """Extract deterministic paragraph/cell text; DOCX has no asserted page map."""

    parser_id = "docx-supplementary-wordprocessingml"

    def __init__(self, *, parser_version: str = "1.0.0") -> None:
        self.parser_version = parser_version
        self.parser_config_hash = content_hash(
            {"format": "wordprocessingml", "layout": "paragraph-and-table-cell-order"}
        )

    def extract(
        self, artifact: RawLiteratureArtifact, artifact_path: Path
    ) -> LiteratureTextExtraction:
        if artifact.media_type != (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            raise ValueError("DOCX extractor requires a DOCX raw artifact")
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError("raw DOCX path must be a regular non-symlink file")
        if artifact_path.stat().st_size != artifact.byte_size:
            raise ValueError("raw DOCX changed before extraction")
        try:
            with ZipFile(artifact_path) as archive:
                member = archive.getinfo("word/document.xml")
                if member.file_size > 20_000_000 or member.flag_bits & 1:
                    raise LiteratureExtractionError("DOCX document XML is unsupported")
                xml = archive.read(member)
            if b"<!DOCTYPE" in xml.upper():
                raise LiteratureExtractionError("DOCX document XML contains a DTD")
            root = ElementTree.fromstring(xml)
        except (BadZipFile, KeyError, ElementTree.ParseError, RuntimeError) as error:
            raise LiteratureExtractionError("DOCX text extraction failed") from error

        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        body = root.find(f"{namespace}body")
        if body is None:
            raise LiteratureExtractionError("DOCX has no document body")
        pieces: list[tuple[str, tuple[str, ...], str]] = []

        def paragraph_text(paragraph: ElementTree.Element) -> str:
            joined = "".join(
                node.text or ""
                for node in paragraph.iter(f"{namespace}t")
            )
            return PypdfLiteratureTextExtractor._canonicalize_page(joined)

        table_index = 0
        paragraph_index = 0
        for child in body:
            if child.tag == f"{namespace}p":
                paragraph_index += 1
                value = paragraph_text(child)
                if value:
                    pieces.append(("docx_paragraph", (f"paragraph:{paragraph_index}",), value))
            elif child.tag == f"{namespace}tbl":
                table_index += 1
                for row_index, row in enumerate(child.findall(f"{namespace}tr"), 1):
                    for cell_index, cell in enumerate(row.findall(f"{namespace}tc"), 1):
                        value = "\n".join(
                            part
                            for paragraph in cell.findall(f"{namespace}p")
                            if (part := paragraph_text(paragraph))
                        )
                        if value:
                            pieces.append(
                                (
                                    "docx_table_cell",
                                    (
                                        f"table:{table_index}",
                                        f"row:{row_index}",
                                        f"cell:{cell_index}",
                                    ),
                                    value,
                                )
                            )
        if not pieces:
            raise LiteratureExtractionError("DOCX has no extractable text")
        separator = "\n\n"
        text = separator.join(piece[2] for piece in pieces)
        blocks: list[CanonicalTextBlock] = []
        offset = 0
        for block_type, section_path, value in pieces:
            identity = {
                "page_number": 1,
                "block_type": block_type,
                "section_path": section_path,
                "start_offset": offset,
                "end_offset": offset + len(value),
                "text_hash": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
            blocks.append(
                CanonicalTextBlock(
                    block_id=f"canonical-block-{content_hash(identity)[:24]}",
                    **identity,
                )
            )
            offset += len(value) + len(separator)
        return LiteratureTextExtraction(
            text=text,
            page_texts=(text,),  # One logical segment; DOCX pagination is not known.
            blocks=tuple(blocks),
            warnings=("DOCX physical page numbers are unavailable",),
            requires_ocr=False,
        )


class LiteratureIngestionService:
    def __init__(
        self,
        extractor: LiteratureTextExtractor | None = None,
    ) -> None:
        self.extractor = extractor or PypdfLiteratureTextExtractor()

    def ingest(
        self,
        pdf_path: Path,
        metadata: LiteratureMetadata | Mapping[str, Any],
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureIngestionOutcome:
        metadata_record = (
            metadata
            if isinstance(metadata, LiteratureMetadata)
            else LiteratureMetadata.model_validate(metadata)
        )
        literature_id = literature_identity_id(
            title=metadata_record.title,
            authors=metadata_record.authors,
            year=metadata_record.year,
            doi=metadata_record.doi,
        )
        raw_artifact = repositories.raw_literature_artifacts.put(
            pdf_path, literature_id
        )
        raw_path = repositories.root.joinpath(
            *PurePosixPath(raw_artifact.stored_path).parts
        )
        try:
            extraction = self.extractor.extract(raw_artifact, raw_path)
        except LiteratureExtractionError:
            ingestion = self._make_ingestion(
                literature_id=literature_id,
                raw_artifact=raw_artifact,
                status=LiteratureIngestionStatus.FAILED,
                warnings=("PDF extraction failed",),
            )
            repositories.literature_ingestions.put(
                ingestion.ingestion_id, ingestion
            )
            return LiteratureIngestionOutcome(
                literature_id=literature_id,
                artifact_id=raw_artifact.artifact_id,
                ingestion_id=ingestion.ingestion_id,
                warnings=ingestion.warnings,
            )
        if extraction.requires_ocr:
            ingestion = self._make_ingestion(
                literature_id=literature_id,
                raw_artifact=raw_artifact,
                status=LiteratureIngestionStatus.REQUIRES_OCR,
                warnings=extraction.warnings,
                extraction=extraction,
            )
            repositories.literature_ingestions.put(
                ingestion.ingestion_id, ingestion
            )
            return LiteratureIngestionOutcome(
                literature_id=literature_id,
                artifact_id=raw_artifact.artifact_id,
                ingestion_id=ingestion.ingestion_id,
                warnings=ingestion.warnings,
                requires_ocr=True,
            )
        if extraction.text is None:
            raise ValueError("extractor returned no canonical text without requires_ocr")
        canonical = repositories.canonical_text_artifacts.put(
            extraction.text,
            literature_id=literature_id,
            raw_artifact=raw_artifact,
            parser_id=self.extractor.parser_id,
            parser_version=self.extractor.parser_version,
            parser_config_hash=self.extractor.parser_config_hash,
            page_count=len(extraction.page_texts),
            blocks=extraction.blocks,
        )
        canonical_path = repositories.root.joinpath(
            *PurePosixPath(canonical.stored_path).parts
        )
        source_id = f"source-{literature_id}"
        source_version = f"canonical-{canonical.canonical_text_id.removeprefix('canonical-text-')}"
        source = evidence_store.ingest_source(
            canonical_path,
            source_id,
            source_version,
            metadata_record.title,
            source_role="literature_author",
            source_type="literature_article",
        )
        ingestion = self._make_ingestion(
            literature_id=literature_id,
            raw_artifact=raw_artifact,
            canonical=canonical,
            source_id=source.source_id,
            source_version=source.version,
            status=LiteratureIngestionStatus.ACCEPTED,
            warnings=extraction.warnings,
            extraction=extraction,
        )
        repositories.literature_ingestions.put(ingestion.ingestion_id, ingestion)
        self._finalize_literature_document(
            metadata_record,
            literature_id,
            raw_artifact,
            canonical,
            source.source_id,
            source.version,
            repositories,
        )
        return LiteratureIngestionOutcome(
            literature_id=literature_id,
            artifact_id=raw_artifact.artifact_id,
            canonical_text_id=canonical.canonical_text_id,
            ingestion_id=ingestion.ingestion_id,
            source_id=source.source_id,
            source_version=source.version,
            warnings=ingestion.warnings,
            requires_ocr=False,
        )

    def _make_ingestion(
        self,
        *,
        literature_id: str,
        raw_artifact: RawLiteratureArtifact,
        status: LiteratureIngestionStatus,
        warnings: tuple[str, ...],
        extraction: LiteratureTextExtraction | None = None,
        canonical: CanonicalTextArtifact | None = None,
        source_id: str | None = None,
        source_version: str | None = None,
    ) -> LiteratureIngestionRecord:
        identity = {
            "literature_id": literature_id,
            "raw_artifact_id": raw_artifact.artifact_id,
            "raw_artifact_hash": raw_artifact.content_hash,
            "canonical_text_id": (
                canonical.canonical_text_id if canonical is not None else None
            ),
            "canonical_text_hash": (
                canonical.content_hash if canonical is not None else None
            ),
            "source_id": source_id,
            "source_version": source_version,
            "parser_id": self.extractor.parser_id,
            "parser_version": self.extractor.parser_version,
            "parser_config_hash": self.extractor.parser_config_hash,
            "ingestion_status": status,
            "total_pages": extraction.total_pages if extraction is not None else 0,
            "pages_with_text": (
                extraction.pages_with_text if extraction is not None else 0
            ),
            "pages_without_text": (
                extraction.pages_without_text if extraction is not None else 0
            ),
            "text_coverage_ratio": (
                extraction.text_coverage_ratio if extraction is not None else 0.0
            ),
            "warnings": warnings,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        ingestion_id = f"literature-ingestion-{content_hash(identity)[:24]}"
        payload = {"ingestion_id": ingestion_id, **identity}
        return LiteratureIngestionRecord(
            **payload,
            content_hash=content_hash(payload),
        )

    @staticmethod
    def _finalize_literature_document(
        metadata: LiteratureMetadata,
        literature_id: str,
        raw_artifact: RawLiteratureArtifact,
        canonical: CanonicalTextArtifact,
        source_id: str,
        source_version: str,
        repositories: KnowledgeRepositories,
    ) -> LiteratureDocument:
        try:
            return repositories.literature_documents.get(literature_id)
        except FileNotFoundError:
            pass
        identity = {
            **metadata.model_dump(mode="json", exclude_none=True),
            "literature_id": literature_id,
            "raw_artifact_ref": raw_artifact.stored_path,
            "canonical_text_ref": canonical.stored_path,
            "source_id": source_id,
            "source_version": source_version,
        }
        document = LiteratureDocument(
            **identity,
            content_hash=content_hash(identity),
        )
        repositories.literature_documents.put(literature_id, document)
        return document


def validate_literature_ingestion_chain(
    ingestion: LiteratureIngestionRecord,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
) -> ValidatedLiteratureIngestionChain:
    if ingestion.ingestion_status != LiteratureIngestionStatus.ACCEPTED:
        raise ValueError("representation selection requires accepted ingestion")
    canonical = repositories.canonical_text_artifacts.get(
        ingestion.canonical_text_id or ""
    )
    raw = repositories.raw_literature_artifacts.get(ingestion.raw_artifact_id)
    if (
        raw.media_type != "application/pdf"
        or ingestion.raw_artifact_hash != raw.content_hash
        or ingestion.canonical_text_hash != canonical.content_hash
        or ingestion.literature_id != canonical.literature_id
        or ingestion.literature_id != raw.literature_id
        or canonical.raw_artifact_id != raw.artifact_id
        or canonical.raw_artifact_hash != raw.content_hash
        or ingestion.parser_id != canonical.parser_id
        or ingestion.parser_version != canonical.parser_version
        or ingestion.parser_config_hash != canonical.parser_config_hash
    ):
        raise ValueError("raw/canonical/ingestion binding is invalid")
    source = evidence_store.get_source(
        ingestion.source_id or "", ingestion.source_version or ""
    )
    evidence_store.verify_source_integrity(source)
    if (
        source.source_id != ingestion.source_id
        or source.version != ingestion.source_version
        or source.content_sha256 != canonical.text_sha256
        or source.source_role != SourceRole.LITERATURE_AUTHOR
        or source.source_type != SourceType.LITERATURE_ARTICLE
    ):
        raise ValueError("canonical/source binding is invalid")
    return ValidatedLiteratureIngestionChain(
        ingestion=ingestion,
        raw_artifact=raw,
        canonical_text=canonical,
        source=source,
    )


class LiteratureRepresentationSelector:
    @staticmethod
    def _pdf_representation(
        ingestion: LiteratureIngestionRecord,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureRepresentationReference:
        chain = validate_literature_ingestion_chain(
            ingestion, repositories, evidence_store
        )
        identity = {
            "representation_kind": LiteratureRepresentationKind.PDF,
            "literature_id": ingestion.literature_id,
            "raw_artifact_id": chain.raw_artifact.artifact_id,
            "raw_artifact_hash": chain.raw_artifact.content_hash,
            "canonical_text_id": chain.canonical_text.canonical_text_id,
            "canonical_text_hash": chain.canonical_text.content_hash,
            "ingestion_id": ingestion.ingestion_id,
            "ingestion_hash": ingestion.content_hash,
            "source_id": chain.source.source_id,
            "source_version": chain.source.version,
        }
        representation_id = (
            f"literature-representation-{content_hash(identity)[:24]}"
        )
        payload = {"representation_id": representation_id, **identity}
        representation = LiteratureRepresentationReference(
            **payload, content_hash=content_hash(payload)
        )
        repositories.literature_representation_refs.put(
            representation.representation_id, representation
        )
        return representation

    @classmethod
    def resolve_reference(
        cls,
        representation_or_ingestion_id: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureRepresentationReference:
        try:
            representation = repositories.literature_representation_refs.get(
                representation_or_ingestion_id
            )
        except FileNotFoundError:
            ingestion = repositories.literature_ingestions.get(
                representation_or_ingestion_id
            )
            representation = cls._pdf_representation(
                ingestion, repositories, evidence_store
            )
        from .acquisition import validate_literature_representation

        return validate_literature_representation(
            representation, repositories, evidence_store
        )

    @classmethod
    def resolve_selection(
        cls,
        selection: LiteratureRepresentationSelection,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureRepresentationReference:
        if selection.representation_id is not None:
            representation = cls.resolve_reference(
                selection.representation_id, repositories, evidence_store
            )
            if (
                selection.representation_hash != representation.content_hash
                or selection.literature_id != representation.literature_id
            ):
                raise ValueError("selection representation binding is invalid")
            return representation
        ingestion = repositories.literature_ingestions.get(
            selection.ingestion_id or ""
        )
        chain = validate_literature_ingestion_chain(
            ingestion, repositories, evidence_store
        )
        if (
            selection.literature_id != ingestion.literature_id
            or selection.ingestion_hash != ingestion.content_hash
            or selection.canonical_text_id != chain.canonical_text.canonical_text_id
            or selection.canonical_text_hash != chain.canonical_text.content_hash
            or selection.source_id != chain.source.source_id
            or selection.source_version != chain.source.version
        ):
            raise ValueError("selection binding is invalid")
        return cls._pdf_representation(ingestion, repositories, evidence_store)

    def select(
        self,
        literature_id: str,
        representation_id: str,
        selected_by: str,
        rationale: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore,
    ) -> LiteratureRepresentationSelectionOutcome:
        representation = self.resolve_reference(
            representation_id, repositories, evidence_store
        )
        if representation.literature_id != literature_id:
            raise ValueError("representation belongs to another literature work")
        repository = repositories.literature_representation_selections
        try:
            current = repository.resolve_current(literature_id)
        except FileNotFoundError:
            current = None
        identity = {
            "literature_id": literature_id,
            "representation_id": representation.representation_id,
            "representation_hash": representation.content_hash,
            "selected_by": selected_by,
            "rationale": rationale,
            "supersedes_selection_id": (
                current.selection_id if current is not None else None
            ),
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        selection_id = f"literature-selection-{content_hash(identity)[:24]}"
        payload = {"selection_id": selection_id, **identity}
        selection = LiteratureRepresentationSelection(
            **payload,
            content_hash=content_hash(payload),
        )
        repository.put(selection.selection_id, selection)
        if representation.representation_kind == LiteratureRepresentationKind.PDF:
            ingestion = repositories.literature_ingestions.get(
                representation.ingestion_id
            )
            total_pages = ingestion.total_pages
            pages_with_text = ingestion.pages_with_text
            pages_without_text = ingestion.pages_without_text
            text_coverage_ratio = ingestion.text_coverage_ratio
            warnings = ingestion.warnings
        else:
            ingestion = repositories.html_literature_ingestions.get(
                representation.ingestion_id
            )
            total_pages = None
            pages_with_text = None
            pages_without_text = None
            text_coverage_ratio = None
            warnings = ()
        return LiteratureRepresentationSelectionOutcome(
            selection=selection,
            ingestion_status=ingestion.ingestion_status,
            total_pages=total_pages,
            pages_with_text=pages_with_text,
            pages_without_text=pages_without_text,
            text_coverage_ratio=text_coverage_ratio,
            warnings=warnings,
        )


def create_evidence_span_from_canonical_text(
    canonical_text_id: str,
    start_offset: int,
    end_offset: int,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
    *,
    locator: str | None = None,
    historical_ingestion: bool = False,
) -> EvidenceSpan:
    pdf_canonical_path = repositories.canonical_text_artifacts.root / canonical_text_id
    if pdf_canonical_path.exists() or pdf_canonical_path.is_symlink():
        canonical = repositories.canonical_text_artifacts.get(canonical_text_id)
        text = repositories.canonical_text_artifacts.read_text(canonical_text_id)
        representation_kind = LiteratureRepresentationKind.PDF
    else:
        canonical = repositories.canonical_html_text_artifacts.get(
            canonical_text_id
        )
        text = repositories.canonical_html_text_artifacts.read_text(
            canonical_text_id
        )
        representation_kind = LiteratureRepresentationKind.HTML
    if start_offset < 0 or end_offset <= start_offset or end_offset > len(text):
        raise ValueError("EvidenceSpan offsets are outside canonical text")
    matching = tuple(
        representation
        for representation in repositories.literature_representation_refs.list()
        if representation.representation_kind == representation_kind
        and representation.canonical_text_id == canonical_text_id
    )
    if len(matching) != 1:
        raise ValueError("canonical text must have exactly one representation binding")
    representation = matching[0]
    if not historical_ingestion:
        selection = repositories.literature_representation_selections.resolve_current(
            canonical.literature_id
        )
        selected_representation = LiteratureRepresentationSelector.resolve_selection(
            selection, repositories, evidence_store
        )
        if selected_representation.representation_id != representation.representation_id:
            raise ValueError("canonical text is not the current selected representation")
    from .acquisition import validate_literature_representation

    validate_literature_representation(
        representation, repositories, evidence_store
    )
    source = evidence_store.get_source(
        representation.source_id, representation.source_version
    )
    evidence_store.verify_source_integrity(source)
    recovered = text[start_offset:end_offset]
    evidence_identity = {
        "canonical_text_id": canonical_text_id,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "text_hash": content_hash({"text": recovered}),
    }
    evidence = EvidenceSpan(
        evidence_id=f"evidence-{content_hash(evidence_identity)[:24]}",
        source_id=source.source_id,
        source_version=source.version,
        content_sha256=source.content_sha256,
        start_offset=start_offset,
        end_offset=end_offset,
        text=recovered,
        locator=locator,
    )
    evidence_store.add_evidence(evidence)
    return evidence


def ingest_literature_supplementary_pdf(
    pdf_path: Path,
    literature_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: KnowledgeEvidenceStore,
    *,
    extractor: LiteratureTextExtractor | None = None,
) -> LiteratureSupplementarySource:
    """Store an independent SI PDF without replacing the selected main article."""
    return _ingest_literature_supplementary(
        pdf_path,
        literature_id,
        repositories,
        evidence_store,
        extractor=extractor or PypdfLiteratureTextExtractor(),
        allow_docx=False,
    )


def ingest_literature_supplementary_docx(
    docx_path: Path,
    literature_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: KnowledgeEvidenceStore,
) -> LiteratureSupplementarySource:
    """Store an original SI DOCX with a deterministic non-paginated text view."""
    return _ingest_literature_supplementary(
        docx_path,
        literature_id,
        repositories,
        evidence_store,
        extractor=DocxSupplementaryTextExtractor(),
        allow_docx=True,
    )


def _ingest_literature_supplementary(
    source_path: Path,
    literature_id: str,
    repositories: KnowledgeRepositories,
    evidence_store: KnowledgeEvidenceStore,
    *,
    extractor: LiteratureTextExtractor,
    allow_docx: bool,
) -> LiteratureSupplementarySource:
    if allow_docx:
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("raw DOCX path must be a regular non-symlink file")
        with source_path.open("rb") as handle:
            if handle.read(4) != b"PK\x03\x04":
                raise ValueError("supplementary DOCX must retain its original DOCX format")
    if evidence_store.root.resolve() != repositories.evidence_store.root.resolve():
        raise ValueError("supplementary sources must use this knowledge repository's evidence store")
    document = repositories.literature_documents.get(literature_id)
    selection = repositories.literature_representation_selections.resolve_current(
        literature_id
    )
    main = LiteratureRepresentationSelector.resolve_selection(
        selection, repositories, evidence_store
    )
    raw = repositories.raw_literature_artifacts.put(
        source_path, literature_id, allow_docx=allow_docx
    )
    if raw.artifact_id == main.raw_artifact_id:
        raise ValueError("supplementary source must differ from the selected main article")
    raw_path = repositories.root.joinpath(*PurePosixPath(raw.stored_path).parts)
    extraction = extractor.extract(raw, raw_path)
    if extraction.requires_ocr or extraction.text is None:
        raise ValueError("supplementary source has no extractable canonical text")
    canonical = repositories.canonical_text_artifacts.put(
        extraction.text,
        literature_id=literature_id,
        raw_artifact=raw,
        parser_id=extractor.parser_id,
        parser_version=extractor.parser_version,
        parser_config_hash=extractor.parser_config_hash,
        page_count=extraction.total_pages,
        blocks=extraction.blocks,
    )
    canonical_path = repositories.root.joinpath(
        *PurePosixPath(canonical.stored_path).parts
    )
    source = evidence_store.ingest_source(
        canonical_path,
        f"source-supplement-{literature_id}",
        f"canonical-{canonical.canonical_text_id.removeprefix('canonical-text-')}",
        f"{document.title} [Supporting Information]",
        source_role=SourceRole.LITERATURE_AUTHOR,
        source_type=SourceType.SUPPORTING_INFORMATION,
    )
    identity = {
        "literature_id": literature_id,
        "literature_hash": document.content_hash,
        "main_selection_id": selection.selection_id,
        "main_selection_hash": selection.content_hash,
        "raw_artifact_id": raw.artifact_id,
        "raw_artifact_hash": raw.content_hash,
        "canonical_text_id": canonical.canonical_text_id,
        "canonical_text_hash": canonical.content_hash,
        "source_id": source.source_id,
        "source_version": source.version,
        "source_sha256": source.content_sha256,
    }
    supplement_id = f"literature-supplement-{content_hash(identity)[:24]}"
    payload = {"supplement_id": supplement_id, **identity}
    supplement = LiteratureSupplementarySource(
        **payload, content_hash=content_hash(payload)
    )
    repositories.literature_supplementary_sources.put(supplement_id, supplement)
    return supplement


def validate_literature_supplementary_source(
    supplement: LiteratureSupplementarySource,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore,
) -> tuple[RawLiteratureArtifact, CanonicalTextArtifact, SourceDocument]:
    """Verify raw/canonical/source binding for an independent SI source."""
    document = repositories.literature_documents.get(supplement.literature_id)
    selection = repositories.literature_representation_selections.get(
        supplement.main_selection_id
    )
    main = LiteratureRepresentationSelector.resolve_selection(
        selection, repositories, evidence_store
    )
    raw = repositories.raw_literature_artifacts.get(supplement.raw_artifact_id)
    canonical = repositories.canonical_text_artifacts.get(
        supplement.canonical_text_id
    )
    source = evidence_store.get_source(
        supplement.source_id, supplement.source_version
    )
    evidence_store.verify_source_integrity(source)
    if (
        document.content_hash != supplement.literature_hash
        or selection.literature_id != supplement.literature_id
        or selection.content_hash != supplement.main_selection_hash
        or main.literature_id != supplement.literature_id
        or main.raw_artifact_id == raw.artifact_id
        or main.source_id == source.source_id
        or raw.literature_id != supplement.literature_id
        or raw.content_hash != supplement.raw_artifact_hash
        or canonical.literature_id != supplement.literature_id
        or canonical.raw_artifact_id != raw.artifact_id
        or canonical.raw_artifact_hash != raw.content_hash
        or canonical.content_hash != supplement.canonical_text_hash
        or source.source_id != f"source-supplement-{supplement.literature_id}"
        or source.version
        != f"canonical-{canonical.canonical_text_id.removeprefix('canonical-text-')}"
        or source.content_sha256 != canonical.text_sha256
        or source.content_sha256 != supplement.source_sha256
        or source.source_role != SourceRole.LITERATURE_AUTHOR
        or source.source_type != SourceType.SUPPORTING_INFORMATION
    ):
        raise ValueError("supplementary raw/canonical/source binding is invalid")
    return raw, canonical, source


def create_supplementary_evidence_span(
    supplement_id: str,
    page_number: int,
    exact_text: str,
    locator: str,
    repositories: KnowledgeRepositories,
    evidence_store: KnowledgeEvidenceStore,
    *,
    block_id: str | None = None,
) -> EvidenceSpan:
    """Bind a unique verbatim SI passage to a verified source segment."""
    if not exact_text or not locator.strip():
        raise ValueError("supplementary evidence requires text and a locator")
    supplement = repositories.literature_supplementary_sources.get(supplement_id)
    raw, canonical, source = validate_literature_supplementary_source(
        supplement, repositories, evidence_store
    )
    text = repositories.canonical_text_artifacts.read_text(canonical.canonical_text_id)
    if raw.media_type == "application/pdf":
        pages = tuple(
            block for block in canonical.blocks if block.page_number == page_number
        )
        if len(pages) != 1:
            raise ValueError("supplementary page has no unique canonical block")
        block = pages[0]
        if block_id is not None and block.block_id != block_id:
            raise ValueError("supplementary block ID does not match the stated page")
        source_locator = f"SI PDF page {page_number}"
    else:
        if page_number != 1:
            raise ValueError("DOCX has no physical page numbers; use logical segment 1")
        matches = tuple(
            block
            for block in canonical.blocks
            if block_id is None or block.block_id == block_id
            if exact_text in text[block.start_offset : block.end_offset]
        )
        if len(matches) != 1:
            raise ValueError("supplementary DOCX quote must identify one block")
        block = matches[0]
        source_locator = f"SI DOCX {'/'.join(block.section_path)}; no physical page"
    block_text = text[block.start_offset : block.end_offset]
    if block_text.count(exact_text) != 1:
        raise ValueError("supplementary quote must occur exactly once in its block")
    start_offset = block.start_offset + block_text.index(exact_text)
    end_offset = start_offset + len(exact_text)
    identity = {
        "canonical_text_id": canonical.canonical_text_id,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "text_hash": content_hash({"text": exact_text}),
    }
    evidence = EvidenceSpan(
        evidence_id=f"evidence-{content_hash(identity)[:24]}",
        source_id=source.source_id,
        source_version=source.version,
        content_sha256=source.content_sha256,
        start_offset=start_offset,
        end_offset=end_offset,
        text=exact_text,
        locator=f"{source_locator}; {locator}",
    )
    evidence_store.add_evidence(evidence)
    return evidence
