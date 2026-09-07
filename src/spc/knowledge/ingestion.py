from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import unicodedata
from typing import Any, Mapping, Protocol

import pypdf
from pydantic import Field

from ..models import (
    CanonicalTextBlock,
    EvidenceSpan,
    LiteratureDocument,
    LiteratureIngestionOutcome,
    LiteratureIngestionRecord,
    LiteratureIngestionStatus,
    LiteratureRepresentationSelection,
    NonBlankStr,
    RawLiteratureArtifact,
    SourceRole,
    SourceType,
    StrictModel,
    literature_identity_id,
)
from ..repositories import KnowledgeRepositories, SourceEvidenceStore
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
        evidence_store: SourceEvidenceStore,
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
        source = evidence_store.ingest(
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
        selection = self._select_representation(ingestion, repositories)
        return LiteratureIngestionOutcome(
            literature_id=literature_id,
            artifact_id=raw_artifact.artifact_id,
            canonical_text_id=canonical.canonical_text_id,
            ingestion_id=ingestion.ingestion_id,
            selection_id=selection.selection_id,
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
        canonical: Any | None = None,
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
    def _select_representation(
        ingestion: LiteratureIngestionRecord,
        repositories: KnowledgeRepositories,
    ) -> LiteratureRepresentationSelection:
        repository = repositories.literature_representation_selections
        try:
            current = repository.resolve_current(ingestion.literature_id)
        except FileNotFoundError:
            current = None
        if current is not None and current.ingestion_id == ingestion.ingestion_id:
            return current
        identity = {
            "literature_id": ingestion.literature_id,
            "ingestion_id": ingestion.ingestion_id,
            "ingestion_hash": ingestion.content_hash,
            "canonical_text_id": ingestion.canonical_text_id,
            "canonical_text_hash": ingestion.canonical_text_hash,
            "source_id": ingestion.source_id,
            "source_version": ingestion.source_version,
            "selected_by": "spc-ingestion-service",
            "rationale": "Selected after deterministic canonical-text ingestion.",
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
        return selection

    @staticmethod
    def _finalize_literature_document(
        metadata: LiteratureMetadata,
        literature_id: str,
        raw_artifact: RawLiteratureArtifact,
        canonical: Any,
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


def create_evidence_span_from_canonical_text(
    canonical_text_id: str,
    start_offset: int,
    end_offset: int,
    repositories: KnowledgeRepositories,
    evidence_store: SourceEvidenceStore,
    *,
    locator: str | None = None,
    historical_ingestion: bool = False,
) -> EvidenceSpan:
    canonical = repositories.canonical_text_artifacts.get(canonical_text_id)
    text = repositories.canonical_text_artifacts.read_text(canonical_text_id)
    if start_offset < 0 or end_offset <= start_offset or end_offset > len(text):
        raise ValueError("EvidenceSpan offsets are outside canonical text")
    matching = tuple(
        ingestion
        for ingestion in repositories.literature_ingestions.list()
        if ingestion.ingestion_status == LiteratureIngestionStatus.ACCEPTED
        and ingestion.canonical_text_id == canonical_text_id
    )
    if len(matching) != 1:
        raise ValueError("canonical text must have exactly one accepted ingestion binding")
    ingestion = matching[0]
    if not historical_ingestion:
        selection = repositories.literature_representation_selections.resolve_current(
            canonical.literature_id
        )
        if selection.ingestion_id != ingestion.ingestion_id:
            raise ValueError("canonical text is not the current selected representation")
        if (
            selection.literature_id != ingestion.literature_id
            or selection.ingestion_hash != ingestion.content_hash
            or selection.canonical_text_id != canonical.canonical_text_id
            or selection.canonical_text_hash != canonical.content_hash
            or selection.source_id != ingestion.source_id
            or selection.source_version != ingestion.source_version
        ):
            raise ValueError("current representation selection binding is invalid")
    raw = repositories.raw_literature_artifacts.get(ingestion.raw_artifact_id)
    if (
        ingestion.raw_artifact_hash != raw.content_hash
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
    source = evidence_store.source_records.get(
        f"{ingestion.source_id}--{ingestion.source_version}"
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
