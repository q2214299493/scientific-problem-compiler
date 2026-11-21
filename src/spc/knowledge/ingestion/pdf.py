"""Born-digital PDF text and physical page offsets."""

from __future__ import annotations

from pathlib import Path
import pypdf

from ...models import CanonicalTextBlock, RawLiteratureArtifact
from ...serialization import content_hash
from ..literature_text import (
    CANONICALIZATION_POLICY,
    LiteratureExtractionError,
    LiteratureTextExtraction,
    PAGE_SEPARATOR,
    canonical_block,
    canonicalize_page,
    has_usable_text,
)


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
                blocks.append(
                    canonical_block(
                        page_text, page_number=page_number, block_type="page_text",
                        section_path=(), start=cursor,
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

    _canonicalize_page = staticmethod(canonicalize_page)
    _has_usable_text = staticmethod(has_usable_text)
