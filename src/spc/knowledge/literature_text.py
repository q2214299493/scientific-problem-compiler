"""Text extraction contracts and canonical block identity for literature sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar
import hashlib
import re
import unicodedata

from pydantic import Field

from ..models import CanonicalTextBlock, NonBlankStr, RawLiteratureArtifact, StrictModel
from ..serialization import content_hash


class AcquisitionError(ValueError):
    """Shared failure type for resource acquisition and canonical HTML extraction."""


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


class LiteratureExtractionError(ValueError):
    """Expected, safely reportable failure while extracting a PDF."""


@dataclass(frozen=True)
class LiteratureTextExtraction:
    text: str | None
    page_texts: tuple[str, ...]
    blocks: tuple[CanonicalTextBlock, ...]
    warnings: tuple[str, ...]
    requires_ocr: bool

    @property
    def canonical_text(self) -> str | None:
        return self.text

    @property
    def total_pages(self) -> int:
        return len(self.page_texts)

    @property
    def pages_with_text(self) -> int:
        return sum(
            1
            for text in self.page_texts
            if has_usable_text(text)
        )

    @property
    def pages_without_text(self) -> int:
        return self.total_pages - self.pages_with_text

    @property
    def text_coverage_ratio(self) -> float:
        return self.pages_with_text / self.total_pages if self.total_pages else 0.0


class CanonicalTextResult(Protocol):
    @property
    def canonical_text(self) -> str | None: ...

    @property
    def blocks(self) -> tuple[CanonicalTextBlock, ...]: ...


_Source = TypeVar("_Source", contravariant=True)
_Payload = TypeVar("_Payload", contravariant=True)
_Result = TypeVar("_Result", bound=CanonicalTextResult, covariant=True)


class CanonicalTextExtractor(Protocol[_Source, _Payload, _Result]):
    """Extract canonical text and offset-bound blocks from format-specific inputs."""

    def extract(self, source: _Source, payload: _Payload, /) -> _Result: ...


class LiteratureTextExtractor(CanonicalTextExtractor[RawLiteratureArtifact, Path, LiteratureTextExtraction], Protocol):
    parser_id: str
    parser_version: str
    parser_config_hash: str

    def extract(
        self,
        artifact: RawLiteratureArtifact,
        artifact_path: Path,
    ) -> LiteratureTextExtraction:
        ...


PAGE_SEPARATOR = "\n\n\f\n\n"
CANONICALIZATION_POLICY = {
    "page_order": "pdf_page_tree_order",
    "line_breaks": "normalize_to_lf",
    "unicode": "NFKC",
    "page_separator": "two_lf_form_feed_two_lf",
    "whitespace": "collapse_horizontal_and_repeated_blank_lines",
    "hyphenation": "preserve",
}


def canonicalize_page(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in normalized.split("\n")]
    compact: list[str] = []
    for line in lines:
        if line or (compact and compact[-1]):
            compact.append(line)
    while compact and not compact[-1]:
        compact.pop()
    return "\n".join(compact)


def has_usable_text(text: str) -> bool:
    return any(character.isalnum() for character in text)


def canonical_block(
    text: str, *, page_number: int, block_type: str, section_path: tuple[str, ...], start: int
) -> CanonicalTextBlock:
    identity = {
        "page_number": page_number,
        "block_type": block_type,
        "section_path": section_path,
        "start_offset": start,
        "end_offset": start + len(text),
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return CanonicalTextBlock(block_id=f"canonical-block-{content_hash(identity)[:24]}", **identity)


@dataclass(frozen=True)
class HTMLTextExtraction:
    source_url: str
    artifact_sha256: str
    canonical_text: str
    text_sha256: str
    blocks: tuple[CanonicalTextBlock, ...]


class HTMLTextExtractor(CanonicalTextExtractor[str, bytes, HTMLTextExtraction], Protocol):
    extractor_id: str
    extractor_version: str
    extractor_config_hash: str
