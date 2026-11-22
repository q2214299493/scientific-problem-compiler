"""HTML canonical text extraction."""

from __future__ import annotations

from html.parser import HTMLParser
import hashlib

from ...models import CanonicalTextBlock
from ...serialization import content_hash
from ..literature_text import AcquisitionError, HTMLTextExtraction, canonical_block


class _CanonicalHTMLParser(HTMLParser):
    BLOCK_TAGS = frozenset(
        {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "li",
            "blockquote",
            "caption",
            "th",
            "td",
            "figcaption",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.active: list[tuple[str, list[str]]] = []
        self.blocks: list[tuple[str, str]] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        elif not self.ignored_depth and normalized in self.BLOCK_TAGS:
            self.active.append((normalized, []))

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
            return
        if self.ignored_depth or normalized not in self.BLOCK_TAGS:
            return
        for index in range(len(self.active) - 1, -1, -1):
            active_tag, content = self.active[index]
            if active_tag == normalized:
                text = " ".join("".join(content).split())
                if text:
                    self.blocks.append((normalized, text))
                del self.active[index]
                break

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth and self.active:
            self.active[-1][1].append(data)


class HTMLLiteratureTextExtractor:
    extractor_id = "html-canonical-text"
    extractor_version = "2.0.0"
    extractor_config_hash = content_hash(
        {
            "block_tags": tuple(sorted(_CanonicalHTMLParser.BLOCK_TAGS)),
            "ignored_tags": ("noscript", "script", "style"),
            "encoding": "utf-8",
            "block_separator": "two_lf",
        }
    )

    def extract(self, source_url: str, html_bytes: bytes) -> HTMLTextExtraction:
        try:
            html_text = html_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AcquisitionError("HTML resource is not UTF-8") from error
        parser = _CanonicalHTMLParser()
        parser.feed(html_text)
        if not parser.blocks:
            raise AcquisitionError("HTML article contains no canonical text blocks")
        texts = [text for _, text in parser.blocks]
        canonical_text = "\n\n".join(texts)
        blocks: list[CanonicalTextBlock] = []
        cursor = 0
        for block_type, block_text in parser.blocks:
            blocks.append(
                canonical_block(
                    block_text, page_number=1, block_type=f"html_{block_type}",
                    section_path=(), start=cursor,
                )
            )
            cursor += len(block_text) + 2
        return HTMLTextExtraction(
            source_url=source_url,
            artifact_sha256=hashlib.sha256(html_bytes).hexdigest(),
            canonical_text=canonical_text,
            text_sha256=hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
            blocks=tuple(blocks),
        )
