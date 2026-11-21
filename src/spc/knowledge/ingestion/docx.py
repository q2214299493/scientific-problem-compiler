"""Supplementary WordprocessingML paragraph and cell extraction."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from ...models import CanonicalTextBlock, RawLiteratureArtifact
from ...serialization import content_hash
from ..literature_text import (
    LiteratureExtractionError,
    LiteratureTextExtraction,
    canonical_block,
    canonicalize_page,
)


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
            return canonicalize_page(joined)

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
            blocks.append(canonical_block(value, page_number=1, block_type=block_type, section_path=section_path, start=offset))
            offset += len(value) + len(separator)
        return LiteratureTextExtraction(
            text=text,
            page_texts=(text,),  # One logical segment; DOCX pagination is not known.
            blocks=tuple(blocks),
            warnings=("DOCX physical page numbers are unavailable",),
            requires_ocr=False,
        )
