from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from pathlib import PurePosixPath
from typing import TypeVar
from zipfile import BadZipFile, ZipFile

from ..models import (
    AcquisitionAttemptRecord,
    CanonicalTextArtifact,
    CanonicalTextBlock,
    CanonicalHTMLTextArtifact,
    DocumentStructureArtifact,
    DocumentStructureBlock,
    DocumentStructureSelection,
    HistoricalLiteratureEvidenceAuthorization,
    HTMLLiteratureIngestionRecord,
    LiteratureAcquisitionRecord,
    LiteratureAcquisitionRequest,
    LiteratureDocument,
    LiteratureIngestionRecord,
    LiteratureRepresentationSelection,
    LiteratureRepresentationReference,
    LiteratureSupplementarySource,
    LiteratureWorkflowPattern,
    MetadataMergeManifest,
    MetadataRetrievalRecord,
    RawLiteratureArtifact,
    RawHTMLLiteratureArtifact,
    ResolvedLiteratureResource,
    StructuredEvidenceLocator,
    TableCellStructure,
    TableStructure,
    FigureStructure,
)
from ..serialization import content_hash, load_model, require_safe_path_component

from .base import (
    ArtifactDirectory,
    CatalogRepository,
    IdentityBoundRepository,
    verify_artifact_bytes,
)
from .evidence import KnowledgeEvidenceStore


class LiteratureWorkflowRepository(CatalogRepository[LiteratureWorkflowPattern]):
    _key_error = "workflow repository key must equal pattern_id"
    _stored_id_error = "stored workflow pattern ID does not match repository key"
    _duplicate_error = "workflow repository contains duplicate record IDs"

    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root / "workflow_patterns", LiteratureWorkflowPattern, "pattern_id")


def raw_artifact_metadata(store: ArtifactDirectory) -> tuple[RawLiteratureArtifact, ...]:
    records: list[RawLiteratureArtifact] = []
    for directory in store._directories():
        metadata_path = directory / "metadata.json"
        store._validate_paths(directory, metadata_path, metadata_path)
        record = load_model(metadata_path, RawLiteratureArtifact)
        content_path = store.knowledge_root.joinpath(
            *PurePosixPath(record.stored_path).parts
        )
        store._validate_paths(directory, content_path, metadata_path)
        if record.artifact_id != directory.name:
            raise ValueError("raw artifact metadata ID does not match its directory")
        records.append(record)
    return tuple(records)


class RawLiteratureArtifactRepository(ArtifactDirectory[RawLiteratureArtifact]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root, "literature_artifacts")

    def put(
        self, source_path: Path, literature_id: str, *, allow_docx: bool = False
    ) -> RawLiteratureArtifact:
        require_safe_path_component(literature_id, field="literature_id")
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("raw literature source must be a regular non-symlink file")
        if allow_docx and source_path.stat().st_size > 250_000_000:
            raise ValueError("supplementary DOCX exceeds the artifact size limit")
        content = source_path.read_bytes()
        media_type = self._detect_media_type(content)
        if media_type != "application/pdf" and not (allow_docx and media_type):
            raise ValueError("raw literature artifact is not a PDF")
        extension = "pdf" if media_type == "application/pdf" else "docx"
        digest = hashlib.sha256(content).hexdigest()
        identity = {"literature_id": literature_id, "sha256": digest}
        artifact_id = f"literature-artifact-{content_hash(identity)[:24]}"
        destination = self.root / artifact_id
        if destination.exists():
            existing = self.get(artifact_id)
            if (
                existing.literature_id != literature_id
                or existing.sha256 != digest
                or existing.byte_size != len(content)
            ):
                raise FileExistsError(
                    f"refusing to reuse conflicting raw artifact: {artifact_id}"
                )
            return existing
        payload = {
            "artifact_id": artifact_id,
            "literature_id": literature_id,
            "original_filename": source_path.name,
            "media_type": media_type,
            "byte_size": len(content),
            "sha256": digest,
            "stored_path": f"literature_artifacts/{artifact_id}/artifact.{extension}",
        }
        record = RawLiteratureArtifact(
            **payload,
            content_hash=content_hash(payload),
        )
        self._write_directory(destination, f"artifact.{extension}", content, record)
        return self.get(artifact_id)

    def get(self, artifact_id: str) -> RawLiteratureArtifact:
        require_safe_path_component(artifact_id, field="artifact_id")
        directory = self.root / artifact_id
        metadata_path = directory / "metadata.json"
        self._validate_paths(directory, metadata_path, metadata_path)
        record = load_model(metadata_path, RawLiteratureArtifact)
        content_path = self.knowledge_root.joinpath(
            *PurePosixPath(record.stored_path).parts
        )
        self._validate_paths(directory, content_path, metadata_path)
        if record.artifact_id != artifact_id:
            raise ValueError("raw artifact metadata ID does not match its directory")
        content = content_path.read_bytes()
        verify_artifact_bytes(content, record.byte_size, record.sha256, "raw literature artifact")
        if self._detect_media_type(content) != record.media_type:
            raise ValueError("stored raw literature artifact media type changed")
        return record


    def list_metadata(self) -> tuple[RawLiteratureArtifact, ...]:
        return raw_artifact_metadata(self)

    @staticmethod
    def _detect_media_type(content: bytes) -> str | None:
        if content.startswith(b"%PDF-"):
            return "application/pdf"
        if not content.startswith(b"PK\x03\x04"):
            return None
        try:
            with ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())
                if not {"[Content_Types].xml", "word/document.xml"} <= names:
                    return None
                document = archive.getinfo("word/document.xml")
                if document.file_size > 20_000_000 or document.flag_bits & 1:
                    return None
        except BadZipFile:
            return None
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


TextArtifactT = TypeVar("TextArtifactT", CanonicalTextArtifact, CanonicalHTMLTextArtifact)


class _CanonicalTextRepository(ArtifactDirectory[TextArtifactT]):
    _model_type: type[TextArtifactT]
    _id_error: str
    _label: str
    _block_error: str

    def get(self, canonical_text_id: str) -> TextArtifactT:
        return self._read_validated_text(canonical_text_id)[0]

    def _read_validated_text(self, canonical_text_id: str) -> tuple[TextArtifactT, str]:
        require_safe_path_component(canonical_text_id, field="canonical_text_id")
        directory = self.root / canonical_text_id
        content_path = directory / "content.txt"
        metadata_path = directory / "metadata.json"
        self._validate_paths(directory, content_path, metadata_path)
        record = load_model(metadata_path, self._model_type)
        if record.canonical_text_id != canonical_text_id:
            raise ValueError(self._id_error)
        try:
            text = content_path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{self._label} is not valid UTF-8") from error
        if len(text) != record.character_count:
            raise ValueError(f"{self._label} character count changed")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != record.text_sha256:
            raise ValueError(f"{self._label} hash changed")
        self._verify_blocks(record, text)
        return record, text

    def read_text(self, canonical_text_id: str) -> str:
        record = self.get(canonical_text_id)
        return self.knowledge_root.joinpath(
            *PurePosixPath(record.stored_path).parts
        ).read_text(encoding="utf-8")

    @classmethod
    def _verify_blocks(cls, record: TextArtifactT, text: str) -> None:
        for block in record.blocks:
            recovered = text[block.start_offset : block.end_offset]
            if hashlib.sha256(recovered.encode("utf-8")).hexdigest() != block.text_hash:
                raise ValueError(
                    f"{cls._block_error}: {block.block_id}"
                )


class CanonicalTextArtifactRepository(_CanonicalTextRepository[CanonicalTextArtifact]):
    _model_type = CanonicalTextArtifact
    _id_error = "canonical text metadata ID does not match its directory"
    _label = "canonical text"
    _block_error = "canonical text block does not recover its text hash"

    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root, "canonical_text")

    def put(
        self,
        text: str,
        *,
        literature_id: str,
        raw_artifact: RawLiteratureArtifact,
        parser_id: str,
        parser_version: str,
        parser_config_hash: str,
        page_count: int,
        blocks: tuple[CanonicalTextBlock, ...],
    ) -> CanonicalTextArtifact:
        require_safe_path_component(literature_id, field="literature_id")
        require_safe_path_component(parser_id, field="parser_id")
        require_safe_path_component(parser_version, field="parser_version")
        encoded = text.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        identity = {
            "literature_id": literature_id,
            "raw_artifact_id": raw_artifact.artifact_id,
            "raw_artifact_hash": raw_artifact.content_hash,
            "parser_id": parser_id,
            "parser_version": parser_version,
            "parser_config_hash": parser_config_hash,
            "text_sha256": digest,
            "character_count": len(text),
            "page_count": page_count,
            "blocks": blocks,
        }
        canonical_text_id = f"canonical-text-{content_hash(identity)[:24]}"
        payload = {
            "canonical_text_id": canonical_text_id,
            **identity,
            "stored_path": f"canonical_text/{canonical_text_id}/content.txt",
        }
        record = CanonicalTextArtifact(
            **payload,
            content_hash=content_hash(payload),
        )
        self._verify_blocks(record, text)
        destination = self.root / canonical_text_id
        if destination.exists():
            existing = self.get(canonical_text_id)
            if existing != record:
                raise FileExistsError(
                    f"refusing to overwrite different canonical text: {canonical_text_id}"
                )
            return existing
        self._write_directory(destination, "content.txt", encoded, record)
        return self.get(canonical_text_id)


    def list_metadata(self) -> tuple[CanonicalTextArtifact, ...]:
        records: list[CanonicalTextArtifact] = []
        for directory in self._directories():
            content_path = directory / "content.txt"
            metadata_path = directory / "metadata.json"
            self._validate_paths(directory, content_path, metadata_path)
            record = load_model(metadata_path, CanonicalTextArtifact)
            if record.canonical_text_id != directory.name:
                raise ValueError(
                    "canonical text metadata ID does not match its directory"
                )
            records.append(record)
        return tuple(records)


class CanonicalHTMLTextArtifactRepository(_CanonicalTextRepository[CanonicalHTMLTextArtifact]):
    _model_type = CanonicalHTMLTextArtifact
    _id_error = "HTML canonical metadata ID does not match directory"
    _label = "HTML canonical text"
    _block_error = "HTML block does not recover exact text"

    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root, "canonical_html_text")

    def put(
        self,
        text: str,
        *,
        raw_artifact: RawHTMLLiteratureArtifact,
        extractor_id: str,
        extractor_version: str,
        extractor_config_hash: str,
        blocks: tuple[CanonicalTextBlock, ...],
    ) -> CanonicalHTMLTextArtifact:
        encoded = text.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        identity = {
            "literature_id": raw_artifact.literature_id,
            "raw_artifact_id": raw_artifact.artifact_id,
            "raw_artifact_hash": raw_artifact.content_hash,
            "source_url": raw_artifact.source_url,
            "extractor_id": extractor_id,
            "extractor_version": extractor_version,
            "extractor_config_hash": extractor_config_hash,
            "text_sha256": digest,
            "character_count": len(text),
            "blocks": blocks,
        }
        canonical_text_id = f"canonical-html-text-{content_hash(identity)[:24]}"
        payload = {
            "canonical_text_id": canonical_text_id,
            **identity,
            "stored_path": (
                f"canonical_html_text/{canonical_text_id}/content.txt"
            ),
        }
        record = CanonicalHTMLTextArtifact(
            **payload, content_hash=content_hash(payload)
        )
        self._verify_blocks(record, text)
        destination = self.root / canonical_text_id
        if destination.exists():
            existing = self.get(canonical_text_id)
            if existing != record:
                raise FileExistsError(
                    f"refusing to overwrite different HTML text: {canonical_text_id}"
                )
            return existing
        self._write_directory(destination, "content.txt", encoded, record)
        return self.get(canonical_text_id)


    def list_metadata(self) -> tuple[CanonicalHTMLTextArtifact, ...]:
        return tuple(load_model(path / "metadata.json", CanonicalHTMLTextArtifact) for path in self._directories())


class RawHTMLLiteratureArtifactRepository(ArtifactDirectory[RawHTMLLiteratureArtifact]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root, "html_literature_artifacts")

    def put(
        self,
        content: bytes,
        *,
        source_url: str,
        literature_id: str,
        media_type: str,
    ) -> RawHTMLLiteratureArtifact:
        require_safe_path_component(literature_id, field="literature_id")
        if media_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("HTML artifact requires an HTML media type")
        digest = hashlib.sha256(content).hexdigest()
        identity = {
            "source_url": source_url,
            "literature_id": literature_id,
            "sha256": digest,
        }
        artifact_id = f"html-artifact-{content_hash(identity)[:24]}"
        payload = {
            "artifact_id": artifact_id,
            **identity,
            "media_type": media_type,
            "byte_size": len(content),
            "stored_path": (
                f"html_literature_artifacts/{artifact_id}/artifact.html"
            ),
        }
        record = RawHTMLLiteratureArtifact(
            **payload, content_hash=content_hash(payload)
        )
        destination = self.root / artifact_id
        if destination.exists():
            existing = self.get(artifact_id)
            if existing != record:
                raise FileExistsError(
                    f"refusing to overwrite different HTML artifact: {artifact_id}"
                )
            return existing
        self._write_directory(destination, "artifact.html", content, record)
        return self.get(artifact_id)

    def get(self, artifact_id: str) -> RawHTMLLiteratureArtifact:
        require_safe_path_component(artifact_id, field="artifact_id")
        directory = self.root / artifact_id
        content_path = directory / "artifact.html"
        metadata_path = directory / "metadata.json"
        self._validate_paths(directory, content_path, metadata_path)
        record = load_model(metadata_path, RawHTMLLiteratureArtifact)
        if record.artifact_id != artifact_id:
            raise ValueError("HTML artifact metadata ID does not match its directory")
        content = content_path.read_bytes()
        verify_artifact_bytes(content, record.byte_size, record.sha256, "raw HTML artifact")
        return record


    def list_metadata(self) -> tuple[RawHTMLLiteratureArtifact, ...]:
        return tuple(load_model(path / "metadata.json", RawHTMLLiteratureArtifact) for path in self._directories())


class LiteratureDocumentRepository(IdentityBoundRepository[LiteratureDocument]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_documents",
            LiteratureDocument,
            "literature_id",
        )


class LiteratureArtifactStore:
    def __init__(self, knowledge_root: Path) -> None:
        self.raw_artifacts = RawLiteratureArtifactRepository(knowledge_root)
        self.canonical_texts = CanonicalTextArtifactRepository(knowledge_root)


class LiteratureIngestionRepository(
    IdentityBoundRepository[LiteratureIngestionRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_ingestions",
            LiteratureIngestionRecord,
            "ingestion_id",
        )


class LiteratureRepresentationSelectionRepository(
    IdentityBoundRepository[LiteratureRepresentationSelection]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_representation_selections",
            LiteratureRepresentationSelection,
            "selection_id",
        )

    def resolve_current(
        self, literature_id: str
    ) -> LiteratureRepresentationSelection:
        require_safe_path_component(literature_id, field="literature_id")
        all_records = {item.selection_id: item for item in self.list()}
        records = tuple(
            item for item in all_records.values() if item.literature_id == literature_id
        )
        if not records:
            raise FileNotFoundError(
                f"no representation selection for literature: {literature_id}"
            )
        successors: dict[str, str] = {}
        for record in records:
            predecessor_id = record.supersedes_selection_id
            if predecessor_id is None:
                continue
            predecessor = all_records.get(predecessor_id)
            if predecessor is None:
                raise ValueError(
                    f"missing superseded representation selection: {predecessor_id}"
                )
            if predecessor.literature_id != literature_id:
                raise ValueError(
                    "representation selection history must retain literature_id"
                )
            if predecessor_id in successors:
                raise ValueError(
                    f"representation selection has multiple successors: {predecessor_id}"
                )
            successors[predecessor_id] = record.selection_id
        heads = [record for record in records if record.selection_id not in successors]
        if len(heads) != 1:
            raise ValueError(
                f"expected exactly one current representation selection: {literature_id}"
            )
        seen: set[str] = set()
        current = heads[0]
        cursor: LiteratureRepresentationSelection | None = current
        while cursor is not None:
            if cursor.selection_id in seen:
                raise ValueError(
                    f"cyclic representation selection history: {literature_id}"
                )
            seen.add(cursor.selection_id)
            predecessor_id = cursor.supersedes_selection_id
            cursor = all_records.get(predecessor_id) if predecessor_id else None
        if len(seen) != len(records):
            raise ValueError(
                f"cyclic or disconnected representation selection history: {literature_id}"
            )
        return current


class HistoricalLiteratureEvidenceAuthorizationRepository(
    IdentityBoundRepository[HistoricalLiteratureEvidenceAuthorization]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "historical_evidence_authorizations",
            HistoricalLiteratureEvidenceAuthorization,
            "authorization_id",
        )


class LiteratureAcquisitionRequestRepository(
    IdentityBoundRepository[LiteratureAcquisitionRequest]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_acquisition_requests",
            LiteratureAcquisitionRequest,
            "request_id",
        )


class ResolvedLiteratureResourceRepository(
    IdentityBoundRepository[ResolvedLiteratureResource]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "resolved_literature_resources",
            ResolvedLiteratureResource,
            "resource_id",
        )


class LiteratureAcquisitionRepository(
    IdentityBoundRepository[LiteratureAcquisitionRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_acquisitions",
            LiteratureAcquisitionRecord,
            "acquisition_id",
        )


class AcquisitionAttemptRepository(
    IdentityBoundRepository[AcquisitionAttemptRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "acquisition_attempts",
            AcquisitionAttemptRecord,
            "attempt_id",
        )


class MetadataRetrievalRepository(
    IdentityBoundRepository[MetadataRetrievalRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "metadata_retrievals",
            MetadataRetrievalRecord,
            "retrieval_id",
        )


class MetadataMergeManifestRepository(
    IdentityBoundRepository[MetadataMergeManifest]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "metadata_merge_manifests",
            MetadataMergeManifest,
            "manifest_id",
        )


class HTMLLiteratureIngestionRepository(
    IdentityBoundRepository[HTMLLiteratureIngestionRecord]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "html_literature_ingestions",
            HTMLLiteratureIngestionRecord,
            "ingestion_id",
        )


class LiteratureRepresentationReferenceRepository(
    IdentityBoundRepository[LiteratureRepresentationReference]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "literature_representation_refs",
            LiteratureRepresentationReference,
            "representation_id",
        )


class DocumentStructureArtifactRepository(
    IdentityBoundRepository[DocumentStructureArtifact]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "document_structure_artifacts",
            DocumentStructureArtifact,
            "structure_id",
        )


class DocumentStructureSelectionRepository(
    IdentityBoundRepository[DocumentStructureSelection]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "document_structure_selections",
            DocumentStructureSelection,
            "selection_id",
        )

    def resolve_current(self, representation_id: str) -> DocumentStructureSelection:
        require_safe_path_component(representation_id, field="representation_id")
        all_records = {item.selection_id: item for item in self.list()}
        records = tuple(
            item
            for item in all_records.values()
            if item.representation_id == representation_id
        )
        if not records:
            raise FileNotFoundError(
                f"no structure selection for representation: {representation_id}"
            )
        successors: dict[str, str] = {}
        for record in records:
            predecessor_id = record.supersedes_selection_id
            if predecessor_id is None:
                continue
            predecessor = all_records.get(predecessor_id)
            if predecessor is None:
                raise ValueError(
                    f"missing superseded structure selection: {predecessor_id}"
                )
            if (
                predecessor.representation_id != representation_id
                or predecessor.literature_id != record.literature_id
            ):
                raise ValueError(
                    "structure selection history must retain representation identity"
                )
            if predecessor_id in successors:
                raise ValueError(
                    f"structure selection has multiple successors: {predecessor_id}"
                )
            successors[predecessor_id] = record.selection_id
        heads = [record for record in records if record.selection_id not in successors]
        if len(heads) != 1:
            raise ValueError(
                f"expected exactly one current structure selection: {representation_id}"
            )
        seen: set[str] = set()
        cursor: DocumentStructureSelection | None = heads[0]
        while cursor is not None:
            if cursor.selection_id in seen:
                raise ValueError(
                    f"cyclic structure selection history: {representation_id}"
                )
            seen.add(cursor.selection_id)
            predecessor_id = cursor.supersedes_selection_id
            cursor = all_records.get(predecessor_id) if predecessor_id else None
        if len(seen) != len(records):
            raise ValueError(
                f"cyclic or disconnected structure selection history: {representation_id}"
            )
        return heads[0]


class DocumentStructureBlockRepository(
    IdentityBoundRepository[DocumentStructureBlock]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "document_structure_blocks",
            DocumentStructureBlock,
            "block_id",
        )


class TableStructureRepository(IdentityBoundRepository[TableStructure]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "table_structures", TableStructure, "table_id"
        )


class TableCellStructureRepository(
    IdentityBoundRepository[TableCellStructure]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "table_cell_structures",
            TableCellStructure,
            "cell_id",
        )


class FigureStructureRepository(IdentityBoundRepository[FigureStructure]):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "figure_structures", FigureStructure, "figure_id"
        )


class StructuredEvidenceLocatorRepository(
    IdentityBoundRepository[StructuredEvidenceLocator]
):
    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(
            knowledge_root / "structured_evidence_locators",
            StructuredEvidenceLocator,
            "locator_id",
        )


class LiteratureRepositories:
    def __init__(
        self,
        root: Path,
        *,
        initialize: bool = True,
        evidence_store: KnowledgeEvidenceStore | None = None,
    ) -> None:
        from ..knowledge.literature_knowledge.repositories import (
            LiteratureKnowledgeChunkRepository,
            LiteratureKnowledgeCompilationInputRepository,
            LiteratureKnowledgeCompilationRepository,
            LiteratureKnowledgeGroundingRepository,
            LiteratureKnowledgeProposalSetRepository,
        )

        self.root = root
        self.evidence_store = (
            KnowledgeEvidenceStore(root, initialize=initialize) if evidence_store is None else evidence_store
        )
        self.workflow_patterns = LiteratureWorkflowRepository(root)
        self.literature_documents = LiteratureDocumentRepository(root)
        self.literature_artifacts = LiteratureArtifactStore(root)
        self.raw_literature_artifacts = self.literature_artifacts.raw_artifacts
        self.canonical_text_artifacts = self.literature_artifacts.canonical_texts
        self.raw_html_literature_artifacts = (
            RawHTMLLiteratureArtifactRepository(root)
        )
        self.canonical_html_text_artifacts = (
            CanonicalHTMLTextArtifactRepository(root)
        )
        self.literature_ingestions = LiteratureIngestionRepository(root)
        self.html_literature_ingestions = HTMLLiteratureIngestionRepository(root)
        self.literature_representation_selections = (
            LiteratureRepresentationSelectionRepository(root)
        )
        self.literature_supplementary_sources = IdentityBoundRepository(
            root / "literature_supplementary_sources",
            LiteratureSupplementarySource,
            "supplement_id",
        )
        self.historical_evidence_authorizations = (
            HistoricalLiteratureEvidenceAuthorizationRepository(root)
        )
        self.acquisition_requests = LiteratureAcquisitionRequestRepository(root)
        self.resolved_literature_resources = (
            ResolvedLiteratureResourceRepository(root)
        )
        self.literature_acquisitions = LiteratureAcquisitionRepository(root)
        self.acquisition_attempts = AcquisitionAttemptRepository(root)
        self.metadata_retrievals = MetadataRetrievalRepository(root)
        self.metadata_merge_manifests = MetadataMergeManifestRepository(root)
        self.literature_representation_refs = (
            LiteratureRepresentationReferenceRepository(root)
        )
        self.document_structure_artifacts = DocumentStructureArtifactRepository(
            root
        )
        self.document_structure_selections = DocumentStructureSelectionRepository(
            root
        )
        self.document_structure_blocks = DocumentStructureBlockRepository(root)
        self.table_structures = TableStructureRepository(root)
        self.table_cell_structures = TableCellStructureRepository(root)
        self.figure_structures = FigureStructureRepository(root)
        self.structured_evidence_locators = StructuredEvidenceLocatorRepository(
            root
        )
        self.literature_knowledge_inputs = LiteratureKnowledgeCompilationInputRepository(root)
        self.literature_knowledge_chunks = LiteratureKnowledgeChunkRepository(root)
        self.literature_knowledge_proposals = LiteratureKnowledgeProposalSetRepository(root)
        self.literature_knowledge_groundings = LiteratureKnowledgeGroundingRepository(root)
        self.literature_knowledge_compilations = LiteratureKnowledgeCompilationRepository(root)
