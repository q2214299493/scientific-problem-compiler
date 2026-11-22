"""Trust checks for literature representations, their sources and historical authorization."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...models import (
    CanonicalHTMLTextArtifact,
    CanonicalTextArtifact,
    CurationStatus,
    HistoricalLiteratureEvidenceAuthorization,
    HTMLLiteratureIngestionRecord,
    LiteratureDocument,
    LiteratureIngestionRecord,
    LiteratureIngestionStatus,
    LiteratureRepresentationKind,
    LiteratureRepresentationReference,
    LiteratureRepresentationSelection,
    LiteratureSupplementarySource,
    RawHTMLLiteratureArtifact,
    RawLiteratureArtifact,
    SourceRole,
    SourceType,
)
from . import (
    Curations,
    RecordKey,
    RememberEvidence,
    RequireRecord,
    TrustedKnowledgeError,
    VerifySource,
    VisitRecord,
)

if TYPE_CHECKING:
    from ...repositories import EvidenceStore, KnowledgeRepositories


def validate_literature(
    document: LiteratureDocument,
    *,
    repositories: KnowledgeRepositories,
    reachable: dict[RecordKey, BaseModel],
    visit: VisitRecord,
    verify_source: VerifySource,
    current: Curations,
) -> None:
    ingestions = tuple(
        ingestion
        for ingestion in repositories.literature_ingestions.list()
        if ingestion.literature_id == document.literature_id
    )
    html_ingestions = tuple(
        ingestion
        for ingestion in repositories.html_literature_ingestions.list()
        if ingestion.literature_id == document.literature_id
    )
    if ingestions or html_ingestions:
        try:
            selection = (
                repositories.literature_representation_selections.resolve_current(
                    document.literature_id
                )
            )
        except (FileNotFoundError, ValueError) as error:
            raise TrustedKnowledgeError(
                "INVALID_LITERATURE_REPRESENTATION_SELECTION",
                str(error),
            ) from error
        selection_curation = current.get(
            ("literature_representation_selection", selection.selection_id)
        )
        if (
            selection_curation is None
            or selection_curation.status != CurationStatus.ACCEPTED
        ):
            raise TrustedKnowledgeError(
                "UNTRUSTED_LITERATURE_REPRESENTATION_SELECTION",
                f"current representation selection is not accepted: {selection.selection_id}",
            )
        visit(("literature_representation_selection", selection.selection_id))
        return
    source = verify_source(document.source_id, document.source_version)
    reachable[("source_document", f"{source.source_id}@{source.version}")] = source


def validate_supplementary_source(
    supplement: LiteratureSupplementarySource,
    *,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None,
    reachable: dict[RecordKey, BaseModel],
    visit: VisitRecord,
    current: Curations,
) -> None:
    if evidence_store is None:
        raise TrustedKnowledgeError(
            "EVIDENCE_STORE_REQUIRED",
            "supplementary source validation requires an EvidenceStore",
        )
    document = visit(("literature_document", supplement.literature_id))
    try:
        from ..ingestion import validate_literature_supplementary_source

        raw, canonical, source = validate_literature_supplementary_source(
            supplement, repositories, evidence_store
        )
    except (FileNotFoundError, ValueError) as error:
        raise TrustedKnowledgeError(
            "INVALID_SUPPLEMENTARY_SOURCE",
            f"supplementary source chain is unavailable: {supplement.supplement_id}",
        ) from error
    if (
        not isinstance(document, LiteratureDocument)
        or document.content_hash != supplement.literature_hash
    ):
        raise TrustedKnowledgeError(
            "INVALID_SUPPLEMENTARY_SOURCE",
            f"supplementary source chain does not match: {supplement.supplement_id}",
        )
    curation = current.get(
        ("literature_supplementary_source", supplement.supplement_id)
    )
    if curation is not None:
        for evidence_id in curation.evidence_refs:
            evidence = evidence_store.get_evidence(evidence_id)
            evidence_store.verify_evidence_integrity(evidence)
            if (evidence.source_id, evidence.source_version) != (
                supplement.source_id,
                supplement.source_version,
            ):
                raise TrustedKnowledgeError(
                    "SUPPLEMENTARY_EVIDENCE_SOURCE_MISMATCH",
                    f"curation evidence belongs to another source: {evidence_id}",
                )
    reachable[("raw_literature_artifact", raw.artifact_id)] = raw
    reachable[("canonical_text_artifact", canonical.canonical_text_id)] = canonical
    reachable[("source_document", f"{source.source_id}@{source.version}")] = source


def validate_representation_selection(
    selection: LiteratureRepresentationSelection,
    *,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None,
    records: dict[RecordKey, BaseModel],
    visit: VisitRecord,
    current: Curations,
) -> None:
    try:
        current_selection = (
            repositories.literature_representation_selections.resolve_current(
                selection.literature_id
            )
        )
    except (FileNotFoundError, ValueError) as error:
        raise TrustedKnowledgeError(
            "INVALID_LITERATURE_REPRESENTATION_SELECTION", str(error)
        ) from error
    selection_curation = current.get(
        ("literature_representation_selection", selection.selection_id)
    )
    if current_selection.selection_id != selection.selection_id:
        raise TrustedKnowledgeError(
            "HISTORICAL_REPRESENTATION_NOT_TRUSTED",
            f"selection is not current: {selection.selection_id}",
        )
    if (
        selection_curation is None
        or selection_curation.status != CurationStatus.ACCEPTED
    ):
        raise TrustedKnowledgeError(
            "UNTRUSTED_LITERATURE_REPRESENTATION_SELECTION",
            f"selection is not explicitly accepted: {selection.selection_id}",
        )
    from ..ingestion import LiteratureRepresentationSelector

    try:
        representation = LiteratureRepresentationSelector.resolve_selection(
            selection, repositories, evidence_store
        )
    except (FileNotFoundError, ValueError) as error:
        raise TrustedKnowledgeError(
            "INVALID_LITERATURE_REPRESENTATION_SELECTION",
            str(error),
        ) from error
    records[
        ("literature_representation_ref", representation.representation_id)
    ] = representation
    visit(("literature_representation_ref", representation.representation_id))


def validate_canonical_text(
    canonical: CanonicalTextArtifact,
    *,
    repositories: KnowledgeRepositories,
    visit: VisitRecord,
) -> None:
    repositories.canonical_text_artifacts.get(canonical.canonical_text_id)
    raw = visit(("raw_literature_artifact", canonical.raw_artifact_id))
    if (
        not isinstance(raw, RawLiteratureArtifact)
        or canonical.raw_artifact_hash != raw.content_hash
        or canonical.literature_id != raw.literature_id
    ):
        raise TrustedKnowledgeError(
            "CANONICAL_RAW_ARTIFACT_MISMATCH",
            f"canonical text does not bind its raw artifact: {canonical.canonical_text_id}",
        )


def validate_html_canonical(
    canonical: CanonicalHTMLTextArtifact,
    *,
    repositories: KnowledgeRepositories,
    visit: VisitRecord,
) -> None:
    repositories.canonical_html_text_artifacts.get(
        canonical.canonical_text_id
    )
    raw = visit(("raw_html_literature_artifact", canonical.raw_artifact_id))
    if (
        not isinstance(raw, RawHTMLLiteratureArtifact)
        or canonical.raw_artifact_hash != raw.content_hash
        or canonical.literature_id != raw.literature_id
        or canonical.source_url != raw.source_url
    ):
        raise TrustedKnowledgeError(
            "HTML_CANONICAL_RAW_ARTIFACT_MISMATCH",
            f"HTML canonical text does not bind its raw artifact: {canonical.canonical_text_id}",
        )


def validate_representation(
    representation: LiteratureRepresentationReference,
    *,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None,
    visit: VisitRecord,
    current: Curations,
) -> None:
    from ..ingestion import LiteratureRepresentationSelector
    from ..representation import validate_literature_representation

    try:
        selection = (
            repositories.literature_representation_selections.resolve_current(
                representation.literature_id
            )
        )
        selection_curation = current.get(
            ("literature_representation_selection", selection.selection_id)
        )
        selected = LiteratureRepresentationSelector.resolve_selection(
            selection, repositories, evidence_store
        )
        if (
            selection_curation is None
            or selection_curation.status != CurationStatus.ACCEPTED
            or selected.representation_id != representation.representation_id
        ):
            raise ValueError("representation is not the accepted current selection")
        validate_literature_representation(
            representation, repositories, evidence_store
        )
    except (FileNotFoundError, ValueError) as error:
        raise TrustedKnowledgeError(
            "INVALID_LITERATURE_REPRESENTATION", str(error)
        ) from error
    ingestion_type = (
        "literature_ingestion"
        if representation.representation_kind == LiteratureRepresentationKind.PDF
        else "html_literature_ingestion"
    )
    ingestion = visit((ingestion_type, representation.ingestion_id))
    if (
        representation.literature_id != getattr(ingestion, "literature_id", None)
        or representation.ingestion_hash
        != getattr(ingestion, "content_hash", None)
    ):
        raise TrustedKnowledgeError(
            "INVALID_LITERATURE_REPRESENTATION",
            f"representation ingestion binding is invalid: {representation.representation_id}",
        )


def validate_html_ingestion(
    ingestion: HTMLLiteratureIngestionRecord,
    *,
    reachable: dict[RecordKey, BaseModel],
    visit: VisitRecord,
    require_record: RequireRecord,
    verify_source: VerifySource,
) -> None:
    if ingestion.ingestion_status != LiteratureIngestionStatus.ACCEPTED:
        raise TrustedKnowledgeError(
            "UNTRUSTED_HTML_LITERATURE_INGESTION",
            f"only accepted HTML ingestion records are trusted: {ingestion.ingestion_id}",
        )
    document = require_record("literature_document", ingestion.literature_id)
    if not isinstance(document, LiteratureDocument):
        raise TrustedKnowledgeError(
            "INVALID_HTML_LITERATURE_INGESTION_BINDING",
            f"HTML ingestion has no LiteratureDocument: {ingestion.ingestion_id}",
        )
    raw = visit(("raw_html_literature_artifact", ingestion.raw_artifact_id))
    canonical = visit(("canonical_html_text_artifact", ingestion.canonical_text_id))
    source = verify_source(ingestion.source_id, ingestion.source_version)
    if (
        not isinstance(raw, RawHTMLLiteratureArtifact)
        or not isinstance(canonical, CanonicalHTMLTextArtifact)
        or ingestion.raw_artifact_hash != raw.content_hash
        or ingestion.canonical_text_hash != canonical.content_hash
        or ingestion.literature_id != raw.literature_id
        or ingestion.literature_id != canonical.literature_id
        or ingestion.extractor_id != canonical.extractor_id
        or ingestion.extractor_version != canonical.extractor_version
        or ingestion.extractor_config_hash != canonical.extractor_config_hash
        or source.content_sha256 != canonical.text_sha256
        or source.source_role != SourceRole.LITERATURE_AUTHOR
        or source.source_type != SourceType.LITERATURE_ARTICLE
    ):
        raise TrustedKnowledgeError(
            "HTML_LITERATURE_INGESTION_HASH_MISMATCH",
            f"HTML ingestion binding is invalid: {ingestion.ingestion_id}",
        )
    reachable[
        ("source_document", f"{source.source_id}@{source.version}")
    ] = source


def validate_ingestion(
    ingestion: LiteratureIngestionRecord,
    *,
    reachable: dict[RecordKey, BaseModel],
    visit: VisitRecord,
    require_record: RequireRecord,
    verify_source: VerifySource,
) -> None:
    if ingestion.ingestion_status != LiteratureIngestionStatus.ACCEPTED:
        raise TrustedKnowledgeError(
            "UNTRUSTED_LITERATURE_INGESTION",
            f"only accepted ingestion records are trusted: {ingestion.ingestion_id}",
        )
    document = require_record("literature_document", ingestion.literature_id)
    if not isinstance(document, LiteratureDocument):
        raise TrustedKnowledgeError(
            "INVALID_LITERATURE_INGESTION_BINDING",
            f"ingestion has no LiteratureDocument: {ingestion.ingestion_id}",
        )
    raw = visit(("raw_literature_artifact", ingestion.raw_artifact_id))
    canonical = visit(("canonical_text_artifact", ingestion.canonical_text_id or ""))
    if (
        not isinstance(raw, RawLiteratureArtifact)
        or not isinstance(canonical, CanonicalTextArtifact)
        or ingestion.raw_artifact_hash != raw.content_hash
        or ingestion.canonical_text_hash != canonical.content_hash
        or ingestion.literature_id != raw.literature_id
        or ingestion.literature_id != canonical.literature_id
        or ingestion.parser_id != canonical.parser_id
        or ingestion.parser_version != canonical.parser_version
        or ingestion.parser_config_hash != canonical.parser_config_hash
    ):
        raise TrustedKnowledgeError(
            "LITERATURE_INGESTION_HASH_MISMATCH",
            f"ingestion artifact/parser binding is invalid: {ingestion.ingestion_id}",
        )
    source = verify_source(
        ingestion.source_id or "", ingestion.source_version or ""
    )
    if (
        source.content_sha256 != canonical.text_sha256
        or source.source_role != SourceRole.LITERATURE_AUTHOR
        or source.source_type != SourceType.LITERATURE_ARTICLE
    ):
        raise TrustedKnowledgeError(
            "CANONICAL_SOURCE_HASH_MISMATCH",
            f"SourceDocument is not the canonical literature source: {ingestion.ingestion_id}",
        )
    reachable[("source_document", f"{source.source_id}@{source.version}")] = source


def validate_historical_authorization(
    authorization: HistoricalLiteratureEvidenceAuthorization,
    *,
    repositories: KnowledgeRepositories,
    evidence_store: EvidenceStore | None,
    reachable: dict[RecordKey, BaseModel],
    verify_source: VerifySource,
    remember_evidence: RememberEvidence,
) -> None:
    if evidence_store is None:
        raise TrustedKnowledgeError(
            "EVIDENCE_STORE_REQUIRED",
            "historical evidence authorization requires an EvidenceStore",
        )
    from ..ingestion import LiteratureRepresentationSelector
    from ..representation import validate_literature_representation

    try:
        if authorization.representation_id is not None:
            representation = LiteratureRepresentationSelector.resolve_reference(
                authorization.representation_id,
                repositories,
                evidence_store,
            )
            if authorization.representation_hash != representation.content_hash:
                raise ValueError("historical representation hash mismatch")
        else:
            representation = LiteratureRepresentationSelector.resolve_reference(
                authorization.ingestion_id or "",
                repositories,
                evidence_store,
            )
            if authorization.ingestion_hash != representation.ingestion_hash:
                raise ValueError("historical ingestion hash mismatch")
        validate_literature_representation(
            representation, repositories, evidence_store
        )
        if (
            authorization.literature_id != representation.literature_id
            or authorization.source_id != representation.source_id
            or authorization.source_version != representation.source_version
        ):
            raise ValueError("historical authorization binding mismatch")
        for evidence_id in authorization.evidence_refs:
            evidence = evidence_store.get_evidence(evidence_id)
            evidence_source = evidence_store.verify_evidence_integrity(evidence)
            if (evidence_source.source_id, evidence_source.version) != (
                authorization.source_id,
                authorization.source_version,
            ):
                raise ValueError("historical evidence comes from another source")
            remember_evidence(evidence, evidence_source)
    except Exception as error:
        if isinstance(error, TrustedKnowledgeError):
            raise
        raise TrustedKnowledgeError(
            "INVALID_HISTORICAL_EVIDENCE_AUTHORIZATION",
            f"historical evidence authorization is invalid: {authorization.authorization_id}",
        ) from error
    reachable[
        ("historical_evidence_authorization", authorization.authorization_id)
    ] = authorization
    reachable[
        ("literature_representation_ref", representation.representation_id)
    ] = representation
    if representation.representation_kind == LiteratureRepresentationKind.PDF:
        ingestion = repositories.literature_ingestions.get(
            representation.ingestion_id
        )
        canonical = repositories.canonical_text_artifacts.get(
            representation.canonical_text_id or ""
        )
        raw = repositories.raw_literature_artifacts.get(
            representation.raw_artifact_id
        )
        record_types = (
            ("literature_ingestion", ingestion.ingestion_id, ingestion),
            ("canonical_text_artifact", canonical.canonical_text_id, canonical),
            ("raw_literature_artifact", raw.artifact_id, raw),
        )
    else:
        ingestion = repositories.html_literature_ingestions.get(
            representation.ingestion_id
        )
        canonical = repositories.canonical_html_text_artifacts.get(
            representation.canonical_text_id or ""
        )
        raw = repositories.raw_html_literature_artifacts.get(
            representation.raw_artifact_id
        )
        record_types = (
            ("html_literature_ingestion", ingestion.ingestion_id, ingestion),
            (
                "canonical_html_text_artifact",
                canonical.canonical_text_id,
                canonical,
            ),
            ("raw_html_literature_artifact", raw.artifact_id, raw),
        )
    for record_type, record_id, record in record_types:
        reachable[(record_type, record_id)] = record
    source = verify_source(
        representation.source_id or "", representation.source_version or ""
    )
    reachable[
        ("source_document", f"{source.source_id}@{source.version}")
    ] = source
