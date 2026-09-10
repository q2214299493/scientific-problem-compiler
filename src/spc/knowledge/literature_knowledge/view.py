from __future__ import annotations

from ...models import CurationStatus, DocumentContentRegion
from ...repositories import EvidenceStore, KnowledgeRepositories
from ...serialization import content_hash
from ..trust import TrustedKnowledgeValidator
from .context import validate_literature_knowledge_input
from .contracts import (
    LiteratureKnowledgeViewMode,
    LiteratureScientificKnowledgeView,
    LiteratureScientificKnowledgeViewRecord,
)
from .repositories import (
    LiteratureKnowledgeCompilationInputRepository,
    LiteratureKnowledgeCompilationRepository,
    LiteratureKnowledgeGroundingRepository,
)
from .validation import validate_grounding_record


RECORD_REPOSITORIES = {
    "source_claim": ("source_claims", "claim_id"),
    "method_fact": ("method_facts", "fact_id"),
    "model_fact": ("model_facts", "fact_id"),
    "reported_result": ("reported_results", "result_id"),
    "knowledge_relation": ("relations", "relation_id"),
}


class LiteratureScientificKnowledgeViewBuilder:
    def build(
        self,
        literature_id: str,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
        *,
        view_mode: LiteratureKnowledgeViewMode | str = LiteratureKnowledgeViewMode.AUDIT,
        record_type: str | None = None,
        curation_status: CurationStatus | str | None = None,
        content_region: DocumentContentRegion | str | None = None,
        section: str | None = None,
    ) -> LiteratureScientificKnowledgeView:
        store = evidence_store or repositories.evidence_store
        mode = LiteratureKnowledgeViewMode(view_mode)
        status_filter = CurationStatus(curation_status) if curation_status is not None else None
        region_filter = DocumentContentRegion(content_region) if content_region is not None else None
        current_curations = TrustedKnowledgeValidator(repositories, store).resolve_current_curations()
        if mode == LiteratureKnowledgeViewMode.TRUSTED_CURRENT:
            TrustedKnowledgeValidator(repositories, store).validate()
        inputs = LiteratureKnowledgeCompilationInputRepository(repositories.root)
        compilations = tuple(
            record
            for record in LiteratureKnowledgeCompilationRepository(repositories.root).list()
            if inputs.get(record.compilation_input_id).literature_id == literature_id
        )
        groundings_repository = LiteratureKnowledgeGroundingRepository(repositories.root)
        records: dict[tuple[str, str], LiteratureScientificKnowledgeViewRecord] = {}
        included_compilations: list[str] = []
        for compilation in compilations:
            compilation_input = inputs.get(compilation.compilation_input_id)
            if mode == LiteratureKnowledgeViewMode.TRUSTED_CURRENT:
                try:
                    validate_literature_knowledge_input(compilation_input, repositories, store)
                except ValueError:
                    continue
            groundings = tuple(
                groundings_repository.get(grounding_id)
                for grounding_id in compilation.grounding_hashes
            )
            for grounding in groundings:
                if grounding.content_hash != compilation.grounding_hashes[grounding.grounding_id]:
                    raise ValueError("compilation grounding hash is invalid")
                validate_grounding_record(
                    grounding,
                    repositories,
                    store,
                    require_current=mode == LiteratureKnowledgeViewMode.TRUSTED_CURRENT,
                )
            grounding_by_quote = {item.quote_id: item for item in groundings}
            grounding_by_evidence = {item.evidence_id: item for item in groundings}
            categories = {
                "source_claim": compilation.source_claim_ids,
                "method_fact": compilation.method_fact_ids,
                "model_fact": compilation.model_fact_ids,
                "reported_result": compilation.reported_result_ids,
                "knowledge_relation": compilation.knowledge_relation_ids,
            }
            for category, record_ids in categories.items():
                repository_name, identity_field = RECORD_REPOSITORIES[category]
                repository = getattr(repositories, repository_name)
                for record_id in record_ids:
                    scientific_record = repository.get(record_id)
                    if getattr(scientific_record, identity_field) != record_id:
                        raise ValueError("scientific knowledge record identity mismatch")
                    curation = current_curations.get((category, record_id))
                    if mode == LiteratureKnowledgeViewMode.TRUSTED_CURRENT and (
                        curation is None or curation.status != CurationStatus.ACCEPTED
                    ):
                        continue
                    related_groundings = []
                    if category == "source_claim":
                        related_groundings = [
                            grounding_by_quote[quote_id]
                            for quote_id in scientific_record.source_quote_refs
                            if quote_id in grounding_by_quote
                        ]
                    else:
                        related_groundings = [
                            grounding_by_evidence[evidence_id]
                            for evidence_id in scientific_record.evidence_refs
                            if evidence_id in grounding_by_evidence
                        ]
                    if not related_groundings:
                        if mode == LiteratureKnowledgeViewMode.TRUSTED_CURRENT:
                            raise ValueError("trusted K1F scientific record has no grounding")
                        continue
                    view_record = LiteratureScientificKnowledgeViewRecord(
                        record_type=category,
                        record_id=record_id,
                        record_hash=getattr(scientific_record, "content_hash", None)
                        or content_hash(scientific_record),
                        curation_status=curation.status if curation is not None else None,
                        grounding_refs=tuple(sorted(item.grounding_id for item in related_groundings)),
                        content_regions=tuple(sorted({item.content_region for item in related_groundings})),
                        section_paths=tuple(
                            sorted(
                                {
                                    repositories.document_structure_blocks.get(item.block_id).section_path
                                    for item in related_groundings
                                }
                            )
                        ),
                    )
                    if record_type is not None and view_record.record_type != record_type:
                        continue
                    if status_filter is not None and view_record.curation_status != status_filter:
                        continue
                    if region_filter is not None and region_filter not in view_record.content_regions:
                        continue
                    if section is not None and not any(section in path for path in view_record.section_paths):
                        continue
                    records[(category, record_id)] = view_record
            included_compilations.append(compilation.compilation_id)
        ordered = tuple(records[key] for key in sorted(records))
        identity = {
            "literature_id": literature_id,
            "view_mode": mode,
            "records": ordered,
            "compilation_ids": tuple(sorted(included_compilations)),
        }
        view_id = f"literature-knowledge-view-{content_hash(identity)[:24]}"
        payload = {"view_id": view_id, **identity}
        return LiteratureScientificKnowledgeView(**payload, content_hash=content_hash(payload))
