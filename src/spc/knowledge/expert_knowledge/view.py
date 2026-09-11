from __future__ import annotations

from ...repositories import KnowledgeRepositories
from ..trust import TrustedKnowledgeValidator
from .contracts import ExpertKnowledgeViewMode


class ExpertKnowledgeViewBuilder:
    def build(
        self,
        expert_id: str,
        repositories: KnowledgeRepositories,
        *,
        view_mode: ExpertKnowledgeViewMode | str = ExpertKnowledgeViewMode.AUDIT,
    ) -> dict:
        mode = ExpertKnowledgeViewMode(view_mode)
        profile = repositories.expert_profiles.get(expert_id)
        validator = TrustedKnowledgeValidator(repositories, repositories.evidence_store)
        current = validator.resolve_current_curations()
        trusted_keys: set[tuple[str, str]] = set()
        if mode == ExpertKnowledgeViewMode.TRUSTED:
            trusted_keys = set(validator.validate().trusted_records)
        source_ids = {
            item.expert_source_id
            for item in repositories.expert_sources.list()
            if item.expert_id == expert_id
        }
        compilations = tuple(
            item
            for item in repositories.expert_knowledge_compilations.list()
            if item.expert_source_id in source_ids
        )
        opinion_ids = {
            record_id for item in compilations for record_id in item.expert_opinion_ids
        }
        case_ids = {record_id for item in compilations for record_id in item.expert_case_ids}
        opinions = []
        for opinion_id in sorted(opinion_ids):
            if mode == ExpertKnowledgeViewMode.TRUSTED and (
                "expert_opinion",
                opinion_id,
            ) not in trusted_keys:
                continue
            opinion = repositories.expert_opinions.get(opinion_id)
            curation = current.get(("expert_opinion", opinion_id))
            quotes = []
            for evidence_id in opinion.evidence_refs:
                evidence = repositories.evidence_store.get_evidence(evidence_id)
                repositories.evidence_store.verify_evidence_integrity(evidence)
                grounding = next(
                    (
                        item
                        for item in repositories.expert_knowledge_groundings.list()
                        if item.evidence_id == evidence_id
                    ),
                    None,
                )
                quotes.append(
                    {
                        "evidence_id": evidence.evidence_id,
                        "exact_text": evidence.text,
                        "locator": evidence.locator,
                        "page_number": grounding.page_number if grounding else None,
                        "section_path": list(grounding.section_path) if grounding else [],
                    }
                )
            opinions.append(
                {
                    "opinion_id": opinion.opinion_id,
                    "statement": opinion.statement,
                    "opinion_category": opinion.opinion_type,
                    "authority_status": opinion.authority_status,
                    "topic": opinion.topic,
                    "scope": opinion.scope,
                    "expert_id": opinion.expert_id,
                    "attribution_refs": list(opinion.attribution_refs),
                    "curation_status": curation.status.value if curation else None,
                    "supporting_quotes": quotes,
                }
            )
        cases = []
        for case_id in sorted(case_ids):
            if mode == ExpertKnowledgeViewMode.TRUSTED and (
                "expert_case",
                case_id,
            ) not in trusted_keys:
                continue
            case = repositories.expert_cases.get(case_id)
            curation = current.get(("expert_case", case_id))
            cases.append(
                {
                    "case_id": case.case_id,
                    "opinion_refs": list(case.opinion_refs),
                    "original_wording": case.original_wording,
                    "latent_concern": case.latent_concern,
                    "atomic_questions": list(case.atomic_questions),
                    "good_question_formulations": list(case.good_question_formulations),
                    "wrong_formulations": list(case.wrong_formulations),
                    "answerability_conditions": list(case.answerability_conditions),
                    "required_evidence_types": list(case.required_evidence_types),
                    "baseline_guidance": list(case.baseline_guidance),
                    "common_misinterpretations": list(case.common_misinterpretations),
                    "resolution_pattern": case.resolution_pattern,
                    "applicability": list(case.applicability),
                    "curation_status": curation.status.value if curation else None,
                }
            )
        return {
            "view_mode": mode.value,
            "expert_profile": profile.model_dump(mode="json"),
            "expert_sources": sorted(source_ids),
            "expert_opinions": opinions,
            "expert_cases": cases,
            "compilation_ids": sorted(item.compilation_id for item in compilations),
            "rejected_proposals": [
                item.model_dump(mode="json")
                for compilation in compilations
                for item in compilation.rejected_proposals
            ],
        }


__all__ = ["ExpertKnowledgeViewBuilder"]
