from __future__ import annotations

from dataclasses import dataclass

from ...models import (
    CurationStatus,
    EvidenceSpan,
    ExpertCase,
    ExpertOpinion,
)
from ...repositories import KnowledgeRepositories
from ...serialization import content_hash
from ..literature_knowledge.validation import curate_knowledge_record
from ..trust import TrustedKnowledgeValidator
from .contracts import (
    ExpertKnowledgeChunk,
    ExpertKnowledgeCompilationInput,
    ExpertKnowledgeCompilationRecord,
    ExpertKnowledgeGroundingRecord,
    ExpertKnowledgeProposalSet,
    RejectedExpertKnowledgeProposal,
)


COMPILER_ID = "spc-expert-knowledge-compiler"
COMPILER_VERSION = "1.0.0"


@dataclass(frozen=True)
class ExpertKnowledgeMaterializationOutcome:
    record: ExpertKnowledgeCompilationRecord
    opinions: tuple[ExpertOpinion, ...]
    cases: tuple[ExpertCase, ...]
    groundings: tuple[ExpertKnowledgeGroundingRecord, ...]


def _put_reuse(repository, key: str, record) -> None:
    try:
        existing = repository.get(key)
    except FileNotFoundError:
        repository.put(key, record)
        return
    if existing != record:
        raise FileExistsError(f"conflicting materialized expert record: {key}")


def _content_bound(model_type, prefix: str, id_field: str, identity: dict):
    record_id = f"{prefix}-{content_hash(identity)[:24]}"
    payload = {id_field: record_id, **identity}
    return model_type(**payload, content_hash=content_hash(payload))


def _reject(
    rejected: list[RejectedExpertKnowledgeProposal],
    reference: str,
    kind: str,
    code: str,
) -> None:
    rejected.append(
        RejectedExpertKnowledgeProposal(
            proposal_ref=reference,
            proposal_kind=kind,
            rejection_code=code,
        )
    )


def _ensure_machine_curation(
    repositories: KnowledgeRepositories,
    target_type: str,
    target_id: str,
) -> None:
    current = TrustedKnowledgeValidator(
        repositories, repositories.evidence_store
    ).resolve_current_curations().get((target_type, target_id))
    if current is not None:
        return
    curate_knowledge_record(
        repositories,
        target_type=target_type,
        target_id=target_id,
        status=CurationStatus.MACHINE_EXTRACTED,
        curator_id=COMPILER_ID,
        rationale="Machine-extracted K1G proposal; requires explicit human review.",
    )


class ExpertKnowledgeMaterializer:
    def materialize(
        self,
        compilation_input: ExpertKnowledgeCompilationInput,
        chunks: tuple[ExpertKnowledgeChunk, ...],
        proposal_set: ExpertKnowledgeProposalSet,
        repositories: KnowledgeRepositories,
    ) -> ExpertKnowledgeMaterializationOutcome:
        if (
            proposal_set.compilation_input_id != compilation_input.compilation_input_id
            or proposal_set.compilation_input_hash != compilation_input.content_hash
        ):
            raise ValueError("expert proposal set is bound to another compilation input")
        _put_reuse(
            repositories.expert_knowledge_proposals,
            proposal_set.proposal_set_id,
            proposal_set,
        )
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        source_record = repositories.expert_sources.get(compilation_input.expert_source_id)
        source = repositories.evidence_store.get_source(
            compilation_input.source_id, compilation_input.source_version
        )
        repositories.evidence_store.verify_source_integrity(source)
        rejected: list[RejectedExpertKnowledgeProposal] = []
        grounding_by_quote: dict[str, ExpertKnowledgeGroundingRecord] = {}
        groundings: list[ExpertKnowledgeGroundingRecord] = []
        for proposal in proposal_set.quote_proposals:
            chunk = chunk_map.get(proposal.chunk_id)
            if chunk is None:
                _reject(rejected, proposal.quote_key, "quote", "UNKNOWN_CHUNK")
                continue
            matches: list[int] = []
            cursor = 0
            while True:
                position = chunk.text.find(proposal.exact_text, cursor)
                if position < 0:
                    break
                matches.append(position)
                cursor = position + 1
            if not matches:
                _reject(rejected, proposal.quote_key, "quote", "QUOTE_NOT_EXACT")
                continue
            if len(matches) != 1:
                _reject(rejected, proposal.quote_key, "quote", "AMBIGUOUS_EXACT_QUOTE")
                continue
            start = chunk.start_offset + matches[0]
            end = start + len(proposal.exact_text)
            evidence_identity = {
                "source_id": source.source_id,
                "source_version": source.version,
                "start_offset": start,
                "end_offset": end,
                "text_hash": content_hash({"text": proposal.exact_text}),
            }
            evidence = EvidenceSpan(
                evidence_id=f"evidence-{content_hash(evidence_identity)[:24]}",
                source_id=source.source_id,
                source_version=source.version,
                content_sha256=source.content_sha256,
                start_offset=start,
                end_offset=end,
                text=proposal.exact_text,
                locator=(f"page {chunk.page_number}" if chunk.page_number else "expert source"),
            )
            repositories.evidence_store.add_evidence(evidence)
            grounding_identity = {
                "expert_source_id": source_record.expert_source_id,
                "expert_source_hash": source_record.content_hash,
                "attribution_id": source_record.attribution_id,
                "attribution_hash": source_record.attribution_hash,
                "chunk_id": chunk.chunk_id,
                "chunk_hash": chunk.content_hash,
                "evidence_id": evidence.evidence_id,
                "evidence_hash": content_hash(evidence),
                "page_number": chunk.page_number,
                "section_path": chunk.section_path,
            }
            grounding_identity = {
                key: value for key, value in grounding_identity.items() if value is not None
            }
            grounding = _content_bound(
                ExpertKnowledgeGroundingRecord,
                "expert-knowledge-grounding",
                "grounding_id",
                grounding_identity,
            )
            _put_reuse(
                repositories.expert_knowledge_groundings,
                grounding.grounding_id,
                grounding,
            )
            grounding_by_quote[proposal.quote_key] = grounding
            groundings.append(grounding)

        opinions_by_key: dict[str, ExpertOpinion] = {}
        opinions: list[ExpertOpinion] = []
        for proposal in proposal_set.opinion_proposals:
            if proposal.expert_id != compilation_input.expert_id:
                _reject(rejected, proposal.opinion_key, "expert_opinion", "EXPERT_ID_MISMATCH")
                continue
            bound = [grounding_by_quote.get(key) for key in proposal.quote_keys]
            if any(item is None for item in bound):
                _reject(rejected, proposal.opinion_key, "expert_opinion", "OPINION_QUOTE_REJECTED")
                continue
            evidence_refs = tuple(sorted({item.evidence_id for item in bound if item is not None}))
            identity = {
                "expert_id": compilation_input.expert_id,
                "domain": compilation_input.domain,
                "topic": proposal.topic,
                "statement": proposal.normalized_text,
                "opinion_type": proposal.opinion_category,
                "scope": proposal.scope,
                "rationale": proposal.rationale,
                "conditions": proposal.conditions,
                "evidence_refs": evidence_refs,
                "attribution_refs": (compilation_input.attribution_id,),
                "related_claim_refs": (),
                "related_workflow_refs": (),
                "authority_status": proposal.authority_status,
            }
            opinion_id = f"expert-opinion-{content_hash(identity)[:24]}"
            payload = {"opinion_id": opinion_id, **identity}
            opinion = ExpertOpinion(**payload, content_hash=content_hash(payload))
            _put_reuse(repositories.expert_opinions, opinion_id, opinion)
            _ensure_machine_curation(repositories, "expert_opinion", opinion_id)
            opinions_by_key[proposal.opinion_key] = opinion
            opinions.append(opinion)

        cases: list[ExpertCase] = []
        for proposal in proposal_set.case_proposals:
            linked = [opinions_by_key.get(key) for key in proposal.opinion_keys]
            if any(item is None for item in linked):
                _reject(rejected, proposal.case_key, "expert_case", "CASE_OPINION_REJECTED")
                continue
            opinion_refs = tuple(sorted({item.opinion_id for item in linked if item is not None}))
            evidence_refs = tuple(
                sorted(
                    {
                        evidence_id
                        for item in linked
                        if item is not None
                        for evidence_id in item.evidence_refs
                    }
                )
            )
            case_identity = {
                "domain": compilation_input.domain,
                "vague_request": proposal.original_wording,
                "translated_questions": proposal.good_question_formulations,
                "positive": True,
                "rationale": proposal.resolution_pattern,
                "evidence_refs": evidence_refs,
                "opinion_refs": opinion_refs,
                "original_wording": proposal.original_wording,
                "latent_concern": proposal.latent_concern,
                "atomic_questions": proposal.atomic_questions,
                "good_question_formulations": proposal.good_question_formulations,
                "wrong_formulations": proposal.wrong_formulations,
                "answerability_conditions": proposal.answerability_conditions,
                "required_evidence_types": proposal.required_evidence_types,
                "baseline_guidance": proposal.baseline_guidance,
                "common_misinterpretations": proposal.common_misinterpretations,
                "resolution_pattern": proposal.resolution_pattern,
                "applicability": proposal.applicability,
            }
            provenance_hash = content_hash(case_identity)
            case = ExpertCase(
                case_id=f"expert-case-{provenance_hash[:24]}",
                **case_identity,
                provenance_hash=provenance_hash,
            )
            _put_reuse(repositories.expert_cases, case.case_id, case)
            _ensure_machine_curation(repositories, "expert_case", case.case_id)
            cases.append(case)

        compilation_identity = {
            "compilation_input_id": compilation_input.compilation_input_id,
            "compilation_input_hash": compilation_input.content_hash,
            "expert_source_id": source_record.expert_source_id,
            "expert_source_hash": source_record.content_hash,
            "source_id": source.source_id,
            "source_version": source.version,
            "source_hash": content_hash(source),
            "attribution_id": source_record.attribution_id,
            "attribution_hash": source_record.attribution_hash,
            "provider_id": proposal_set.provider_id,
            "provider_version": proposal_set.provider_version,
            "provider_config_hash": proposal_set.provider_config_hash,
            "proposal_set_id": proposal_set.proposal_set_id,
            "proposal_set_hash": proposal_set.content_hash,
            "chunk_hashes": {item.chunk_id: item.content_hash for item in chunks},
            "batch_invocations": proposal_set.batch_invocations,
            "expert_opinion_ids": tuple(sorted(item.opinion_id for item in opinions)),
            "expert_case_ids": tuple(sorted(item.case_id for item in cases)),
            "grounding_hashes": {
                item.grounding_id: item.content_hash for item in sorted(
                    groundings, key=lambda value: value.grounding_id
                )
            },
            "rejected_proposals": tuple(rejected),
            "compiler_id": COMPILER_ID,
            "compiler_version": COMPILER_VERSION,
        }
        compilation = _content_bound(
            ExpertKnowledgeCompilationRecord,
            "expert-knowledge-compilation",
            "compilation_id",
            compilation_identity,
        )
        _put_reuse(
            repositories.expert_knowledge_compilations,
            compilation.compilation_id,
            compilation,
        )
        return ExpertKnowledgeMaterializationOutcome(
            record=compilation,
            opinions=tuple(opinions),
            cases=tuple(cases),
            groundings=tuple(groundings),
        )


__all__ = [
    "ExpertKnowledgeMaterializationOutcome",
    "ExpertKnowledgeMaterializer",
]
