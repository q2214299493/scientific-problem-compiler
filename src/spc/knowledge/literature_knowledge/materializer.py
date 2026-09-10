from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

from pydantic import BaseModel

from ...interpretation.claim_extractor import source_quote_id
from ...interpretation.validators import (
    result_context_record_issues,
    source_claim_binding_issues,
    source_quote_record_issues,
)
from ...models import (
    EpistemicStatus,
    KnowledgeRelation,
    MethodFact,
    ModelFact,
    ReportedResult,
    ResultContext,
    SourceClaim,
    SourceQuote,
)
from ...repositories import EvidenceStore, KnowledgeRepositories, ModelRepository
from ...serialization import content_hash
from ...backends.rebinding import exact_normalized_matches
from ..ingestion import create_evidence_span_from_canonical_text
from ..structure import _create_locator
from .context import (
    resolve_figure_ownership,
    resolve_table_ownership,
    validate_literature_knowledge_chunk,
    validate_literature_knowledge_input,
)
from .contracts import (
    LiteratureKnowledgeCompilationInput,
    LiteratureKnowledgeCompilationRecord,
    LiteratureKnowledgeGroundingRecord,
    LiteratureKnowledgeProposalSet,
    RejectedLiteratureKnowledgeProposal,
)
from .repositories import (
    LiteratureKnowledgeCompilationRepository,
    LiteratureKnowledgeGroundingRepository,
    LiteratureKnowledgeProposalSetRepository,
)
from .validation import ensure_machine_curation


COMPILER_ID = "spc-literature-scientific-knowledge-compiler"
COMPILER_VERSION = "1.0.0"
NUMERIC_TOKEN = re.compile(
    r"(?<![\w.])(?P<number>[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)(?![\w.])"
)


@dataclass(frozen=True)
class LiteratureKnowledgeCompilationOutcome:
    record: LiteratureKnowledgeCompilationRecord
    source_quotes: tuple[SourceQuote, ...]
    source_claims: tuple[SourceClaim, ...]
    method_facts: tuple[MethodFact, ...]
    model_facts: tuple[ModelFact, ...]
    reported_results: tuple[ReportedResult, ...]
    relations: tuple[KnowledgeRelation, ...]
    groundings: tuple[LiteratureKnowledgeGroundingRecord, ...]


def _put_reuse(repository: ModelRepository, key: str, record: BaseModel) -> None:
    try:
        existing = repository.get(key)
    except FileNotFoundError:
        repository.put(key, record)
        return
    if existing != record:
        raise FileExistsError(f"conflicting materialized record: {key}")


def _reject(target: list, reference: str, kind: str, code: str) -> None:
    target.append(
        RejectedLiteratureKnowledgeProposal(
            proposal_ref=reference,
            proposal_kind=kind,
            rejection_code=code,
        )
    )


def _claim_status_compatible(claim_type: str, status: EpistemicStatus) -> bool:
    normalized = claim_type.casefold().replace("-", "_").replace(" ", "_")
    required = None
    if "hypothesis" in normalized:
        required = EpistemicStatus.SOURCE_HYPOTHESIS
    elif "interpretation" in normalized:
        required = EpistemicStatus.SOURCE_INTERPRETATION
    elif "method" in normalized:
        required = EpistemicStatus.METHOD_STATEMENT
    elif "model" in normalized:
        required = EpistemicStatus.MODEL_STATEMENT
    elif "result" in normalized:
        required = EpistemicStatus.REPORTED_RESULT
    return required is None or status == required


def _matching_numeric_evidence(
    evidence_refs: tuple[str, ...],
    value: float,
    store: EvidenceStore,
) -> tuple[str, ...]:
    expected = Decimal(str(value))
    matches = []
    for evidence_id in evidence_refs:
        text = store.get_evidence(evidence_id).text
        for token in NUMERIC_TOKEN.finditer(text):
            try:
                parsed = Decimal(token.group("number"))
            except InvalidOperation:
                continue
            if parsed == expected:
                matches.append(evidence_id)
                break
    return tuple(matches)


def _contains_unit(text: str, unit: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(unit)}(?!\w)", text, flags=re.I) is not None


def _table_result_supports_unit_and_context(
    numeric_evidence_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
    unit: str,
    grounding_by_evidence: dict,
    repositories: KnowledgeRepositories,
    store: EvidenceStore,
) -> bool:
    located = {}
    for evidence_id in evidence_refs:
        grounding = grounding_by_evidence.get(evidence_id)
        if grounding is None:
            continue
        located[evidence_id] = repositories.structured_evidence_locators.get(
            grounding.locator_id
        )
    for numeric_evidence_id in numeric_evidence_ids:
        value_locator = located.get(numeric_evidence_id)
        if value_locator is None or value_locator.table_cell_id is None:
            continue
        value_cell = repositories.table_cell_structures.get(
            value_locator.table_cell_id
        )
        unit_supported = False
        row_context_supported = False
        for evidence_id, locator in located.items():
            if locator.table_id != value_locator.table_id:
                continue
            text = store.get_evidence(evidence_id).text
            if locator.table_cell_id is None:
                unit_supported |= _contains_unit(text, unit)
                continue
            cell = repositories.table_cell_structures.get(locator.table_cell_id)
            if cell.table_id != value_cell.table_id:
                continue
            if cell.is_header and (
                cell.column_index
                <= value_cell.column_index
                < cell.column_index + cell.column_span
            ):
                unit_supported |= _contains_unit(text, unit)
            if (
                not cell.is_header
                and cell.row_index == value_cell.row_index
                and cell.column_index != value_cell.column_index
            ):
                row_context_supported = True
        if unit_supported and row_context_supported:
            return True
    return False


def _result_is_explicitly_supported(
    *,
    evidence_refs: tuple[str, ...],
    value: float,
    unit: str,
    grounding_by_evidence: dict,
    repositories: KnowledgeRepositories,
    store: EvidenceStore,
) -> bool:
    numeric_evidence_ids = _matching_numeric_evidence(evidence_refs, value, store)
    if not numeric_evidence_ids:
        return False
    if any(
        _contains_unit(store.get_evidence(evidence_id).text, unit)
        for evidence_id in numeric_evidence_ids
    ):
        return True
    return _table_result_supports_unit_and_context(
        numeric_evidence_ids,
        evidence_refs,
        unit,
        grounding_by_evidence,
        repositories,
        store,
    )


def _grounding_record(
    compilation_input: LiteratureKnowledgeCompilationInput,
    *,
    block,
    evidence,
    locator,
    quote: SourceQuote,
) -> LiteratureKnowledgeGroundingRecord:
    identity = {
        "literature_id": compilation_input.literature_id,
        "literature_hash": compilation_input.literature_hash,
        "representation_selection_id": compilation_input.representation_selection_id,
        "representation_selection_hash": compilation_input.representation_selection_hash,
        "representation_id": compilation_input.representation_id,
        "representation_hash": compilation_input.representation_hash,
        "structure_selection_id": compilation_input.structure_selection_id,
        "structure_selection_hash": compilation_input.structure_selection_hash,
        "structure_id": compilation_input.structure_id,
        "structure_hash": compilation_input.structure_hash,
        "block_id": block.block_id,
        "block_hash": block.content_hash,
        "evidence_id": evidence.evidence_id,
        "evidence_hash": content_hash(evidence),
        "locator_id": locator.locator_id,
        "locator_hash": locator.content_hash,
        "quote_id": quote.quote_id,
        "content_region": block.content_region,
    }
    grounding_id = f"literature-knowledge-grounding-{content_hash(identity)[:24]}"
    payload = {"grounding_id": grounding_id, **identity}
    return LiteratureKnowledgeGroundingRecord(**payload, content_hash=content_hash(payload))


class LiteratureKnowledgeMaterializer:
    def materialize(
        self,
        compilation_input: LiteratureKnowledgeCompilationInput,
        chunks,
        proposal_set: LiteratureKnowledgeProposalSet,
        repositories: KnowledgeRepositories,
        evidence_store: EvidenceStore | None = None,
    ) -> LiteratureKnowledgeCompilationOutcome:
        store = evidence_store or repositories.evidence_store
        validate_literature_knowledge_input(compilation_input, repositories, store)
        if (
            proposal_set.compilation_input_id != compilation_input.compilation_input_id
            or proposal_set.compilation_input_hash != compilation_input.content_hash
        ):
            raise ValueError("literature knowledge proposal does not bind the compilation input")
        LiteratureKnowledgeProposalSetRepository(repositories.root).put(
            proposal_set.proposal_set_id,
            proposal_set,
        )
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(chunks_by_id) != len(chunks):
            raise ValueError("literature knowledge chunks must be unique")
        for chunk in chunks:
            validate_literature_knowledge_chunk(chunk, compilation_input, repositories)
        structure = repositories.document_structure_artifacts.get(
            compilation_input.structure_id
        )
        rejected: list[RejectedLiteratureKnowledgeProposal] = []
        quotes_by_key: dict[str, SourceQuote] = {}
        grounding_by_quote: dict[str, LiteratureKnowledgeGroundingRecord] = {}

        for proposal in proposal_set.quote_proposals:
            chunk = chunks_by_id.get(proposal.chunk_id)
            if chunk is None or proposal.block_id not in chunk.block_refs:
                _reject(rejected, proposal.quote_key, "quote", "QUOTE_OUTSIDE_ALLOWED_CHUNK")
                continue
            block = repositories.document_structure_blocks.get(proposal.block_id)
            if (
                block.structure_id != compilation_input.structure_id
                or block.content_region not in compilation_input.included_content_regions
            ):
                _reject(rejected, proposal.quote_key, "quote", "QUOTE_OUTSIDE_CONTENT_POLICY")
                continue
            matches = exact_normalized_matches(proposal.exact_text, chunk.text)
            if len(matches) != 1:
                code = "QUOTE_NOT_FOUND" if not matches else "QUOTE_AMBIGUOUS"
                _reject(rejected, proposal.quote_key, "quote", code)
                continue
            relative_start, relative_end = matches[0]
            start = chunk.canonical_start_offset + relative_start
            end = chunk.canonical_start_offset + relative_end
            if start < block.start_offset or end > block.end_offset:
                _reject(rejected, proposal.quote_key, "quote", "QUOTE_OUTSIDE_BLOCK")
                continue
            evidence = create_evidence_span_from_canonical_text(
                compilation_input.canonical_text_id,
                start,
                end,
                repositories,
                store,
                locator=f"K1F exact quote in {block.block_id}",
            )
            table_ownership = resolve_table_ownership(block, structure, repositories)
            table, cell = table_ownership if table_ownership is not None else (None, None)
            figure = resolve_figure_ownership(block, structure, repositories)
            evidence, locator = _create_locator(
                evidence=evidence,
                artifact=structure,
                block=block,
                repositories=repositories,
                evidence_store=store,
                table=table,
                cell=cell,
                figure=figure,
            )
            source = store.get_source(evidence.source_id, evidence.source_version)
            quote = SourceQuote(
                quote_id=source_quote_id(evidence.evidence_id, 0, len(evidence.text), evidence.text),
                evidence_ref=evidence.evidence_id,
                relative_start_offset=0,
                relative_end_offset=len(evidence.text),
                text=evidence.text,
                source_id=source.source_id,
                source_version=source.version,
                source_role=source.source_role,
                source_type=source.source_type,
            )
            issues, _ = source_quote_record_issues(quote, store, path=f"quote:{proposal.quote_key}")
            if issues:
                _reject(rejected, proposal.quote_key, "quote", issues[0].code)
                continue
            _put_reuse(repositories.source_quotes, quote.quote_id, quote)
            grounding = _grounding_record(
                compilation_input,
                block=block,
                evidence=evidence,
                locator=locator,
                quote=quote,
            )
            _put_reuse(
                LiteratureKnowledgeGroundingRepository(repositories.root),
                grounding.grounding_id,
                grounding,
            )
            quotes_by_key[proposal.quote_key] = quote
            grounding_by_quote[quote.quote_id] = grounding

        claims_by_key: dict[str, SourceClaim] = {}
        for proposal in proposal_set.claim_proposals:
            quotes = tuple(quotes_by_key[key] for key in proposal.quote_keys if key in quotes_by_key)
            if len(quotes) != len(proposal.quote_keys):
                _reject(rejected, proposal.claim_key, "claim", "CLAIM_QUOTE_REJECTED")
                continue
            if not _claim_status_compatible(proposal.claim_type, proposal.epistemic_status):
                _reject(rejected, proposal.claim_key, "claim", "EPISTEMIC_STATUS_MISMATCH")
                continue
            roles = {quote.source_role for quote in quotes}
            if len(roles) != 1:
                _reject(rejected, proposal.claim_key, "claim", "CLAIM_SOURCE_ROLE_AMBIGUOUS")
                continue
            evidence_refs = tuple(sorted({quote.evidence_ref for quote in quotes}))
            quote_refs = tuple(sorted(quote.quote_id for quote in quotes))
            identity = {
                "text": proposal.text,
                "claim_type": proposal.claim_type,
                "source_role": next(iter(roles)),
                "evidence_refs": evidence_refs,
                "source_quote_refs": quote_refs,
                "claim_strength": proposal.claim_strength,
                "epistemic_status": proposal.epistemic_status,
            }
            claim = SourceClaim(
                claim_id=f"claim-{content_hash(identity)[:24]}",
                **identity,
            )
            issues = source_claim_binding_issues(
                claim,
                {quote.quote_id: quote for quote in quotes},
                path=f"claim:{proposal.claim_key}",
            )
            if issues:
                _reject(rejected, proposal.claim_key, "claim", issues[0].code)
                continue
            _put_reuse(repositories.source_claims, claim.claim_id, claim)
            ensure_machine_curation(repositories, "source_claim", claim.claim_id)
            claims_by_key[proposal.claim_key] = claim

        method_by_key: dict[str, MethodFact] = {}
        for proposal in proposal_set.method_fact_proposals:
            claims = tuple(claims_by_key[key] for key in proposal.claim_keys if key in claims_by_key)
            if len(claims) != len(proposal.claim_keys):
                _reject(rejected, proposal.fact_key, "method_fact", "FACT_CLAIM_REJECTED")
                continue
            if not any(claim.epistemic_status == EpistemicStatus.METHOD_STATEMENT for claim in claims):
                _reject(rejected, proposal.fact_key, "method_fact", "METHOD_NOT_EXPLICIT")
                continue
            evidence_refs = tuple(sorted({ref for claim in claims for ref in claim.evidence_refs}))
            identity = {
                "text": proposal.text,
                "attributes": proposal.attributes,
                "evidence_refs": evidence_refs,
                "epistemic_status": EpistemicStatus.METHOD_STATEMENT,
            }
            fact = MethodFact(fact_id=f"method-fact-{content_hash(identity)[:24]}", **identity)
            _put_reuse(repositories.method_facts, fact.fact_id, fact)
            ensure_machine_curation(repositories, "method_fact", fact.fact_id)
            method_by_key[proposal.fact_key] = fact

        model_by_key: dict[str, ModelFact] = {}
        for proposal in proposal_set.model_fact_proposals:
            claims = tuple(claims_by_key[key] for key in proposal.claim_keys if key in claims_by_key)
            if len(claims) != len(proposal.claim_keys):
                _reject(rejected, proposal.fact_key, "model_fact", "FACT_CLAIM_REJECTED")
                continue
            if not any(claim.epistemic_status == EpistemicStatus.MODEL_STATEMENT for claim in claims):
                _reject(rejected, proposal.fact_key, "model_fact", "MODEL_NOT_EXPLICIT")
                continue
            evidence_refs = tuple(sorted({ref for claim in claims for ref in claim.evidence_refs}))
            identity = {
                "text": proposal.text,
                "attributes": proposal.attributes,
                "evidence_refs": evidence_refs,
                "epistemic_status": EpistemicStatus.MODEL_STATEMENT,
            }
            fact = ModelFact(fact_id=f"model-fact-{content_hash(identity)[:24]}", **identity)
            _put_reuse(repositories.model_facts, fact.fact_id, fact)
            ensure_machine_curation(repositories, "model_fact", fact.fact_id)
            model_by_key[proposal.fact_key] = fact

        result_by_key: dict[str, ReportedResult] = {}
        grounding_by_evidence = {
            grounding.evidence_id: grounding
            for grounding in grounding_by_quote.values()
        }
        for proposal in proposal_set.reported_result_proposals:
            claims = tuple(claims_by_key[key] for key in proposal.claim_keys if key in claims_by_key)
            if len(claims) != len(proposal.claim_keys):
                _reject(rejected, proposal.result_key, "reported_result", "RESULT_CLAIM_REJECTED")
                continue
            if proposal.unit is None:
                _reject(rejected, proposal.result_key, "reported_result", "MISSING_RESULT_UNIT")
                continue
            if not any(claim.epistemic_status == EpistemicStatus.REPORTED_RESULT for claim in claims):
                _reject(rejected, proposal.result_key, "reported_result", "RESULT_NOT_EXPLICIT")
                continue
            evidence_refs = tuple(sorted({ref for claim in claims for ref in claim.evidence_refs}))
            if not _result_is_explicitly_supported(
                evidence_refs=evidence_refs,
                value=proposal.value,
                unit=proposal.unit,
                grounding_by_evidence=grounding_by_evidence,
                repositories=repositories,
                store=store,
            ):
                _reject(rejected, proposal.result_key, "reported_result", "RESULT_NOT_PRESENT_IN_SOURCE")
                continue
            method_refs = tuple(
                sorted(method_by_key[key].fact_id for key in proposal.method_fact_keys if key in method_by_key)
            )
            model_refs = tuple(
                sorted(model_by_key[key].fact_id for key in proposal.model_fact_keys if key in model_by_key)
            )
            if len(method_refs) != len(proposal.method_fact_keys) or len(model_refs) != len(
                proposal.model_fact_keys
            ):
                _reject(rejected, proposal.result_key, "reported_result", "RESULT_CONTEXT_FACT_REJECTED")
                continue
            context_identity = {
                "system_context": proposal.system_context,
                "method_context": proposal.method_context,
                "method_fact_refs": method_refs,
                "model_fact_refs": model_refs,
            }
            result_context = ResultContext(
                context_id=f"result-context-{content_hash(context_identity)[:24]}",
                **context_identity,
            )
            result_identity = {
                "quantity": proposal.quantity,
                "value": proposal.value,
                "unit": proposal.unit,
                "system_context": proposal.system_context,
                "method_context": proposal.method_context,
                "result_context": result_context,
                "evidence_refs": evidence_refs,
                "result_status": proposal.result_status,
            }
            result = ReportedResult(
                result_id=f"result-{content_hash(result_identity)[:24]}",
                **result_identity,
            )
            issues = result_context_record_issues(
                result,
                {fact.fact_id: fact for fact in method_by_key.values()},
                {fact.fact_id: fact for fact in model_by_key.values()},
                path=f"result:{proposal.result_key}",
                require_context=True,
            )
            if issues:
                _reject(rejected, proposal.result_key, "reported_result", issues[0].code)
                continue
            _put_reuse(repositories.reported_results, result.result_id, result)
            ensure_machine_curation(repositories, "reported_result", result.result_id)
            result_by_key[proposal.result_key] = result

        endpoint_maps = {
            "source_claim": claims_by_key,
            "method_fact": method_by_key,
            "model_fact": model_by_key,
            "reported_result": result_by_key,
        }
        relations: list[KnowledgeRelation] = []
        for proposal in proposal_set.relation_proposals:
            subject = endpoint_maps[proposal.subject_type.value].get(proposal.subject_key)
            object_record = endpoint_maps[proposal.object_type.value].get(proposal.object_key)
            if subject is None or object_record is None:
                _reject(rejected, proposal.relation_key, "relation", "RELATION_ENDPOINT_REJECTED")
                continue
            subject_id = getattr(subject, "claim_id", None) or getattr(subject, "fact_id", None) or getattr(
                subject, "result_id"
            )
            object_id = getattr(object_record, "claim_id", None) or getattr(
                object_record, "fact_id", None
            ) or getattr(object_record, "result_id")
            evidence_refs = tuple(
                sorted(set(subject.evidence_refs) | set(object_record.evidence_refs))
            )
            identity = {
                "subject_type": proposal.subject_type.value,
                "subject_id": subject_id,
                "predicate": proposal.predicate,
                "object_type": proposal.object_type.value,
                "object_id": object_id,
                "domain": repositories.literature_documents.get(
                    compilation_input.literature_id
                ).domain,
                "evidence_refs": evidence_refs,
                "rationale": proposal.rationale,
            }
            relation_id = f"knowledge-relation-{content_hash(identity)[:24]}"
            payload = {"relation_id": relation_id, **identity}
            relation = KnowledgeRelation(**payload, content_hash=content_hash(payload))
            _put_reuse(repositories.relations, relation.relation_id, relation)
            ensure_machine_curation(repositories, "knowledge_relation", relation.relation_id)
            relations.append(relation)

        groundings = tuple(sorted(grounding_by_quote.values(), key=lambda item: item.grounding_id))
        compilation_identity = {
            "compilation_input_id": compilation_input.compilation_input_id,
            "compilation_input_hash": compilation_input.content_hash,
            "provider_id": proposal_set.provider_id,
            "provider_version": proposal_set.provider_version,
            "provider_config_hash": proposal_set.provider_config_hash,
            "proposal_set_id": proposal_set.proposal_set_id,
            "proposal_set_hash": proposal_set.content_hash,
            "representation_selection_id": compilation_input.representation_selection_id,
            "representation_selection_hash": compilation_input.representation_selection_hash,
            "structure_selection_id": compilation_input.structure_selection_id,
            "structure_selection_hash": compilation_input.structure_selection_hash,
            "source_quote_ids": tuple(sorted(quote.quote_id for quote in quotes_by_key.values())),
            "source_claim_ids": tuple(sorted(claim.claim_id for claim in claims_by_key.values())),
            "method_fact_ids": tuple(sorted(fact.fact_id for fact in method_by_key.values())),
            "model_fact_ids": tuple(sorted(fact.fact_id for fact in model_by_key.values())),
            "reported_result_ids": tuple(sorted(result.result_id for result in result_by_key.values())),
            "knowledge_relation_ids": tuple(sorted(relation.relation_id for relation in relations)),
            "grounding_hashes": {item.grounding_id: item.content_hash for item in groundings},
            "rejected_proposals": tuple(rejected),
            "compiler_id": COMPILER_ID,
            "compiler_version": COMPILER_VERSION,
        }
        compilation_id = f"literature-knowledge-compilation-{content_hash(compilation_identity)[:24]}"
        payload = {"compilation_id": compilation_id, **compilation_identity}
        record = LiteratureKnowledgeCompilationRecord(**payload, content_hash=content_hash(payload))
        _put_reuse(
            LiteratureKnowledgeCompilationRepository(repositories.root),
            record.compilation_id,
            record,
        )
        return LiteratureKnowledgeCompilationOutcome(
            record=record,
            source_quotes=tuple(sorted(quotes_by_key.values(), key=lambda item: item.quote_id)),
            source_claims=tuple(sorted(claims_by_key.values(), key=lambda item: item.claim_id)),
            method_facts=tuple(sorted(method_by_key.values(), key=lambda item: item.fact_id)),
            model_facts=tuple(sorted(model_by_key.values(), key=lambda item: item.fact_id)),
            reported_results=tuple(sorted(result_by_key.values(), key=lambda item: item.result_id)),
            relations=tuple(sorted(relations, key=lambda item: item.relation_id)),
            groundings=groundings,
        )
