from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import BaseModel

from ..interpretation.validators import (
    result_context_record_issues,
    source_claim_binding_issues,
    source_quote_record_issues,
)
from ..models import (
    CurationStatus,
    EvidenceSpan,
    ExpertAttributionRecord,
    ExpertCase,
    ExpertOpinion,
    ExpertProfile,
    KnowledgeCurationRecord,
    KnowledgeRelation,
    LiteratureDocument,
    LiteratureWorkflowPattern,
    MethodFact,
    ModelFact,
    ReportedResult,
    ScientificCapability,
    SourceClaim,
    SourceDocument,
    SourceQuote,
)
from ..serialization import content_hash

if TYPE_CHECKING:
    from ..repositories import KnowledgeRepositories, SourceEvidenceStore


REPOSITORY_NODE_SPECS = (
    ("literature_document", "literature_documents", "literature_id"),
    ("expert_profile", "expert_profiles", "expert_id"),
    ("expert_attribution", "expert_attributions", "attribution_id"),
    ("expert_opinion", "expert_opinions", "opinion_id"),
    ("expert_case", "expert_cases", "case_id"),
    ("workflow_pattern", "workflow_patterns", "pattern_id"),
    ("scientific_capability", "capabilities", "capability_id"),
    ("source_quote", "source_quotes", "quote_id"),
    ("source_claim", "source_claims", "claim_id"),
    ("reported_result", "reported_results", "result_id"),
    ("method_fact", "method_facts", "fact_id"),
    ("model_fact", "model_facts", "fact_id"),
    ("knowledge_relation", "relations", "relation_id"),
)

CURATION_REQUIRED_TYPES = frozenset(
    {"literature_document", "expert_opinion", "knowledge_relation"}
)


class TrustedKnowledgeError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class TrustedKnowledgeSelection:
    literature_documents: tuple[LiteratureDocument, ...]
    expert_profiles: tuple[ExpertProfile, ...]
    expert_attributions: tuple[ExpertAttributionRecord, ...]
    expert_opinions: tuple[ExpertOpinion, ...]
    relations: tuple[KnowledgeRelation, ...]
    curations: tuple[KnowledgeCurationRecord, ...]
    current_curations: Mapping[tuple[str, str], KnowledgeCurationRecord]
    trusted_records: Mapping[tuple[str, str], BaseModel]


class TrustedKnowledgeValidator:
    def __init__(
        self,
        repositories: KnowledgeRepositories,
        evidence_store: SourceEvidenceStore | None,
    ) -> None:
        self.repositories = repositories
        self.evidence_store = evidence_store
        self.records = self.record_index(repositories)
        self._reachable: dict[tuple[str, str], BaseModel] = {}

    @staticmethod
    def record_index(
        repositories: KnowledgeRepositories,
    ) -> dict[tuple[str, str], BaseModel]:
        records: dict[tuple[str, str], BaseModel] = {}
        for record_type, repository_name, identity_field in REPOSITORY_NODE_SPECS:
            repository: Any = getattr(repositories, repository_name)
            for record in repository.list():
                key = (record_type, getattr(record, identity_field))
                if key in records:
                    raise TrustedKnowledgeError(
                        "DUPLICATE_KNOWLEDGE_RECORD",
                        f"duplicate record {key[0]}:{key[1]}",
                    )
                records[key] = record
        return records

    def resolve_current_curations(
        self,
    ) -> Mapping[tuple[str, str], KnowledgeCurationRecord]:
        curations = self.repositories.curations.list()
        by_id = {item.curation_id: item for item in curations}
        groups: dict[tuple[str, str], list[KnowledgeCurationRecord]] = {}
        successors: dict[str, str] = {}
        for curation in curations:
            key = (curation.target_type, curation.target_id)
            target = self.records.get(key)
            if target is None:
                raise TrustedKnowledgeError(
                    "UNKNOWN_CURATION_TARGET",
                    f"curation target does not exist: {key[0]}:{key[1]}",
                )
            if curation.target_hash != self._record_hash(target):
                raise TrustedKnowledgeError(
                    "CURATION_TARGET_HASH_MISMATCH",
                    f"curation target hash is stale: {key[0]}:{key[1]}",
                )
            groups.setdefault(key, []).append(curation)
            predecessor_id = curation.supersedes_curation_id
            if predecessor_id is None:
                continue
            predecessor = by_id.get(predecessor_id)
            if predecessor is None:
                raise TrustedKnowledgeError(
                    "MISSING_CURATION_PREDECESSOR",
                    f"missing superseded curation {predecessor_id}",
                )
            if (predecessor.target_type, predecessor.target_id) != key:
                raise TrustedKnowledgeError(
                    "CURATION_TARGET_MISMATCH",
                    "a curation transition must retain the same target",
                )
            if predecessor_id in successors:
                raise TrustedKnowledgeError(
                    "AMBIGUOUS_CURATION_CHAIN",
                    f"curation {predecessor_id} has multiple successors",
                )
            successors[predecessor_id] = curation.curation_id

        current: dict[tuple[str, str], KnowledgeCurationRecord] = {}
        for key, group in groups.items():
            heads = [item for item in group if item.curation_id not in successors]
            if len(heads) != 1:
                raise TrustedKnowledgeError(
                    "AMBIGUOUS_CURATION_CHAIN",
                    f"expected one current curation for {key[0]}:{key[1]}",
                )
            self._assert_acyclic(heads[0], by_id)
            current[key] = heads[0]
        return MappingProxyType(current)

    def validate(self) -> TrustedKnowledgeSelection:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires a SourceEvidenceStore",
            )
        self._reachable = {}
        current = self.resolve_current_curations()
        accepted = {
            key: curation
            for key, curation in current.items()
            if curation.status == CurationStatus.ACCEPTED
        }
        for curation in accepted.values():
            self._verify_evidence_refs(curation.evidence_refs)
            self._reachable[("knowledge_curation", curation.curation_id)] = curation

        roots = set(accepted)
        roots.update(
            ("expert_case", item.case_id)
            for item in self.repositories.expert_cases.list()
        )
        roots.update(
            ("workflow_pattern", item.pattern_id)
            for item in self.repositories.workflow_patterns.list()
        )
        roots.update(
            ("scientific_capability", item.capability_id)
            for item in self.repositories.capabilities.list()
        )
        for key in sorted(roots):
            self._validate_record(key, current, set())

        documents = self._records_of_type("literature_document", LiteratureDocument)
        profiles = self._records_of_type("expert_profile", ExpertProfile)
        attributions = self._records_of_type(
            "expert_attribution", ExpertAttributionRecord
        )
        opinions = self._records_of_type("expert_opinion", ExpertOpinion)
        relations = self._records_of_type("knowledge_relation", KnowledgeRelation)
        accepted_curations = tuple(
            sorted(accepted.values(), key=lambda item: item.curation_id)
        )
        return TrustedKnowledgeSelection(
            literature_documents=documents,
            expert_profiles=profiles,
            expert_attributions=attributions,
            expert_opinions=opinions,
            relations=relations,
            curations=accepted_curations,
            current_curations=current,
            trusted_records=MappingProxyType(dict(sorted(self._reachable.items()))),
        )

    def _validate_record(
        self,
        key: tuple[str, str],
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> BaseModel:
        if key in self._reachable:
            return self._reachable[key]
        if key in visiting:
            return self._require_record(*key)
        record = self._require_record(*key)
        visiting.add(key)
        try:
            if isinstance(record, LiteratureDocument):
                self._validate_literature(record)
            elif isinstance(record, ExpertProfile):
                pass
            elif isinstance(record, ExpertAttributionRecord):
                self._validate_attribution(record)
            elif isinstance(record, ExpertOpinion):
                self._validate_opinion(record, current, visiting)
            elif isinstance(record, KnowledgeRelation):
                self._validate_relation(record, current, visiting)
            elif isinstance(record, SourceQuote):
                self._validate_source_quote(record)
            elif isinstance(record, SourceClaim):
                self._validate_source_claim(record, current, visiting)
            elif isinstance(record, MethodFact | ModelFact):
                self._verify_evidence_refs(record.evidence_refs)
            elif isinstance(record, ReportedResult):
                self._validate_reported_result(record, current, visiting)
            elif isinstance(record, ExpertCase | LiteratureWorkflowPattern):
                self._verify_evidence_refs(record.evidence_refs)
            elif isinstance(record, ScientificCapability):
                pass
            else:
                raise TrustedKnowledgeError(
                    "UNSUPPORTED_TRUSTED_RECORD",
                    f"no provenance validator for {key[0]}:{key[1]}",
                )
            self._reachable[key] = record
            return record
        finally:
            visiting.remove(key)

    def _validate_literature(self, document: LiteratureDocument) -> None:
        source = self._verify_source(document.source_id, document.source_version)
        self._reachable[("source_document", f"{source.source_id}@{source.version}")] = source

    def _validate_attribution(self, attribution: ExpertAttributionRecord) -> None:
        source = self._verify_source(attribution.source_id, attribution.source_version)
        for evidence_id in attribution.evidence_refs:
            evidence, evidence_source = self._verify_evidence(evidence_id)
            if (evidence.source_id, evidence.source_version) != (
                attribution.source_id,
                attribution.source_version,
            ):
                raise TrustedKnowledgeError(
                    "EXPERT_ATTRIBUTION_SOURCE_MISMATCH",
                    f"attribution evidence comes from another source: {evidence_id}",
                )
            self._remember_evidence(evidence, evidence_source)
        self._reachable[("source_document", f"{source.source_id}@{source.version}")] = source

    def _validate_opinion(
        self,
        opinion: ExpertOpinion,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        self._validate_record(("expert_profile", opinion.expert_id), current, visiting)
        if not opinion.attribution_refs:
            raise TrustedKnowledgeError(
                "MISSING_EXPERT_ATTRIBUTION",
                f"accepted ExpertOpinion has no attribution: {opinion.opinion_id}",
            )
        attributed_evidence: set[str] = set()
        for attribution_id in opinion.attribution_refs:
            attribution = self._validate_record(
                ("expert_attribution", attribution_id), current, visiting
            )
            if not isinstance(attribution, ExpertAttributionRecord):
                raise TrustedKnowledgeError(
                    "INVALID_EXPERT_ATTRIBUTION",
                    f"record is not an ExpertAttributionRecord: {attribution_id}",
                )
            if attribution.expert_id != opinion.expert_id:
                raise TrustedKnowledgeError(
                    "EXPERT_ATTRIBUTION_OWNER_MISMATCH",
                    f"attribution belongs to another expert: {attribution_id}",
                )
            attributed_evidence.update(attribution.evidence_refs)
        if not set(opinion.evidence_refs).issubset(attributed_evidence):
            raise TrustedKnowledgeError(
                "EXPERT_OPINION_EVIDENCE_NOT_ATTRIBUTED",
                f"opinion evidence is not covered by its attributions: {opinion.opinion_id}",
            )
        self._verify_evidence_refs(opinion.evidence_refs)
        for claim_id in opinion.related_claim_refs:
            self._validate_record(("source_claim", claim_id), current, visiting)
        for workflow_id in opinion.related_workflow_refs:
            self._validate_record(("workflow_pattern", workflow_id), current, visiting)
        if opinion.supersedes is not None:
            self._validate_record(("expert_opinion", opinion.supersedes), current, visiting)

    def _validate_relation(
        self,
        relation: KnowledgeRelation,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        self._verify_evidence_refs(relation.evidence_refs)
        for key in (
            (relation.subject_type, relation.subject_id),
            (relation.object_type, relation.object_id),
        ):
            self._require_record(*key)
            curation = current.get(key)
            if (key[0] in CURATION_REQUIRED_TYPES or curation is not None) and (
                curation is None or curation.status != CurationStatus.ACCEPTED
            ):
                raise TrustedKnowledgeError(
                    "UNTRUSTED_RELATION_ENDPOINT",
                    f"accepted relation points to untrusted target {key[0]}:{key[1]}",
                )
            self._validate_record(key, current, visiting)

    def _validate_source_quote(self, quote: SourceQuote) -> None:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires a SourceEvidenceStore",
            )
        issues, source = source_quote_record_issues(
            quote,
            self.evidence_store,
            path=f"source_quote:{quote.quote_id}",
        )
        self._raise_issues(issues)
        evidence, evidence_source = self._verify_evidence(quote.evidence_ref)
        self._remember_evidence(evidence, evidence_source)
        if source is not None:
            self._reachable[("source_document", f"{source.source_id}@{source.version}")] = source

    def _validate_source_claim(
        self,
        claim: SourceClaim,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        quotes_by_id = {
            quote_id: quote
            for quote_id in claim.source_quote_refs
            if isinstance(
                (quote := self.records.get(("source_quote", quote_id))), SourceQuote
            )
        }
        self._raise_issues(
            source_claim_binding_issues(
                claim,
                quotes_by_id,
                path=f"source_claim:{claim.claim_id}",
            )
        )
        self._verify_evidence_refs(claim.evidence_refs)
        for quote_id in claim.source_quote_refs:
            self._validate_record(("source_quote", quote_id), current, visiting)

    def _validate_reported_result(
        self,
        result: ReportedResult,
        current: Mapping[tuple[str, str], KnowledgeCurationRecord],
        visiting: set[tuple[str, str]],
    ) -> None:
        self._verify_evidence_refs(result.evidence_refs)
        result_context = result.result_context
        if result_context is None:
            return
        method_facts = {
            fact_id: fact
            for fact_id in result_context.method_fact_refs
            if isinstance((fact := self.records.get(("method_fact", fact_id))), MethodFact)
        }
        model_facts = {
            fact_id: fact
            for fact_id in result_context.model_fact_refs
            if isinstance((fact := self.records.get(("model_fact", fact_id))), ModelFact)
        }
        self._raise_issues(
            result_context_record_issues(
                result,
                method_facts,
                model_facts,
                path=f"reported_result:{result.result_id}.result_context",
                require_context=False,
            )
        )
        for fact_id in result_context.method_fact_refs:
            self._validate_record(("method_fact", fact_id), current, visiting)
        for fact_id in result_context.model_fact_refs:
            self._validate_record(("model_fact", fact_id), current, visiting)

    def _verify_evidence_refs(self, evidence_refs: tuple[str, ...]) -> None:
        for evidence_id in evidence_refs:
            evidence, source = self._verify_evidence(evidence_id)
            self._remember_evidence(evidence, source)

    def _verify_evidence(self, evidence_id: str) -> tuple[EvidenceSpan, SourceDocument]:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires a SourceEvidenceStore",
            )
        try:
            evidence = self.evidence_store.get(evidence_id)
            source = self.evidence_store.verify_evidence_integrity(evidence)
        except Exception as error:
            raise TrustedKnowledgeError(
                "INVALID_KNOWLEDGE_EVIDENCE",
                f"evidence is unavailable or invalid: {evidence_id}",
            ) from error
        return evidence, source

    def _verify_source(self, source_id: str, source_version: str) -> SourceDocument:
        if self.evidence_store is None:
            raise TrustedKnowledgeError(
                "EVIDENCE_STORE_REQUIRED",
                "trusted knowledge validation requires a SourceEvidenceStore",
            )
        try:
            source = self.evidence_store.source_records.get(
                f"{source_id}--{source_version}"
            )
            if (source.source_id, source.version) != (source_id, source_version):
                raise ValueError("source identity mismatch")
            self.evidence_store.verify_source_integrity(source)
        except Exception as error:
            raise TrustedKnowledgeError(
                "INVALID_LITERATURE_SOURCE",
                f"source is unavailable or invalid: {source_id}@{source_version}",
            ) from error
        return source

    def _remember_evidence(
        self, evidence: EvidenceSpan, source: SourceDocument
    ) -> None:
        self._reachable[("evidence_span", evidence.evidence_id)] = evidence
        self._reachable[("source_document", f"{source.source_id}@{source.version}")] = source

    def _require_record(self, record_type: str, record_id: str) -> BaseModel:
        record = self.records.get((record_type, record_id))
        if record is None:
            raise TrustedKnowledgeError(
                "UNKNOWN_KNOWLEDGE_ENDPOINT",
                f"record does not exist: {record_type}:{record_id}",
            )
        return record

    def _records_of_type(self, record_type: str, model_type: type[Any]) -> tuple[Any, ...]:
        return tuple(
            record
            for (kind, _), record in sorted(self._reachable.items())
            if kind == record_type and isinstance(record, model_type)
        )

    @staticmethod
    def _raise_issues(issues: list[Any]) -> None:
        if issues:
            issue = issues[0]
            raise TrustedKnowledgeError(issue.code, issue.message)

    @staticmethod
    def _record_hash(record: BaseModel) -> str:
        return getattr(record, "content_hash", None) or content_hash(record)

    @staticmethod
    def _assert_acyclic(
        head: KnowledgeCurationRecord,
        by_id: Mapping[str, KnowledgeCurationRecord],
    ) -> None:
        seen: set[str] = set()
        current: KnowledgeCurationRecord | None = head
        while current is not None:
            if current.curation_id in seen:
                raise TrustedKnowledgeError(
                    "CURATION_CHAIN_CYCLE",
                    f"curation chain contains a cycle at {current.curation_id}",
                )
            seen.add(current.curation_id)
            predecessor_id = current.supersedes_curation_id
            current = by_id.get(predecessor_id) if predecessor_id is not None else None
