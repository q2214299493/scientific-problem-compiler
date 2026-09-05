from __future__ import annotations

from ..models import ScientificContextPacket, ScientificEvidencePacket, SourceQuote
from ..serialization import content_hash
from ..validators import EvidenceSpanRepository, ValidationReport
from .provider import InterpretationProvider
from .validators import validate_evidence_packet_integrity


class EvidencePacketIntegrityError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        codes = ", ".join(issue.code for issue in report.issues)
        super().__init__(f"ScientificEvidencePacket validation failed: {codes}")


class ScientificEvidencePacketBuilder:
    def __init__(self, provider: InterpretationProvider) -> None:
        self.provider = provider

    def build(
        self,
        context: ScientificContextPacket,
        evidence_repository: EvidenceSpanRepository,
    ) -> ScientificEvidencePacket:
        proposal = self.provider.interpret(context)
        if proposal.context_id != context.context_id or proposal.context_hash != context.content_hash:
            raise ValueError("InterpretationProposal is not bound to the supplied context")
        proposal_payload = proposal.model_dump(mode="json", exclude={"proposal_id"})
        if proposal.proposal_id != f"interpretation-{content_hash(proposal_payload)[:24]}":
            raise ValueError("InterpretationProposal proposal_id is not content-bound")

        source_quotes: list[SourceQuote] = []
        source_document_hashes: dict[str, str] = {}
        for evidence_id in sorted({hit.record_id for hit in context.evidence_hits}):
            evidence = evidence_repository.get(evidence_id)
            source = evidence_repository.verify_evidence_integrity(evidence)
            source_document_hashes[f"{source.source_id}@{source.version}"] = content_hash(source)
            source_quotes.append(
                SourceQuote(
                    quote_id=f"quote-{content_hash({'evidence_ref': evidence_id})[:24]}",
                    text=evidence.text,
                    evidence_ref=evidence_id,
                    source_id=source.source_id,
                    source_version=source.version,
                    source_role=source.source_role,
                    source_type=source.source_type,
                )
            )
        quotes_by_id = {quote.quote_id: quote for quote in source_quotes}
        source_claims = []
        for claim in proposal.source_claims:
            explicit_roles = {
                quotes_by_id[quote_id].source_role
                for quote_id in claim.source_quote_refs
                if quote_id in quotes_by_id
                and quotes_by_id[quote_id].source_role != "unspecified"
            }
            source_claims.append(
                claim.model_copy(update={"source_role": next(iter(explicit_roles))})
                if len(explicit_roles) == 1
                else claim
            )

        manifest = {
            "context_id": context.context_id,
            "context_hash": context.content_hash,
            "retrieval_id": context.retrieval_manifest.retrieval_id,
            "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
            "provider_id": proposal.provider_id,
            "provider_version": proposal.provider_version,
            "proposal_id": proposal.proposal_id,
            "evidence_record_ids": tuple(hit.record_id for hit in context.evidence_hits),
            "source_document_hashes": source_document_hashes,
        }
        identity = {
            "context_id": context.context_id,
            "context_hash": context.content_hash,
            "source_quotes": tuple(source_quotes),
            "source_claims": tuple(source_claims),
            "reported_results": proposal.reported_results,
            "method_facts": proposal.method_facts,
            "model_facts": proposal.model_facts,
            "evidence_assessments": proposal.evidence_assessments,
            "conflict_sets": proposal.conflict_sets,
            "comparison_constraints": proposal.comparison_constraints,
            "evidence_gaps": proposal.evidence_gaps,
            "unknowns": proposal.unknowns,
            "assumption_candidates": proposal.assumption_candidates,
            "capability_candidates": proposal.capability_candidates,
            "provenance_manifest": manifest,
        }
        packet_id = f"evidence-packet-{content_hash(identity)[:24]}"
        packet_payload = {"packet_id": packet_id, **identity}
        packet = ScientificEvidencePacket(
            **packet_payload,
            content_hash=content_hash(packet_payload),
        )
        report = validate_evidence_packet_integrity(packet, context, evidence_repository)
        if not report.valid:
            raise EvidencePacketIntegrityError(report)
        return packet
