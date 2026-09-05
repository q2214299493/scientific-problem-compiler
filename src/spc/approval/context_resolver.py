from __future__ import annotations

from ..domains import DomainPackLoader
from ..models import (
    ApprovalReviewInput,
    PlanValidationRecord,
    ScientificContextPacket,
    ScientificEvidencePacket,
    ScientificPlanningInput,
    ScientificQuestionPlan,
)
from ..planning.context_resolver import PlanningContextError, PlanningContextResolver
from ..repositories import KnowledgeRepositories
from ..serialization import content_hash, to_primitive
from ..validators import (
    EvidenceSpanRepository,
    validate_plan_validation_record,
    validate_question_plan,
)

APPROVAL_CONTEXT_RESOLVER_VERSION = "approval-context-resolver-1.0.0"


class ApprovalContextError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class ApprovalContextResolver:
    def __init__(self, domain_loader: DomainPackLoader | None = None) -> None:
        self.domain_loader = domain_loader or DomainPackLoader()

    def resolve(
        self,
        context: ScientificContextPacket,
        evidence_packet: ScientificEvidencePacket,
        planning_input: ScientificPlanningInput,
        candidate_plan: ScientificQuestionPlan,
        validation_record: PlanValidationRecord,
        knowledge: KnowledgeRepositories,
        evidence_repository: EvidenceSpanRepository,
    ) -> ApprovalReviewInput:
        try:
            current_planning_input = PlanningContextResolver(self.domain_loader).resolve(
                context, evidence_packet, knowledge, evidence_repository
            )
        except PlanningContextError as error:
            raise ApprovalContextError(error.code, str(error)) from error
        if current_planning_input != planning_input:
            raise ApprovalContextError(
                "STALE_PLANNING_INPUT",
                "ScientificPlanningInput does not match the current trusted context",
            )
        if (
            candidate_plan.domain != planning_input.domain
            or candidate_plan.domain_pack_version != planning_input.domain_pack_version
        ):
            raise ApprovalContextError(
                "CANDIDATE_DOMAIN_MISMATCH",
                "candidate domain binding does not match ScientificPlanningInput",
            )
        required_provenance = {
            planning_input.context_id,
            planning_input.evidence_packet_id,
            planning_input.planning_input_id,
        }
        if not required_provenance.issubset(set(candidate_plan.source_query_manifest)):
            raise ApprovalContextError(
                "CANDIDATE_PROVENANCE_MISMATCH",
                "candidate is not provenance-bound to the supplied planning artifacts",
            )

        candidate_hash = content_hash(candidate_plan)
        if validation_record.plan_content_hash != candidate_hash:
            raise ApprovalContextError(
                "STALE_CANDIDATE_HASH",
                "PlanValidationRecord does not bind the current candidate content",
            )
        candidate_identity = {
            field_name: getattr(candidate_plan, field_name)
            for field_name in type(candidate_plan).model_fields
            if field_name != "plan_id"
        }
        expected_candidate_id = f"plan-{content_hash(candidate_identity)[:24]}"
        if candidate_plan.plan_id != expected_candidate_id:
            raise ApprovalContextError(
                "TAMPERED_CANDIDATE_ID",
                "candidate plan_id is not bound to its current scientific content",
            )
        candidate_claim_ids = {
            entry.removeprefix("claim:")
            for entry in candidate_plan.source_query_manifest
            if entry.startswith("claim:")
        }
        if not candidate_claim_ids.issubset(set(planning_input.allowed_claim_ids)):
            raise ApprovalContextError(
                "FABRICATED_CANDIDATE_CLAIM_REF",
                "candidate provenance references a non-allowlisted SourceClaim",
            )
        current_report = validate_question_plan(
            candidate_plan,
            planning_input.scientific_capabilities,
            evidence_repository,
        )
        validation_report = validate_plan_validation_record(
            candidate_plan, validation_record, current_report
        )
        binding_issues = tuple(
            issue
            for issue in validation_report.issues
            if issue.code != "PLAN_VALIDATION_FAILED"
        )
        if binding_issues:
            raise ApprovalContextError(
                binding_issues[0].code,
                "; ".join(issue.message for issue in binding_issues),
            )
        current_validator_version = PlanValidationRecord.model_fields[
            "validator_version"
        ].default
        if validation_record.validator_version != current_validator_version:
            raise ApprovalContextError(
                "STALE_VALIDATOR_VERSION",
                "PlanValidationRecord was produced by a different validator version",
            )

        provenance_manifest = {
            "resolver_version": APPROVAL_CONTEXT_RESOLVER_VERSION,
            "retrieval_id": context.retrieval_manifest.retrieval_id,
            "knowledge_snapshot_id": context.knowledge_snapshot.snapshot_id,
            "context_id": context.context_id,
            "context_hash": context.content_hash,
            "evidence_packet_id": evidence_packet.packet_id,
            "evidence_packet_hash": evidence_packet.content_hash,
            "planning_input_id": planning_input.planning_input_id,
            "planning_input_hash": planning_input.content_hash,
            "candidate_plan_id": candidate_plan.plan_id,
            "candidate_plan_hash": candidate_hash,
            "plan_validation_id": validation_record.validation_id,
            "plan_validation_hash": content_hash(validation_record),
        }
        identity = to_primitive({
            "original_request": context.original_request,
            "domain": planning_input.domain,
            "domain_pack_version": planning_input.domain_pack_version,
            "context_id": context.context_id,
            "context_hash": context.content_hash,
            "evidence_packet_id": evidence_packet.packet_id,
            "evidence_packet_hash": evidence_packet.content_hash,
            "planning_input_id": planning_input.planning_input_id,
            "planning_input_hash": planning_input.content_hash,
            "candidate_plan": candidate_plan,
            "candidate_plan_hash": candidate_hash,
            "plan_validation_record": validation_record,
            "plan_validation_hash": content_hash(validation_record),
            "source_quotes": evidence_packet.source_quotes,
            "source_claims": evidence_packet.source_claims,
            "reported_results": evidence_packet.reported_results,
            "method_facts": evidence_packet.method_facts,
            "model_facts": evidence_packet.model_facts,
            "evidence_assessments": evidence_packet.evidence_assessments,
            "conflict_sets": evidence_packet.conflict_sets,
            "comparison_constraints": evidence_packet.comparison_constraints,
            "evidence_gaps": evidence_packet.evidence_gaps,
            "expert_cases": planning_input.expert_cases,
            "workflow_patterns": planning_input.workflow_patterns,
            "scientific_capabilities": planning_input.scientific_capabilities,
            "allowed_evidence_ids": planning_input.allowed_evidence_ids,
            "allowed_claim_ids": planning_input.allowed_claim_ids,
            "allowed_task_ids": tuple(task.task_id for task in candidate_plan.tasks),
            "allowed_capability_ids": planning_input.allowed_capability_ids,
            "provenance_manifest": provenance_manifest,
        })
        review_input_id = f"approval-review-input-{content_hash(identity)[:24]}"
        payload = {"review_input_id": review_input_id, **identity}
        return ApprovalReviewInput(**payload, content_hash=content_hash(payload))
