from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from ..downstream import DownstreamImportError, DownstreamImportValidator
from ..models import (
    CurationStatus,
    EvidenceSpan,
    ExecutionProposal,
    FingerprintDifference,
    KnowledgeCurationRecord,
    MethodFact,
    ModelFact,
    ReportedResult,
    ReportedObservation,
    ResultContext,
    ResultContextComparison,
    ResultContextComparisonStatus,
    ResultEvidenceAssessment,
    ResultEvidenceIntakeStatus,
    ResultEvidenceMaterializationReceipt,
    ResultEvidenceSubmission,
    ResultEvidenceType,
    ResultExecutionOutcome,
    ResultIntakeIssue,
    ScientificQuestionPlan,
    SourceRole,
    SourceType,
)
from ..repositories import (
    CompositeEvidenceStore,
    KnowledgeEvidenceStore,
    ProjectEvidenceStore,
)
from ..serialization import content_hash, to_primitive
from ..workflow.repository import ScientificProblemRunRepository
from .artifacts import ResultArtifacts
from .parent_authority import ApprovedParent, ParentAuthority
from .submission import ResultFeedbackError, build_content_bound


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) or hasattr(value, "items"):
        flattened: dict[str, Any] = {}
        for key, item in sorted(dict(value).items()):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(item, path))
        return flattened
    return {prefix: value}


def _normalized_difference_field(field: str) -> str:
    parts = field.split(".", 1)
    if len(parts) == 1:
        return field
    prefix, suffix = parts
    if prefix in {"system", "system_context"}:
        return f"system.{suffix}"
    if prefix in {"method", "method_context"}:
        return f"method.{suffix}"
    return field


def _canonical_value(value: Any) -> str:
    return content_hash(to_primitive(value))


def _same_difference(
    actual: FingerprintDifference,
    declared: FingerprintDifference,
) -> bool:
    actual_field = _normalized_difference_field(actual.field)
    declared_field = _normalized_difference_field(declared.field)
    field_matches = declared_field == actual_field or (
        "." not in declared.field and declared.field == actual_field.split(".", 1)[-1]
    )
    return (
        declared.disclosed_deviation
        and field_matches
        and _canonical_value(actual.left) == _canonical_value(declared.left)
        and _canonical_value(actual.right) == _canonical_value(declared.right)
    )


class ResultEvidenceIntakeService:
    def __init__(self, *, state_dir: Path, knowledge_dir: Path) -> None:
        self.state_dir = state_dir.resolve()
        self.knowledge_dir = knowledge_dir.resolve()
        self.runs = ScientificProblemRunRepository(self.state_dir)
        self.project_evidence = ProjectEvidenceStore(self.state_dir)
        self.composite_evidence = CompositeEvidenceStore(
            KnowledgeEvidenceStore(self.knowledge_dir),
            self.project_evidence,
        )
        self.artifacts = ResultArtifacts(self.runs)
        self.parent_authority = ParentAuthority(self.runs, self.composite_evidence, self.artifacts)

    @staticmethod
    def _comparison(
        submission: ResultEvidenceSubmission,
        plan: ScientificQuestionPlan,
    ) -> ResultContextComparison:
        differences: list[FingerprintDifference] = []
        insufficient = False
        for category, expected, actual in (
            (
                "system",
                _flatten(plan.system_fingerprint.attributes),
                _flatten(submission.system_fingerprint.attributes),
            ),
            (
                "method",
                _flatten(plan.method_fingerprint.attributes),
                _flatten(submission.method_fingerprint.attributes),
            ),
        ):
            for field in sorted(set(expected) | set(actual)):
                expected_value = expected.get(field)
                actual_value = actual.get(field)
                if field not in actual:
                    insufficient = True
                    differences.append(
                        FingerprintDifference(
                            field=f"{category}.{field}",
                            left=expected_value,
                            right=None,
                            disclosed_deviation=False,
                        )
                    )
                elif field not in expected or actual_value != expected_value:
                    differences.append(
                        FingerprintDifference(
                            field=f"{category}.{field}",
                            left=expected_value,
                            right=actual_value,
                            disclosed_deviation=False,
                        )
                    )
        approved = tuple(
            item
            for item in differences
            if any(
                _same_difference(item, declared)
                for declared in submission.declared_deviations
            )
            and any(
                _same_difference(item, allowed)
                for allowed in plan.fingerprint_differences
            )
        )
        if not differences:
            status = ResultContextComparisonStatus.MATCHED
        elif insufficient:
            status = ResultContextComparisonStatus.INSUFFICIENT_CONTEXT
        elif len(approved) == len(differences):
            status = ResultContextComparisonStatus.EXPLICIT_APPROVED_DEVIATION
        elif any(item.field.startswith("system.") for item in differences):
            status = ResultContextComparisonStatus.UNDECLARED_SYSTEM_CHANGE
        else:
            status = ResultContextComparisonStatus.UNDECLARED_METHOD_CHANGE
        payload = {
            "submission_id": submission.submission_id,
            "submission_hash": submission.content_hash,
            "parent_plan_id": plan.plan_id,
            "parent_plan_hash": content_hash(plan),
            "status": status,
            "differences": tuple(differences),
            "approved_deviation_refs": tuple(
                f"fingerprint-difference:{content_hash(item)[:24]}"
                for item in approved
            ),
        }
        return build_content_bound(
            ResultContextComparison,
            "result-context-comparison",
            payload,
        )

    @staticmethod
    def _admissibility_issues(
        submission: ResultEvidenceSubmission,
        parent: ApprovedParent,
    ) -> tuple[ResultIntakeIssue, ...]:
        issues: list[ResultIntakeIssue] = []
        observed = {
            item.observable_key for item in submission.reported_observables
        } | {item.quantity for item in submission.reported_observables}
        missing = tuple(output for output in parent.task.outputs if output not in observed)
        if (
            submission.execution_outcome == ResultExecutionOutcome.COMPLETED
            and missing
        ):
            issues.append(
                ResultIntakeIssue(
                    code="REQUIRED_TASK_OUTPUT_MISSING",
                    message="completed result omits required task outputs: "
                    + ", ".join(missing),
                )
            )
        if submission.result_type in {
            ResultEvidenceType.COMPUTED_RESULT,
            ResultEvidenceType.PARTIAL_RESULT,
            ResultEvidenceType.NULL_OR_NEGATIVE_RESULT,
        }:
            converged = submission.convergence_metadata.get("converged")
            if submission.result_type == ResultEvidenceType.COMPUTED_RESULT and converged is not True:
                issues.append(
                    ResultIntakeIssue(
                        code="CONVERGENCE_METADATA_INSUFFICIENT",
                        message=(
                            "computed_result requires an explicit converged=true declaration; "
                            "this declaration remains subject to curation"
                        ),
                    )
                )
        expected_status = {
            ResultEvidenceType.COMPUTED_RESULT: "computed_reported",
            ResultEvidenceType.EXPERIMENTAL_RESULT: "experimental_reported",
        }.get(submission.result_type)
        if expected_status is not None and any(
            item.result_status.value != expected_status
            for item in submission.reported_observables
        ):
            issues.append(
                ResultIntakeIssue(
                    code="RESULT_ORIGIN_STATUS_MISMATCH",
                    message="reported observable origin does not match result_type",
                )
            )
        return tuple(issues)

    def intake(
        self,
        submission: ResultEvidenceSubmission,
        *,
        artifact_root: Path,
        execution_proposal: ExecutionProposal | None = None,
    ) -> ResultEvidenceAssessment:
        try:
            self.runs.get(submission.parent_run_id)
        except FileNotFoundError as error:
            raise ResultFeedbackError(
                "PARENT_RUN_NOT_FOUND", submission.parent_run_id
            ) from error
        prefix = self.artifacts.submission_prefix(submission.submission_id)
        with self.runs.acquire_run_lock(submission.parent_run_id):
            self.runs.write_immutable_artifact(
                submission.parent_run_id,
                f"{prefix}/submission.yaml",
                "result_evidence_submission",
                submission.submission_id,
                submission,
            )
            content_issues = list(
                self.artifacts.verify_and_archive_artifacts(submission, artifact_root)
            )
            parent: ApprovedParent | None = None
            plan_issues: list[ResultIntakeIssue] = []
            if not content_issues:
                try:
                    parent = self.parent_authority.load_parent(submission)
                    if submission.execution_proposal_id is not None:
                        if execution_proposal is None:
                            raise ResultFeedbackError(
                                "EXECUTION_PROPOSAL_MISSING",
                                "submission names an execution proposal but none was supplied",
                            )
                        if (
                            execution_proposal.proposal_id
                            != submission.execution_proposal_id
                            or content_hash(execution_proposal)
                            != submission.execution_proposal_hash
                            or execution_proposal.source_plan_id != parent.plan.plan_id
                            or execution_proposal.source_plan_hash
                            != content_hash(parent.plan)
                            or execution_proposal.task_id != parent.task.task_id
                            or execution_proposal.scientific_capability_id
                            != parent.task.capability_id
                        ):
                            raise ResultFeedbackError(
                                "EXECUTION_PROPOSAL_BINDING_INVALID",
                                "execution proposal does not bind the exact plan/task/capability",
                            )
                    if submission.export_id is not None:
                        if parent.run.export_path is None:
                            raise ResultFeedbackError(
                                "EXPORT_BINDING_MISSING",
                                "submission names an export but the parent run has none",
                            )
                        package = DownstreamImportValidator().import_package(
                            Path(parent.run.export_path)
                        )
                        manifest = package.export_manifest
                        if (
                            manifest.export_id != submission.export_id
                            or content_hash(manifest)
                            != submission.export_manifest_hash
                            or package.source_plan.plan_id != parent.plan.plan_id
                            or content_hash(package.source_plan)
                            != content_hash(parent.plan)
                            or not any(
                                item.scientific_capability_id
                                == parent.task.capability_id
                                for item in package.capability_bindings
                            )
                        ):
                            raise ResultFeedbackError(
                                "EXPORT_BINDING_INVALID",
                                "submission export binding does not match the parent export",
                            )
                except (
                    DownstreamImportError,
                    FileNotFoundError,
                    OSError,
                    ValueError,
                ) as error:
                    plan_issues.append(
                        ResultIntakeIssue(
                            code=getattr(error, "code", "PARENT_PLAN_BINDING_INVALID"),
                            message=str(error),
                        )
                    )

            comparison = self._comparison(submission, parent.plan) if parent else None
            context_compatible = comparison is not None and comparison.status in {
                ResultContextComparisonStatus.MATCHED,
                ResultContextComparisonStatus.EXPLICIT_APPROVED_DEVIATION,
            }
            context_issues = (
                ()
                if context_compatible or comparison is None
                else (
                    ResultIntakeIssue(
                        code=comparison.status.value,
                        message="result system/method context is not compatible with the parent task",
                    ),
                )
            )
            admissibility_issues = (
                self._admissibility_issues(submission, parent)
                if parent is not None and context_compatible
                else ()
            )
            issues = tuple(
                (*content_issues, *plan_issues, *context_issues, *admissibility_issues)
            )
            content_bound = not content_issues
            plan_bound = parent is not None and not plan_issues
            scientifically_admissible = (
                plan_bound and context_compatible and not admissibility_issues
            )
            if content_issues:
                status = ResultEvidenceIntakeStatus.BLOCKED_CONTENT_BINDING
            elif plan_issues:
                status = ResultEvidenceIntakeStatus.BLOCKED_PLAN_BINDING
            elif context_issues:
                status = ResultEvidenceIntakeStatus.BLOCKED_CONTEXT_COMPATIBILITY
            elif admissibility_issues:
                status = ResultEvidenceIntakeStatus.BLOCKED_SCIENTIFIC_ADMISSIBILITY
            else:
                status = ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
            payload = {
                "submission_id": submission.submission_id,
                "submission_hash": submission.content_hash,
                "parent_run_id": submission.parent_run_id,
                "parent_plan_id": submission.parent_plan_id,
                "parent_plan_hash": submission.parent_plan_hash,
                "task_id": submission.task_id,
                "capability_id": submission.capability_id,
                "parent_authority_origin": parent.origin if parent else None,
                "parent_authority_ref": parent.origin_ref if parent else None,
                "content_bound": content_bound,
                "plan_bound": plan_bound,
                "context_compatible": context_compatible,
                "scientifically_admissible": scientifically_admissible,
                "context_comparison": comparison,
                "issues": issues,
                "status": status,
            }
            assessment = build_content_bound(
                ResultEvidenceAssessment, "result-assessment", payload
            )
            self.runs.write_immutable_artifact(
                submission.parent_run_id,
                f"{prefix}/assessment.yaml",
                "result_evidence_assessment",
                assessment.assessment_id,
                assessment,
            )
            if comparison is not None:
                self.runs.write_immutable_artifact(
                    submission.parent_run_id,
                    f"{prefix}/context-comparison.yaml",
                    "result_context_comparison",
                    comparison.comparison_id,
                    comparison,
                )
            return assessment

    @staticmethod
    def _curation_record(
        submission: ResultEvidenceSubmission,
        *,
        status: CurationStatus,
        curator_id: str,
        rationale: str,
        supersedes: KnowledgeCurationRecord | None,
    ) -> KnowledgeCurationRecord:
        identity = {
            "target_type": "result_evidence_submission",
            "target_id": submission.submission_id,
            "target_hash": submission.content_hash,
            "status": status,
            "curator_id": curator_id,
            "rationale": rationale,
            "evidence_refs": (),
            "supersedes_curation_id": (
                supersedes.curation_id if supersedes is not None else None
            ),
        }
        normalized = KnowledgeCurationRecord.model_construct(
            curation_id="pending",
            **identity,
            content_hash="0" * 64,
        ).model_dump(
            mode="json",
            exclude={"curation_id", "content_hash"},
            exclude_none=True,
        )
        curation_id = f"knowledge-curation-{content_hash(normalized)[:24]}"
        payload = {"curation_id": curation_id, **normalized}
        return KnowledgeCurationRecord(
            **payload, content_hash=content_hash(payload)
        )

    def curate(
        self,
        *,
        parent_run_id: str,
        submission_id: str,
        status: CurationStatus,
        curator_id: str,
        rationale: str,
    ) -> tuple[KnowledgeCurationRecord, ResultEvidenceMaterializationReceipt | None]:
        if status not in {CurationStatus.ACCEPTED, CurationStatus.REJECTED}:
            raise ResultFeedbackError(
                "INVALID_RESULT_CURATION_STATUS",
                "result intake may only be explicitly accepted or rejected",
            )
        with self.runs.acquire_run_lock(parent_run_id):
            submission = self.artifacts.load_submission(parent_run_id, submission_id)
            assessment = self.artifacts.load_assessment(parent_run_id, submission_id)
            if (
                assessment.submission_hash != submission.content_hash
                or assessment.parent_run_id != parent_run_id
            ):
                raise ResultFeedbackError(
                    "RESULT_ASSESSMENT_BINDING_INVALID",
                    "assessment does not bind the requested submission and run",
                )
            if (
                status == CurationStatus.ACCEPTED
                and assessment.status
                != ResultEvidenceIntakeStatus.RESULT_REQUIRES_CURATION
            ):
                raise ResultFeedbackError(
                    "RESULT_NOT_ADMISSIBLE_FOR_ACCEPTANCE",
                    "blocked result intake cannot be accepted",
                )
            if status == CurationStatus.ACCEPTED:
                self.artifacts.verify_archived_artifacts(submission)
            current = self.artifacts.current_curation(parent_run_id, submission_id)
            if current is not None and current.status == status:
                receipt = (
                    self.artifacts.load_receipt(parent_run_id, submission_id)
                    if status == CurationStatus.ACCEPTED
                    else None
                )
                return current, receipt
            curation = self._curation_record(
                submission,
                status=status,
                curator_id=curator_id,
                rationale=rationale,
                supersedes=current,
            )
            self.runs.write_immutable_artifact(
                parent_run_id,
                f"result-evidence/curations/{submission_id}/"
                f"c-{curation.content_hash[:16]}.yaml",
                "knowledge_curation_record",
                curation.curation_id,
                curation,
            )
            if status == CurationStatus.REJECTED:
                return curation, None
            return curation, self._materialize(
                submission, assessment, curation
            )

    def _materialize(
        self,
        submission: ResultEvidenceSubmission,
        assessment: ResultEvidenceAssessment,
        curation: KnowledgeCurationRecord,
    ) -> ResultEvidenceMaterializationReceipt:
        parent = self.parent_authority.load_parent(submission)
        method_text = "method_context = " + json.dumps(
            to_primitive(submission.method_fingerprint.attributes),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system_text = "system_context = " + json.dumps(
            to_primitive(submission.system_fingerprint.attributes),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        outcome_text = (
            f"execution_outcome = {submission.execution_outcome.value}; "
            f"result_type = {submission.result_type.value}"
        )
        observable_lines = tuple(
            (
                f"{item.quantity} = {item.value} {item.unit}"
                if item.value is not None
                else f"{item.quantity} = {item.qualitative_value}"
            )
            for item in submission.reported_observables
        )
        lines = (system_text, method_text, outcome_text, *observable_lines)
        text = "\n".join(lines) + "\n"
        run_dir = self.runs.run_dir(submission.parent_run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=".result-summary-",
                suffix=".txt",
                dir=run_dir,
                delete=False,
            ) as handle:
                handle.write(text)
                temporary_path = Path(handle.name)
            source_id = f"result-source-{submission.submission_id.removeprefix('result-submission-')}"
            source_version = submission.content_hash[:24]
            source = self.project_evidence.ingest_source(
                temporary_path,
                source_id,
                source_version,
                title=f"Accepted downstream result {submission.submission_id}",
                source_role=SourceRole.SYSTEM,
                source_type=SourceType.CALCULATION_ARCHIVE,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        evidence_by_label: dict[str, EvidenceSpan] = {}
        offset = 0
        labels = ("system", "method", "outcome", *(
            f"observable:{item.observable_key}" for item in submission.reported_observables
        ))
        for label, line in zip(labels, lines, strict=True):
            evidence_id = (
                "result-evidence-"
                + content_hash(
                    {"submission_id": submission.submission_id, "label": label}
                )[:24]
            )
            evidence = EvidenceSpan(
                evidence_id=evidence_id,
                source_id=source.source_id,
                source_version=source.version,
                content_sha256=source.content_sha256,
                start_offset=offset,
                end_offset=offset + len(line),
                text=line,
                locator=f"result submission {submission.submission_id}; {label}",
            )
            self.project_evidence.add_evidence(evidence)
            evidence_by_label[label] = evidence
            offset += len(line) + 1

        method_evidence = evidence_by_label["method"].evidence_id
        system_evidence = evidence_by_label["system"].evidence_id
        method_payload = {
            "text": method_text,
            "attributes": submission.method_fingerprint.attributes,
            "evidence_refs": (method_evidence,),
        }
        method_fact = MethodFact(
            fact_id=f"method-fact-{content_hash(method_payload)[:24]}",
            **method_payload,
        )
        model_payload = {
            "text": system_text,
            "attributes": submission.system_fingerprint.attributes,
            "evidence_refs": (system_evidence,),
        }
        model_fact = ModelFact(
            fact_id=f"model-fact-{content_hash(model_payload)[:24]}",
            **model_payload,
        )
        results: list[ReportedResult] = []
        observations: list[ReportedObservation] = []
        if submission.result_type in {
            ResultEvidenceType.COMPUTED_RESULT,
            ResultEvidenceType.EXPERIMENTAL_RESULT,
            ResultEvidenceType.PARTIAL_RESULT,
            ResultEvidenceType.NULL_OR_NEGATIVE_RESULT,
        }:
            for observable in submission.reported_observables:
                observable_evidence = evidence_by_label[
                    f"observable:{observable.observable_key}"
                ].evidence_id
                result_context_payload = {
                    "system_context": submission.system_fingerprint.attributes,
                    "method_context": submission.method_fingerprint.attributes,
                    "method_fact_refs": (method_fact.fact_id,),
                    "model_fact_refs": (model_fact.fact_id,),
                }
                result_context = ResultContext(
                    context_id=(
                        "result-context-"
                        + content_hash(result_context_payload)[:24]
                    ),
                    **result_context_payload,
                )
                shared_payload = {
                    "quantity": observable.quantity,
                    "system_context": submission.system_fingerprint.attributes,
                    "method_context": submission.method_fingerprint.attributes,
                    "result_context": result_context,
                    "evidence_refs": (
                        observable_evidence,
                        method_evidence,
                        system_evidence,
                    ),
                    "result_status": observable.result_status,
                }
                if observable.value is not None:
                    result_payload = {
                        **shared_payload,
                        "value": observable.value,
                        "unit": observable.unit,
                    }
                    results.append(
                        ReportedResult(
                            result_id=(
                                "reported-result-"
                                + content_hash(result_payload)[:24]
                            ),
                            **result_payload,
                        )
                    )
                else:
                    observation_payload = {
                        **shared_payload,
                        "qualitative_value": observable.qualitative_value,
                    }
                    observations.append(
                        ReportedObservation(
                            observation_id=(
                                "reported-observation-"
                                + content_hash(observation_payload)[:24]
                            ),
                            **observation_payload,
                        )
                    )

        prefix = self.artifacts.accepted_prefix(submission.submission_id)
        self.runs.write_immutable_artifact(
            submission.parent_run_id,
            f"{prefix}/m/{method_fact.fact_id[-16:]}.yaml",
            "method_fact",
            method_fact.fact_id,
            method_fact,
        )
        self.runs.write_immutable_artifact(
            submission.parent_run_id,
            f"{prefix}/d/{model_fact.fact_id[-16:]}.yaml",
            "model_fact",
            model_fact.fact_id,
            model_fact,
        )
        for result in results:
            self.runs.write_immutable_artifact(
                submission.parent_run_id,
                f"{prefix}/r/{result.result_id[-16:]}.yaml",
                "reported_result",
                result.result_id,
                result,
            )
        for observation in observations:
            self.runs.write_immutable_artifact(
                submission.parent_run_id,
                f"{prefix}/o/{observation.observation_id[-16:]}.yaml",
                "reported_observation",
                observation.observation_id,
                observation,
            )
        observed = {
            item.observable_key for item in submission.reported_observables
        } | {item.quantity for item in submission.reported_observables}
        missing = tuple(output for output in parent.task.outputs if output not in observed)
        payload = {
            "submission_id": submission.submission_id,
            "submission_hash": submission.content_hash,
            "assessment_id": assessment.assessment_id,
            "assessment_hash": assessment.content_hash,
            "curation_id": curation.curation_id,
            "curation_hash": curation.content_hash,
            "parent_run_id": submission.parent_run_id,
            "parent_plan_id": submission.parent_plan_id,
            "parent_plan_hash": submission.parent_plan_hash,
            "task_id": submission.task_id,
            "parent_authority_origin": parent.origin,
            "parent_authority_ref": parent.origin_ref,
            "source_id": source.source_id,
            "source_version": source.version,
            "accepted_evidence_ids": tuple(
                evidence.evidence_id for evidence in evidence_by_label.values()
            ),
            "reported_result_hashes": {
                item.result_id: content_hash(item) for item in results
            },
            "reported_observation_hashes": {
                item.observation_id: content_hash(item) for item in observations
            },
            "method_fact_hashes": {method_fact.fact_id: content_hash(method_fact)},
            "model_fact_hashes": {model_fact.fact_id: content_hash(model_fact)},
            "missing_observables": missing,
        }
        receipt = build_content_bound(
            ResultEvidenceMaterializationReceipt,
            "result-materialization",
            payload,
        )
        self.runs.write_immutable_artifact(
            submission.parent_run_id,
            f"{prefix}/receipt.yaml",
            "result_evidence_materialization_receipt",
            receipt.receipt_id,
            receipt,
        )
        return receipt
