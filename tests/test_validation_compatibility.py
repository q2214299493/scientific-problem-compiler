from __future__ import annotations

import pytest

from spc import validators
from spc.approval.validators import ApprovalResponseError
from spc.downstream.validator import DownstreamImportError
from spc.export import ExportError
from spc.planning.validators import PlanningProposalError
from spc.validation import approval, common, evidence, export, plans


@pytest.mark.parametrize(("owner", "names"), [
    (common, ("ValidationIssue", "ValidationReport")),
    (evidence, ("EvidenceSpanRepository", "validate_evidence_links")),
    (plans, (
        "validate_dag", "compare_method_fingerprints", "validate_question_plan",
        "validate_candidate_set", "validate_plan_compilation_receipt",
        "build_plan_validation_record", "validate_plan_validation_record", "validate_phase1_boundary",
    )),
    (approval, (
        "validate_approval_boundary", "requires_independent_approval", "validate_approval_claim_evidence_coherence",
        "validate_independent_approval_chain", "validate_conditional_approval", "validate_approval_state",
        "validate_approved_plan_authority",
    )),
    (export, (
        "validate_handoff_package", "validate_export", "REQUIRED_EXPORT_FILES", "INDEPENDENT_APPROVAL_EXPORT_FILES",
    )),
])
def test_original_validation_names_reference_their_owners(owner, names):
    for name in names:
        assert getattr(validators, name) is getattr(owner, name)


def test_ordered_issues_preserve_boundary_error_formats():
    first = validators.ValidationIssue(code="B", message="first", path="tasks[1]")
    second = validators.ValidationIssue(code="A", message="second")
    report = common.validation_report(iter((first, second, first)))
    assert report.issues == (first, second, first)
    assert not report.valid
    assert common.validation_report(iter(())).valid
    assert str(ExportError(report)) == str(DownstreamImportError(report)) == "B: first; A: second; B: first"
    assert str(PlanningProposalError(report)) == "PlanningProposalSet validation failed: B, A, B"
    assert str(ApprovalResponseError(report)) == "ApprovalLLMResponse validation failed: B, A, B"
