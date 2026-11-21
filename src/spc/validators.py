"""Compatibility exports for the original SPC validators module."""

from __future__ import annotations

import json as json
import re as re
from collections import Counter as Counter, deque as deque
from pathlib import Path as Path, PurePosixPath as PurePosixPath
from typing import Any as Any, Iterable as Iterable, Protocol as Protocol

from pydantic import BaseModel as BaseModel, ConfigDict as ConfigDict

from .models import (
    AgentHandoffPackage as AgentHandoffPackage,
    ApprovalDecision as ApprovalDecision,
    ApprovalMode as ApprovalMode,
    ApprovalReviewInput as ApprovalReviewInput,
    ApprovalReviewRecord as ApprovalReviewRecord,
    ApprovalScores as ApprovalScores,
    ApprovalVerdict as ApprovalVerdict,
    DAGTask as DAGTask,
    EvidenceClassification as EvidenceClassification,
    EvidenceSpan as EvidenceSpan,
    ExecutionPolicy as ExecutionPolicy,
    ExportManifest as ExportManifest,
    GateVerdict as GateVerdict,
    GroundedStatement as GroundedStatement,
    IndependentApprovalReceipt as IndependentApprovalReceipt,
    PlanCompilationReceipt as PlanCompilationReceipt,
    PlanValidationRecord as PlanValidationRecord,
    ProjectTrustPolicy as ProjectTrustPolicy,
    RevisionApprovalLLMResponse as RevisionApprovalLLMResponse,
    RevisionApprovalReviewInput as RevisionApprovalReviewInput,
    RevisionIssueStatus as RevisionIssueStatus,
    ScientificCapability as ScientificCapability,
    ScientificQuestionPlan as ScientificQuestionPlan,
    SourceDocument as SourceDocument,
    parse_approval_review_input as parse_approval_review_input,
)
from .serialization import (
    content_hash as content_hash,
    file_sha256 as file_sha256,
    load_data as load_data,
)
from .validation import common as _common
from .validation.approval import (
    requires_independent_approval as requires_independent_approval,
    validate_approval_boundary as validate_approval_boundary,
    validate_approval_claim_evidence_coherence as validate_approval_claim_evidence_coherence,
    validate_approval_state as validate_approval_state,
    validate_approved_plan_authority as validate_approved_plan_authority,
    validate_conditional_approval as validate_conditional_approval,
    validate_independent_approval_chain as validate_independent_approval_chain,
)
from .validation.common import ValidationIssue as ValidationIssue, ValidationReport as ValidationReport
from .validation.evidence import (
    EvidenceSpanRepository as EvidenceSpanRepository,
    validate_evidence_links as validate_evidence_links,
)
from .validation.export import (
    INDEPENDENT_APPROVAL_EXPORT_FILES as INDEPENDENT_APPROVAL_EXPORT_FILES,
    REQUIRED_EXPORT_FILES as REQUIRED_EXPORT_FILES,
    _load_jsonl as _load_jsonl,
    _safe_export_relative_path as _safe_export_relative_path,
    _semantic_export_issues as _semantic_export_issues,
    validate_export as validate_export,
    validate_handoff_package as validate_handoff_package,
)
from .validation.plans import (
    build_plan_validation_record as build_plan_validation_record,
    compare_method_fingerprints as compare_method_fingerprints,
    validate_candidate_set as validate_candidate_set,
    validate_dag as validate_dag,
    validate_phase1_boundary as validate_phase1_boundary,
    validate_plan_compilation_receipt as validate_plan_compilation_receipt,
    validate_plan_validation_record as validate_plan_validation_record,
    validate_question_plan as validate_question_plan,
)

_report = _common.validation_report
_walk = _common.walk_values
