"""Digest contracts captured from 9435a5f and immutable pre-refactor records."""

import json
from pathlib import Path
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict
import pytest
import yaml

from spc.knowledge.collection import _make_bound
from spc.models import (
    CollectionPageRecord,
    ResultEvidenceMaterializationReceipt,
    ResultEvidenceSubmission,
    ResultEvidenceType,
    SuccessorPlanningCycle,
)
from spc.result_feedback import build_result_evidence_submission
from spc.result_feedback.submission import build_content_bound
from spc.serialization import build_content_bound_record


FIXTURES = Path(__file__).parent / "fixtures"


class SampleRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    record_id: str
    content_hash: str
    count: int = 2


class NestedRecord(BaseModel):
    label: str = "é"
    note: str | None = None


IDENTITY = {"label": "文献-é", "kind": ResultEvidenceType.COMPUTED_RESULT, "items": [2, 1]}
BASE_ID = "sample-2916431aa90dec53e21d6bd6"
BASE_HASH = "88d9c5c2e62f82aadee4925ab825739cb840d2dd6301a80056956407012669b9"


@pytest.mark.parametrize(
    ("identity", "expected_id", "expected_hash"),
    [
        (IDENTITY, BASE_ID, BASE_HASH),
        (dict(reversed(IDENTITY.items())), BASE_ID, BASE_HASH),
        ({**IDENTITY, "kind": ResultEvidenceType.COMPUTED_RESULT.value}, BASE_ID, BASE_HASH),
        (
            {**IDENTITY, "items": [1, 2]}, "sample-9d2139ceb7eafbed6f172547",
            "32ba5b0fbd8d4b9863c34b59a190b430d297e400c6723e206bd3f6f958667241",
        ),
        (
            {**IDENTITY, "note": None}, "sample-777c9b6e7c967357f655bbb9",
            "ef9ba30b42d354e8a2b2fab3970f482b06d7fb69da839aaf4d3a9e4dee72e3d2",
        ),
        (
            {**IDENTITY, "count": 2}, "sample-6a0282f558ff4ffebd37f2e5",
            "504ed48f83910b6de3cb97e1609cd5cf92195eb1683ce24d6eb9c8da29a9aac3",
        ),
        (
            {**IDENTITY, "label": "文献-e\u0301"}, "sample-9a36d359b35dd9d1bc0e3573",
            "f745aa633082b6b37afb9a8117ba38c87b308b83db72ef410e814b95b1348acb",
        ),
        (
            {**IDENTITY, "child": NestedRecord()}, "sample-1116ac38f1a872e73f8139d3",
            "d66a0c375a599821557f87c354bb72b3645f4424983c95b0d33e01b8d75b49cb",
        ),
    ],
    ids=["original", "key-order", "enum-value", "list-order", "null", "default", "nfd", "nested"],
)
def test_content_bound_record_preserves_both_digests(identity, expected_id, expected_hash):
    record = build_content_bound_record(SampleRecord, "sample", "record_id", identity)
    assert (record.record_id, record.content_hash) == (expected_id, expected_hash)


def test_historical_collection_page_can_be_reconstructed():
    member = "collection_pages/collection-page-a528966052fce62888fb4eb4.json"
    with ZipFile(FIXTURES / "repositories-ac25bd4.zip") as archive:
        saved = json.loads(archive.read(member))
    identity = {key: value for key, value in saved.items() if key not in ("page_id", "content_hash")}
    page = _make_bound(CollectionPageRecord, "collection-page", identity)
    assert page == CollectionPageRecord.model_validate(saved)


def historical_result_record(model):
    with ZipFile(FIXTURES / "model-history-f5cfdd7.zip") as archive:
        return model.model_validate(yaml.safe_load(archive.read(f"{model.__name__}.yaml")))


def test_historical_submission_with_omitted_and_explicit_defaults():
    saved = historical_result_record(ResultEvidenceSubmission)
    values = {
        name: getattr(saved, name) for name in ResultEvidenceSubmission.model_fields
        if name not in ("submission_id", "content_hash")
    }
    assert build_result_evidence_submission(**values) == saved
    assert values.pop("export_id") is None
    assert values.pop("declared_deviations") == ()
    assert build_result_evidence_submission(**values) == saved


@pytest.mark.parametrize(("model", "prefix", "id_field"), [
    (ResultEvidenceMaterializationReceipt, "result-materialization", "receipt_id"),
    (SuccessorPlanningCycle, "successor-cycle", "cycle_id"),
])
def test_historical_result_records_with_inferred_id_field(model, prefix, id_field):
    saved = historical_result_record(model)
    values = {
        name: getattr(saved, name) for name in model.model_fields
        if name not in (id_field, "content_hash") and getattr(saved, name) is not None
    }
    assert build_content_bound(model, prefix, values) == saved
