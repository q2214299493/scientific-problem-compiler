"""Compatibility checks against the model contracts captured at f5cfdd7."""

from __future__ import annotations

import csv
from enum import Enum
import gzip
from importlib import import_module
import json
from pathlib import Path
from zipfile import ZipFile

from pydantic import BaseModel, TypeAdapter
import pytest

from spc import models
from spc.serialization import content_hash, load_model


FIXTURES = Path(__file__).parent / "fixtures"
INVENTORY = Path(__file__).parents[1] / "docs" / "model-inventory-f5cfdd7.csv"
with gzip.open(FIXTURES / "model-contracts-f5cfdd7.json.gz", "rt", encoding="utf-8") as stream:
    BASELINE = json.load(stream)


def field_contract(cls):
    annotations = {}
    for parent in reversed(cls.__mro__):
        annotations.update(parent.__dict__.get("__annotations__", {}))
    return [
        {
            "name": name,
            "annotation": annotations[name],
            "required": field.is_required(),
            "default": repr(field.default),
            "factory": field.default_factory.__qualname__ if field.default_factory else None,
            "alias": field.alias,
            "validation_alias": repr(field.validation_alias),
            "serialization_alias": field.serialization_alias,
        }
        for name, field in cls.model_fields.items()
    ]


@pytest.mark.parametrize("name", sorted(BASELINE["classes"]))
def test_model_contract(name):
    cls = getattr(models, name)
    saved = BASELINE["classes"][name]
    if issubclass(cls, Enum):
        assert [[key, member.value] for key, member in cls.__members__.items()] == saved["members"]
        schema = TypeAdapter(cls).json_schema
    else:
        assert issubclass(cls, BaseModel)
        assert field_contract(cls) == saved["fields"]
        schema = cls.model_json_schema

    assert schema() == saved["validation"]
    assert schema(mode="validation") == saved["validation"]
    assert schema(mode="serialization") == saved.get("serialization", saved["validation"])


def test_legacy_exports_are_the_domain_definitions():
    assert sorted(name for name in vars(models) if not name.startswith("_")) == BASELINE["public_names"]
    with INVENTORY.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    domain_modules = {
        domain: import_module(f"spc.model_definitions.{domain}")
        for domain in {row["domain_module"] for row in rows}
    }
    for row in rows:
        name = row["name"]
        assert getattr(models, name) is getattr(domain_modules[row["domain_module"]], name)

    definitions = [
        name
        for module in domain_modules.values()
        for name, value in vars(module).items()
        if isinstance(value, type) and value.__module__ == module.__name__
    ]
    assert sorted(definitions) == sorted(BASELINE["classes"])


def test_persisted_model_hashes(tmp_path):
    with ZipFile(FIXTURES / "model-history-f5cfdd7.zip") as archive:
        expected = json.loads(archive.read("hashes.json"))
        for filename, expected_hash in expected.items():
            path = tmp_path / filename
            path.write_bytes(archive.read(filename))
            model = load_model(path, getattr(models, path.stem))
            assert content_hash(model) == expected_hash, filename
