from __future__ import annotations

from collections.abc import Iterable

from ...models import CurationStatus
from ...repositories import KnowledgeRepositories
from ...serialization import content_hash
from ..trust import TrustedKnowledgeValidator
from .contracts import (
    ExpertKnowledgeChunk,
    ExpertKnowledgeCompilationInput,
    ExpertSourcePage,
)
from .source import validate_expert_source


CHUNK_POLICY_ID = "spc-expert-source-offset-chunks"
CHUNK_POLICY_VERSION = "1.0.0"
MAX_CHUNK_CHARACTERS = 2_000


def _content_bound(model_type, prefix: str, id_field: str, identity: dict):
    record_id = f"{prefix}-{content_hash(identity)[:24]}"
    payload = {id_field: record_id, **identity}
    return model_type(**payload, content_hash=content_hash(payload))


def _put_reuse(repository, key: str, record) -> None:
    try:
        existing = repository.get(key)
    except FileNotFoundError:
        repository.put(key, record)
        return
    if existing != record:
        raise FileExistsError(f"conflicting immutable expert compilation record: {key}")


def resolve_expert_knowledge_input(
    expert_source_id: str,
    repositories: KnowledgeRepositories,
) -> ExpertKnowledgeCompilationInput:
    source_record = validate_expert_source(
        repositories.expert_sources.get(expert_source_id), repositories
    )
    profile = repositories.expert_profiles.get(source_record.expert_id)
    attribution = repositories.expert_attributions.get(source_record.attribution_id)
    if attribution.attribution_status != "confirmed":
        raise ValueError("expert source attribution is unresolved")
    current = TrustedKnowledgeValidator(
        repositories, repositories.evidence_store
    ).resolve_current_curations()
    for target_type, target_id in (
        ("expert_source", source_record.expert_source_id),
        ("expert_attribution", attribution.attribution_id),
    ):
        curation = current.get((target_type, target_id))
        if curation is None or curation.status != CurationStatus.ACCEPTED:
            raise ValueError(f"{target_type} source authority is not accepted")
    source = repositories.evidence_store.get_source(
        source_record.source_id, source_record.source_version
    )
    identity = {
        "expert_id": profile.expert_id,
        "expert_profile_hash": profile.content_hash,
        "expert_source_id": source_record.expert_source_id,
        "expert_source_hash": source_record.content_hash,
        "source_id": source.source_id,
        "source_version": source.version,
        "source_hash": content_hash(source),
        "attribution_id": attribution.attribution_id,
        "attribution_hash": attribution.content_hash,
        "domain": source_record.domain,
        "chunk_policy_id": CHUNK_POLICY_ID,
        "chunk_policy_version": CHUNK_POLICY_VERSION,
    }
    record = _content_bound(
        ExpertKnowledgeCompilationInput,
        "expert-knowledge-input",
        "compilation_input_id",
        identity,
    )
    _put_reuse(
        repositories.expert_knowledge_inputs,
        record.compilation_input_id,
        record,
    )
    return record


def _bounded_ranges(start: int, end: int, text: str) -> Iterable[tuple[int, int]]:
    cursor = start
    while cursor < end:
        boundary = min(cursor + MAX_CHUNK_CHARACTERS, end)
        if boundary < end:
            relative = text[cursor:boundary]
            split = max(relative.rfind("\n"), relative.rfind(" "))
            if split > MAX_CHUNK_CHARACTERS // 2:
                boundary = cursor + split + 1
        while cursor < boundary and text[cursor].isspace():
            cursor += 1
        while boundary > cursor and text[boundary - 1].isspace():
            boundary -= 1
        if boundary > cursor:
            yield cursor, boundary
        cursor = max(boundary, cursor + 1)


def _page_segments(length: int, pages: tuple[ExpertSourcePage, ...]):
    if pages:
        return tuple((page.start_offset, page.end_offset, page.page_number) for page in pages)
    return ((0, length, None),)


def build_expert_knowledge_chunks(
    compilation_input: ExpertKnowledgeCompilationInput,
    repositories: KnowledgeRepositories,
) -> tuple[ExpertKnowledgeChunk, ...]:
    source_record = repositories.expert_sources.get(compilation_input.expert_source_id)
    source = repositories.evidence_store.get_source(
        compilation_input.source_id, compilation_input.source_version
    )
    repositories.evidence_store.verify_source_integrity(source)
    content_path = repositories.evidence_store.state_root.joinpath(*source.stored_path.split("/"))
    text = content_path.read_text(encoding="utf-8")
    chunks: list[ExpertKnowledgeChunk] = []
    for start, end, page_number in _page_segments(len(text), source_record.page_ranges):
        for chunk_start, chunk_end in _bounded_ranges(start, end, text):
            identity = {
                "expert_source_id": source_record.expert_source_id,
                "expert_id": source_record.expert_id,
                "attribution_id": source_record.attribution_id,
                "attribution_hash": source_record.attribution_hash,
                "source_id": source.source_id,
                "source_version": source.version,
                "source_type": source_record.source_type,
                "start_offset": chunk_start,
                "end_offset": chunk_end,
                "text": text[chunk_start:chunk_end],
                "page_number": page_number,
                "section_path": (),
            }
            identity = {key: value for key, value in identity.items() if value is not None}
            chunk = _content_bound(
                ExpertKnowledgeChunk,
                "expert-knowledge-chunk",
                "chunk_id",
                identity,
            )
            _put_reuse(repositories.expert_knowledge_chunks, chunk.chunk_id, chunk)
            chunks.append(chunk)
    if not chunks:
        raise ValueError("expert source produced no bounded chunks")
    return tuple(chunks)


__all__ = [
    "MAX_CHUNK_CHARACTERS",
    "build_expert_knowledge_chunks",
    "resolve_expert_knowledge_input",
]
