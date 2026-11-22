from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self, TypeVar

from pydantic import BaseModel

from ...repositories import IdentityBoundRepository
from .contracts import (
    LiteratureKnowledgeChunk,
    LiteratureKnowledgeCompilationInput,
    LiteratureKnowledgeCompilationRecord,
    LiteratureKnowledgeGroundingRecord,
    LiteratureKnowledgeProposalSet,
)


if TYPE_CHECKING:
    from ...repositories import KnowledgeRepositories

_Record = TypeVar("_Record", bound=BaseModel)


class _LiteratureKnowledgeRepository(IdentityBoundRepository[_Record]):
    _directory_name: str
    _record_type: type[_Record]
    _key_field: str

    def __init__(self, knowledge_root: Path) -> None:
        super().__init__(knowledge_root / self._directory_name, self._record_type, self._key_field)

    @classmethod
    def for_operation(cls, repositories: KnowledgeRepositories) -> Self:
        """Reuse only a native filesystem handle with the original root and behavior."""
        candidate = getattr(repositories, cls._directory_name, None)
        owner = getattr(repositories, "literature", None)
        if type(candidate) is not cls or candidate is not getattr(owner, cls._directory_name, None):
            return cls(repositories.root)
        # An instance-level reader/writer override or other extra state also
        # disqualifies the handle. These sites previously opened it from disk.
        expected_root = repositories.root / cls._directory_name
        state = vars(candidate)
        if (
            state.keys() == {"root", "model_type", "identity_field"}
            and candidate.model_type is cls._record_type
            and candidate.identity_field == cls._key_field
            and type(candidate.root) is type(expected_root)
            # WindowsPath equality folds case even in a case-sensitive directory.
            and str(candidate.root) == str(expected_root)
        ):
            return candidate
        return cls(repositories.root)


class LiteratureKnowledgeCompilationInputRepository(
    _LiteratureKnowledgeRepository[LiteratureKnowledgeCompilationInput]
):
    _directory_name = "literature_knowledge_inputs"
    _record_type = LiteratureKnowledgeCompilationInput
    _key_field = "compilation_input_id"


class LiteratureKnowledgeChunkRepository(_LiteratureKnowledgeRepository[LiteratureKnowledgeChunk]):
    _directory_name = "literature_knowledge_chunks"
    _record_type = LiteratureKnowledgeChunk
    _key_field = "chunk_id"


class LiteratureKnowledgeProposalSetRepository(_LiteratureKnowledgeRepository[LiteratureKnowledgeProposalSet]):
    _directory_name = "literature_knowledge_proposals"
    _record_type = LiteratureKnowledgeProposalSet
    _key_field = "proposal_set_id"


class LiteratureKnowledgeGroundingRepository(_LiteratureKnowledgeRepository[LiteratureKnowledgeGroundingRecord]):
    _directory_name = "literature_knowledge_groundings"
    _record_type = LiteratureKnowledgeGroundingRecord
    _key_field = "grounding_id"


class LiteratureKnowledgeCompilationRepository(_LiteratureKnowledgeRepository[LiteratureKnowledgeCompilationRecord]):
    _directory_name = "literature_knowledge_compilations"
    _record_type = LiteratureKnowledgeCompilationRecord
    _key_field = "compilation_id"
