from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

from .models import DomainProfile, ExpertCase, LiteratureWorkflowPattern, ScientificCapability
from .serialization import load_data, require_safe_path_component


@dataclass(frozen=True)
class DomainPack:
    profile: DomainProfile
    capabilities: tuple[ScientificCapability, ...]
    expert_cases: tuple[ExpertCase, ...]
    workflow_patterns: tuple[LiteratureWorkflowPattern, ...]


class DomainPackLoader:
    def __init__(self, search_paths: tuple[Path, ...] = ()) -> None:
        self.search_paths = search_paths

    def _resolve(self, domain_id: str) -> Path:
        require_safe_path_component(domain_id, field="domain_id")
        for root in self.search_paths:
            candidate = root / domain_id
            if (candidate / "profile.yaml").is_file():
                return candidate
        resource = files("spc.domain_packs").joinpath(domain_id)
        with as_file(resource) as candidate:
            if not (candidate / "profile.yaml").is_file():
                raise FileNotFoundError(f"domain pack not found: {domain_id}")
            return Path(candidate)

    def load(self, domain_id: str) -> DomainPack:
        root = self._resolve(domain_id)
        profile = DomainProfile.model_validate(load_data(root / "profile.yaml"))
        if profile.domain_id != domain_id:
            raise ValueError("domain directory and profile domain_id differ")
        capabilities = tuple(
            ScientificCapability.model_validate(item) for item in (load_data(root / "capabilities.yaml") or [])
        )
        expert_cases = tuple(
            ExpertCase.model_validate(item) for item in (load_data(root / "expert_cases.yaml") or [])
        )
        patterns = tuple(
            LiteratureWorkflowPattern.model_validate(item)
            for item in (load_data(root / "workflow_patterns.yaml") or [])
        )
        return DomainPack(profile, capabilities, expert_cases, patterns)
