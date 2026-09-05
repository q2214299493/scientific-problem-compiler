from __future__ import annotations

import re

from ..models import EpistemicStatus, SourceRole, SourceType


def classify_claim(
    text: str,
    source_role: SourceRole | str,
    source_type: SourceType | str,
) -> tuple[str, SourceRole, str, EpistemicStatus]:
    """Classify source wording without promoting it to an SPC-established fact."""
    source_role = SourceRole(source_role)
    source_type = SourceType(source_type)
    normalized = text.casefold()
    if source_type == SourceType.AUTHOR_RESPONSE:
        source_role = SourceRole.AUTHOR
    if source_type == SourceType.REVIEWER_COMMENT or source_role == SourceRole.REVIEWER:
        source_role = SourceRole.REVIEWER
        claim_type = "reviewer_question" if text.strip().endswith("?") else "reviewer_statement"
        return claim_type, source_role, "unresolved", EpistemicStatus.UNRESOLVED
    if "reviewer" in normalized and re.search(r"\b(?:asks?|questions?)\b", normalized):
        claim_type = "reviewer_question" if "whether" in normalized or text.strip().endswith("?") else "reviewer_statement"
        return claim_type, source_role, "unresolved", EpistemicStatus.UNRESOLVED
    if text.strip().endswith("?"):
        return "source_question", source_role, "unresolved", EpistemicStatus.UNRESOLVED
    if re.search(r"\b(hypothes(?:is|ize|izes|ized)|we propose|may|might|could)\b", normalized):
        return "hypothesis", source_role, "tentative", EpistemicStatus.SOURCE_HYPOTHESIS
    if re.search(r"\b(game|bep|model|descriptor|scaling relation)\b", normalized):
        return "model_statement", source_role, "source_reported", EpistemicStatus.MODEL_STATEMENT
    if re.search(r"\b(dft|pbe|vasp|neb|functional|basis set|calculation)\b", normalized):
        return "method_statement", source_role, "source_reported", EpistemicStatus.METHOD_STATEMENT
    if re.search(r"[-+]?\d+(?:\.\d+)?\s*(?:mev|ev|kj\s*/\s*mol|kcal\s*/\s*mol|k|bar|pa|atm|%)\b", normalized):
        return "reported_result", source_role, "source_reported", EpistemicStatus.REPORTED_RESULT
    return "source_statement", source_role, "source_reported", EpistemicStatus.SOURCE_REPORTED
