from __future__ import annotations

import re

from ..models import EpistemicStatus


def classify_claim(text: str) -> tuple[str, str, str, EpistemicStatus]:
    """Classify source wording without promoting it to an SPC-established fact."""
    normalized = text.casefold()
    if "reviewer" in normalized or text.strip().endswith("?"):
        return "reviewer_question", "reviewer", "unresolved", EpistemicStatus.UNRESOLVED
    if re.search(r"\b(hypothes(?:is|ize|izes|ized)|we propose|may|might|could)\b", normalized):
        return "hypothesis", "author", "tentative", EpistemicStatus.SOURCE_HYPOTHESIS
    if re.search(r"\b(game|bep|model|descriptor|scaling relation)\b", normalized):
        return "model_statement", "author", "source_reported", EpistemicStatus.MODEL_STATEMENT
    if re.search(r"\b(dft|pbe|vasp|neb|functional|basis set|calculation)\b", normalized):
        return "method_statement", "author", "source_reported", EpistemicStatus.METHOD_STATEMENT
    if re.search(r"[-+]?\d+(?:\.\d+)?\s*(?:mev|ev|kj\s*/\s*mol|kcal\s*/\s*mol|k|bar|pa|atm|%)\b", normalized):
        return "reported_result", "author", "source_reported", EpistemicStatus.REPORTED_RESULT
    return "source_statement", "author", "source_reported", EpistemicStatus.SOURCE_REPORTED
