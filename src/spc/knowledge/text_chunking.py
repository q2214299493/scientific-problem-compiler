"""Character ranges shared by the expert and literature chunk builders."""

from collections.abc import Iterator


def bounded_text_ranges(
    text: str, start: int, end: int, *, max_characters: int
) -> Iterator[tuple[int, int]]:
    """Yield trimmed, absolute half-open ranges within the supplied source span."""
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    cursor = start
    while cursor < end:
        boundary = min(cursor + max_characters, end)
        if boundary < end:
            relative = text[cursor:boundary]
            split = max(relative.rfind("\n"), relative.rfind(" "))
            if split > max_characters // 2:
                boundary = cursor + split + 1
        while cursor < boundary and text[cursor].isspace():
            cursor += 1
        while boundary > cursor and text[boundary - 1].isspace():
            boundary -= 1
        if boundary > cursor:
            yield cursor, boundary
        cursor = max(boundary, cursor + 1)
