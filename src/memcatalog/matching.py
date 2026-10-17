"""Question-to-session entity matching."""

from __future__ import annotations

from typing import List

from .heuristics import question_entities


def session_matches(question: str, sessions_text: List[str]) -> List[List[str]]:
    """Return up to five matching question entities for each session."""
    entities = sorted(question_entities(question))
    if not entities:
        return [[] for _ in sessions_text]

    matches: List[List[str]] = []
    for text in sessions_text:
        lowered = text.lower()
        matches.append(
            [entity for entity in entities if entity.lower() in lowered][:5]
        )
    return matches
