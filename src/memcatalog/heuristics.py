"""Shared lexical heuristics for catalog planning."""

from __future__ import annotations

import re
from typing import List, Optional, Set

from .catalog import REVISION_CUES, STOP_WORDS, extract_entities


def question_entities(question: str) -> Set[str]:
    """Return normalized named entities and useful content tokens."""
    entities = extract_entities(question)
    tokens = {
        token
        for token in re.findall(r"[a-z]{5,}", question.lower())
        if token not in STOP_WORDS
    }
    return entities | tokens


def entity_scores(question: str, sessions_text: List[str]) -> List[int]:
    """Count question-entity hits in each session."""
    entities = question_entities(question)
    if not entities:
        return [0] * len(sessions_text)
    return [
        sum(1 for entity in entities if entity.lower() in text.lower())
        for text in sessions_text
    ]


def revision_flags(sessions_text: List[str]) -> List[bool]:
    """Mark sessions containing a lexical revision cue."""
    return [bool(REVISION_CUES.search(text)) for text in sessions_text]


def normalize_question_type(question_type: Optional[str]) -> str:
    """Normalize named types and the common numeric type labels."""
    if not question_type:
        return ""
    normalized = str(question_type).lower().strip()
    aliases = {
        "1": "temporal",
        "2": "single-hop",
        "3": "open-ended",
        "4": "multi-hop",
        "5": "adversarial",
    }
    return aliases.get(normalized, normalized)
