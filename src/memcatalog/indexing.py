"""Derived temporal, entity, event, and fact indexes."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Set

from .heuristics import question_entities
from .models import FactRecord, Session


class MemoryIndex:
    def __init__(self, sessions: Iterable[Session], facts: Iterable[FactRecord]) -> None:
        self.sessions = sorted(sessions, key=lambda item: (item.timestamp, item.session_id))
        self.facts = list(facts)
        self.entity_sessions: DefaultDict[str, Set[str]] = defaultdict(set)
        self.entity_facts: DefaultDict[str, Set[str]] = defaultdict(set)
        self.event_facts: DefaultDict[str, Set[str]] = defaultdict(set)
        self._build()

    def _build(self) -> None:
        for session in self.sessions:
            for entity in self._tokens(session.text):
                self.entity_sessions[entity].add(session.session_id)
        for fact in self.facts:
            for entity in self._tokens(" ".join((fact.subject, fact.predicate, fact.object))):
                self.entity_facts[entity].add(fact.fact_id)
            if fact.event_id:
                self.event_facts[fact.event_id].add(fact.fact_id)

    def session_candidates(self, question: str, limit: Optional[int] = None) -> List[str]:
        scores: Dict[str, int] = defaultdict(int)
        for entity in question_entities(question):
            for session_id in self.entity_sessions.get(entity.casefold(), set()):
                scores[session_id] += 1
        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        if not ranked:
            ranked = [session.session_id for session in self.sessions]
        return ranked[:limit] if limit is not None else ranked

    def fact_candidates(self, question: str) -> List[str]:
        scores: Dict[str, int] = defaultdict(int)
        for entity in question_entities(question):
            for fact_id in self.entity_facts.get(entity.casefold(), set()):
                scores[fact_id] += 1
        return sorted(scores, key=lambda item: (-scores[item], item))

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {token.casefold() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", text)}
