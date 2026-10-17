"""Evidence grounding against cached facts and source snapshots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .experience import ExperienceModel
from .indexing import MemoryIndex
from .models import EvidenceBundle, FactRecord, Request, SupportStatus
from .storage import JsonMemoryStore


@dataclass
class GroundingResult:
    request_id: str
    facts: List[FactRecord]
    evidence: EvidenceBundle
    relevance: Dict[str, float]


class Grounder:
    def __init__(self, store: JsonMemoryStore, experience: ExperienceModel) -> None:
        self.store = store
        self.experience = experience

    def ground(
        self,
        user_id: str,
        request: Request,
        context_key: str,
        snapshot_at: Optional[str],
        max_facts: int = 20,
    ) -> GroundingResult:
        sessions = self.store.sessions(user_id, snapshot_at)
        facts = self.store.facts(user_id, snapshot_at, include_superseded=False)
        index = MemoryIndex(sessions, facts)
        question = str(request.scope.get("question", ""))
        indexed_ids = set(index.fact_candidates(question))
        relevance = {
            fact.fact_id: self._relevance(question, fact, indexed_ids)
            for fact in facts
        }
        eligible = [fact for fact in facts if relevance[fact.fact_id] > 0]
        selected = self.experience.prioritize(
            user_id, context_key, eligible, relevance, limit=max_facts
        )
        source_sessions = list(
            dict.fromkeys(
                span.session_id for fact in selected for span in fact.source_spans
            )
        )

        gaps: List[str] = []
        if not selected:
            gaps.append(f"no grounded facts for request {request.request_id}")
            status = SupportStatus.UNKNOWN
        elif request.kind == "population":
            gaps.append("population coverage is not independently certified")
            status = SupportStatus.PARTIAL
        else:
            status = SupportStatus.SUPPORTED

        return GroundingResult(
            request_id=request.request_id,
            facts=selected,
            evidence=EvidenceBundle(
                fact_ids=[fact.fact_id for fact in selected],
                session_ids=source_sessions,
                gaps=gaps,
                status=status,
            ),
            relevance=relevance,
        )

    @staticmethod
    def _relevance(question: str, fact: FactRecord, indexed_ids: set) -> float:
        question_tokens = set(re.findall(r"[a-z0-9_-]{2,}", question.casefold()))
        fact_tokens = set(
            re.findall(
                r"[a-z0-9_-]{2,}",
                " ".join((fact.subject, fact.predicate, fact.object)).casefold(),
            )
        )
        if not question_tokens or not fact_tokens:
            return 0.0
        overlap = len(question_tokens & fact_tokens) / len(question_tokens)
        if fact.fact_id in indexed_ids:
            overlap = max(overlap, 0.25)
        return min(1.0, overlap)
