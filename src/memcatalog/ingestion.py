"""Session ingestion and query-driven fact acquisition."""

from __future__ import annotations

import hashlib
from typing import Callable, Iterable, List, Optional, Sequence

from .catalog import REVISION_CUES
from .facts import parse_triples
from .models import FactRecord, Session, SourceSpan, Turn
from .storage import JsonMemoryStore


FactExtractor = Callable[[str], str]


ACQUISITION_PROMPT = """\
Extract only source-backed facts needed for the requested evidence.
Keep speaker identity: assistant suggestions are not user facts unless the user confirms them.
Resolve relative time expressions against the session timestamp.
Return a JSON array of [subject, predicate, object] triples.

REQUIRED EVIDENCE:
{requirements}

SESSION TIMESTAMP: {timestamp}
SESSION:
{session}

JSON:
"""


class MemoryIngestor:
    def __init__(self, store: JsonMemoryStore, extractor: Optional[FactExtractor] = None) -> None:
        self.store = store
        self.extractor = extractor

    def append_session(
        self,
        user_id: str,
        session_id: str,
        timestamp: str,
        turns: Sequence[Turn],
        metadata: Optional[dict] = None,
    ) -> Session:
        if not user_id or not session_id or not timestamp:
            raise ValueError("user_id, session_id, and timestamp are required")
        if not turns:
            raise ValueError("a session must contain at least one turn")
        session = Session(
            session_id=session_id,
            user_id=user_id,
            timestamp=timestamp,
            turns=list(turns),
            closed=True,
            metadata=dict(metadata or {}),
        )
        self.store.append_session(session)
        return session

    def acquire(
        self,
        user_id: str,
        session_ids: Iterable[str],
        requirements: Sequence[str],
    ) -> List[FactRecord]:
        if self.extractor is None:
            return []
        acquired: List[FactRecord] = []
        requirement_text = "\n".join(f"- {item}" for item in requirements) or "- relevant facts"
        sessions = [
            session
            for session_id in dict.fromkeys(session_ids)
            for session in [self.store.session(user_id, session_id)]
            if session is not None
        ]
        sessions.sort(key=lambda item: (item.timestamp, item.session_id))
        for session in sessions:
            prompt = ACQUISITION_PROMPT.format(
                requirements=requirement_text,
                timestamp=session.timestamp,
                session=session.text[:12000],
            )
            raw = self.extractor(prompt)
            for subject, predicate, object_value in parse_triples(raw):
                fact = self._make_fact(session, subject, predicate, object_value)
                self._reconcile_revision(session, fact)
                self.store.put_fact(fact)
                acquired.append(fact)
        return acquired

    def _make_fact(
        self, session: Session, subject: str, predicate: str, object_value: str
    ) -> FactRecord:
        identity = "\x1f".join(
            (session.user_id, session.session_id, subject, predicate, object_value)
        )
        fact_id = "fact-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        quote = session.text[:1000]
        event_id = None
        if any(token in predicate.casefold() for token in ("organizer", "duration", "location", "date")):
            event_id = "event-" + hashlib.sha256(subject.casefold().encode("utf-8")).hexdigest()[:12]
        return FactRecord(
            fact_id=fact_id,
            user_id=session.user_id,
            subject=subject,
            predicate=predicate,
            object=object_value,
            source_spans=[SourceSpan(session.session_id, 0, len(quote), quote)],
            valid_from=session.timestamp,
            event_id=event_id,
        )

    def _reconcile_revision(self, session: Session, fact: FactRecord) -> None:
        if not REVISION_CUES.search(session.text):
            return
        candidates = [
            existing
            for existing in self.store.facts(session.user_id, include_superseded=False)
            if existing.key() == fact.key() and existing.object.casefold() != fact.object.casefold()
        ]
        if not candidates:
            return
        previous = max(candidates, key=lambda item: (item.valid_from or "", item.fact_id))
        previous.valid_to = session.timestamp
        fact.supersedes = previous.fact_id
        self.store.put_fact(previous)
