"""Thread-safe JSON persistence for all three memory levels."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import ExecutionRecord, FactRecord, Session


def _empty_state() -> Dict[str, Any]:
    return {
        "sessions": {},
        "facts": {},
        "executions": {},
        "fact_stats": {},
        "action_stats": [],
    }


class JsonMemoryStore:
    """Persistent, user-isolated memory state with atomic file replacement."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return _empty_state()
        with self.path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        state = _empty_state()
        state.update(loaded)
        return state

    def flush(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self.path.name}.", dir=str(self.path.parent), text=True
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(self._state, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            except Exception:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise

    def append_session(self, session: Session) -> None:
        if not session.closed:
            raise ValueError("only closed sessions can be appended to memory")
        with self._lock:
            bucket = self._state["sessions"].setdefault(session.user_id, {})
            existing = bucket.get(session.session_id)
            payload = session.to_dict()
            if existing and existing != payload:
                raise ValueError(f"session ID already exists: {session.session_id}")
            bucket[session.session_id] = payload

    def sessions(self, user_id: str, snapshot_at: Optional[str] = None) -> List[Session]:
        with self._lock:
            payloads = list(self._state["sessions"].get(user_id, {}).values())
        sessions = [Session.from_dict(payload) for payload in payloads]
        if snapshot_at is not None:
            sessions = [session for session in sessions if session.timestamp <= snapshot_at]
        return sorted(sessions, key=lambda session: (session.timestamp, session.session_id))

    def session(self, user_id: str, session_id: str) -> Optional[Session]:
        with self._lock:
            payload = self._state["sessions"].get(user_id, {}).get(session_id)
        return Session.from_dict(payload) if payload else None

    def put_fact(self, fact: FactRecord) -> None:
        with self._lock:
            bucket = self._state["facts"].setdefault(fact.user_id, {})
            bucket[fact.fact_id] = fact.to_dict()

    def facts(
        self,
        user_id: str,
        as_of: Optional[str] = None,
        include_superseded: bool = True,
    ) -> List[FactRecord]:
        with self._lock:
            payloads = list(self._state["facts"].get(user_id, {}).values())
        facts = [FactRecord.from_dict(payload) for payload in payloads]
        if not include_superseded:
            facts = [fact for fact in facts if fact.is_valid_at(as_of)]
        elif as_of is not None:
            facts = [fact for fact in facts if fact.is_valid_at(as_of)]
        return sorted(facts, key=lambda fact: (fact.valid_from or "", fact.fact_id))

    def fact(self, user_id: str, fact_id: str) -> Optional[FactRecord]:
        with self._lock:
            payload = self._state["facts"].get(user_id, {}).get(fact_id)
        return FactRecord.from_dict(payload) if payload else None

    def record_execution(self, record: ExecutionRecord) -> None:
        with self._lock:
            bucket = self._state["executions"].setdefault(record.user_id, {})
            if record.query_id in bucket:
                raise ValueError(f"query ID already exists: {record.query_id}")
            bucket[record.query_id] = record.to_dict()
            for action in record.actions:
                action_payload = action.to_dict()
                action_payload["user_id"] = record.user_id
                action_payload["plan_class"] = record.plan.get("plan_class", "")
                self._state["action_stats"].append(action_payload)

    def execution(self, user_id: str, query_id: str) -> Optional[ExecutionRecord]:
        with self._lock:
            payload = self._state["executions"].get(user_id, {}).get(query_id)
        return ExecutionRecord.from_dict(payload) if payload else None

    def executions(self, user_id: str) -> List[ExecutionRecord]:
        with self._lock:
            payloads = list(self._state["executions"].get(user_id, {}).values())
        return [ExecutionRecord.from_dict(payload) for payload in payloads]

    def report_use(
        self,
        user_id: str,
        query_id: str,
        used_fact_ids: Iterable[str],
        complete: bool,
    ) -> ExecutionRecord:
        with self._lock:
            payload = self._state["executions"].get(user_id, {}).get(query_id)
            if payload is None:
                raise KeyError(f"unknown query ID: {query_id}")
            record = ExecutionRecord.from_dict(payload)
            used = list(dict.fromkeys(used_fact_ids))
            if not set(used).issubset(record.returned_fact_ids):
                raise ValueError("used facts must be a subset of returned facts")
            if record.complete_feedback:
                if record.used_fact_ids != used or not complete:
                    raise ValueError("a completed use report is immutable")
                return record
            if complete:
                for fact_id in record.returned_fact_ids:
                    key = self._stat_key(user_id, record.context_key, fact_id)
                    stats = self._state["fact_stats"].setdefault(
                        key, {"exposures": 0, "uses": 0}
                    )
                    stats["exposures"] += 1
                    if fact_id in used:
                        stats["uses"] += 1
            record.complete_feedback = complete
            record.used_fact_ids = used
            self._state["executions"][user_id][query_id] = record.to_dict()
            return record

    def fact_stats(self, user_id: str, context_key: str, fact_id: str) -> Dict[str, int]:
        key = self._stat_key(user_id, context_key, fact_id)
        with self._lock:
            stats = self._state["fact_stats"].get(key, {"exposures": 0, "uses": 0})
            return dict(stats)

    def action_history(
        self, user_id: str, kind: Optional[str] = None, plan_class: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [
                dict(row)
                for row in self._state["action_stats"]
                if row.get("user_id") == user_id
            ]
        if kind is not None:
            rows = [row for row in rows if row.get("kind") == kind]
        if plan_class is not None:
            rows = [row for row in rows if row.get("plan_class") == plan_class]
        return rows

    def summary(self, user_id: str) -> Dict[str, int]:
        return {
            "sessions": len(self.sessions(user_id)),
            "facts": len(self.facts(user_id)),
            "executions": len(self.executions(user_id)),
        }

    @staticmethod
    def _stat_key(user_id: str, context_key: str, fact_id: str) -> str:
        return "\x1f".join((user_id, context_key, fact_id))
