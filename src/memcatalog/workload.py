"""Memory-of-Memory: a persistent query log that adapts future plans.

DBMS analogy:
    pg_stat_user_indexes + pg_stat_all_tables
    → adaptive query optimization based on usage patterns.

Design (gold-free — no F1/label peeking):
  - For each Q processed, log q_type, Q entities, and which sessions
    were matched (by entity overlap or [matches:] hits).
  - For a new Q, retrieve past Qs sharing ≥1 entity. Aggregate the
    session-usage counts from those "similar" past Qs.
  - Boost sessions with high usage count — reflected by prepending a
    "[popular: N]" tag to their header, giving the LLM a soft prior.

This is *not* self-supervised training. It's usage-driven ordering,
mirroring how Postgres reorders index preference by `pg_stat_all_indexes.idx_scan`.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


_LOG_LOCK = threading.Lock()
_LOG_PATH = Path(os.environ.get("MEMCAT_QUERY_LOG", "results/query_log.json"))


# ============================================================
# Log persistence
# ============================================================

def _load_log() -> Dict[str, List[Dict[str, Any]]]:
    try:
        with _LOG_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_log(log: Dict[str, List[Dict[str, Any]]]) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False)
    except Exception:
        pass


# Cache of the log grouped by scope
_LOG_CACHE: Dict[str, Any] = {"data": None}


def _get_log() -> Dict[str, List[Dict[str, Any]]]:
    with _LOG_LOCK:
        if _LOG_CACHE["data"] is None:
            _LOG_CACHE["data"] = _load_log()
        return _LOG_CACHE["data"]


# ============================================================
# API
# ============================================================

@dataclass
class LogEntry:
    scope: str                    # caller-defined conversation/user partition
    q_type: str
    q_entities: List[str]         # lower-cased entity strings
    matched_sessions: List[str]   # session_label list where any Q entity appeared


def record(scope: str, q_type: str, q_entities: List[str],
           matched_sessions: List[str]) -> None:
    log = _get_log()
    with _LOG_LOCK:
        log.setdefault(scope, []).append({
            "q_type": q_type,
            "q_entities": [e.lower() for e in q_entities],
            "matched_sessions": list(matched_sessions),
        })


def flush_log() -> None:
    with _LOG_LOCK:
        if _LOG_CACHE["data"] is not None:
            _save_log(_LOG_CACHE["data"])


def popularity_score(scope: str, q_entities: List[str], q_type: Optional[str] = None,
                     min_overlap: int = 1) -> Dict[str, int]:
    """Return a session_label → usage_count map, based on past Q's within
    the same scope whose q_entities overlap by at least `min_overlap` with
    the current Q's entities.

    Entries from other scopes never contribute to the score.
    """
    log = _get_log()
    entries = log.get(scope, [])
    if not entries:
        return {}
    q_set = {e.lower() for e in q_entities if e}
    counts: Counter = Counter()
    for e in entries:
        past_ents = set(e.get("q_entities", []))
        if not q_set:
            overlap = 0
        else:
            overlap = len(q_set & past_ents)
        if overlap < min_overlap:
            continue
        # also give a small boost for same q_type match
        weight = 1
        if q_type and e.get("q_type") == q_type:
            weight = 2
        for s in e.get("matched_sessions", []):
            counts[s] += weight
    return dict(counts)
