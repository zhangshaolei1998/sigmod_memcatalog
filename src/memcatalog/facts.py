"""Fact triple extraction — materialized view over raw turns.

DBMS analogy:
    CREATE MATERIALIZED VIEW facts AS
        SELECT session_id, session_date, subject, predicate, object
        FROM llm_extract_triples(session);

Cost: one model call per unique session, memoized by content hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Persistent triple cache
# ============================================================

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, List[List[str]]] = {}
_CACHE_PATH = Path(os.environ.get("MEMCAT_FACTS_CACHE", "results/facts_cache.json"))


def _load_cache() -> None:
    global _CACHE
    try:
        with _CACHE_PATH.open(encoding="utf-8") as f:
            _CACHE = json.load(f)
    except Exception:
        _CACHE = {}


def _save_cache_now() -> None:
    try:
        with _CACHE_LOCK:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _CACHE_PATH.open("w", encoding="utf-8") as f:
                json.dump(_CACHE, f, ensure_ascii=False)
    except Exception:
        pass


_load_cache()


def _content_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ============================================================
# Triple extraction
# ============================================================

FACT_EXTRACTION_PROMPT = """\
Extract factual claims stated in this conversation session as (subject, predicate, object) triples.
Focus on facts the speakers state about themselves, their preferences, activities, relationships, and life events.
Skip small-talk, questions, and assistant advice that isn't a factual claim.

CRITICAL: Resolve every relative date/time expression ("last year", "yesterday", "two weeks ago", "next month", "in a few days") to an absolute calendar date using the session date shown as "[Session date: ...]" at the top of the session. Never leave a relative time expression in the extracted triples.

Return a JSON array. Each element: ["subject", "predicate", "object"].
Keep each triple to <= 8 words per field. Aim for 3-10 triples per session.

SESSION:
{session_text}

JSON:"""


def parse_triples(raw: str) -> List[Tuple[str, str, str]]:
    """Parse LLM output into a list of (s, p, o) triples. Tolerant to
    prose before/after the JSON array and truncated output."""
    if not raw:
        return []
    # find the outermost [...] block
    start = raw.find("[")
    if start < 0:
        return []
    # take from '[' to the last ']' if present; otherwise attempt repair
    end = raw.rfind("]")
    if end < 0:
        # try to auto-close truncated JSON
        payload = raw[start:]
        # ensure balanced brackets
        payload = payload.rstrip().rstrip(",")
        payload = payload + "]"
    else:
        payload = raw[start:end + 1]
    try:
        data = json.loads(payload)
    except Exception:
        # coarse regex fallback: (?:\["?)([^"]+)("?,\s*"?)...
        triples = []
        for m in re.finditer(r'\[\s*"([^"]{1,80})"\s*,\s*"([^"]{1,80})"\s*,\s*"([^"]{1,120})"\s*\]', raw):
            triples.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
        return triples
    triples: List[Tuple[str, str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list) and len(item) >= 3:
                s, p, o = str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()
                if s and p and o and len(s) < 200 and len(p) < 200 and len(o) < 400:
                    triples.append((s, p, o))
    return triples


def extract_triples(session_text: str, generate_fn) -> List[Tuple[str, str, str]]:
    """Return memoized (s, p, o) triples for one session."""
    if not session_text or not session_text.strip():
        return []
    key = _content_key(session_text)
    with _CACHE_LOCK:
        if key in _CACHE:
            return [tuple(t) for t in _CACHE[key]]
    try:
        raw = generate_fn(FACT_EXTRACTION_PROMPT.format(session_text=session_text[:6000]))
    except Exception:
        raw = ""
    triples = parse_triples(raw)
    with _CACHE_LOCK:
        _CACHE[key] = [list(t) for t in triples]
    return triples


def flush_cache() -> None:
    _save_cache_now()


# ============================================================
# Fact-table rendering
# ============================================================

@dataclass
class FactRow:
    subject: str
    predicate: str
    object: str
    session_label: str
    session_date: Optional[str]

    def key(self) -> Tuple[str, str]:
        return (self.subject.lower().strip(), self.predicate.lower().strip())


def build_fact_table(sessions_text: List[str],
                     session_labels: List[str],
                     session_dates: List[Optional[str]],
                     generate_fn) -> List[FactRow]:
    rows: List[FactRow] = []
    for i, txt in enumerate(sessions_text):
        triples = extract_triples(txt, generate_fn)
        d = session_dates[i] if i < len(session_dates) else None
        lbl = session_labels[i] if i < len(session_labels) else f"S{i+1}"
        for s, p, o in triples:
            rows.append(FactRow(subject=s, predicate=p, object=o,
                                session_label=lbl, session_date=d))
    return rows


def enforce_revision(rows: List[FactRow]) -> Tuple[List[FactRow], List[FactRow]]:
    """MVCC-style dedup: keep only the latest row per (subject, predicate).
    Returns (kept, superseded)."""
    by_key: Dict[Tuple[str, str], List[FactRow]] = {}
    for r in rows:
        by_key.setdefault(r.key(), []).append(r)
    kept: List[FactRow] = []
    superseded: List[FactRow] = []
    for _, group in by_key.items():
        group.sort(key=lambda r: r.session_date or "")
        kept.append(group[-1])
        superseded.extend(group[:-1])
    return kept, superseded


def render_fact_table(rows: List[FactRow], title: str = "FACTS",
                      max_rows: int = 60) -> str:
    if not rows:
        return ""
    sel = rows[:max_rows]
    lines = [f"=== {title} (materialized fact-triple view) ==="]
    for r in sel:
        date_tag = f" [{r.session_date}]" if r.session_date else ""
        lines.append(f"  ({r.subject}, {r.predicate}, {r.object}) @ {r.session_label}{date_tag}")
    if len(rows) > max_rows:
        lines.append(f"  ...(+{len(rows) - max_rows} more)")
    return "\n".join(lines)
