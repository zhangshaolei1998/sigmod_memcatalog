"""End-to-end memory query pipeline.

M0 base memory  = raw sessions
M1 catalog:
    - temporal index (chronological ordering)
    - entity-match tag
    - revision marker
    - fact-triple materialized view
    - hard revision enforcement
M2 memory-of-memory:
    - query log (per scope)
    - popularity index

Plan dispatch selects which M1/M2 layers to render per Q type:
    Q type              | matches | rev | facts | mom
    ----------------------+---------+-----+-------+-----
    temporal / open-ended | –       | –   | –     | –
    single-hop / user     | ✓       | –   | ✓     | ✓
    multi-hop / multi-sess| –       | ✓   | ✓     | ✓
    knowledge-update      | ✓       | ✓   | ✓ (hard-dedup) | ✓
    preference            | –       | –   | ✓     | –
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .matching import session_matches
from .heuristics import normalize_question_type, question_entities, revision_flags
from .facts import build_fact_table, enforce_revision, render_fact_table, flush_cache as flush_facts_cache
from .workload import record as record_mom, popularity_score, flush_log as flush_mom_log

# Q phrasing that asks about historical state — hard revision would answer wrong
_HISTORICAL_Q = re.compile(
    r"\b(initial|initially|first|originally|original|used to|previously|at first|before|earlier|old|former)\b",
    re.IGNORECASE,
)
# Q phrasing that asks about current state — hard revision is appropriate
_CURRENT_Q = re.compile(
    r"\b(now|currently|current|latest|recent|these days|nowadays|today|as of now)\b",
    re.IGNORECASE,
)
# Q that needs exhaustive coverage — facts (which are a summary) would miss items
_COUNT_Q = re.compile(
    r"\b(how many|number of|count of|total|all of|list all|which of the|multiple|each of)\b",
    re.IGNORECASE,
)


def _q_wants_current(question: str) -> bool:
    if _HISTORICAL_Q.search(question):
        return False
    return bool(_CURRENT_Q.search(question))


def _q_needs_coverage(question: str) -> bool:
    """Return True when the Q needs exhaustive session coverage; facts (a
    lossy summary) can miss items and produce wrong counts."""
    return bool(_COUNT_Q.search(question))


# Cost-based facts gate: only worthwhile on large haystacks where the
# fact-triple summary can concentrate scattered info. Small haystacks
# don't benefit — the raw sessions already fit easily in context.
FACTS_MIN_SESSIONS = 8


PLAN_TABLE: Dict[str, Dict[str, bool]] = {
    "temporal":               {"matches": False, "revision": False, "facts": False, "hard_rev": False, "mom": False},
    "temporal-reasoning":     {"matches": False, "revision": False, "facts": False, "hard_rev": False, "mom": False},

    "open-ended":             {"matches": False, "revision": False, "facts": False, "hard_rev": False, "mom": False},
    "adversarial":            {"matches": False, "revision": False, "facts": False, "hard_rev": False, "mom": False},

    "preference":                 {"matches": False, "revision": False, "facts": True,  "hard_rev": False, "mom": False},
    "single-session-preference":  {"matches": False, "revision": False, "facts": True,  "hard_rev": False, "mom": False},

    "single-hop":                 {"matches": True,  "revision": False, "facts": True,  "hard_rev": False, "mom": True},
    "single-session-user":        {"matches": True,  "revision": False, "facts": True,  "hard_rev": False, "mom": True},
    "single-session-assistant":   {"matches": True,  "revision": False, "facts": True,  "hard_rev": False, "mom": True},

    "knowledge-update":       {"matches": True,  "revision": True,  "facts": True,  "hard_rev": True,  "mom": True},

    "multi-hop":              {"matches": False, "revision": True,  "facts": True,  "hard_rev": False, "mom": True},
    "multi-session":          {"matches": False, "revision": True,  "facts": True,  "hard_rev": False, "mom": True},
}

DEFAULT_PLAN = {"matches": False, "revision": False, "facts": False, "hard_rev": False, "mom": False}


MEMCATALOG_PROMPT_HEAD = """\
You are answering a question about a user based on their conversation history.

QUESTION: {question}

"""

MEMCATALOG_PROMPT_CONTEXT_HDR_BASE = "CONVERSATION HISTORY (chronological"
MEMCATALOG_PROMPT_TAIL = """
Answer the question directly and concisely. Give ONLY the factual answer — no explanation, no hedging.
If the answer is a name, date, number, or yes/no, respond with JUST that.
If the question asks about a specific date, format it as "15 July 2023" and give a specific calendar date.{extra_rules}"""


def _build_prompt(question: str, facts_block: str, context: str,
                  used_matches: bool, used_revision: bool,
                  used_popular: bool, used_facts: bool) -> str:
    hdr_bits = []
    tail_bits = []
    if used_matches:
        hdr_bits.append("session headers may include [matches: ...] with Q keywords")
    if used_revision:
        hdr_bits.append("[REVISION] marking updates over earlier statements")
        tail_bits.append(
            "\nWhen a session is marked [REVISION], prefer its later statement over earlier statements about the same fact."
        )
    if used_popular:
        hdr_bits.append("[popular: N] indicating a session frequently useful in past similar queries")
    if used_facts:
        tail_bits.append(
            "\nThe FACTS block above is a helpful digest of the sessions; verify against the raw sessions before answering, especially for counts or lists."
        )

    hdr = MEMCATALOG_PROMPT_CONTEXT_HDR_BASE
    if hdr_bits:
        hdr = hdr + "; " + "; ".join(hdr_bits)
    hdr += "):"

    return (
        MEMCATALOG_PROMPT_HEAD.format(question=question)
        + facts_block
        + hdr + "\n" + context + "\n"
        + MEMCATALOG_PROMPT_TAIL.format(extra_rules="".join(tail_bits))
    )


@dataclass
class PromptPlan:
    q_type: str
    matches: bool
    revision: bool
    facts: bool
    hard_rev: bool
    mom: bool
    n_facts_rendered: int = 0
    n_facts_superseded: int = 0
    n_matches_sessions: int = 0
    n_popular_sessions: int = 0
    entities_used: List[str] = field(default_factory=list)


@dataclass
class PromptResult:
    prompt: str
    plan: PromptPlan
    scope: str
    matched_sessions: List[str] = field(default_factory=list)
    q_entities: List[str] = field(default_factory=list)


def build_memory_query(question: str,
                     sessions_text: List[str],
                     session_labels: List[str],
                     session_dates: List[Optional[str]],
                     generate_fn: Callable[[str], str],
                     scope: str,                       # conv-id or question-id
                     q_type_hint: Optional[str] = None,
                     today: Optional[str] = None) -> PromptResult:
    q_type = normalize_question_type(q_type_hint)
    if not (len(sessions_text) == len(session_labels) == len(session_dates)):
        raise ValueError("sessions_text, session_labels, and session_dates must have equal length")
    if not callable(generate_fn):
        raise TypeError("generate_fn must be callable")

    p = PLAN_TABLE.get(q_type, DEFAULT_PLAN)
    plan = PromptPlan(q_type=q_type, matches=p["matches"], revision=p["revision"],
                    facts=p["facts"], hard_rev=p["hard_rev"], mom=p["mom"])

    n = len(sessions_text)
    q_entities = sorted(question_entities(question))
    plan.entities_used = q_entities[:10]

    # M1: matches
    match_lists = session_matches(question, sessions_text) if plan.matches else [[] for _ in range(n)]
    plan.n_matches_sessions = sum(1 for m in match_lists if m)

    # M1: revision markers
    rev = revision_flags(sessions_text) if plan.revision else [False] * n

    # M2: popularity scores from Memory-of-Memory log
    if plan.mom:
        pop = popularity_score(scope, q_entities, q_type=q_type, min_overlap=1)
    else:
        pop = {}
    plan.n_popular_sessions = sum(1 for v in pop.values() if v > 0)

    # M1: fact triples
    facts_block = ""
    # Coverage-guard: exhaustive Qs (counting / list-all) shouldn't rely on a
    # summary view — force fall-back to raw sessions only.
    facts_disabled_by_coverage = _q_needs_coverage(question)
    # CBO gate: only build the fact-triple view when the haystack has enough
    # sessions to make the summarization worthwhile.
    facts_disabled_by_size = n < FACTS_MIN_SESSIONS
    if plan.facts and not facts_disabled_by_coverage and not facts_disabled_by_size:
        rows = build_fact_table(sessions_text, session_labels, session_dates, generate_fn)
        # Only apply hard-revision when Q phrasing actually asks about current state.
        # For historical Qs ("originally", "initially", "used to") keeping all versions
        # is required — otherwise MVCC dedup answers the wrong version.
        apply_hard_rev = plan.hard_rev and _q_wants_current(question)
        if apply_hard_rev:
            kept, superseded = enforce_revision(rows)
            plan.n_facts_superseded = len(superseded)
            fact_rows = kept
            fact_title = "FACTS (revision-enforced: only latest per (subject, predicate))"
        else:
            fact_rows = rows
            fact_title = "FACTS"
        # optional filter: keep triples whose subject/object match a Q entity
        if q_entities:
            q_low = {e.lower() for e in q_entities}
            def _match(r) -> bool:
                return any(e in r.subject.lower() or e in r.object.lower()
                           for e in q_low)
            filtered = [r for r in fact_rows if _match(r)]
            # fall back to full list if the filter over-prunes
            if len(filtered) >= 3:
                fact_rows = filtered
        block = render_fact_table(fact_rows, title=fact_title, max_rows=25)
        plan.n_facts_rendered = len(fact_rows)
        facts_block = (block + "\n\n") if block else ""

    # Order: chronological
    order = sorted(range(n), key=lambda i: session_dates[i] or "")
    matched_labels: List[str] = []
    parts: List[str] = []
    for i in order:
        d = session_dates[i]
        tags = []
        if d:
            tags.append(f"[{d}]")
        if match_lists[i]:
            tags.append(f"[matches: {', '.join(match_lists[i])}]")
            matched_labels.append(session_labels[i])
        if rev[i]:
            tags.append("[REVISION]")
        popc = pop.get(session_labels[i], 0)
        if popc > 0:
            tags.append(f"[popular: {popc}]")
        header_tags = " " + " ".join(tags) if tags else ""
        parts.append(f"=== {session_labels[i]}{header_tags} ===\n{sessions_text[i]}")
    context = "\n\n".join(parts)

    q_with_today = f"[Today: {today}]\n{question}" if today else question
    used_matches = any(match_lists)
    used_revision = any(rev)
    used_popular = plan.n_popular_sessions > 0
    used_facts = bool(facts_block)
    prompt = _build_prompt(
        q_with_today, facts_block, context,
        used_matches=used_matches, used_revision=used_revision,
        used_popular=used_popular, used_facts=used_facts,
    )

    return PromptResult(prompt=prompt, plan=plan, scope=scope,
                            matched_sessions=matched_labels, q_entities=q_entities)


def commit_query(result: PromptResult) -> None:
    """Record the query into Memory-of-Memory log for future retrievals."""
    if not result.plan.mom:
        return
    record_mom(result.scope, result.plan.q_type,
               result.q_entities, result.matched_sessions)


def flush_stores() -> None:
    flush_facts_cache()
    flush_mom_log()
