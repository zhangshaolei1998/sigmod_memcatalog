"""Typed catalog views over conversation sessions.

Each catalog is materialized as a compact, human-readable table that a model
can consult while answering a memory question.

    Cat-T (temporal index) : session_id  → date
    Cat-E (entity index)   : query_entity → [sessions where it appears]
    Cat-R (revision log)   : session → detected "supersession" markers

Together these views form the catalog layer of the memory system.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ============================================================
# Lightweight entity extraction (regex only — reproducible)
# ============================================================

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "have", "has", "had", "did",
    "do", "does", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "as", "into", "through", "and", "or", "but", "not", "no", "yes", "you",
    "me", "my", "your", "our", "their", "they", "we", "he", "she", "it", "his",
    "her", "its", "this", "that", "these", "those", "there", "here", "what",
    "when", "where", "who", "which", "how", "why", "if", "will", "would",
    "can", "could", "should", "may", "might", "shall", "am", "be", "been",
    "being", "than", "then", "so", "such", "also",
}


_MONTHS = {"january","february","march","april","may","june","july","august",
           "september","october","november","december",
           "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec"}


def extract_entities(text: str) -> Set[str]:
    """Cheap NER: capitalized multi-word phrases, dates, quoted phrases,
    and any all-caps token >=2 chars. Lowercased for indexing."""
    out: Set[str] = set()
    if not text:
        return out
    # Multi-word capitalized (proper nouns): "New York", "Sarah Palmer"
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text):
        phrase = m.group(1).strip()
        if phrase.lower() in STOP_WORDS:
            continue
        out.add(phrase.lower())
    # ISO / numeric dates: 2023-04-15, 15/04/2023
    for m in re.finditer(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", text):
        out.add(m.group(1))
    # Verbal dates: "15 July 2023" / "July 2023"
    for m in re.finditer(r"\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{4})\b", text):
        out.add(m.group(1).lower())
    # Quoted strings
    for m in re.finditer(r'"([^"]{2,40})"', text):
        out.add(m.group(1).lower())
    # Cardinal numbers >= 2 digits
    for m in re.finditer(r"\b(\d{2,})\b", text):
        out.add(m.group(1))
    # Drop bare month names / years - too broad
    return {e for e in out if e not in _MONTHS and len(e) >= 2}


# ============================================================
# Cat-T Temporal Catalog
# ============================================================

def render_temporal_catalog(session_labels: List[str],
                            session_dates: List[Optional[str]]) -> str:
    valid = [(l, d) for l, d in zip(session_labels, session_dates) if d]
    if not valid:
        return ""
    lines = [f"  {l}: {d}" for l, d in valid]
    return "=== Cat-T Temporal Catalog (session → date) ===\n" + "\n".join(lines)


# ============================================================
# Cat-E Entity Catalog (inverted index over Q entities)
# ============================================================

def render_entity_catalog(question: str,
                          sessions_text: List[str],
                          session_labels: List[str]) -> str:
    q_ents = extract_entities(question)
    if not q_ents:
        return ""
    # Also treat lowercased content tokens > 4 chars from question as entities
    q_tokens = {t for t in re.findall(r"[a-z]{5,}", question.lower())
                if t not in STOP_WORDS}
    q_ents = q_ents | q_tokens

    coverage: Dict[str, List[str]] = {}
    for ent in sorted(q_ents):
        needle = ent.lower()
        hit = []
        for i, txt in enumerate(sessions_text):
            if needle in txt.lower():
                hit.append(session_labels[i])
        if hit:
            coverage[ent] = hit
    if not coverage:
        return ""
    lines = [f"  {ent!r} → {', '.join(sess)}" for ent, sess in coverage.items()]
    return ("=== Cat-E Entity Catalog (Q entity → sessions) ===\n"
            + "\n".join(lines))


# ============================================================
# Cat-R Revision Catalog (supersession markers)
# ============================================================

REVISION_CUES = re.compile(
    r"\b(actually|changed my mind|instead|now|no longer|updated|revised|"
    r"correction|but wait|scratch that|nope,?\s*(?:actually)?|"
    r"i (?:meant|misspoke)|new plan)\b",
    re.IGNORECASE,
)


def render_revision_catalog(sessions_text: List[str],
                            session_labels: List[str],
                            session_dates: List[Optional[str]]) -> str:
    hits = []
    for i, txt in enumerate(sessions_text):
        matches = REVISION_CUES.findall(txt)
        if not matches:
            continue
        cues = sorted(set(m.lower() if isinstance(m, str) else m[0].lower()
                          for m in matches))
        d = session_dates[i] if i < len(session_dates) else None
        date_tag = f" [{d}]" if d else ""
        hits.append(f"  {session_labels[i]}{date_tag}: markers = {', '.join(cues)}")
    if not hits:
        return ""
    return ("=== Cat-R Revision Catalog (potential fact supersession) ===\n"
            "The following sessions contain revision cues — treat their "
            "content as OVERRIDING earlier statements about the same fact.\n"
            + "\n".join(hits))


# ============================================================
# Prompt assembly
# ============================================================

MEMCATALOG_PROMPT = """\
You are answering a question about a user based on their conversation history.
The conversation memory is exposed via a DBMS-style catalog: consult the catalog
tables below to locate evidence, then produce the answer.

{catalogs}
QUESTION: {question}

CONVERSATION HISTORY (materialized view, sessions in chronological order):
{context}

Answer the question directly and concisely. Give ONLY the factual answer — no explanation, no hedging.
If the answer is a name, date, number, or yes/no, respond with JUST that.
If the question asks about a specific date, format it as "15 July 2023" and give a specific calendar date.
If the Revision Catalog flags a session, prefer that session's later statement over earlier ones."""


@dataclass
class MemCatalogResult:
    prompt: str
    active_layers: List[str]
    entity_coverage: Dict[str, List[str]] = field(default_factory=dict)


def build_memcatalog(question: str,
                     sessions_text: List[str],
                     session_labels: List[str],
                     session_dates: List[Optional[str]],
                     enable_T: bool = True,
                     enable_E: bool = True,
                     enable_R: bool = True,
                     today: Optional[str] = None) -> MemCatalogResult:
    """Assemble a prompt with the requested catalog layers."""
    catalogs = []
    active: List[str] = []
    if enable_T:
        t = render_temporal_catalog(session_labels, session_dates)
        if t:
            catalogs.append(t); active.append("T")
    if enable_E:
        e = render_entity_catalog(question, sessions_text, session_labels)
        if e:
            catalogs.append(e); active.append("E")
    if enable_R:
        r = render_revision_catalog(sessions_text, session_labels, session_dates)
        if r:
            catalogs.append(r); active.append("R")
    catalog_block = ("\n\n".join(catalogs) + "\n\n") if catalogs else ""

    # Materialized view: sessions in chronological order
    context_parts = []
    order = sorted(range(len(sessions_text)),
                   key=lambda i: session_dates[i] or "")
    for i in order:
        d = session_dates[i]
        header = f"=== {session_labels[i]}" + (f" [{d}]" if d else "") + " ==="
        context_parts.append(f"{header}\n{sessions_text[i]}")
    context = "\n\n".join(context_parts)

    q_with_today = f"[Today: {today}]\n{question}" if today else question
    prompt = MEMCATALOG_PROMPT.format(
        catalogs=catalog_block, question=q_with_today, context=context,
    )
    return MemCatalogResult(prompt=prompt, active_layers=active)
