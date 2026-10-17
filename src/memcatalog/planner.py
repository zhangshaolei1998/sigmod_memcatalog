"""Query-class planner for multiple-choice memory questions.

The planner routes a multiple-choice question to one of three execution
classes using only the question text and the shape of its answer choices:

``POINT``
    Resolve one stored fact directly.
``AGGREGATE``
    Enumerate relevant items and calculate a count or amount.
``COMPARE``
    Gather candidate evidence and compare the candidates explicitly.

The module has no model, dataset, or benchmark dependency.
"""
from __future__ import annotations

import re
from typing import List, Optional

POINT = "POINT"
AGGREGATE = "AGGREGATE"
COMPARE = "COMPARE"


# --- Question-side cues -----------------------------------------------------

# Explicit counting / quantity language.
_AGG_Q = re.compile(
    r"\b(how many|how much|how often|number of|count of|total|sum of|"
    r"combined|altogether|in total|on average|average)\b",
    re.IGNORECASE,
)

# Comparative / superlative / ranking language: needs a scan over candidates.
_CMP_Q = re.compile(
    r"\b(which\s+\w+\s+(?:is|was|has|had|does|did)\s+(?:the\s+)?(?:most|least|"
    r"more|fewer|less|greater|higher|lower|best|worst|earliest|latest)|"
    r"most|least|highest|lowest|largest|smallest|longest|shortest|"
    r"earliest|latest|best|worst|more than|less than|fewer than|"
    r"compared? (?:to|with)|than any|rank|order them)\b",
    re.IGNORECASE,
)

# Superlative adjectives ending in -est are a strong ranking cue.
_EST = re.compile(r"\b\w{3,}est\b", re.IGNORECASE)

_EST_STOP = {
    "best", "worst", "latest", "earliest",  # already covered above
    "interest", "rest", "test", "guest", "request", "suggest", "invest",
    "honest", "protest", "forest", "nearest", "modest", "harvest",
}

# --- Binary comparison ------------------------------------------------------
# The most common comparison form is "Who is taller, X or Y?" /
# "Which event lasts longer, A or B?" — a gradable predicate plus two
# alternatives joined by "or". This is ordinary English structure, not a
# property of any one benchmark, so detecting it transfers.

_INTERROGATIVE = re.compile(r"\b(who|which|what|whose)\b", re.IGNORECASE)

# Irregular comparatives plus common gradable -er adjectives/adverbs.
_COMPARATIVE_WORDS = {
    "more", "less", "fewer", "better", "worse", "further", "farther",
    "taller", "shorter", "longer", "older", "younger", "larger", "smaller",
    "bigger", "higher", "lower", "earlier", "later", "faster", "slower",
    "greater", "heavier", "lighter", "closer", "nearer", "cheaper",
    "stronger", "weaker", "wider", "narrower", "deeper", "shallower",
    "richer", "poorer", "warmer", "cooler", "hotter", "colder", "newer",
    "busier", "quieter", "louder", "harder", "easier", "softer", "denser",
}

# Tokens that end in -er but are not comparatives.
_ER_STOP = {
    "other", "another", "over", "after", "never", "ever", "whether",
    "together", "member", "order", "water", "paper", "number", "under",
    "either", "neither", "computer", "user", "owner", "answer", "consider",
    "however", "whatever", "whenever", "wherever", "her", "per", "server",
    "quarter", "character", "partner", "matter", "letter", "center",
    "manager", "teacher", "speaker", "leader", "provider", "customer",
    "summer", "winter", "father", "mother", "brother", "sister", "daughter",
    "offer", "differ", "cover", "power", "lower_", "trigger", "counter",
    "enter", "master", "chapter", "filter", "folder", "header", "footer",
    "border", "corner", "dinner", "banner", "poster", "roster", "shelter",
}

_ER_TOKEN = re.compile(r"\b([a-z]{4,}er)\b", re.IGNORECASE)


def _has_comparative_word(question: str) -> bool:
    q = (question or "").lower()
    toks = set(re.findall(r"[a-z]+", q))
    if toks & _COMPARATIVE_WORDS:
        return True
    for m in _ER_TOKEN.finditer(q):
        w = m.group(1).lower()
        if w not in _ER_STOP and w not in _COMPARATIVE_WORDS:
            # Unknown -er token in an interrogative with alternatives: treat
            # it as a comparative rather than miss the class.
            return True
    return False


def _is_binary_comparison(question: str) -> bool:
    """True for 'Who is taller, X or Y?' style two-candidate comparisons."""
    q = question or ""
    if not _INTERROGATIVE.search(q):
        return False
    if not re.search(r"\bor\b", q, re.IGNORECASE):
        return False
    return _has_comparative_word(q)


# --- Choice-side cues -------------------------------------------------------

_NUMERIC_CHOICE = re.compile(r"^\W*(\d[\d,\.]*)\s*\w*\W*$")


def _choice_texts(choices: object) -> List[str]:
    """Normalize a choices container into a list of option strings."""
    if not choices:
        return []
    if isinstance(choices, dict):
        vals = [choices[k] for k in sorted(choices)]
    else:
        vals = list(choices)
    out = []
    for v in vals:
        s = str(v).strip()
        # Strip a leading "A." / "B)" label if present.
        s = re.sub(r"^[A-D][\.\)]\s*", "", s)
        if s:
            out.append(s)
    return out


def choices_are_quantities(choices: object) -> bool:
    """True when most options are bare numbers/quantities.

    A numeric answer set means the question resolves to a count or an
    amount, regardless of how the question is phrased.
    """
    texts = _choice_texts(choices)
    if len(texts) < 2:
        return False
    numeric = sum(1 for t in texts if _NUMERIC_CHOICE.match(t))
    return numeric >= max(2, int(0.75 * len(texts)))


def _has_superlative(question: str) -> bool:
    for m in _EST.finditer(question or ""):
        if m.group(0).lower() not in _EST_STOP:
            return True
    return False


def classify_query(question: str, choices: object = None) -> str:
    """Classify an MC question into POINT / AGGREGATE / COMPARE.

    AGGREGATE — resolves to a count or amount; needs a full scan plus a
                running tally.
    COMPARE   — needs the candidate set enumerated and ranked against each
                other before one can be picked.
    POINT     — resolves to a single stored fact; execute directly.
    """
    q = question or ""
    if _AGG_Q.search(q) or choices_are_quantities(choices):
        return AGGREGATE
    if _is_binary_comparison(q) or _CMP_Q.search(q) or _has_superlative(q):
        return COMPARE
    return POINT


# --- Plan prompts -----------------------------------------------------------

AGGREGATE_TAIL = """

Work through this as a counting problem:
1. Scan the history and list EVERY item relevant to the question.
2. For each item, state briefly whether it satisfies the condition.
3. Tally the items that do.
4. On the LAST LINE output only: "Answer: X" (X being A, B, C or D)."""

COMPARE_TAIL = """

Work through this as a comparison:
1. Identify the candidates the question is comparing.
2. For each candidate, pull the specific evidence from the history that
   bears on the comparison (dates, amounts, frequencies, or attributes).
3. Compare them directly on that dimension and say which one wins.
4. On the LAST LINE output only: "Answer: X" (X being A, B, C or D)."""


def build_plan_prompt(base_prompt: str, qclass: str) -> Optional[str]:
    """Return the plan-specific prompt, or ``None`` for direct POINT lookup.

    Raises:
        ValueError: If ``qclass`` is not one of the three public classes.
    """
    if qclass == AGGREGATE:
        return base_prompt.rstrip() + AGGREGATE_TAIL
    if qclass == COMPARE:
        return base_prompt.rstrip() + COMPARE_TAIL
    if qclass == POINT:
        return None
    raise ValueError(f"unknown query class: {qclass!r}")


_ANSWER_LINE = re.compile(r"Answer\s*[:：]\s*([ABCD])", re.IGNORECASE)


def extract_plan_letter(output: str) -> Optional[str]:
    """Pull the final letter out of a plan-CoT response."""
    if not output:
        return None
    s = str(output)
    hits = _ANSWER_LINE.findall(s)
    if hits:
        return hits[-1].upper()  # The final answer line is the conclusion.
    m = re.search(r"\\boxed\{\s*([ABCD])\s*\}", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None
