"""Deterministic result operators over grounded evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .grounding import GroundingResult
from .models import EvidenceBundle, QueryPlan, SupportStatus


@dataclass
class OperatorResult:
    value: Any
    evidence: EvidenceBundle
    status: SupportStatus


def apply_operator(plan: QueryPlan, grounded: Iterable[GroundingResult]) -> OperatorResult:
    results = list(grounded)
    evidence = EvidenceBundle(status=SupportStatus.SUPPORTED)
    for result in results:
        evidence = evidence.merge(result.evidence)

    if plan.result_operator == "project":
        values = [fact.object for result in results for fact in result.facts]
        value: Any = values[0] if len(values) == 1 else values
    elif plan.result_operator == "count":
        positives = sum(len(result.facts) for result in results)
        unresolved = sum(1 for result in results if result.evidence.gaps)
        value = positives if unresolved == 0 else {"lower": positives, "upper": positives + unresolved}
    elif plan.result_operator == "compare":
        value = _compare(results)
    elif plan.result_operator == "bind-and-project":
        value = _dependent_projection(results)
    else:
        value = None
        evidence.gaps.append(f"unsupported result operator: {plan.result_operator}")
        evidence.status = SupportStatus.UNKNOWN

    status = evidence.status
    if value in (None, [], {}):
        status = SupportStatus.UNKNOWN if not evidence.fact_ids else SupportStatus.PARTIAL
    return OperatorResult(value=value, evidence=evidence, status=status)


def _compare(results: List[GroundingResult]) -> Dict[str, Any]:
    candidates = []
    for result in results:
        for fact in result.facts:
            number = _first_number(fact.object)
            if number is not None:
                candidates.append((number, fact.subject, fact.fact_id))
    if not candidates:
        return {}
    best = max(number for number, _, _ in candidates)
    return {
        "value": best,
        "winners": sorted({subject for number, subject, _ in candidates if number == best}),
        "support": [fact_id for number, _, fact_id in candidates if number == best],
    }


def _dependent_projection(results: List[GroundingResult]) -> Any:
    if len(results) < 2 or not results[-1].facts:
        return None
    values = [fact.object for fact in results[-1].facts]
    return values[0] if len(values) == 1 else values


def _first_number(value: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else None
