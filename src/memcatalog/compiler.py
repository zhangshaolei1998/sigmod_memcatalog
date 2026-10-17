"""Compile natural-language memory questions into explicit query plans."""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence

from .heuristics import question_entities
from .models import Binding, PlanClass, QueryPlan, Request
from .planner import AGGREGATE, COMPARE, POINT, classify_query


_DEPENDENT = re.compile(
    r"\b(who|what|where)\b.*\b(longest|shortest|earliest|latest|most|least|winner)\b",
    re.IGNORECASE,
)


class PlanCompiler:
    """Deterministic compiler with an explicit, validated plan graph."""

    def compile(
        self,
        question: str,
        choices: Optional[Sequence[str]] = None,
        snapshot_at: Optional[str] = None,
    ) -> QueryPlan:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        query_class = classify_query(question, choices)
        entities = sorted(question_entities(question))
        plan_id = "plan-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]

        if _DEPENDENT.search(question):
            requests = [
                Request(
                    request_id="select-candidate",
                    kind="population",
                    target="candidate",
                    scope={"entities": entities, "question": question},
                ),
                Request(
                    request_id="lookup-property",
                    kind="lookup",
                    target="bound-candidate",
                    scope={"question": question},
                    slots={"entity": None},
                ),
            ]
            plan = QueryPlan(
                plan_id=plan_id,
                plan_class=PlanClass.DEPENDENT,
                requests=requests,
                bindings=[Binding("select-candidate", "lookup-property", "entity")],
                result_operator="bind-and-project",
                snapshot_at=snapshot_at,
                metadata={"query_class": query_class, "question": question},
            )
        elif query_class == POINT:
            plan = QueryPlan(
                plan_id=plan_id,
                plan_class=PlanClass.RETRIEVE,
                requests=[
                    Request(
                        request_id="lookup",
                        kind="lookup",
                        target=entities[0] if entities else "fact",
                        scope={"entities": entities, "question": question},
                    )
                ],
                bindings=[],
                result_operator="project",
                snapshot_at=snapshot_at,
                metadata={"query_class": query_class, "question": question},
            )
        else:
            operator = "count" if query_class == AGGREGATE else "compare"
            plan = QueryPlan(
                plan_id=plan_id,
                plan_class=PlanClass.INDEPENDENT,
                requests=[
                    Request(
                        request_id="population",
                        kind="population",
                        target="candidate",
                        scope={"entities": entities, "question": question},
                    )
                ],
                bindings=[],
                result_operator=operator,
                snapshot_at=snapshot_at,
                metadata={"query_class": query_class, "question": question},
            )

        plan.validate()
        return plan
