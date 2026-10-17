"""High-level lifecycle API for the complete memory system."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .budget import ActionTemplate, BudgetController, BudgetState
from .compiler import PlanCompiler
from .config import MemoryConfig
from .experience import ExperienceModel
from .grounding import Grounder, GroundingResult
from .indexing import MemoryIndex
from .ingestion import FactExtractor, MemoryIngestor
from .models import ActionRecord, ExecutionRecord, QueryPlan, SupportStatus, Turn
from .operators import OperatorResult, apply_operator
from .planner import build_plan_prompt
from .storage import JsonMemoryStore


AnswerFunction = Callable[[str], str]


@dataclass
class MemoryQueryResponse:
    query_id: str
    plan: QueryPlan
    result: OperatorResult
    prompt: str
    answer: Optional[str]
    memory_calls: int
    actions: List[ActionRecord]


class MemoryEngine:
    """Orchestrates session history, facts, experience, and execution."""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        fact_extractor: Optional[FactExtractor] = None,
        answer_fn: Optional[AnswerFunction] = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.store = JsonMemoryStore(self.config.store_path)
        self.experience = ExperienceModel(self.store)
        self.ingestor = MemoryIngestor(self.store, fact_extractor)
        self.compiler = PlanCompiler()
        self.grounder = Grounder(self.store, self.experience)
        self.budget_controller = BudgetController(self.experience)
        self.answer_fn = answer_fn

    def remember(
        self,
        user_id: str,
        session_id: str,
        timestamp: str,
        turns: Sequence[Turn],
        metadata: Optional[dict] = None,
    ) -> None:
        self.ingestor.append_session(
            user_id, session_id, timestamp, turns, metadata=metadata
        )
        self._flush_if_needed()

    def query(
        self,
        user_id: str,
        question: str,
        choices: Optional[Sequence[str]] = None,
        snapshot_at: Optional[str] = None,
        query_id: Optional[str] = None,
    ) -> MemoryQueryResponse:
        sessions = self.store.sessions(user_id, snapshot_at)
        if not sessions:
            raise ValueError(f"no closed sessions for user: {user_id}")
        effective_snapshot = snapshot_at or sessions[-1].timestamp
        plan = self.compiler.compile(question, choices, effective_snapshot)
        context_key = self._context_key(plan)
        query_id = query_id or f"query-{uuid.uuid4().hex}"
        budget = BudgetState(
            max_calls=self.config.max_memory_calls,
            price=self.config.cost_price,
        )
        action_records: List[ActionRecord] = []
        grounded: List[GroundingResult] = []

        for request in plan.requests:
            self._bind_request(request, grounded)
            result = self.grounder.ground(
                user_id,
                request,
                context_key,
                effective_snapshot,
                max_facts=self.config.max_facts_per_request,
            )
            if result.evidence.status is not SupportStatus.SUPPORTED:
                result, action = self._try_acquire(
                    user_id,
                    question,
                    plan,
                    request,
                    context_key,
                    effective_snapshot,
                    budget,
                    result,
                )
                if action is not None:
                    action_records.append(action)
            grounded.append(result)

        operator_result = apply_operator(plan, grounded)
        prompt = self._build_answer_prompt(question, plan, grounded, operator_result)
        answer = self.answer_fn(prompt) if self.answer_fn is not None else None
        record = ExecutionRecord(
            query_id=query_id,
            user_id=user_id,
            question=question,
            context_key=context_key,
            plan=plan.to_dict(),
            returned_fact_ids=operator_result.evidence.fact_ids,
            returned_session_ids=operator_result.evidence.session_ids,
            status=operator_result.status.value,
            prompt=prompt,
            answer=answer,
            actions=action_records,
            memory_calls=budget.spent,
        )
        self.store.record_execution(record)
        self._flush_if_needed()
        return MemoryQueryResponse(
            query_id=query_id,
            plan=plan,
            result=operator_result,
            prompt=prompt,
            answer=answer,
            memory_calls=budget.spent,
            actions=action_records,
        )

    def report_use(
        self,
        user_id: str,
        query_id: str,
        used_fact_ids: Iterable[str],
        complete: bool = True,
    ) -> None:
        self.experience.report_use(
            user_id, query_id, used_fact_ids, complete=complete
        )
        self._flush_if_needed()

    def summary(self, user_id: str) -> Dict[str, int]:
        return self.store.summary(user_id)

    def flush(self) -> None:
        self.store.flush()

    def _try_acquire(
        self,
        user_id: str,
        question: str,
        plan: QueryPlan,
        request: Any,
        context_key: str,
        snapshot_at: str,
        budget: BudgetState,
        before: GroundingResult,
    ) -> tuple:
        if self.ingestor.extractor is None or budget.remaining == 0:
            return before, None
        index = MemoryIndex(
            self.store.sessions(user_id, snapshot_at),
            self.store.facts(user_id, snapshot_at),
        )
        candidates = index.session_candidates(
            question,
            limit=min(self.config.max_source_sessions_per_action, budget.remaining),
        )
        if not candidates:
            return before, None
        action = ActionTemplate(
            kind="acquire",
            context=f"{plan.plan_class.value}:{request.kind}",
            max_calls=len(candidates),
            prior_reduction=0.5,
            prior_cost=float(len(candidates)),
        )
        selected = self.budget_controller.choose(
            user_id, plan.plan_class.value, [action], budget
        )
        if selected is None:
            return before, None
        self.ingestor.acquire(
            user_id,
            candidates,
            [str(request.target), str(request.predicate or ""), question],
        )
        budget.charge(len(candidates))
        after = self.grounder.ground(
            user_id,
            request,
            context_key,
            snapshot_at,
            max_facts=self.config.max_facts_per_request,
        )
        reduction = self._gap_fraction(before) - self._gap_fraction(after)
        return after, ActionRecord(
            kind=selected.kind,
            context=selected.context,
            predicted_reduction=selected.prior_reduction,
            observed_reduction=max(0.0, reduction),
            call_cost=len(candidates),
            reservation=selected.max_calls,
            completed=True,
        )

    @staticmethod
    def _bind_request(request: Any, grounded: List[GroundingResult]) -> None:
        if not request.slots or not grounded:
            return
        predecessors = grounded[-1].facts
        if predecessors:
            value = predecessors[0].subject
            request.slots["entity"] = value
            request.scope["question"] = f"{request.scope.get('question', '')} {value}".strip()

    @staticmethod
    def _gap_fraction(result: GroundingResult) -> float:
        if result.evidence.status is SupportStatus.SUPPORTED:
            return 0.0
        if result.evidence.status is SupportStatus.PARTIAL:
            return 0.5
        return 1.0

    @staticmethod
    def _context_key(plan: QueryPlan) -> str:
        query_class = str(plan.metadata.get("query_class", "unknown"))
        temporal = "temporal" if plan.snapshot_at else "current"
        return f"{plan.plan_class.value}:{query_class}:{temporal}"

    @staticmethod
    def _build_answer_prompt(
        question: str,
        plan: QueryPlan,
        grounded: List[GroundingResult],
        result: OperatorResult,
    ) -> str:
        fact_lines = []
        for request_result in grounded:
            for fact in request_result.facts:
                sources = ", ".join(span.session_id for span in fact.source_spans)
                fact_lines.append(
                    f"- [{fact.fact_id}] ({fact.subject}, {fact.predicate}, {fact.object}) source={sources}"
                )
        evidence = "\n".join(fact_lines) if fact_lines else "- No cached fact satisfied the request."
        gaps = "\n".join(f"- {gap}" for gap in result.evidence.gaps) or "- none"
        base = (
            "Answer the question using only the source-backed evidence below.\n\n"
            f"QUESTION: {question}\n\n"
            f"PLAN: {plan.plan_class.value} / {plan.result_operator}\n"
            f"DETERMINISTIC RESULT: {json.dumps(result.value, ensure_ascii=False)}\n\n"
            f"EVIDENCE:\n{evidence}\n\n"
            f"UNRESOLVED GAPS:\n{gaps}\n\n"
            "Do not turn an unresolved gap into a definite claim."
        )
        query_class = str(plan.metadata.get("query_class", "POINT"))
        return build_plan_prompt(base, query_class) or base

    def _flush_if_needed(self) -> None:
        if self.config.auto_flush:
            self.store.flush()
