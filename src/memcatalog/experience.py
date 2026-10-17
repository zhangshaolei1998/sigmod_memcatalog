"""Second-order memory for evidence use and action outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from .models import FactRecord
from .storage import JsonMemoryStore


@dataclass(frozen=True)
class ActionEstimate:
    reduction: float
    cost: float
    observations: int


class ExperienceModel:
    def __init__(
        self,
        store: JsonMemoryStore,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ) -> None:
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("experience priors must be positive")
        self.store = store
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def use_propensity(self, user_id: str, context_key: str, fact_id: str) -> float:
        stats = self.store.fact_stats(user_id, context_key, fact_id)
        return (self.alpha_prior + stats["uses"]) / (
            self.alpha_prior + self.beta_prior + stats["exposures"]
        )

    def prioritize(
        self,
        user_id: str,
        context_key: str,
        facts: Iterable[FactRecord],
        relevance: Dict[str, float],
        limit: int,
    ) -> List[FactRecord]:
        facts = list(facts)
        if limit <= 0 or not facts:
            return []
        ranked = sorted(
            facts,
            key=lambda fact: (
                -relevance.get(fact.fact_id, 0.0)
                * self.use_propensity(user_id, context_key, fact.fact_id),
                fact.fact_id,
            ),
        )
        selected = ranked[:limit]

        # Reserve one position for a relevant, least-exposed fact.
        relevant = [fact for fact in facts if relevance.get(fact.fact_id, 0.0) > 0]
        if relevant:
            least_exposed = min(
                relevant,
                key=lambda fact: (
                    self.store.fact_stats(user_id, context_key, fact.fact_id)["exposures"],
                    fact.fact_id,
                ),
            )
            if least_exposed not in selected:
                if len(selected) == limit:
                    selected[-1] = least_exposed
                else:
                    selected.append(least_exposed)
        return selected

    def estimate_action(
        self,
        user_id: str,
        kind: str,
        plan_class: str,
        prior_reduction: float,
        prior_cost: float,
    ) -> ActionEstimate:
        detailed = self.store.action_history(user_id, kind=kind, plan_class=plan_class)
        history = detailed or self.store.action_history(user_id, kind=kind)
        reductions = [prior_reduction] + [float(row["observed_reduction"]) for row in history]
        costs = [prior_cost] + [float(row["call_cost"]) for row in history]
        return ActionEstimate(mean(reductions), mean(costs), len(history))

    def report_use(
        self,
        user_id: str,
        query_id: str,
        used_fact_ids: Iterable[str],
        complete: bool = True,
    ) -> None:
        self.store.report_use(user_id, query_id, used_fact_ids, complete=complete)
