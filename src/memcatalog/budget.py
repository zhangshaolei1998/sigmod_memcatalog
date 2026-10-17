"""Cost-aware admission control for bounded memory actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .experience import ActionEstimate, ExperienceModel


@dataclass(frozen=True)
class ActionTemplate:
    kind: str
    context: str
    max_calls: int
    prior_reduction: float
    prior_cost: float


@dataclass
class BudgetState:
    max_calls: int
    price: float = 0.0
    spent: int = 0
    admitted: List[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.spent)

    def charge(self, calls: int) -> None:
        if calls < 0 or calls > self.remaining:
            raise ValueError("action cost exceeds remaining hard cap")
        self.spent += calls


class BudgetController:
    def __init__(self, experience: ExperienceModel) -> None:
        self.experience = experience

    def choose(
        self,
        user_id: str,
        plan_class: str,
        actions: List[ActionTemplate],
        budget: BudgetState,
    ) -> Optional[ActionTemplate]:
        scored = []
        for action in actions:
            if action.max_calls > budget.remaining:
                continue
            estimate = self.experience.estimate_action(
                user_id,
                action.kind,
                plan_class,
                action.prior_reduction,
                action.prior_cost,
            )
            benefit = estimate.reduction - budget.price * estimate.cost
            if benefit > 0:
                scored.append((benefit, -estimate.cost, action.kind, action))
        if not scored:
            return None
        selected = max(scored)[-1]
        budget.admitted.append(selected.kind)
        return selected
