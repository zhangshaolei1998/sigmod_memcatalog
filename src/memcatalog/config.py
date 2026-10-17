"""Runtime configuration for the memory engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    store_path: str = "results/memory.json"
    max_memory_calls: int = 8
    max_facts_per_request: int = 20
    max_source_sessions_per_action: int = 4
    cost_price: float = 0.0
    auto_flush: bool = True

    def __post_init__(self) -> None:
        if self.max_memory_calls < 0:
            raise ValueError("max_memory_calls must be non-negative")
        if self.max_facts_per_request <= 0:
            raise ValueError("max_facts_per_request must be positive")
        if self.max_source_sessions_per_action <= 0:
            raise ValueError("max_source_sessions_per_action must be positive")
        if self.cost_price < 0:
            raise ValueError("cost_price must be non-negative")
