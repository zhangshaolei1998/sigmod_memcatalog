"""Self-contained demonstration of the complete memory lifecycle."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from memcatalog import MemoryConfig, MemoryEngine, Turn


def synthetic_extractor(prompt: str) -> str:
    """Deterministic stand-in for a fact-extraction model."""
    if "coffee" in prompt:
        return '[["user", "prefers", "coffee"]]'
    if "tea" in prompt:
        return '[["user", "prefers", "tea"]]'
    return "[]"


def describe(label: str, response) -> None:
    print(f"\n{label}")
    print(f"  query_id:     {response.query_id}")
    print(f"  plan:         {response.plan.plan_class.value}")
    print(f"  status:       {response.result.status.value}")
    print(f"  value:        {json.dumps(response.result.value, ensure_ascii=False)}")
    print(f"  facts:        {response.result.evidence.fact_ids}")
    print(f"  memory calls: {response.memory_calls}")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        engine = MemoryEngine(
            MemoryConfig(store_path=str(Path(directory) / "memory.json")),
            fact_extractor=synthetic_extractor,
        )
        engine.remember(
            "demo-user",
            "S1",
            "2024-01-01",
            [Turn("user", "I prefer tea in the morning.")],
        )
        engine.remember(
            "demo-user",
            "S2",
            "2024-02-01",
            [Turn("user", "Actually, I now prefer coffee in the morning.")],
        )

        cold = engine.query(
            "demo-user",
            "What does the user prefer in the morning now?",
            query_id="cold-query",
        )
        describe("Cold query: source acquisition and revision reconciliation", cold)
        engine.report_use(
            "demo-user",
            cold.query_id,
            cold.result.evidence.fact_ids,
            complete=True,
        )

        warm = engine.query(
            "demo-user",
            "What does the user prefer in the morning now?",
            query_id="warm-query",
        )
        describe("Warm query: verified fact reuse", warm)
        print(f"\nPersistent state: {engine.summary('demo-user')}")


if __name__ == "__main__":
    main()
