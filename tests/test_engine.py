"""End-to-end tests for persistent memory lifecycle operations."""

import tempfile
import unittest
from pathlib import Path

from memcatalog import MemoryConfig, MemoryEngine, PlanClass, Turn


class SyntheticExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if "coffee" in prompt:
            return '[["Alex", "prefers", "coffee"]]'
        if "tea" in prompt:
            return '[["Alex", "prefers", "tea"]]'
        return "[]"


class MemoryEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store_path = str(Path(self.temporary.name) / "memory.json")
        self.extractor = SyntheticExtractor()
        self.engine = MemoryEngine(
            MemoryConfig(store_path=self.store_path, max_memory_calls=4),
            fact_extractor=self.extractor,
        )
        self.engine.remember(
            "user-a",
            "S1",
            "2024-01-01",
            [Turn("user", "Alex prefers tea.")],
        )
        self.engine.remember(
            "user-a",
            "S2",
            "2024-02-01",
            [Turn("user", "Actually, Alex now prefers coffee.")],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_query_acquires_versions_and_records_execution(self) -> None:
        response = self.engine.query(
            "user-a",
            "What does Alex prefer now?",
            query_id="query-a",
        )
        self.assertEqual(response.plan.plan_class, PlanClass.RETRIEVE)
        self.assertEqual(response.result.value, "coffee")
        self.assertEqual(response.memory_calls, 2)
        self.assertEqual(self.engine.summary("user-a"), {"sessions": 2, "facts": 2, "executions": 1})

        self.engine.report_use(
            "user-a", "query-a", response.result.evidence.fact_ids, complete=True
        )
        stats = self.engine.store.fact_stats(
            "user-a",
            "retrieve:POINT:temporal",
            response.result.evidence.fact_ids[0],
        )
        self.assertEqual(stats, {"exposures": 1, "uses": 1})

    def test_warm_query_reuses_facts_without_calls(self) -> None:
        self.engine.query("user-a", "What does Alex prefer now?", query_id="cold")
        calls_after_cold = self.extractor.calls
        response = self.engine.query(
            "user-a", "What does Alex prefer now?", query_id="warm"
        )
        self.assertEqual(response.memory_calls, 0)
        self.assertEqual(self.extractor.calls, calls_after_cold)

    def test_state_survives_engine_restart(self) -> None:
        self.engine.query("user-a", "What does Alex prefer now?", query_id="persisted")
        reopened = MemoryEngine(MemoryConfig(store_path=self.store_path))
        self.assertEqual(reopened.summary("user-a"), {"sessions": 2, "facts": 2, "executions": 1})

    def test_users_are_isolated(self) -> None:
        self.engine.remember(
            "user-b", "S1", "2024-01-01", [Turn("user", "Jordan likes hiking.")]
        )
        self.assertEqual(self.engine.summary("user-b")["sessions"], 1)
        self.assertEqual(self.engine.summary("user-a")["sessions"], 2)


if __name__ == "__main__":
    unittest.main()
