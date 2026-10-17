"""Integration tests for the catalog, fact, workload, and pipeline layers."""

import unittest

import memcatalog.workload as workload
from memcatalog import (
    FactRow,
    build_memcatalog,
    build_memory_query,
    commit_query,
    enforce_revision,
    extract_entities,
    popularity_score,
)


class EmptyGenerator:
    def __call__(self, prompt: str) -> str:
        return ""


class CatalogTests(unittest.TestCase):
    def test_extracts_named_entities_and_dates(self) -> None:
        entities = extract_entities("Alex visited New York on 2024-05-02.")
        self.assertIn("new york", entities)
        self.assertIn("2024-05-02", entities)

    def test_catalog_prompt_contains_all_views(self) -> None:
        result = build_memcatalog(
            "What did Alex update?",
            ["Alex changed my mind and selected tea."],
            ["S1"],
            ["2024-05-02"],
        )
        self.assertIn("Temporal Catalog", result.prompt)
        self.assertIn("Entity Catalog", result.prompt)
        self.assertIn("Revision Catalog", result.prompt)


class FactTests(unittest.TestCase):
    def test_revision_keeps_latest_fact(self) -> None:
        rows = [
            FactRow("user", "prefers", "tea", "S1", "2024-01-01"),
            FactRow("user", "prefers", "coffee", "S2", "2024-02-01"),
        ]
        kept, superseded = enforce_revision(rows)
        self.assertEqual([row.object for row in kept], ["coffee"])
        self.assertEqual([row.object for row in superseded], ["tea"])


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        workload._LOG_CACHE["data"] = {}

    def test_builds_and_commits_complete_query(self) -> None:
        result = build_memory_query(
            "What does Alex prefer now?",
            ["Alex prefers tea.", "Actually, Alex now prefers coffee."],
            ["S1", "S2"],
            ["2024-01-01", "2024-02-01"],
            EmptyGenerator(),
            scope="synthetic-conversation",
            q_type_hint="knowledge-update",
        )
        self.assertIn("[REVISION]", result.prompt)
        self.assertEqual(result.plan.n_matches_sessions, 2)

        commit_query(result)
        scores = popularity_score(
            "synthetic-conversation",
            result.q_entities,
            q_type=result.plan.q_type,
        )
        self.assertEqual(scores, {"S1": 2, "S2": 2})

    def test_rejects_misaligned_session_fields(self) -> None:
        with self.assertRaises(ValueError):
            build_memory_query(
                "Question?",
                ["one session"],
                [],
                ["2024-01-01"],
                EmptyGenerator(),
                scope="synthetic-conversation",
            )


if __name__ == "__main__":
    unittest.main()
