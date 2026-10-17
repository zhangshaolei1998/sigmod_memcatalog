"""Tests for the query-class planner."""

import unittest

from memcatalog import (
    AGGREGATE,
    COMPARE,
    POINT,
    build_plan_prompt,
    choices_are_quantities,
    classify_query,
    extract_plan_letter,
)


class ClassificationTests(unittest.TestCase):
    def test_point_question(self) -> None:
        self.assertEqual(classify_query("Where was the meeting held?"), POINT)

    def test_explicit_count_question(self) -> None:
        self.assertEqual(classify_query("How many meetings were held?"), AGGREGATE)

    def test_numeric_choices_imply_aggregate(self) -> None:
        choices = {"A": "1 day", "B": "2 days", "C": "3 days", "D": "4 days"}
        self.assertTrue(choices_are_quantities(choices))
        self.assertEqual(classify_query("What was the duration?", choices), AGGREGATE)

    def test_binary_comparison(self) -> None:
        question = "Which meeting lasted longer, the first or the second?"
        self.assertEqual(classify_query(question), COMPARE)

    def test_superlative_question(self) -> None:
        self.assertEqual(classify_query("Which trip was shortest?"), COMPARE)


class PromptTests(unittest.TestCase):
    def test_point_uses_direct_path(self) -> None:
        self.assertIsNone(build_plan_prompt("base", POINT))

    def test_aggregate_adds_counting_plan(self) -> None:
        prompt = build_plan_prompt("base", AGGREGATE)
        self.assertIsNotNone(prompt)
        self.assertIn("counting problem", prompt)

    def test_compare_adds_comparison_plan(self) -> None:
        prompt = build_plan_prompt("base", COMPARE)
        self.assertIsNotNone(prompt)
        self.assertIn("comparison", prompt)

    def test_unknown_class_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_plan_prompt("base", "UNKNOWN")

    def test_extracts_last_answer_line(self) -> None:
        self.assertEqual(extract_plan_letter("Answer: A\nAnswer: C"), "C")

    def test_extracts_boxed_answer(self) -> None:
        self.assertEqual(extract_plan_letter(r"Therefore, \boxed{B}"), "B")


if __name__ == "__main__":
    unittest.main()
