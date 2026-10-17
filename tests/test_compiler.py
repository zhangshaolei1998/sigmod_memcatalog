"""Tests for normalized query-plan compilation."""

import unittest

from memcatalog import Binding, PlanClass, PlanCompiler, QueryPlan, Request


class CompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = PlanCompiler()

    def test_atomic_lookup(self) -> None:
        plan = self.compiler.compile("Where was the meeting held?")
        self.assertEqual(plan.plan_class, PlanClass.RETRIEVE)
        self.assertEqual(len(plan.requests), 1)

    def test_independent_population(self) -> None:
        plan = self.compiler.compile("How many meetings were held?")
        self.assertEqual(plan.plan_class, PlanClass.INDEPENDENT)
        self.assertEqual(plan.result_operator, "count")

    def test_dependent_binding(self) -> None:
        plan = self.compiler.compile("Who organized the longest trip?")
        self.assertEqual(plan.plan_class, PlanClass.DEPENDENT)
        self.assertEqual(len(plan.bindings), 1)

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.compiler.compile("   ")

    def test_binding_cycles_are_rejected(self) -> None:
        plan = QueryPlan(
            plan_id="cycle",
            plan_class=PlanClass.DEPENDENT,
            requests=[
                Request("a", "lookup", "a", slots={"input": None}),
                Request("b", "lookup", "b", slots={"input": None}),
            ],
            bindings=[Binding("a", "b", "input"), Binding("b", "a", "input")],
            result_operator="bind-and-project",
        )
        with self.assertRaises(ValueError):
            plan.validate()


if __name__ == "__main__":
    unittest.main()
