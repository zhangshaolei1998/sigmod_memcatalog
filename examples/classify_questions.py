"""Small, synthetic example for the query-class planner."""

from memcatalog import build_plan_prompt, classify_query


EXAMPLES = [
    ("Where was the meeting held?", ["A. Park", "B. Library"]),
    ("How many meetings were held?", ["A. 1", "B. 2", "C. 3"]),
    ("Which meeting lasted longer, the first or the second?", None),
]


def main() -> None:
    base_prompt = "Answer the question from the supplied memory."
    for question, choices in EXAMPLES:
        query_class = classify_query(question, choices)
        planned = build_plan_prompt(base_prompt, query_class)
        mode = "direct" if planned is None else "planned"
        print(f"{query_class:9} {mode:7} {question}")


if __name__ == "__main__":
    main()
