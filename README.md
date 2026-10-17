# MemCatalog

MemCatalog is a complete, DBMS-inspired memory system for language-model
agents. It manages raw conversation history, source-backed and temporally
versioned facts, query plans, evidence support, memory-use experience, and
cost-aware execution under a hard model-call budget.

The repository contains no development history, private data, credentials,
machine-specific paths, or author metadata.

## What is implemented

- Append-only, user-isolated, timestamped conversation sessions.
- Query-driven fact extraction with stable IDs, source spans, event links, and
  validity intervals.
- Explicit revision chains that preserve historical values.
- Temporal, entity, event, and fact indexes.
- Validated query plans for atomic retrieval, independent composition, and
  dependency-bound composition.
- Grounding with snapshot, source, version, and support checks.
- Count, comparison, projection, and dependent-result operators.
- Evidence states: `supported`, `partial`, and `unknown`.
- Second-order exposure/use statistics with validated, idempotent use reports.
- Per-user action outcomes and cost estimates.
- Hard call caps and benefit-versus-cost action admission.
- Atomic JSON persistence and a complete CLI lifecycle.
- A lightweight prompt-only API retained for experiments and ablations.

## Structure

```text
.
├── src/memcatalog/
│   ├── models.py        # sessions, fact versions, plans, evidence, records
│   ├── storage.py       # atomic persistent state and user isolation
│   ├── ingestion.py     # session append and query-driven fact acquisition
│   ├── indexing.py      # temporal/entity/event/fact indexes
│   ├── compiler.py      # normalized query-plan construction
│   ├── grounding.py     # fact selection and evidence checks
│   ├── operators.py     # lookup, count, comparison, dependency projection
│   ├── experience.py    # exposure/use priorities and action estimates
│   ├── budget.py        # hard caps and action admission
│   ├── engine.py        # full memory lifecycle orchestration
│   ├── cli.py           # persistent command-line interface
│   ├── catalog.py       # human-readable catalog views
│   ├── facts.py         # triple parsing and lightweight fact cache
│   ├── pipeline.py      # lightweight prompt pipeline
│   ├── planner.py       # POINT / AGGREGATE / COMPARE prompt routing
│   ├── heuristics.py, matching.py, workload.py
│   ├── __init__.py      # public API
│   └── __main__.py      # `python -m memcatalog`
├── docs/architecture.md
├── examples/
├── tests/
├── Makefile
└── ANONYMIZATION.md
```

The implementation uses only the Python standard library and supports Python
3.8 or newer.

## High-level API

```python
from memcatalog import MemoryConfig, MemoryEngine, Turn

engine = MemoryEngine(
    MemoryConfig(
        store_path="results/memory.json",
        max_memory_calls=8,
        cost_price=0.05,
    ),
    fact_extractor=my_fact_extractor,  # prompt -> JSON triple array
    answer_fn=my_answer_model,         # optional prompt -> answer
)

engine.remember(
    user_id="user-001",
    session_id="session-001",
    timestamp="2024-01-01T10:00:00Z",
    turns=[
        Turn("user", "I prefer tea in the morning."),
        Turn("assistant", "I will remember that."),
    ],
)

response = engine.query(
    user_id="user-001",
    question="What does the user prefer in the morning?",
)

# Credit only evidence actually consumed by the agent.
engine.report_use(
    "user-001",
    response.query_id,
    response.result.evidence.fact_ids,
    complete=True,
)
```

The extractor is optional. Without it, queries reuse existing facts and expose
unresolved gaps instead of fabricating support. The answer function is also
optional; when absent, `response.prompt` can be sent to any model by the caller.

## CLI lifecycle

```bash
export PYTHONPATH=src

python3 -m memcatalog --store results/memory.json init

python3 -m memcatalog --store results/memory.json ingest \
  --user user-001 \
  --session session-001 \
  --timestamp 2024-01-01T10:00:00Z \
  --file examples/session_turns.json

python3 -m memcatalog --store results/memory.json query \
  --user user-001 \
  --question 'What does the user prefer in the morning?'

python3 -m memcatalog --store results/memory.json report-use \
  --user user-001 \
  --query QUERY_ID \
  --facts FACT_ID

python3 -m memcatalog --store results/memory.json stats --user user-001
```

Use `--fact-command` and `--answer-command` on `query` to connect external
model commands. Commands receive their prompt on standard input and must write
the result on standard output. They are executed directly without a shell.

## Complete demo

The self-contained engine demo exercises cold acquisition, fact revision,
validated use feedback, persistence, and zero-call warm reuse:

```bash
PYTHONPATH=src python3 examples/full_engine_demo.py
```

`examples/run_demo.py` separately exercises the lightweight prompt pipeline
over a longer synthetic conversation.

## Support and execution semantics

The memory engine never treats ranking or budget exhaustion as factual support.
A lookup is supported only by eligible source-backed facts. Population
operations remain partial until coverage is closed; unresolved count judgments
produce an interval. Dependent requests execute only after predecessor evidence
is available. Complete use feedback must name a subset of returned fact IDs.

See [docs/architecture.md](docs/architecture.md) for the state model, support
invariants, experience updates, and cost-control behavior.

## Tests and audit

```bash
make test
make demo
make cli-check
make audit
```
