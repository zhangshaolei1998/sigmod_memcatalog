# Architecture

MemCatalog maintains three user-isolated levels of state:

```text
closed sessions (H)
        │ query-driven acquisition
        ▼
source-backed fact versions (F)
        │ exposure, reported use, action outcome
        ▼
execution experience (U)
```

## Session history

A `Session` is an immutable, closed, timestamped sequence of speaker-labelled
turns. The store rejects reuse of a session identifier with different content.
Queries read a fixed timestamp snapshot, so later ingestion cannot silently
change the source set of an in-flight query.

## First-order facts

`MemoryIngestor.acquire` asks the configured extractor only for evidence needed
by the current request. Every `FactRecord` includes:

- a deterministic fact identifier;
- subject, predicate, and object;
- one or more source spans;
- validity bounds;
- an optional event identifier;
- a link to the fact it supersedes.

An explicit revision closes the prior fact's validity interval instead of
deleting it. Historical queries can therefore address old values, while current
queries select open fact versions.

## Query compilation

`PlanCompiler` emits a validated `QueryPlan` containing requests, bindings, and
a result operator. Plans are partitioned into:

- `retrieve`: one atomic lookup;
- `independent`: population enumeration followed by count or comparison;
- `dependent`: a predecessor result binds an input slot of a later request.

Bindings must reference existing requests and self-bindings are rejected.

## Grounding and support

`Grounder` applies identity, snapshot, fact-validity, and source checks before
returning evidence. Each result is `supported`, `partial`, or `unknown`. A
population operation remains partial unless coverage is independently closed;
ranking a subset never proves an exact count.

`operators.py` propagates evidence and unresolved gaps through lookup, count,
comparison, and dependent projection. It returns intervals for counts when
unresolved judgments remain.

## Second-order experience

An `ExecutionRecord` distinguishes returned evidence from facts reported as
used. A complete use report must be a subset of returned fact IDs. Updates are
idempotent per query and immutable after complete feedback.

For a fact and query context, `ExperienceModel` estimates reported-use
propensity with positive prior counts. Ranking reserves one position for a
relevant, least-exposed fact to prevent starvation. Experience affects work
order, never source support or population obligations.

## Cost-aware execution

The `BudgetController` predicts diagnostic reduction and call cost using the
same user's matching action history plus a declared prior observation. An
action is admitted only when it has positive predicted net benefit and its
maximum reservation fits the remaining hard cap.

Every model-backed acquisition is charged once. The final answer model is kept
outside the memory-call ledger. Exhausting the cap preserves `partial` or
`unknown`; it never converts missing evidence into support.

## Persistence

`JsonMemoryStore` writes all levels atomically through a temporary file and
`os.replace`. It stores sessions, facts, execution records, fact-use statistics,
and action outcomes in one portable JSON document. Every lookup is scoped by
user identifier.
