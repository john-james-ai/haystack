# DAT Sprint 2 — Summary

---

## 1. Implementation Summary

Sprint 2 delivered `InMemoryDATHybridRetriever`, a self-contained Haystack component
that wraps the full DAT hybrid retrieval pipeline (BM25 + embedding + DAT fusion) into
a single component backed by `InMemoryDocumentStore`. It reduces a three-component
pipeline to one, following the established naming convention of `InMemoryBM25Retriever`
and `InMemoryEmbeddingRetriever`.

Sprint 2 also delivered integration tests that verify `DATDocumentJoiner` correctly
wires into a full Haystack pipeline and that both new components survive serialisation
round-trips.

**Files created / modified:**

| File | Status | Lines |
|------|--------|------:|
| `haystack/components/retrievers/in_memory/dat_hybrid_retriever.py` | New | 330 |
| `test/components/retrievers/test_in_memory_dat_hybrid_retriever.py` | New | 530 |
| `test/components/joiners/test_dat_document_joiner_integration.py` | New | 225 |
| `haystack/components/retrievers/in_memory/__init__.py` | Modified | +4 |
| `haystack/components/retrievers/__init__.py` | Modified | +2 |

---

## 2. Implementation Detail

### Component interface

```python
@component
class InMemoryDATHybridRetriever:
    def __init__(
        self,
        document_store: InMemoryDocumentStore,  # single store for both retrievers
        chat_generator: ChatGenerator,           # LLM effectiveness scorer
        top_k: int = 10,
        scoring_top_k: int = 1,
        scale_score: bool = False,
        filters: dict[str, Any] | None = None,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
        raise_on_failure: bool = True,
    ): ...

    @component.output_types(documents=list[Document], alpha=float)
    def run(self, query: str, query_embedding: list[float],
            filters: dict | None = None, top_k: int | None = None) -> dict: ...

    @component.output_types(documents=list[Document], alpha=float)
    async def run_async(self, query: str, query_embedding: list[float],
                        filters: dict | None = None, top_k: int | None = None) -> dict: ...
```

### Internal architecture

`InMemoryDATHybridRetriever` delegates all DAT logic to an internal
`DATDocumentJoiner` instance (`self._joiner`), constructed at `__init__` time
from the same `chat_generator`, `scoring_top_k`, and `raise_on_failure` params.
This avoids code duplication while keeping the public interface clean.

**`run()` flow:**
1. Resolve effective filters via `filter_policy` (REPLACE or MERGE).
2. Call `document_store.bm25_retrieval(query, ...)` → `bm25_docs`.
3. Call `document_store.embedding_retrieval(query_embedding, ...)` → `dense_docs`.
4. Delegate to `self._joiner.run(query, dense_docs, bm25_docs, top_k)`.

**`run_async()` flow:**
1. Resolve effective filters.
2. Launch both store calls concurrently with `asyncio.gather()`.
3. Delegate to `self._joiner.run_async(...)`.

### Serialisation

- `to_dict()` serialises `document_store` (via `default_to_dict` auto-call of
  `document_store.to_dict()`), `chat_generator` (via `component_to_dict`), and all
  scalar params. `filter_policy` is stored as its string value.
- `from_dict()` restores `filter_policy` via `FilterPolicy.from_str()`, deserialises
  `chat_generator` via `deserialize_chatgenerator_inplace`, and delegates the rest to
  `default_from_dict` (which auto-restores `document_store` by calling
  `InMemoryDocumentStore.from_dict()`).

### Package export

`InMemoryDATHybridRetriever` is exported from both:
- `haystack.components.retrievers.in_memory` (primary location)
- `haystack.components.retrievers` (convenience re-export, consistent with the two
  existing InMemory retrievers)

---

## 3. Insights

### 3.1 Design Decisions

---

#### DD-1 — Internal `DATDocumentJoiner` vs. duplicating algorithm logic

**Context.** `InMemoryDATHybridRetriever` needs the same normalisation, LLM-call,
α-computation, and fusion logic already implemented in `DATDocumentJoiner`. Options
were to duplicate the logic or reuse the existing component.

**Options considered**

| Option | Pros | Cons |
|--------|------|------|
| (A) Duplicate algorithm code | No internal dependency | Maintenance burden; two copies to keep in sync |
| (B) Call static methods of `DATDocumentJoiner` directly | Minimal coupling | `_parse_scores` and `_call_llm` are instance methods, not static — awkward to import |
| (C) Hold a `DATDocumentJoiner` as `self._joiner` (CHOSEN) | Full reuse; one source of truth; `warm_up` delegation is natural | `_joiner` is not serialised (private); slight indirection |

**Recommendation: Option C — internal private joiner.**

The private `_joiner` is not part of `to_dict()` — it is reconstructed in
`__init__` from the same top-level parameters. All behaviour and bug-fixes in
`DATDocumentJoiner` propagate to `InMemoryDATHybridRetriever` automatically.

---

#### DD-2 — `asyncio.gather` for concurrent dual retrieval

**Context.** `run_async` must call both `bm25_retrieval_async` and
`embedding_retrieval_async`. Both calls are I/O-bound (in production, the store
may be remote). Sequential calls would double the latency.

**Options considered**

| Option | Pros | Cons |
|--------|------|------|
| (A) Sequential `await` for each call | Simple | Doubles retrieval latency |
| (B) `asyncio.gather(bm25_call, dense_call)` (CHOSEN) | Concurrent; halves latency for equal-cost calls | Errors from either call propagate as exceptions |

**Recommendation: Option B — `asyncio.gather`.**

The two retrievals are fully independent; concurrent execution is strictly better.
`InMemoryDocumentStore.bm25_retrieval_async` and `embedding_retrieval_async` both
delegate to `run_in_executor`, so they can genuinely run concurrently.

---

#### DD-3 — `scale_score` defaults to `False`

**Context.** `InMemoryBM25Retriever` and `InMemoryEmbeddingRetriever` both default
`scale_score=False`. DAT's min-max normalisation step makes score scaling redundant
(it would normalise the already-scaled scores again). Keeping `False` avoids double
normalisation while matching peer retriever defaults.

**Recommendation: `scale_score=False` default.** Matches peer components; DAT
normalisation is applied regardless.

---

### 3.2 Open and Clarifying Questions

---

#### Q-1 — `_joiner` not serialised: is there a risk of state drift?

**Context.** `self._joiner` is reconstructed from top-level parameters on every
`__init__` and `from_dict`. If `DATDocumentJoiner`'s defaults change in a future
version, deserialized `InMemoryDATHybridRetriever` instances will pick up new defaults
only if the serialised dict was created with the old defaults (they are stored
explicitly).

**Assessment:** Low risk. All params (`top_k`, `scoring_top_k`, `raise_on_failure`)
are serialised in the top-level dict and passed explicitly to `DATDocumentJoiner`.
No drift is possible.

---

#### Q-2 — `warm_up` triggering on `run` vs `warm_up`

**Context.** `DATDocumentJoiner.warm_up` is triggered automatically on the first
`run()` call. `InMemoryDATHybridRetriever` delegates `warm_up()` to `self._joiner`,
so calling `run()` on the retriever also triggers `_joiner.warm_up()` internally.

**Assessment:** Correct by design. No action needed.

---

### 3.3 Challenges Faced

1. **`FilterPolicy` string value mismatch.** The test initially asserted
   `ip["filter_policy"] == "replace_or_merge"`, but the actual enum value is `"merge"`.
   Fixed by asserting `== FilterPolicy.MERGE.value` instead of a hard-coded string.

---

### 3.4 Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R-1 | `InMemoryDocumentStore`-specific: does not generalise to other stores | By design | Low | Generic pipeline (DATDocumentJoiner) covers all other store types |
| R-2 | Documents without embeddings cause empty dense results → α=0.0 silently | Medium | Medium | Documented in class docstring; InMemoryDocumentStore logs INFO when no embedding docs found |
| R-3 | `asyncio.gather` masks which retrieval failed when one raises | Low | Low | Python propagates the first exception from gather; stack trace identifies source |

---

## 4. Statistics

### 4.1 Development Stats

| Field | Value |
|-------|-------|
| Start Timestamp | 2026-02-12 02:30:23 |
| End Timestamp | 2026-02-12 02:35:51 |
| Duration | 5 min 28 s |
| Modules | 3 new; 2 modified |
| Classes | 1 production (`InMemoryDATHybridRetriever`); 10 test classes (7 unit, 3 integration) |
| LOC | 330 (impl) + 530 (retriever tests) + 225 (integration tests) + 6 (init files) = **1 091 total** |

### 4.2 Testing Status

| Tests | Tested | Passed | % Passed |
|-------|-------:|-------:|---------:|
| Unit Test | 53 + 32 = **85** | 85 | 100% |
| Integration Test | 8 | 8 | 100% |
| Regression Test | 0 | 0 | — |

- Unit tests include all 53 Sprint 1 tests (no regressions) + 32 new Sprint 2 unit tests.
- Integration tests cover pipeline wiring, serialisation round-trips, α propagation,
  and known perfect-hit scenarios.

#### Sprint 2 unit test breakdown

| Test Class | Tests | Focus |
|------------|------:|-------|
| `TestInMemoryDATHybridRetrieverInit` | 6 | Defaults, custom params, invalid store/top_k/scoring_top_k, internal joiner |
| `TestFilterResolution` | 4 | REPLACE uses runtime, falls back to init; MERGE combines; no-filter → None |
| `TestRunUnit` | 7 | Both retrievals called, top_k passed, top_k override, filters passed, empty → α=0.5, parse failure raises, α in output |
| `TestRunAsyncUnit` | 2 | Both empty → α=0.5; concurrent async retrievals + correct α |
| `TestWarmUp` | 2 | Delegates to joiner; called only once |
| `TestTelemetry` | 1 | Data shape (store + generator class names) |
| `TestSerialization` | 3 | `to_dict`, `to_dict` defaults, `from_dict` round-trip |
| `TestPackageExport` | 2 | `from haystack.components.retrievers.in_memory`; `from haystack.components.retrievers` |
| `TestIntegration` | 6 | Full store + mocked LLM; α reflected; sorted; async; raise on failure; fallback |
| **Retriever subtotal** | **33** | |
| `TestDATJoinerPipelineWiring` | 5 | End-to-end pipeline; α from LLM; dense-perfect; BM25-perfect; sorted output |
| `TestPipelineSerializationRoundTrip` | 2 | `to_dict`; `from_dict` preserves all params |
| **Integration subtotal** | **7** | |
| **Sprint 2 total** | **40** | |
