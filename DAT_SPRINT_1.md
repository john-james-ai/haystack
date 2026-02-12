# DAT Sprint 1 — Summary

---

## 1. Implementation Summary

Sprint 1 delivered `DATDocumentJoiner`, a new Haystack component that implements
the Dynamic Alpha Tuning (DAT) algorithm from Hsu & Tzeng (2025 — arXiv:2503.23013)
for hybrid dense + BM25 retrieval fusion.

The component accepts ranked document lists from a dense (embedding) retriever and a
BM25 retriever, uses a pluggable `ChatGenerator` to assign per-query effectiveness
scores to the top-1 result from each retriever, computes a dynamic weighting
coefficient α, and returns a fused, ranked document list.

**Files created / modified:**

| File | Status | Lines |
|------|--------|------:|
| `haystack/components/joiners/dat_document_joiner.py` | New | 547 |
| `test/components/joiners/test_dat_document_joiner.py` | New | 546 |
| `haystack/components/joiners/__init__.py` | Modified | +3 |

---

## 2. Implementation Detail

### Component interface

```python
@component
class DATDocumentJoiner:
    def __init__(
        self,
        chat_generator: ChatGenerator,  # any Haystack ChatGenerator
        top_k: int = 10,                # max documents returned after fusion
        scoring_top_k: int = 1,         # docs passed to LLM per retriever
        sort_by_score: bool = True,
        raise_on_failure: bool = True,  # ComponentError on unparseable LLM reply
    ): ...

    @component.output_types(documents=list[Document], alpha=float)
    def run(self, query: str, dense_documents: list[Document],
            bm25_documents: list[Document], top_k: int | None = None) -> dict: ...

    @component.output_types(documents=list[Document], alpha=float)
    async def run_async(self, query, dense_documents, bm25_documents,
                        top_k=None) -> dict: ...
```

### DAT algorithm (Equations 1–6 from the paper)

1. **Min-max normalise** scores from each retriever independently into [0, 1].
   When all scores are equal (delta = 0) every normalised score is set to 0.0,
   consistent with the DBSF behaviour in `DocumentJoiner`.
2. **Build LLM prompt** — verbatim from Appendix A of the paper — embedding the
   query and the concatenated content of the top `scoring_top_k` documents from
   each retriever.
3. **Invoke LLM** via `chat_generator.run(messages)` (or `run_async` when
   available).
4. **Parse reply** with regex `\b([0-5])\s+([0-5])\b`. A regex search (not
   match) tolerates minor surrounding text from verbose LLMs.
5. **Compute α** (Eq. 6):
   - Both scores 0 → α = 0.5
   - Dense = 5, BM25 ≠ 5 → α = 1.0
   - BM25 = 5, Dense ≠ 5 → α = 0.0
   - Otherwise → α = round(S_v / (S_v + S_b), 1)
6. **Fuse** `R(q,d) = α · S̃_dense + (1−α) · S̃_BM25`; deduplicate by `doc.id`;
   sort descending; truncate to `top_k`.

### Edge-case handling

| Condition | α | LLM called? |
|-----------|---|-------------|
| Both lists empty | 0.5 | No |
| Dense list empty | 0.0 | No |
| BM25 list empty | 1.0 | No |
| Parse failure, `raise_on_failure=True` | — | Yes → raises `ComponentError` |
| Parse failure, `raise_on_failure=False` | 0.5 | Yes → logs warning |

### Other implementation details

- **No input mutation** — `_normalize_scores` and `_fuse` create new `Document`
  instances via `doc.to_dict()` + `Document.from_dict()`. The caller's lists are
  never modified.
- **warm_up lifecycle** — delegates to `chat_generator.warm_up()` on first
  `run()` call (or when called explicitly); guards against double warm-up.
- **Async fallback** — `_call_llm_async` uses `run_async` when the generator
  exposes it; falls back to synchronous `run` otherwise.
- **Serialisation** — follows Haystack convention: `component_to_dict` for the
  nested generator in `to_dict`; `deserialize_chatgenerator_inplace` in
  `from_dict` — matching `LLMEvaluator` exactly.
- **Package export** — `DATDocumentJoiner` registered in
  `haystack/components/joiners/__init__.py` via `LazyImporter`.
- **Telemetry** — `_get_telemetry_data` reports the generator class name to
  Posthog, consistent with other Haystack components.

---

## 3. Insights

### 3.1 Design Decisions

---

#### DD-1 — Standalone component vs extending `DocumentJoiner`

**Context.** `DocumentJoiner` is the existing joiner for hybrid retrieval. DAT
needs (a) two *named* document inputs (`dense_documents`, `bm25_documents`) to
distinguish retriever outputs, (b) the user query as a separate input, and (c) an
injected `ChatGenerator`. `DocumentJoiner` uses a `Variadic[list[Document]]`
input and carries no LLM dependency.

**Options considered**

| Option | Pros | Cons |
|--------|------|------|
| (A) Extend `DocumentJoiner` | Reuse parent serialisation / output plumbing | `Variadic` input is incompatible with named inputs; LLM injection would break parent interface; requires fragile overrides |
| (B) Standalone `@component` class | Clean named-input API; no inheritance debt; independently testable | Small amount of normalisation code cannot be trivially shared |

**Recommendation: Option B — standalone component.**

The DAT algorithm requires a fundamentally different input contract. Forcing it
into `DocumentJoiner`'s interface would produce a confusing API and a fragile
subclass. A standalone component is the standard Haystack pattern for novel
retrieval strategies.

---

#### DD-2 — Default value for `raise_on_failure`

**Context.** When the LLM returns output that does not match the expected format,
the component can either raise a `ComponentError` or silently fall back to α = 0.5.

**Options considered**

| Option | Pros | Cons |
|--------|------|------|
| (A) `raise_on_failure=True` (default) | Surfaces LLM misconfiguration immediately; prevents silent DAT bypass; consistent with `LLMEvaluator` precedent | Requires explicit opt-out for fault-tolerant pipelines |
| (B) `raise_on_failure=False` (default) | More forgiving in production | Silently bypasses DAT if the LLM is misconfigured or prompt drifts — this was explicitly rejected by the user |

**Recommendation: Option A — `raise_on_failure=True`.**

The user stated: *"This may be a bug, and we don't want to silently bypass the DAT
functionality."* Raising by default makes failure visible. Users who want
fault-tolerant behaviour opt in explicitly with `raise_on_failure=False`.

---

#### DD-3 — Document deep-copy strategy for score mutation

**Context.** Normalisation and fusion must assign new scores to documents without
mutating the caller's input lists, which may be used again downstream.

**Options considered**

| Option | Pros | Cons |
|--------|------|------|
| (A) In-place mutation | No allocation overhead | Side effects on caller data; violates pipeline contract |
| (B) Shallow copy + replace `score` field | Less allocation | Metadata dicts still shared between original and copy |
| (C) Deep copy via `doc.to_dict()` + `Document.from_dict()` | Fully isolated; guaranteed no shared references | Slightly higher allocation |

**Recommendation: Option C — deep copy.**

Pipeline components must not produce side effects on their inputs. The
`to_dict/from_dict` round-trip is the idiomatic Haystack way to clone a
`Document` and is already tested in `DocumentJoiner`.

---

#### DD-4 — LLM reply parsing strategy

**Context.** The paper specifies format `"3 4"` but LLMs frequently emit
surrounding commentary (e.g. `"Vector: 3, BM25: 4\nScores: 3 4"`).

**Options considered**

| Option | Pros | Cons |
|--------|------|------|
| (A) Exact / `str.split()` parse | Maximally strict | Fails on minor LLM verbosity; high false-failure rate in practice |
| (B) Regex search `\b([0-5])\s+([0-5])\b` | Tolerates surrounding text; still validates range [0, 5] | Could match first valid pair in ambiguous multi-line output |

**Recommendation: Option B — regex search.**

Real LLM outputs frequently include brief context despite explicit instructions.
The regex anchors on word boundaries and validates the integer range, giving a
good signal-to-noise balance. Confirmed by test: `"The scores are: 2 3."` parses
correctly.

---

### 3.2 Open and Clarifying Questions

---

#### Q-1 — `scoring_top_k > 1` behaviour

**Context.** The paper defines the scoring step using only the top-1 result from
each retriever. The implementation supports `scoring_top_k > 1` by concatenating
document content, but this diverges from the paper.

**Options considered**

| Option | Pros | Cons |
|--------|------|------|
| (A) Hard-restrict to `scoring_top_k = 1` | Strict paper compliance | Loses generality; prevents experimentation |
| (B) Allow any value; concatenate content | Flexible | Multi-doc prompt behaviour is undefined in the paper |
| (C) Allow any value; document the deviation in docstring | Best of both | Requires empirical validation |

**Recommendation: Option C** — `scoring_top_k` defaults to `1` as the paper
recommends, with a docstring note that higher values are experimental. Empirical
evaluation is planned for Sprint 3.

---

#### Q-2 — Document ID deduplication across retrievers

**Context.** `_fuse` deduplicates by `doc.id`. If the same underlying document
is returned by both retrievers but with different IDs (or `None`), the scores
will not be summed and the document will appear twice.

**Options considered**

| Option | Pros | Cons |
|--------|------|------|
| (A) Use content hash as fallback deduplication key | Handles edge cases | Expensive; hash collisions possible |
| (B) Accept current behaviour; document the assumption | Zero overhead | Caller must ensure consistent IDs |
| (C) Warn at runtime when potential duplicates detected | Visible signal | Adds noise for normal operation |

**Recommendation: Option B** — document that IDs must be consistent.
`InMemoryDocumentStore` assigns IDs at write time, so both retrievers return the
same ID for the same document. This is confirmed behaviour and eliminates the
risk in practice. Revisit if external vector stores are supported.

---

### 3.3 Challenges Faced

1. **`MagicMock` attribute auto-creation.** Python's `MagicMock` generates any
   attribute on first access, including `run_async`. The async-fallback test
   initially failed because `hasattr(mock, "run_async")` was always `True`.
   Fixed by explicitly deleting the attribute (`del mock_generator.run_async`)
   in the test to simulate a synchronous-only generator.

2. **`OpenAIChatGenerator` requires a live API key at construction.** The
   `test_from_dict` test instantiates a real `OpenAIChatGenerator` as part of
   the deserialization round-trip. The OpenAI client raises immediately if
   `OPENAI_API_KEY` is absent. Fixed with
   `monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")`.

---

### 3.4 Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R-1 | LLM cost — one extra call per query increases operating cost | High (by design) | Medium | Use `run_async`; consider query-level caching (future sprint) |
| R-2 | LLM latency — extra round-trip adds 200–1 500 ms per query | High (by design) | Medium | Async interface minimises pipeline blocking |
| R-3 | LLM output quality — weaker models produce noisy α values | Medium | High | Scoring prompt is verbatim from the paper; evaluate empirically in Sprint 3 |
| R-4 | ID deduplication gap — same document with different IDs scored twice | Low (InMemoryDocumentStore guarantees stable IDs) | Low | Document assumption; revisit for external stores |

---

## 4. Statistics

### 4.1 Development Stats

| Field | Value |
|-------|-------|
| Start Timestamp | 2026-02-12 02:13:34 |
| End Timestamp | 2026-02-12 02:20:54 |
| Duration | 7 min 20 s |
| Modules | 2 new (`dat_document_joiner.py`, `test_dat_document_joiner.py`); 1 modified (`__init__.py`) |
| Classes | 1 production (`DATDocumentJoiner`); 12 test classes |
| LOC | 547 (implementation) + 546 (tests) + 3 (init) = **1 096 total** |

### 4.2 Testing Status

| Tests | Tested | Passed | % Passed |
|-------|-------:|-------:|---------:|
| Unit Test | 53 | 53 | 100% |
| Integration Test | 0 | 0 | — |
| Regression Test | 0 | 0 | — |

Integration and regression tests are scheduled for Sprint 2 and Sprint 3
respectively.

#### Unit test breakdown

| Test Class | Tests | Focus |
|------------|------:|-------|
| `TestDATDocumentJoinerInit` | 4 | Defaults, custom params, invalid `top_k` / `scoring_top_k` |
| `TestNormalizeScores` | 6 | Standard, arbitrary range, delta=0, empty, no-mutation, `None` score |
| `TestComputeAlpha` | 7 | All paper edge cases, proportional weighting, rounding, full grid sweep |
| `TestParseScores` | 5 | Valid response, surrounding text, `ComponentError`, fallback, out-of-range |
| `TestFuse` | 5 | Combined scores, α=0 (pure BM25), α=1 (pure dense), empty, no-mutation |
| `TestRunEdgeCases` | 5 | Both empty, empty dense, empty BM25, `top_k` limit, `top_k` override |
| `TestRunStandard` | 9 | Return shape, α value, sort order, LLM called once, prompt content, parse-failure raise/fallback, perfect-hit shortcuts |
| `TestRunAsync` | 4 | Both empty, empty dense, uses `run_async`, sync fallback |
| `TestWarmUp` | 3 | Called once, absent method, triggered by `run` |
| `TestTelemetry` | 1 | Data shape |
| `TestSerialization` | 3 | `to_dict`, `from_dict` round-trip, default params |
| `TestPackageExport` | 1 | `from haystack.components.joiners import DATDocumentJoiner` |
| **Total** | **53** | |
