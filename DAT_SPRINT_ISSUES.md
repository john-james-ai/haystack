# DAT Sprint Issues

**Date:** 2026-02-12
**Status:** Resolved
**Scope:** Post-Sprint-2 critical review of all sprint designs

---

## Summary

| ID | Sprint | Severity | Category | Status |
|----|--------|----------|----------|--------|
| I-01 | S2 | High | Missing test | Resolved |
| I-02 | S2 | High | Missing test | Resolved |
| I-03 | S3 | Medium | Design concern | Resolved |
| I-04 | S3 | Medium | Test scoping | Resolved |
| I-05 | S4 | Low | Scope concern | Resolved |
| I-06 | S1/S2 | Low | Functional gap | Resolved |
| I-07 | S2 | Low | Functional gap | Resolved |
| I-08 | S1/S2 | Low | Documentation gap | Resolved |

---

## I-01 — `InMemoryDATHybridRetriever` pipeline wiring test missing

**Sprint:** 2
**Severity:** High
**Category:** Missing integration test

**Description.**
The `TestIntegration` class in `test_in_memory_dat_hybrid_retriever.py` tests
`retriever.run()` directly. There is no test that wires `InMemoryDATHybridRetriever`
into an actual `Pipeline` object (`pipeline.add_component` / `pipeline.connect` /
`pipeline.run`).

This matters because the Haystack component machinery — input/output type checking,
socket wiring validation, and component lifecycle management — is only exercised
through the `Pipeline`. Calling `run()` directly bypasses it entirely. A wiring
error (e.g. a wrong output socket name, missing required input, type mismatch) would
not be caught by any current test.

**Strategy reference:**
> `test_inmemorydathybrid_pipeline_wiring` (Sprint 2 test plan)

**Expected test:**
```python
def test_inmemorydathybrid_pipeline_wiring(populated_store, mock_generator):
    pipeline = Pipeline()
    pipeline.add_component(
        "dat_retriever",
        InMemoryDATHybridRetriever(
            document_store=populated_store,
            chat_generator=mock_generator,
            top_k=3,
        ),
    )
    result = pipeline.run({
        "dat_retriever": {
            "query": "capital of France",
            "query_embedding": [1.0, 0.0, 0.0],
        },
    })
    assert "dat_retriever" in result
    assert "documents" in result["dat_retriever"]
    assert "alpha" in result["dat_retriever"]
```

**Resolution:** Add test to
`test/components/retrievers/test_in_memory_dat_hybrid_retriever.py` before or
during Sprint 3.

---

## I-02 — `InMemoryDATHybridRetriever` pipeline serialisation round-trip test missing

**Sprint:** 2
**Severity:** High
**Category:** Missing integration test

**Description.**
The serialisation round-trip for a pipeline containing `DATDocumentJoiner` is tested
in `test_dat_joiner_pipeline_round_trip`. No equivalent test exists for a pipeline
containing `InMemoryDATHybridRetriever`.

`InMemoryDATHybridRetriever.to_dict` / `from_dict` is more complex than
`DATDocumentJoiner`'s because it nests two serialisable objects: `InMemoryDocumentStore`
(auto-serialised by `default_to_dict`) and `chat_generator` (serialised via
`component_to_dict` + `deserialize_chatgenerator_inplace`). The component-level
`test_from_dict` unit test validates `InMemoryDATHybridRetriever.from_dict` in
isolation, but does not exercise the full `Pipeline.to_dict` → `Pipeline.from_dict`
path, which includes additional registry lookups and nested deserialization logic.

**Strategy reference:**
> `test_pipeline_serialization_roundtrip (InMemoryDATHybridRetriever)` (Sprint 2 test plan)

**Expected test:**
```python
def test_inmemorydathybrid_pipeline_round_trip(populated_store, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")
    from haystack.components.generators.chat import OpenAIChatGenerator

    pipeline = Pipeline()
    pipeline.add_component(
        "dat_retriever",
        InMemoryDATHybridRetriever(
            document_store=populated_store,
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            top_k=4,
            raise_on_failure=False,
        ),
    )
    serialised = pipeline.to_dict()
    restored = Pipeline.from_dict(serialised)

    restored_retriever = restored.get_component("dat_retriever")
    assert isinstance(restored_retriever, InMemoryDATHybridRetriever)
    assert restored_retriever.top_k == 4
    assert restored_retriever.raise_on_failure is False
```

**Resolution:** Add test to
`test/components/retrievers/test_in_memory_dat_hybrid_retriever.py` (or a new
`TestPipelineSerializationRoundTrip` class there) before or during Sprint 3.

---

## I-03 — `DocumentPrecisionEvaluator` vs. existing `DocumentMAPEvaluator`

**Sprint:** 3
**Severity:** Medium
**Category:** Design concern

**Description.**
`DocumentMAPEvaluator` already exists. MAP (Mean Average Precision) and Precision@K
are frequently confused. They are distinct metrics:

| Metric | Formula | Paper use |
|--------|---------|-----------|
| MAP | Mean of average precision at every rank where a relevant doc appears | Not used in DAT paper |
| Precision@K | (# relevant docs in top-K) / K | Primary metric in DAT paper |

Precision@1 (the DAT paper's primary metric) is **not** covered by any existing
evaluator in the suite. The new `DocumentPrecisionEvaluator` is still warranted, but
Sprint 3 planning documents should explicitly state:

1. Why `DocumentPrecisionEvaluator` is distinct from `DocumentMAPEvaluator`.
2. That the default `k=1` directly matches the paper's Precision@1 metric.
3. The formula: `Precision@K = |{relevant docs} ∩ {top-K retrieved docs}| / K`.

The `DocumentMAPEvaluator` docstring should be referenced in the new evaluator's
docstring to help users choose the right metric.

**Resolution:** Add explicit note to Sprint 3 planning and `DocumentPrecisionEvaluator`
docstring before implementation.

---

## I-04 — Regression test for Sprint 3 needs deterministic scoping

**Sprint:** 3
**Severity:** Medium
**Category:** Test scoping

**Description.**
The strategy lists:
> `regression_test_dat_vs_fixed_alpha_on_sample_corpus`

Without a real embedding model and real LLM, any regression test will use synthetic
data. The risk is that an under-scoped test either:
- **Passes trivially** regardless of algorithm correctness (vacuous mock), or
- **Is too coupled to hand-crafted fixtures** and breaks on minor refactors.

**Proposed deterministic regression test design:**

Use two complementary scenarios, each with hand-crafted embeddings and a mocked LLM,
that exercise the end-to-end value proposition of DAT:

**Scenario A — Dense wins:**
- Index documents such that the correct answer has embedding `[1.0, 0.0]`, and a
  distracting document has high BM25 score but embedding `[0.0, 1.0]`.
- Query embedding: `[1.0, 0.0]`.
- LLM returns `"5 0"` → α = 1.0 → DAT ranks the correct answer first.
- Fixed hybrid (α = 0.5) may not rank the correct answer first (BM25 noise pulls it down).
- Assert: `Precision@1(DAT) >= Precision@1(fixed α=0.5)`.

**Scenario B — BM25 wins:**
- Mirror of Scenario A: correct answer has high BM25 score; poor embedding similarity.
- LLM returns `"0 5"` → α = 0.0 → DAT ranks the correct answer first.
- Assert: `Precision@1(DAT) >= Precision@1(fixed α=0.5)`.

This design is fully deterministic (no real models), directly validates the paper's
central claim, and is robust to refactors.

**Resolution:** Define fixture design above in Sprint 3 planning before implementation.

---

## I-05 — Sprint 4 documentation scope may be redundant

**Sprint:** 4
**Severity:** Low
**Category:** Scope concern

**Description.**
Sprint 4 as planned ("update docstrings for all three new components") may produce
little net value because `DATDocumentJoiner` and `InMemoryDATHybridRetriever` already
have comprehensive class-level docstrings, `__init__` param docs, `run` param/returns/raises
docs, and usage examples.

**Recommended Sprint 4 additions to provide real value:**

| Item | Value |
|------|-------|
| `DAT_USAGE_GUIDE.md` — end-to-end walkthrough: indexing, hybrid retrieval, evaluation | High — users need a copy-paste guide |
| Verify all docstring code examples are executable (doc-test or manual check) | Medium — prevents stale examples |
| `CHANGELOG` entry / release note stub for all three new components | Low-Medium — needed for upstream PR |
| Add `:note:` in `DATDocumentJoiner` docstring that `scoring_top_k > 1` is experimental | Low |

**Resolution:** Revise Sprint 4 scope in `DAT_STRATEGY_V2.md` to include the usage
guide and docstring example verification.

---

## I-06 — `scoring_top_k > 1` concatenated content shape not tested

**Sprint:** S1 / S2
**Severity:** Low
**Category:** Functional gap

**Description.**
When `scoring_top_k > 1`, `_call_llm` (and `_call_llm_async`) concatenate document
content for both retrievers:
```python
vector_reference = " ".join(d.content or "" for d in dense_top).strip()
bm25_reference   = " ".join(d.content or "" for d in bm25_top).strip()
```

The DAT prompt template labels these as `"dense retrieval Top1 Result"` and
`"BM25 retrieval Top1 Result"`, which is misleading when `scoring_top_k > 1`.
No unit test currently verifies:
1. That the concatenated content string is correctly constructed.
2. That the prompt sent to the LLM reflects multiple documents when `scoring_top_k > 1`.
3. That `vector_reference` / `bm25_reference` are empty strings when all doc content is `None`
   (which would send an empty reference to the LLM, potentially causing a parse failure).

**Resolution:** Add 2–3 targeted unit tests in `TestRunStandard` for `scoring_top_k=2`
and all-`None`-content edge case. Consider adding a `:note:` in the `scoring_top_k`
param docstring that prompt labels say "Top1" regardless of `scoring_top_k`.

---

## I-07 — Same `scale_score` applied to both BM25 and embedding retrieval

**Sprint:** S2
**Severity:** Low
**Category:** Functional gap

**Description.**
`InMemoryDATHybridRetriever` applies a single `scale_score` value to both
`bm25_retrieval()` and `embedding_retrieval()`. BM25 and cosine similarity scores
have fundamentally different raw ranges:
- BM25 (raw): typically 0–20+, right-skewed, query-length dependent
- Cosine similarity (raw): typically 0–1

When `scale_score=False` (default), DAT's own min-max normalisation handles the
difference correctly. However, when `scale_score=True`, both scores are individually
scaled to [0, 1] before DAT normalises them again — resulting in redundant double
normalisation.

There is no test that documents this behaviour or validates that `scale_score=True`
produces reasonable results.

**Resolution:** Add a note to the `scale_score` docstring in `InMemoryDATHybridRetriever`
explaining the redundancy with DAT normalisation. Consider adding a test that verifies
`scale_score=True` and `scale_score=False` both produce valid α values (as a smoke test).

---

## I-08 — Missing docstring note: `query_embedding` dimension must match indexed embeddings

**Sprint:** S1 / S2
**Severity:** Low
**Category:** Documentation gap

**Description.**
Neither `DATDocumentJoiner` nor `InMemoryDATHybridRetriever` documents that
`query_embedding` must have the same dimension as the embeddings stored in the
document store. Dimension mismatch is caught downstream by `InMemoryDocumentStore`
with an `IndexError` or silent empty result (depending on the numpy broadcast path),
which produces a confusing error message.

**Resolution:** Add a `:note:` to the `query_embedding` param docstring in
`InMemoryDATHybridRetriever.run()` and `run_async()`.

---

## Prioritised Resolution Order

| Priority | Issue | Rationale |
|----------|-------|-----------|
| 1 | I-01 | High-severity, missing test, 20 lines to fix |
| 2 | I-02 | High-severity, missing test, 25 lines to fix |
| 3 | I-03 | Needed before Sprint 3 implementation starts |
| 4 | I-04 | Needed before Sprint 3 test implementation starts |
| 5 | I-05 | Needed before Sprint 4 scope is finalised |
| 6 | I-06 | Small test additions, low risk |
| 7 | I-07 | Documentation only |
| 8 | I-08 | Documentation only |
