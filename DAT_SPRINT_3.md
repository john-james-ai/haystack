# DAT Sprint 3 Summary

**Sprint:** 3 — Evaluation Framework
**Date:** 2026-02-12
**Status:** Complete

---

## 1. Implementation Summary

Sprint 3 delivers the `DocumentPrecisionEvaluator` component — the final building block of the DAT evaluation framework. Together with the existing `DocumentMRREvaluator`, `DocumentRecallEvaluator`, and `DocumentNDCGEvaluator`, it now provides full metric coverage for benchmarking hybrid retrieval systems against the DAT paper's evaluation protocol.

All 8 issues from `DAT_SPRINT_ISSUES.md` (I-01 through I-08) were resolved prior to this sprint.

---

## 2. Implementation Detail

### New Files

| File | Description |
|------|-------------|
| `haystack/components/evaluators/document_precision.py` | `DocumentPrecisionEvaluator` component |
| `test/components/evaluators/test_document_precision.py` | 20 tests: unit, serialisation, export, regression |

### Modified Files

| File | Change |
|------|--------|
| `haystack/components/evaluators/__init__.py` | Added `DocumentPrecisionEvaluator` to `_import_structure` and `TYPE_CHECKING` block |
| `DAT_SPRINT_ISSUES.md` | All 8 issues marked as Resolved |

### `DocumentPrecisionEvaluator` Interface

```python
@component
class DocumentPrecisionEvaluator:
    def __init__(self, k: int = 1): ...

    @component.output_types(score=float, individual_scores=list[float])
    def run(
        self,
        ground_truth_documents: list[list[Document]],
        retrieved_documents: list[list[Document]],
    ) -> dict[str, Any]: ...

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentPrecisionEvaluator": ...
```

**Formula:** `Precision@K = |{relevant docs} ∩ {top-K retrieved docs}| / K`

**Default `k=1`** directly matches the Precision@1 metric used in the DAT paper
(Hsu & Tzeng, 2025, arXiv:2503.23013) as the primary evaluation criterion.

**Distinction from `DocumentMAPEvaluator`:**

| Metric | Formula | Paper use |
|--------|---------|-----------|
| MAP | Mean of average precision at every rank where a relevant doc appears | Not used in DAT paper |
| Precision@K | `\|{relevant} ∩ {top-K}\| / K` — position-insensitive within top-K | Primary metric in DAT paper |

At K=1, Precision@1 equals Hit Rate@1 and cannot be derived from MAP.

---

## 3. Insights

### Design Decisions

**DD8 — `DocumentPrecisionEvaluator` over extending existing evaluators.**
`DocumentMAPEvaluator` was considered as a base, but MAP and Precision@K are mathematically distinct. A new stateless component is the correct choice, matching the pattern of all existing evaluators.

**DD9 — Denominator is always `k`, not `len(retrieved[:k])`.**
Using `k` as the fixed denominator (even when fewer than `k` docs are retrieved) is consistent with the standard information retrieval definition of Precision@K. It correctly penalises under-retrieval by treating missing slots as misses.

**DD10 — Regression tests use `DATDocumentJoiner` directly (not `InMemoryDocumentStore`).**
Bypassing the document store gives exact control over dense and BM25 scores, making the regression tests fully deterministic without any BM25 tokenisation heuristics. The two-scenario design (Scenario A: dense wins; Scenario B: BM25 wins) validates the DAT paper's central claim end-to-end.

### Open Questions

None outstanding. Sprint 3 closes all design concerns flagged in `DAT_SPRINT_ISSUES.md`.

### Challenges

**Regression test score design.** With two documents and exactly orthogonal normalized scores (correct has dense=1.0/BM25=0.0; decoy has dense=0.0/BM25=1.0), the fixed hybrid always ties at 0.5 each, making Precision@1 comparison vacuous. Resolution: introduce a third "noise" document and use asymmetric intermediate scores (e.g., `dense_decoy=0.7` vs `bm25_correct=1.0`) so that the fixed hybrid ranks the decoy first while DAT correctly ranks the true answer first.

### Risks

**Denominator edge case:** When `k > len(retrieved)` for a query, `Precision@K < 1.0` even when all retrieved docs are relevant (e.g., 1 relevant doc with `k=3` → P@3 = 1/3, not 1.0). This is mathematically correct per the standard definition but may surprise users expecting a "perfect" score. Addressed in docstring.

---

## 4. Statistics

### Development

| Metric | Value |
|--------|-------|
| Sprint start | 2026-02-12 |
| Sprint end | 2026-02-12 |
| New source files | 1 |
| Modified source files | 1 |
| New test files | 1 |
| Lines of source code (new) | ~130 |
| Lines of test code (new) | ~220 |

### Testing

| Category | Tests | Pass | Fail |
|----------|-------|------|------|
| Unit — init & validation | 4 | 4 | 0 |
| Unit — precision computation | 9 | 9 | 0 |
| Unit — serialisation | 3 | 3 | 0 |
| Unit — package export | 1 | 1 | 0 |
| Regression — DAT vs fixed hybrid | 2 | 2 | 0 |
| **Sprint 3 total** | **20** | **20** | **0** |
| **Cumulative total** | **119** | **119** | **0** |
