# DAT-RAG Strategy V2
## Dynamic Alpha Tuning for Hybrid Retrieval in Haystack

**Author:** Senior Architect & Developer
**Date:** 2026-02-12
**Status:** Revised — Addresses V1 Review Comments
**Reference:** Hsu & Tzeng (2025), *DAT: Dynamic Alpha Tuning for Hybrid Retrieval in RAG*, arXiv:2503.23013

### V2 Changes from V1
| Section | Change |
|---------|--------|
| Risk R2 | **Breaking change**: LLM parse failure now raises `ComponentError` by default (`raise_on_failure=True`); silent fallback removed. Matches `LLMEvaluator` precedent. |
| Architecture | Added `InMemoryDATHybridRetriever` — a self-contained hybrid retriever for `InMemoryDocumentStore` (responds to review feedback that `DocumentJoiner` is functionally a hybrid retriever). |
| Package Structure | Added `retrievers/in_memory/dat_hybrid_retriever.py` and associated test files. |
| Impact Analysis | Added `InMemoryDATHybridRetriever`, updated test entries. |
| Design Decisions | Added DD6: `InMemoryDATHybridRetriever` rationale. |
| Sprint Roadmap | Sprint 2 now includes `InMemoryDATHybridRetriever`. Added Sprint 4: Documentation Sprint. |

---

## Table of Contents
1. [Algorithm Summary](#1-algorithm-summary)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Package Structure](#3-package-structure)
4. [Impact Analysis](#4-impact-analysis)
5. [Risk Analysis](#5-risk-analysis)
6. [Clarifying Questions](#6-clarifying-questions)
7. [Design Decisions](#7-design-decisions)
8. [Sprint Roadmap](#8-sprint-roadmap)

---

## 1. Algorithm Summary

DAT (Dynamic Alpha Tuning) replaces the static weighting coefficient (α) in hybrid retrieval with a per-query, LLM-determined weight that reflects each retriever's effectiveness for the specific query–document relationship.

### 1.1 Step-by-Step Algorithm

**Inputs:** query `q`, document corpus `D`
**Output:** ranked list of top-K documents `D_final(q)`

**Step 1 — Dual Retrieval**
Retrieve top-K candidates from both retrievers:
- `D_dense(q)`: top-K documents from dense (embedding) retriever
- `D_BM25(q)`: top-K documents from BM25 (sparse) retriever

**Step 2 — Score Normalization (min-max scaling)**
Normalize scores within each retriever's result set to [0, 1]:

```
S̃_dense(q,d) = [S_dense(q,d) − min(S_dense)] / [max(S_dense) − min(S_dense)]
S̃_BM25(q,d)  = [S_BM25(q,d)  − min(S_BM25)]  / [max(S_BM25)  − min(S_BM25)]
```

**Step 3 — LLM Effectiveness Scoring**
Using only the top-1 document from each retriever:
- `d_v,1 ∈ D_dense(q)`: top-1 dense result
- `d_b,1 ∈ D_BM25(q)`: top-1 BM25 result

Prompt the LLM with `(q, d_v,1, d_b,1)` using the scoring rubric (0–5):
- **5**: Direct hit — document directly answers the question
- **3–4**: Good wrong result — conceptually close, likely correct answers nearby
- **1–2**: Bad wrong result — loosely related but misleading
- **0**: Completely off-track — totally unrelated

Returns: `S_v(q)` (dense score), `S_b(q)` (BM25 score)

**Step 4 — Dynamic Alpha Calculation**

```
α(q) = 0.5                       if S_v = 0 AND S_b = 0
     = 1.0                       if S_v = 5 AND S_b ≠ 5    (prefer dense)
     = 0.0                       if S_b = 5 AND S_v ≠ 5    (prefer BM25)
     = S_v / (S_v + S_b)         otherwise
```

α(q) is rounded to 1 decimal place.

**Step 5 — Score Fusion & Ranking**

```
R(q, d) = α(q) · S̃_dense(q,d) + (1 − α(q)) · S̃_BM25(q,d)
```

Documents are ranked by `R(q, d)` descending; top-K returned.

### 1.2 LLM Prompt Template (verbatim from paper, Appendix A)

```
You are an evaluator assessing the retrieval effectiveness of dense
retrieval (Cosine Distance) and BM25 retrieval for finding the correct answer.

## Task:
Given a question and two top1 search results (one from dense retrieval,
one from BM25 retrieval), score each retrieval method from **0 to 5**
based on whether the correct answer is likely to appear in top2, top3, etc.

### **Scoring Criteria:**
1. **Direct hit --> 5 points**
   - If the retrieved document directly answers the question, assign **5 points**.
2. **Good wrong result (High likelihood correct answer is nearby) --> 3-4 points**
   - If the top1 result is **conceptually close** to the correct answer
     (e.g., mentions relevant entities, related events, partial answer),
     it indicates the search method is in the right direction.
   - Give **4** if it's very close, **3** if somewhat close.
3. **Bad wrong result (Low likelihood correct answer is nearby) --> 1-2 points**
   - If the top1 result is **loosely related but misleading** (e.g.,
     shares keywords but changes context), correct answers might not be in top2, top3.
   - Give **2** if there's a small chance correct answers are nearby, **1** if unlikely.
4. **Completely off-track --> 0 points**
   - If the result is **totally unrelated**, it means the retrieval method is failing.

---
### **Given Data:**
- **Question:** "{question}"
- **dense retrieval Top1 Result:** "{vector_reference}"
- **BM25 retrieval Top1 Result:** "{bm25_reference}"

---
### **Output Format:**
Return two integers separated by a space:
- **First number:** dense retrieval score.
- **Second number:** BM25 retrieval score.
- Example output: 3 4
(Vector: 3, BM25: 4)
**Do not output any other text.**
```

### 1.3 Empirical Results Summary

| Method | SQuAD P@1 | SQuAD MRR@20 | DRCD P@1 | DRCD MRR@20 |
|--------|-----------|--------------|----------|-------------|
| BM25 Only (α=0.0) | 0.7594 | 0.8223 | 0.7630 | 0.8134 |
| Dense Only (α=1.0) | 0.7396 | 0.8119 | 0.5743 | 0.6708 |
| Fixed Hybrid (α=0.6) | 0.8461 | 0.8997 | 0.8113 | 0.8619 |
| **DAT (GPT-4o)** | **0.8740** | **0.9130** | **0.8440** | **0.8807** |
| DAT (GPT-4o-mini) | 0.8676 | 0.9093 | 0.8417 | 0.8796 |
| DAT (DeepSeek-14B) | 0.8663 | 0.9079 | 0.8347 | 0.8711 |

---

## 2. Architecture Diagram

### 2.1 Generic DAT Pipeline (DATDocumentJoiner)

```
╔══════════════════════════════════════════════════════════════════╗
║             Generic DAT Hybrid Retrieval Pipeline               ║
║          (works with any document store / retriever pair)       ║
╚══════════════════════════════════════════════════════════════════╝

  User Query
      │
      ├─────────────────────────────────────────────┐
      │                                             │
      ▼                                             ▼
┌─────────────────────┐              ┌──────────────────────────┐
│  TextEmbedder       │              │  [Any] BM25 Retriever    │
│  (embedding model)  │              │  e.g. InMemoryBM25Retr.  │
└──────────┬──────────┘              └────────────┬─────────────┘
           │ query_embedding                      │ bm25_documents
           ▼                                      │
┌─────────────────────────┐                       │
│  [Any] Dense Retriever  │                       │
│  e.g. InMemoryEmbRetr.  │                       │
└──────────┬──────────────┘                       │
           │ dense_documents                       │
           └──────────────┬────────────────────────┘
                          │
                          ▼
          ┌───────────────────────────────────────┐
          │         DATDocumentJoiner             │
          │                                       │
          │  [ChatGenerator injected at init]     │
          │                                       │
          │  1. Normalize scores (min-max)         │
          │  2. LLM scores top-1 from each        │
          │  3. Compute α(q) ∈ [0.0, 1.0]        │
          │  4. Fuse: R = α·S̃_v + (1-α)·S̃_b   │
          │  5. Rank → top-K                      │
          └───────────────────────────────────────┘
                          │
                          ▼
              Documents (top-K) + alpha
```

### 2.2 InMemory DAT Pipeline (InMemoryDATHybridRetriever) ← NEW

```
╔══════════════════════════════════════════════════════════════════╗
║           InMemory DAT Hybrid Retrieval Pipeline                ║
║      (self-contained: one component replaces three)             ║
╚══════════════════════════════════════════════════════════════════╝

  User Query
      │
      ▼
┌─────────────────────┐
│   TextEmbedder      │
└──────────┬──────────┘
           │ query + query_embedding
           ▼
┌──────────────────────────────────────────────┐
│        InMemoryDATHybridRetriever            │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │        InMemoryDocumentStore           │  │
│  │  bm25_retrieval(query)        → docs_b │  │
│  │  embedding_retrieval(embed)   → docs_v │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  [ChatGenerator injected at init]            │
│                                              │
│  1. Normalize scores (min-max)               │
│  2. LLM scores top-1 from each              │
│  3. Compute α(q) ∈ [0.0, 1.0]             │
│  4. Fuse: R = α·S̃_v + (1-α)·S̃_b        │
│  5. Rank → top-K                            │
└──────────────────────────────────────────────┘
           │
           ▼
  Documents (top-K) + alpha
```

### 2.3 Evaluation Pipeline

```
  Ground Truth Docs ──► DocumentPrecisionEvaluator  (Precision@K) ← NEW
  Retrieved Docs    ──► DocumentMRREvaluator         (MRR@K)
                   ──► DocumentRecallEvaluator       (Recall@K)
                   ──► DocumentNDCGEvaluator         (nDCG)
```

### 2.4 Component Summary

| Component | Role | Sprint | Source |
|-----------|------|--------|--------|
| `TextEmbedder` | Encodes query for dense retrieval | — | Existing |
| `InMemoryBM25Retriever` | Lexical retrieval (generic pipeline) | — | Existing |
| `InMemoryEmbeddingRetriever` | Semantic retrieval (generic pipeline) | — | Existing |
| `InMemoryDocumentStore` | Storage with both BM25 + embedding retrieval | — | Existing |
| `DATDocumentJoiner` | Generic DAT fusion (any two retrievers) | S1 | **NEW** |
| `InMemoryDATHybridRetriever` | Self-contained DAT for InMemoryDocumentStore | S2 | **NEW** |
| `OpenAIChatGenerator` | LLM effectiveness scorer (injected) | — | Existing |
| `DocumentPrecisionEvaluator` | Precision@K metric | S3 | **NEW** |
| `DocumentMRREvaluator` | MRR metric | — | Existing |
| `DocumentRecallEvaluator` | Recall@K metric | — | Existing |
| `DocumentNDCGEvaluator` | nDCG metric | — | Existing |

---

## 3. Package Structure

```
haystack/
├── components/
│   ├── joiners/
│   │   ├── __init__.py                                  [MODIFIED — S1]
│   │   ├── answer_joiner.py
│   │   ├── branch.py
│   │   ├── dat_document_joiner.py                       [NEW — S1]
│   │   ├── document_joiner.py
│   │   ├── list_joiner.py
│   │   └── string_joiner.py
│   ├── retrievers/
│   │   ├── __init__.py                                  [MODIFIED — S2]
│   │   ├── in_memory/
│   │   │   ├── __init__.py                              [MODIFIED — S2]
│   │   │   ├── bm25_retriever.py
│   │   │   ├── dat_hybrid_retriever.py                  [NEW — S2]
│   │   │   └── embedding_retriever.py
│   │   └── ... (other retrievers unchanged)
│   └── evaluators/
│       ├── __init__.py                                  [MODIFIED — S3]
│       ├── document_precision.py                        [NEW — S3]
│       └── ... (other evaluators unchanged)
test/
├── components/
│   ├── joiners/
│   │   ├── test_dat_document_joiner.py                  [NEW — S1]
│   │   └── test_dat_document_joiner_integration.py      [NEW — S2]
│   ├── retrievers/
│   │   └── test_in_memory_dat_hybrid_retriever.py       [NEW — S2]
│   └── evaluators/
│       └── test_document_precision.py                   [NEW — S3]
```

---

## 4. Impact Analysis

### 4.1 New Files

| File | Class | Sprint | Description |
|------|-------|--------|-------------|
| `haystack/components/joiners/dat_document_joiner.py` | `DATDocumentJoiner` | S1 | Generic DAT joiner. Accepts `query + dense_documents + bm25_documents` from any retriever pair. Normalizes scores, calls LLM, computes α, fuses and ranks. |
| `haystack/components/retrievers/in_memory/dat_hybrid_retriever.py` | `InMemoryDATHybridRetriever` | S2 | Self-contained DAT retriever for `InMemoryDocumentStore`. Internally calls `bm25_retrieval()` and `embedding_retrieval()` then applies DAT. Reduces pipeline from 3 components to 1. |
| `haystack/components/evaluators/document_precision.py` | `DocumentPrecisionEvaluator` | S3 | Precision@K evaluator. Configurable K (default 1, matching paper). Stateless `@component` following existing evaluator pattern. |
| `test/components/joiners/test_dat_document_joiner.py` | `TestDATDocumentJoiner` | S1 | Unit tests: all α cases, normalization edge cases, LLM parsing, `raise_on_failure`, serialization, async. |
| `test/components/joiners/test_dat_document_joiner_integration.py` | `TestDATDocumentJoinerIntegration` | S2 | Integration tests: generic pipeline wiring with InMemory stores + mocked LLM. |
| `test/components/retrievers/test_in_memory_dat_hybrid_retriever.py` | `TestInMemoryDATHybridRetriever` | S2 | Unit + integration tests for self-contained retriever: init, serialization, run, async, full pipeline. |
| `test/components/evaluators/test_document_precision.py` | `TestDocumentPrecisionEvaluator` | S3 | Unit tests for Precision@K evaluator. |

### 4.2 Modified Files

| File | Change | Sprint |
|------|--------|--------|
| `haystack/components/joiners/__init__.py` | Add `DATDocumentJoiner` to `_import_structure` | S1 |
| `haystack/components/retrievers/in_memory/__init__.py` | Add `InMemoryDATHybridRetriever` | S2 |
| `haystack/components/retrievers/__init__.py` | Add `InMemoryDATHybridRetriever` | S2 |
| `haystack/components/evaluators/__init__.py` | Add `DocumentPrecisionEvaluator` | S3 |

### 4.3 Unchanged

`InMemoryBM25Retriever`, `InMemoryEmbeddingRetriever`, `DocumentJoiner`, all existing evaluators — used as-is with no modification.

---

## 5. Risk Analysis

| # | Risk | Probability | Impact | Mitigation |
|---|------|-------------|--------|------------|
| R1 | **LLM Latency**: One LLM call per query adds 100ms–2s overhead | HIGH | HIGH | Async support (`run_async`); users can wrap with Haystack's `CacheChecker`; small models (GPT-4o-mini) shown to match GPT-4o performance |
| **R2** | **LLM Response Parsing Failure**: LLM returns output not matching `"INT INT"` format | MEDIUM | **HIGH** | **Raise `ComponentError` by default** (`raise_on_failure=True`, matching `LLMEvaluator` precedent). Silent α=0.5 fallback would mask a failure in the core DAT mechanism and produce silently degraded, non-adaptive retrieval. When `raise_on_failure=False`, log a `WARNING` and fall back to α=0.5 — reserved for production fault-tolerance use only, not the default. |
| R3 | **Score Normalization Division-by-Zero**: All documents from one retriever have identical scores | LOW | LOW | If `max_score == min_score` (delta=0), set all normalized scores to 0.0. Consistent with `_distribution_based_rank_fusion` precedent in `DocumentJoiner`. |
| R4 | **Empty Document Lists**: One or both retrievers return no documents | MEDIUM | MEDIUM | If dense returns 0 docs: α=0.0 (pure BM25, skip LLM). If BM25 returns 0 docs: α=1.0 (pure dense, skip LLM). If both return 0: return `{"documents": [], "alpha": 0.5}` immediately. |
| R5 | **API Key Dependency in Tests**: LLM API credentials required for end-to-end tests | HIGH | MEDIUM | Unit tests use a mock `ChatGenerator` (implements the Protocol with a fixed response). Integration tests also mock the LLM. E2E tests (if any) guarded with `pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, ...)`. |
| R6 | **Algorithm Generalization**: Paper tested only SQuAD (English) and DRCD (Chinese). May not generalize universally | MEDIUM | MEDIUM | Document limitation clearly in docstrings. Evaluation framework (Sprint 3) allows users to benchmark on their own corpora. Modular design: injectable `ChatGenerator` allows domain-specific prompt customization. |
| R7 | **Production Cost**: One LLM call per query at scale | HIGH | HIGH | Document tradeoff prominently. Recommend GPT-4o-mini as default (cost-efficient, near-equivalent performance). Async design enables concurrent execution. Response caching (stable for same query+top-1 pair) is a natural optimization users can apply. |
| R8 | **Score Rounding Artifacts**: Rounding α to 1 decimal creates discrete jumps | LOW | LOW | Match paper exactly (1-decimal rounding). Expose `round_alpha: bool = True` only if demand arises post-Sprint 1. |
| R9 | **ChatGenerator Protocol Evolution** | LOW | LOW | Type-hint against the `ChatGenerator` Protocol (not a concrete class); mock tests implement minimal Protocol only. |

---

## 6. Clarifying Questions

### Q1. LLM Provider Default
**Recommendation: No default.** Accept any `ChatGenerator` (keyword-only arg). Consistent with `LLMEvaluator`. Provider-agnostic.

---

### Q2. Is DocumentJoiner a Hybrid Retriever? *(Addressed in V2)*

**Context (V1 review):** The reviewer noted that `DocumentJoiner` is functionally a hybrid retriever and should be treated as such. The documentation points to `InMemoryDocumentStore` as supporting both BM25 and embedding retrieval natively.

**Analysis:**
`DocumentJoiner` accepts the *results* of two independent retrievers and fuses them — it is the fusion layer of a hybrid retrieval system. However, it is not itself a retriever: it does not query a document store. The `InMemoryDocumentStore` exposes both `bm25_retrieval(query)` and `embedding_retrieval(query_embedding)` methods, enabling a single component to perform both retrieval phases internally.

**Decision:** Implement at **two levels**:

| Level | Component | Location | Use Case |
|-------|-----------|----------|----------|
| Generic (store-agnostic) | `DATDocumentJoiner` | `joiners/` | Any two retriever types (external stores, cloud retrievers, custom pipelines) |
| Convenience (InMemory) | `InMemoryDATHybridRetriever` | `retrievers/in_memory/` | `InMemoryDocumentStore` users; single-component simplicity |

`InMemoryDATHybridRetriever` follows the same naming convention as `InMemoryBM25Retriever` and `InMemoryEmbeddingRetriever`, and internally orchestrates both `bm25_retrieval()` and `embedding_retrieval()` from the same store.

---

### Q3. Top-K for Retrieval vs. Scoring
**Recommendation: `scoring_top_k=1` default (matching paper), separate from `top_k`.** Paper explicitly justifies top-1 as sufficient signal.

---

### Q4. Alpha Rounding
**Recommendation: Always round to 1 decimal (matching paper).** Configurable rounding deferred to a future sprint.

---

### Q5. Component Placement
**Resolved by Q2:** `DATDocumentJoiner` → `joiners/`; `InMemoryDATHybridRetriever` → `retrievers/in_memory/`.

---

## 7. Design Decisions

### DD1. Standalone DATDocumentJoiner vs. Extending DocumentJoiner
**Decision: Standalone component.** DAT requires named inputs, query string, and an injected LLM — incompatible with `DocumentJoiner`'s `Variadic`-input, query-unaware interface without breaking changes.

---

### DD2. Named vs. Variadic Document Inputs
**Decision: Named `dense_documents` / `bm25_documents`.** DAT is explicitly a two-retriever algorithm. Named inputs are unambiguous and self-documenting.

---

### DD3. LLM Integration: Injected ChatGenerator
**Decision: Inject any `ChatGenerator` via `__init__`.** Follows `LLMEvaluator` pattern. Provider-agnostic, fully testable.

---

### DD4. Async Support
**Decision: Both `run()` and `run_async()`.** Consistent with `InMemoryBM25Retriever` and `InMemoryEmbeddingRetriever`. `InMemoryDocumentStore` exposes `bm25_retrieval_async()` and `embedding_retrieval_async()`.

---

### DD5. Precision@K Evaluator
**Decision: New `DocumentPrecisionEvaluator`.** Fills a gap in the existing evaluator suite. Directly matches the paper's primary metric (Precision@1). Stateless `@component`, consistent with all existing evaluators.

---

### DD6. InMemoryDATHybridRetriever *(New in V2)*

**Context:** `InMemoryDocumentStore` natively exposes both retrieval methods:
- `bm25_retrieval(query, filters, top_k, scale_score) → list[Document]`
- `embedding_retrieval(query_embedding, filters, top_k, scale_score, return_embedding) → list[Document]`
- Both have async counterparts.

This enables a single component to orchestrate the full hybrid retrieval + DAT fusion.

**Options:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. Use 3-component pipeline only | BM25Retr + EmbRetr + DATDocumentJoiner | Maximally flexible | 3 components to wire; more pipeline boilerplate |
| B. Also provide `InMemoryDATHybridRetriever` (CHOSEN) | Self-contained hybrid retriever for InMemoryDocumentStore | Single component; simpler pipeline; follows naming convention (`InMemoryBM25Retriever`, `InMemoryEmbeddingRetriever`) | Specific to InMemoryDocumentStore only |

**Decision: B — implement both.** The generic `DATDocumentJoiner` remains for use with any retriever pair. `InMemoryDATHybridRetriever` provides a convenience path for the most common development/prototyping scenario and models a true "hybrid retriever" in the spirit of the reviewer's observation.

**Interface:**

```python
@component
class InMemoryDATHybridRetriever:
    def __init__(
        self,
        document_store: InMemoryDocumentStore,  # Single store for both retrieval types
        chat_generator: ChatGenerator,           # LLM scorer
        top_k: int = 10,
        scoring_top_k: int = 1,
        scale_score: bool = False,
        filters: dict[str, Any] | None = None,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
        raise_on_failure: bool = True,
    ): ...

    @component.output_types(documents=list[Document], alpha=float)
    def run(
        self,
        query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict: ...

    async def run_async(self, ...) -> dict: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InMemoryDATHybridRetriever": ...
```

---

### DD7. raise_on_failure Behavior *(New in V2)*

**Context (V1 review):** V1 proposed silently falling back to α=0.5 when the LLM returns unparseable output. The reviewer correctly identified this as a potential bug: silently bypassing DAT means the component delivers non-adaptive retrieval without the user knowing.

**Options:**

| Option | Behavior | Pros | Cons |
|--------|----------|------|------|
| A. Always raise | No fallback | Forces users to handle failures | Breaks production pipelines on transient errors |
| B. Always fall back silently | α=0.5 on failure | Never crashes | Masks bugs; DAT is silently non-functional |
| C. `raise_on_failure=True` default (CHOSEN) | Raise by default; fall back only when explicitly opted out | Safe default; transparent failures; matches `LLMEvaluator` | Slightly more user configuration |

**Decision: C.** Default behavior raises `ComponentError` with a clear message. Users who need fault-tolerant production behavior can set `raise_on_failure=False`, which falls back to α=0.5 with a `WARNING` log. This matches `LLMEvaluator`'s `raise_on_failure: bool = True` parameter exactly.

---

## 8. Sprint Roadmap

### Sprint Overview

| Sprint | Title | Key Deliverables |
|--------|-------|-----------------|
| S1 | Core DATDocumentJoiner | Generic joiner + unit tests |
| S2 | InMemoryDATHybridRetriever + Integration | Self-contained retriever + integration tests |
| S3 | Evaluation Framework | Precision@K evaluator + evaluation pipeline |
| S4 | Documentation | Comprehensive docstrings + usage guides |

---

### Sprint 1: Core DATDocumentJoiner

**Goal:** Production-quality generic DAT joiner component with full algorithm, error handling, serialization, and unit test coverage.

**Files Created:**
- `haystack/components/joiners/dat_document_joiner.py`
- `test/components/joiners/test_dat_document_joiner.py`

**Files Modified:**
- `haystack/components/joiners/__init__.py`

**Key Implementation Notes:**
- `raise_on_failure=True` default — raises `ComponentError` on LLM parse failure
- `_get_telemetry_data()` method returning `{"chat_generator": type(self.chat_generator).__name__}`
- Normalization handles delta=0 edge case (all normalized to 0.0)
- Alpha rounded to 1 decimal place
- Empty list handling for both inputs
- `to_dict()` uses `component_to_dict(self.chat_generator)` for generator serialization
- Both `run()` and `run_async()` implemented

**Acceptance Criteria:**
- All 4 alpha cases from Eq. 6 correctly implemented
- `raise_on_failure=True` raises `ComponentError`; `raise_on_failure=False` logs `WARNING` and falls back to α=0.5
- `to_dict()` / `from_dict()` round-trip succeeds
- `run()` and `run_async()` both pass tests

**Test Plan:**
```
test_init_defaults
test_init_custom_params
test_to_dict / test_from_dict
test_normalize_scores_standard
test_normalize_scores_all_identical (delta=0)
test_normalize_scores_empty_list
test_alpha_both_zero → 0.5
test_alpha_dense_perfect → 1.0
test_alpha_bm25_perfect → 0.0
test_alpha_proportional → S_v/(S_v+S_b)
test_alpha_rounding
test_run_returns_fused_documents
test_run_empty_dense_docs (skip LLM, α=0.0)
test_run_empty_bm25_docs (skip LLM, α=1.0)
test_run_both_empty
test_run_llm_parse_failure_raises (raise_on_failure=True)
test_run_llm_parse_failure_fallback (raise_on_failure=False)
test_run_top_k_override
test_run_async
test_run_alpha_in_output
test_telemetry_data
```

---

### Sprint 2: InMemoryDATHybridRetriever + Integration Tests

**Goal:** Self-contained hybrid retriever for `InMemoryDocumentStore` and verified integration of both components in full pipelines.

**Files Created:**
- `haystack/components/retrievers/in_memory/dat_hybrid_retriever.py`
- `test/components/retrievers/test_in_memory_dat_hybrid_retriever.py`
- `test/components/joiners/test_dat_document_joiner_integration.py`

**Files Modified:**
- `haystack/components/retrievers/in_memory/__init__.py`
- `haystack/components/retrievers/__init__.py`

**Key Implementation Notes:**
- `InMemoryDATHybridRetriever.run()` calls `document_store.bm25_retrieval()` and `document_store.embedding_retrieval()` directly — no pipeline wiring needed
- Internally reuses `DATDocumentJoiner`'s normalization and alpha logic (shared private utility or inheritance — see test for decision)
- `run_async()` uses `bm25_retrieval_async()` and `embedding_retrieval_async()` with `asyncio.gather()`
- Filter policy (REPLACE / MERGE) applied consistently

**Acceptance Criteria:**
- `InMemoryDATHybridRetriever` correctly exported from `haystack.components.retrievers`
- Pipeline `TextEmbedder → InMemoryDATHybridRetriever` produces ranked documents
- Generic pipeline `TextEmbedder + BM25Retriever + EmbeddingRetriever → DATDocumentJoiner` verified
- Both pipelines serializable and deserializable
- Alpha value exposed in component output

**Test Plan:**
```
InMemoryDATHybridRetriever:
  test_init_defaults
  test_init_custom_params
  test_to_dict / test_from_dict
  test_run_calls_both_retrieval_methods
  test_run_filter_policy_replace
  test_run_filter_policy_merge
  test_run_empty_store
  test_run_async
  test_raises_on_invalid_document_store_type
  test_export_from_package

Integration:
  test_dat_joiner_pipeline_wiring
  test_inmemorydathybrid_pipeline_wiring
  test_pipeline_serialization_roundtrip (DATDocumentJoiner)
  test_pipeline_serialization_roundtrip (InMemoryDATHybridRetriever)
  test_alpha_reflected_in_output
  test_known_documents_correct_alpha_selection
```

---

### Sprint 3: Evaluation Framework

**Goal:** `DocumentPrecisionEvaluator` and a deterministic regression test validating
DAT against static hybrid retrieval on synthetic data.

**Files Created:**
- `haystack/components/evaluators/document_precision.py`
- `test/components/evaluators/test_document_precision.py`

**Files Modified:**
- `haystack/components/evaluators/__init__.py`

**Key Implementation Notes:**
- Precision@K formula: `|{relevant docs} ∩ {top-K retrieved}| / K`
- Precision@1 (default `k=1`) directly matches the DAT paper's primary metric (P@1).
- Same interface as `DocumentMRREvaluator`:
  `run(ground_truth_documents, retrieved_documents)` → `{"score": float, "individual_scores": list[float]}`
- Stateless — `__init__` takes only `k: int = 1`.

**Distinction from `DocumentMAPEvaluator` (I-03):**
`DocumentMAPEvaluator` already exists and computes *Mean Average Precision* — the mean
of average precision values computed at every rank where a relevant document appears.
`DocumentPrecisionEvaluator` computes *Precision@K* — the fraction of the top-K
retrieved documents that are relevant. These are different metrics:

| Metric | Considers rank beyond K? | Paper use |
|--------|--------------------------|-----------|
| MAP | Yes (averages over all relevant ranks) | Not used in DAT paper |
| Precision@K | No (only top-K) | Primary metric in DAT paper |

At K=1, Precision@1 = Hit Rate@1, which is not computable from MAP.
The `DocumentPrecisionEvaluator` docstring must reference `DocumentMAPEvaluator` to
help users choose the right metric.

**Acceptance Criteria:**
- `DocumentPrecisionEvaluator(k=1).run(...)` returns Precision@1 matching paper definition
- Exported from `haystack.components.evaluators`
- Regression test passes (see below)

**Regression Test Design (I-04):**
The regression test must be fully deterministic — no real embedding model or LLM.
Use two complementary synthetic scenarios with hand-crafted embeddings and a mocked LLM:

*Scenario A — Dense retrieval wins:*
- Corpus: correct answer has embedding `[1.0, 0.0]`; a distractor has high BM25 score
  but embedding `[0.0, 1.0]`.
- Query embedding: `[1.0, 0.0]` (matches correct answer exactly).
- Mocked LLM returns `"5 0"` → α = 1.0 (full dense weight).
- Assert: DAT Precision@1 == 1.0; fixed hybrid (α=0.5) Precision@1 < 1.0 (distractor
  pulls correct answer below rank 1).

*Scenario B — BM25 retrieval wins:*
- Mirror of Scenario A: correct answer has top BM25 score; poor embedding similarity.
- Query embedding: `[0.0, 1.0]` (far from correct answer).
- Mocked LLM returns `"0 5"` → α = 0.0 (full BM25 weight).
- Assert: DAT Precision@1 == 1.0; fixed hybrid (α=0.5) Precision@1 < 1.0.

**Test Plan:**
```
test_precision_at_1_perfect_retrieval → 1.0
test_precision_at_1_complete_miss → 0.0
test_precision_at_k_partial
test_precision_multi_query_averaging
test_precision_mismatched_list_lengths (raises ValueError)
test_precision_empty_retrieved_docs
test_to_dict / test_from_dict
test_export_from_package
regression_test_dat_vs_fixed_alpha_dense_wins  (Scenario A)
regression_test_dat_vs_fixed_alpha_bm25_wins   (Scenario B)
```

---

### Sprint 4: Documentation Sprint

**Goal:** Produce user-facing documentation and verify existing docstrings are accurate
and executable. Docstrings for `DATDocumentJoiner` and `InMemoryDATHybridRetriever`
are already comprehensive from earlier sprints; the primary new deliverable is the
end-to-end usage guide (I-05).

**Files Created:**
- `DAT_USAGE_GUIDE.md` — end-to-end walkthrough covering indexing, hybrid retrieval,
  and evaluation with copy-paste examples

**Files Modified (docstring review + `DocumentPrecisionEvaluator` docs):**
- `haystack/components/joiners/dat_document_joiner.py`
- `haystack/components/retrievers/in_memory/dat_hybrid_retriever.py`
- `haystack/components/evaluators/document_precision.py`

**Documentation Standards (matching existing components):**
Each component must include:

1. **Class docstring** with:
   - Brief description
   - Algorithm reference (paper citation)
   - Limitations section
   - Full `### Usage example:` code block that runs end-to-end

2. **`__init__` docstring** with `:param name:` for every parameter including defaults

3. **`run()` docstring** with:
   - `:param name:` for every input
   - `:returns:` describing the output dict structure
   - `:raises:` listing `ComponentError` and `ValueError` conditions

**Deliverables:**

| Deliverable | Description | Priority |
|-------------|-------------|----------|
| `DAT_USAGE_GUIDE.md` | End-to-end walkthrough: indexing, generic pipeline, self-contained pipeline, evaluation | High |
| Docstring review: `DATDocumentJoiner` | Verify all params, usage example correct (already comprehensive) | Low |
| Docstring review: `InMemoryDATHybridRetriever` | Verify all params, usage example correct (already comprehensive) | Low |
| Docstring: `DocumentPrecisionEvaluator` | Write from scratch following Haystack standard | High |
| Executable doc-test verification | Manually verify all `### Usage example` blocks run without error | Medium |
| `CHANGELOG` entry stub | Short release note for all three new components | Low |

**`DAT_USAGE_GUIDE.md` outline:**
1. Overview — what DAT is and why it works
2. Installation / prerequisites
3. Indexing documents (shared setup)
4. Generic pipeline: `TextEmbedder + BM25Retriever + EmbeddingRetriever + DATDocumentJoiner`
5. Self-contained pipeline: `TextEmbedder + InMemoryDATHybridRetriever`
6. Evaluating with `DocumentPrecisionEvaluator`
7. Choosing a model (cost / quality trade-off, citing paper's empirical results)
8. FAQ: `raise_on_failure`, `scoring_top_k`, `scale_score`, caching

**Acceptance Criteria:**
- `DAT_USAGE_GUIDE.md` exists and all code examples are correct
- All `__init__` parameters documented
- All `run()` inputs, outputs, and exceptions documented
- `DocumentPrecisionEvaluator` docstring passes review against Haystack standard

---

## Appendix A: DATDocumentJoiner Interface

```python
@component
class DATDocumentJoiner:
    """
    Dynamic Alpha Tuning Document Joiner for hybrid retrieval.

    Implements DAT (Hsu & Tzeng, 2025) which uses an LLM to dynamically
    determine the optimal weighting coefficient α for combining dense and
    BM25 retrieval results on a per-query basis.

    ### Usage example:
    ...
    """

    def __init__(
        self,
        chat_generator: ChatGenerator,      # LLM for effectiveness scoring
        top_k: int = 10,
        scoring_top_k: int = 1,             # Docs scored per retriever (paper: 1)
        sort_by_score: bool = True,
        raise_on_failure: bool = True,       # Raise ComponentError on LLM parse failure
    ): ...

    @component.output_types(documents=list[Document], alpha=float)
    def run(
        self,
        query: str,
        dense_documents: list[Document],
        bm25_documents: list[Document],
        top_k: int | None = None,
    ) -> dict: ...

    @component.output_types(documents=list[Document], alpha=float)
    async def run_async(self, query: str, dense_documents: list[Document],
                        bm25_documents: list[Document], top_k: int | None = None) -> dict: ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DATDocumentJoiner": ...
    def _get_telemetry_data(self) -> dict[str, Any]: ...
```

---

## Appendix B: InMemoryDATHybridRetriever Interface

```python
@component
class InMemoryDATHybridRetriever:
    """
    Self-contained DAT hybrid retriever for InMemoryDocumentStore.

    Combines BM25 and embedding retrieval from a single InMemoryDocumentStore,
    then applies Dynamic Alpha Tuning (Hsu & Tzeng, 2025) to optimally weight
    the results. Reduces a 3-component hybrid pipeline to a single component.

    ### Usage example:
    ...
    """

    def __init__(
        self,
        document_store: InMemoryDocumentStore,
        chat_generator: ChatGenerator,
        top_k: int = 10,
        scoring_top_k: int = 1,
        scale_score: bool = False,
        filters: dict[str, Any] | None = None,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
        raise_on_failure: bool = True,
    ): ...

    @component.output_types(documents=list[Document], alpha=float)
    def run(
        self,
        query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict: ...

    @component.output_types(documents=list[Document], alpha=float)
    async def run_async(self, query: str, query_embedding: list[float],
                        filters: dict[str, Any] | None = None,
                        top_k: int | None = None) -> dict: ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InMemoryDATHybridRetriever": ...
    def _get_telemetry_data(self) -> dict[str, Any]: ...
```

---

## Appendix C: Usage Examples

### C.1 Generic Pipeline (DATDocumentJoiner)

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DATDocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
# ... index documents with embeddings ...

pipeline = Pipeline()
pipeline.add_component("text_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
pipeline.add_component("bm25_retriever",
    InMemoryBM25Retriever(document_store=document_store, top_k=10))
pipeline.add_component("dense_retriever",
    InMemoryEmbeddingRetriever(document_store=document_store, top_k=10))
pipeline.add_component("dat_joiner",
    DATDocumentJoiner(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        top_k=5,
    ))

pipeline.connect("text_embedder.embedding", "dense_retriever.query_embedding")
pipeline.connect("bm25_retriever.documents", "dat_joiner.bm25_documents")
pipeline.connect("dense_retriever.documents", "dat_joiner.dense_documents")

query = "What gun did the Royal Navy start using?"
result = pipeline.run({
    "text_embedder": {"text": query},
    "bm25_retriever": {"query": query},
    "dat_joiner": {"query": query},
})
print(f"Alpha: {result['dat_joiner']['alpha']}")
print(f"Top documents: {result['dat_joiner']['documents']}")
```

### C.2 Self-Contained Pipeline (InMemoryDATHybridRetriever)

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryDATHybridRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
# ... index documents with embeddings ...

pipeline = Pipeline()
pipeline.add_component("text_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
pipeline.add_component("dat_retriever",
    InMemoryDATHybridRetriever(
        document_store=document_store,
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        top_k=5,
    ))

pipeline.connect("text_embedder.embedding", "dat_retriever.query_embedding")

query = "What gun did the Royal Navy start using?"
result = pipeline.run({
    "text_embedder": {"text": query},
    "dat_retriever": {"query": query},
})
print(f"Alpha: {result['dat_retriever']['alpha']}")
print(f"Top documents: {result['dat_retriever']['documents']}")
```
