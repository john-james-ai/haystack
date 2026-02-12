# DAT-RAG Strategy V1
## Dynamic Alpha Tuning for Hybrid Retrieval in Haystack

**Author:** Senior Architect & Developer
**Date:** 2026-02-12
**Status:** Draft — Awaiting Review
**Reference:** Hsu & Tzeng (2025), *DAT: Dynamic Alpha Tuning for Hybrid Retrieval in RAG*, arXiv:2503.23013

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

DAT (Dynamic Alpha Tuning) replaces the static weighting coefficient (α) in hybrid retrieval with a per-query, LLM-determined weight that reflects each retriever's effectiveness for the specific query-document relationship.

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

DAT consistently outperforms fixed-weight hybrid by ~2–3% on full datasets and ~5–7.5% on hybrid-sensitive subsets.

---

## 2. Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║                    DAT-RAG Retrieval Pipeline                    ║
╚══════════════════════════════════════════════════════════════════╝

  User Query
      │
      ├─────────────────────────────────────────────┐
      │                                             │
      ▼                                             ▼
┌─────────────────────┐              ┌──────────────────────────┐
│  TextEmbedder       │              │  InMemoryBM25Retriever   │
│  (embedding model)  │              │  (sparse retrieval)      │
└──────────┬──────────┘              └────────────┬─────────────┘
           │ query_embedding                      │ bm25_documents
           ▼                                      │ (top-K, with scores)
┌─────────────────────────┐                       │
│ InMemoryEmbeddingRetr.  │                       │
│ (dense retrieval)       │                       │
└──────────┬──────────────┘                       │
           │ dense_documents                       │
           │ (top-K, with scores)                  │
           └──────────────┬────────────────────────┘
                          │
                          ▼
          ┌───────────────────────────────────────┐
          │         DATDocumentJoiner             │
          │                                       │
          │  1. Normalize scores (min-max)         │
          │     dense: S̃_v ∈ [0,1]               │
          │     bm25:  S̃_b ∈ [0,1]               │
          │                                       │
          │  2. LLM Effectiveness Scoring         │
          │     ┌─────────────────────────┐       │
          │     │  OpenAIChatGenerator    │       │
          │     │  (or any ChatGenerator) │       │
          │     └──────────┬──────────────┘       │
          │                │ scores: S_v, S_b ∈ {0..5}
          │                                       │
          │  3. Dynamic Alpha Calculation         │
          │     α(q) = f(S_v, S_b) ∈ [0.0, 1.0] │
          │     (rounded to 1 decimal)            │
          │                                       │
          │  4. Score Fusion                      │
          │     R(q,d) = α·S̃_v + (1-α)·S̃_b    │
          │                                       │
          │  5. Rank & truncate → top-K           │
          └───────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Documents (top-K)    │
              │  with fused scores    │
              │  + metadata:          │
              │    - alpha value      │
              │    - dense_score      │
              │    - bm25_score       │
              └───────────────────────┘


  ┌──────────────────────────────────────────────────────────────┐
  │                   Evaluation Pipeline                        │
  │                                                              │
  │  Ground Truth ──► DocumentPrecisionEvaluator (Precision@K)  │
  │  Retrieved Docs ─► DocumentMRREvaluator      (MRR@K)        │
  │                ──► DocumentRecallEvaluator   (Recall@K)     │
  │                ──► DocumentNDCGEvaluator     (nDCG)         │
  └──────────────────────────────────────────────────────────────┘
```

**Component Interaction Summary:**

| Component | Role | Source |
|-----------|------|--------|
| `TextEmbedder` | Encodes query for dense retrieval | Existing |
| `InMemoryBM25Retriever` | Lexical retrieval | Existing |
| `InMemoryEmbeddingRetriever` | Semantic retrieval | Existing |
| `DATDocumentJoiner` | Dynamic alpha fusion | **NEW** |
| `OpenAIChatGenerator` | LLM effectiveness scoring | Existing (injected) |
| `DocumentPrecisionEvaluator` | Precision@K metric | **NEW** |
| `DocumentMRREvaluator` | MRR metric | Existing |
| `DocumentRecallEvaluator` | Recall@K metric | Existing |
| `DocumentNDCGEvaluator` | nDCG metric | Existing |

---

## 3. Package Structure

```
haystack/
├── components/
│   ├── joiners/
│   │   ├── __init__.py                          [MODIFIED]
│   │   ├── answer_joiner.py
│   │   ├── branch.py
│   │   ├── dat_document_joiner.py               [NEW — Sprint 1]
│   │   ├── document_joiner.py
│   │   ├── list_joiner.py
│   │   └── string_joiner.py
│   └── evaluators/
│       ├── __init__.py                          [MODIFIED — Sprint 3]
│       ├── answer_exact_match.py
│       ├── context_relevance.py
│       ├── document_map.py
│       ├── document_mrr.py
│       ├── document_ndcg.py
│       ├── document_precision.py                [NEW — Sprint 3]
│       ├── document_recall.py
│       ├── faithfulness.py
│       ├── llm_evaluator.py
│       └── sas_evaluator.py
test/
├── components/
│   ├── joiners/
│   │   ├── test_dat_document_joiner.py          [NEW — Sprint 1]
│   │   └── test_dat_document_joiner_integration.py  [NEW — Sprint 2]
│   └── evaluators/
│       └── test_document_precision.py           [NEW — Sprint 3]
```

---

## 4. Impact Analysis

### 4.1 New Files

| File | Class / Artefact | Sprint | Description |
|------|-----------------|--------|-------------|
| `haystack/components/joiners/dat_document_joiner.py` | `DATDocumentJoiner` | 1 | Core DAT component. Accepts query + two document lists. Scores top-1 from each with LLM. Computes dynamic alpha. Fuses and returns top-K documents. |
| `haystack/components/evaluators/document_precision.py` | `DocumentPrecisionEvaluator` | 3 | Precision@K evaluator. Measures fraction of retrieved documents that are relevant at position K. Supports configurable K. |
| `test/components/joiners/test_dat_document_joiner.py` | `TestDATDocumentJoiner` | 1 | Unit tests for all DAT logic paths: alpha cases, normalization edge cases, LLM parsing, serialization, async. |
| `test/components/joiners/test_dat_document_joiner_integration.py` | `TestDATDocumentJoinerIntegration` | 2 | Integration tests with InMemoryDocumentStore + real retrievers + mocked LLM. Tests pipeline wiring. |
| `test/components/evaluators/test_document_precision.py` | `TestDocumentPrecisionEvaluator` | 3 | Unit tests for Precision@K evaluator. |

### 4.2 Modified Files

| File | Change | Sprint | Justification |
|------|--------|--------|---------------|
| `haystack/components/joiners/__init__.py` | Add `DATDocumentJoiner` to `_import_structure` and TYPE_CHECKING block | 1 | Consistent with Haystack's LazyImporter export pattern |
| `haystack/components/evaluators/__init__.py` | Add `DocumentPrecisionEvaluator` to `_import_structure` and TYPE_CHECKING block | 3 | Consistent with existing evaluator exports |

### 4.3 No Modifications Required

The following existing components are used as-is:
- `InMemoryBM25Retriever` — consumes its output directly
- `InMemoryEmbeddingRetriever` — consumes its output directly
- `OpenAIChatGenerator` / any `ChatGenerator` — injected via constructor
- `DocumentJoiner` — DAT is a separate, complementary component
- All existing evaluators — used alongside new `DocumentPrecisionEvaluator`

---

## 5. Risk Analysis

| # | Risk | Probability | Impact | Mitigation |
|---|------|-------------|--------|------------|
| R1 | **LLM Latency**: One LLM call per query adds 100ms–2s overhead per query, degrading throughput | HIGH | HIGH | (a) Async support allows concurrent execution with other pipeline stages; (b) LLM cache layer (Haystack's `CacheChecker`) can be wired in by users; (c) Use of small models (e.g., GPT-4o-mini) shown to perform comparably to GPT-4o |
| R2 | **LLM Response Parsing Failure**: LLM may not return two space-separated integers | MEDIUM | MEDIUM | Implement robust regex parsing; fallback to α=0.5 (neutral weighting) on parse failure; log a warning; never raise exception on malformed LLM output |
| R3 | **Score Normalization Division-by-Zero**: When all documents from a retriever have identical scores (delta=0) | LOW | LOW | Explicitly handle: if `max_score == min_score`, set all normalized scores to 0.0 (consistent with DBSF behavior in existing `DocumentJoiner._distribution_based_rank_fusion`) |
| R4 | **Empty Document Lists**: One or both retrievers return no documents | MEDIUM | MEDIUM | If dense returns 0 docs: skip LLM call, set α=0.0 (pure BM25). If BM25 returns 0 docs: skip LLM call, set α=1.0 (pure dense). If both return 0: return empty list immediately. |
| R5 | **API Key Dependency in Tests**: Integration and regression tests require LLM API credentials | HIGH | MEDIUM | Unit tests mock the ChatGenerator; integration tests use pytest fixtures with mock ChatGenerator; E2E tests guarded by `@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ)` |
| R6 | **Algorithm Generalization**: Paper tested only SQuAD (English) and DRCD (Chinese). May not generalize to all domains | MEDIUM | MEDIUM | Document the limitation clearly; provide evaluation tooling so users can benchmark on their own datasets; the modular design (injectable ChatGenerator) allows users to customize prompts for domain-specific scoring |
| R7 | **Production Cost**: Each query requires one additional LLM call; at scale this is significant | HIGH | HIGH | Document the cost tradeoff clearly; recommend GPT-4o-mini as cost-efficient default; the async design enables parallel execution; cache-friendly design (query → scores are deterministic) |
| R8 | **Score Rounding Artifacts**: Rounding alpha to 1 decimal can cause abrupt behavior changes | LOW | LOW | Match paper exactly (1-decimal rounding) in the core algorithm; expose `round_alpha` parameter to allow users to disable rounding if needed |
| R9 | **ChatGenerator Protocol Versioning**: The `ChatGenerator` Protocol could evolve | LOW | LOW | Type-hint against the Protocol (not a concrete class); unit tests with a mock that implements the minimal Protocol |

---

## 6. Clarifying Questions

### Q1. LLM Provider Default
**Context:** The paper uses OpenAI's GPT-4o. Haystack is intentionally LLM-provider-agnostic (every generator component accepts any `ChatGenerator`).

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| A. Default to `OpenAIChatGenerator` | Matches paper; easiest to get started | Couples implementation to OpenAI; breaks for users without OpenAI key |
| B. No default — require explicit `chat_generator` arg | Provider-agnostic; consistent with `LLMEvaluator`'s design | Slightly more verbose user code |

**Recommendation: B — No default.** Consistent with `LLMEvaluator`, which also requires an explicit `chat_generator`. This maintains Haystack's provider-agnostic philosophy and avoids implicit dependencies.

---

### Q2. Top-K for Retrieval vs. Top-K for Scoring
**Context:** DAT uses top-1 for LLM scoring but top-K for final ranking. These could be decoupled parameters.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| A. Single `top_k` parameter | Simple; users set once | LLM always scores only top-1 regardless; could expose `scoring_top_k` confusion |
| B. Separate `top_k` (retrieval) and `scoring_top_k` (LLM, defaults to 1) | Flexible; allows scoring top-2 or top-3 if desired | More parameters; paper's design is strictly top-1 |

**Recommendation: B** with `scoring_top_k=1` as the default (matching the paper). The paper explicitly justifies evaluating only top-1 as sufficient signal while minimizing cost. Keep it as an advanced parameter, defaulting to the paper's specification.

---

### Q3. Alpha Rounding
**Context:** The paper rounds α to 1 decimal place before applying fusion. This creates discrete alpha values {0.0, 0.1, ..., 1.0}.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| A. Always round to 1 decimal | Matches paper; more predictable behavior | Loses granularity |
| B. Configurable rounding (`round_alpha: bool = True`) | Flexible | Deviates from paper when disabled |

**Recommendation: A** — round to 1 decimal as the default and only behavior in the initial implementation. This matches the paper and avoids premature configurability. Can be made configurable in a later sprint if demand arises.

---

### Q4. Handling When LLM Returns Scores of Equal Value (non-zero tie)
**Context:** If `S_v = S_b = 3` (both "good wrong result"), the formula gives α = 3/(3+3) = 0.5. This is the correct paper behavior — equal weighting — but should be made explicit.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| A. Apply formula as-is (α = 0.5 when tied) | Mathematically correct; paper's intent | Could confuse users expecting asymmetry |
| B. Special-case ties explicitly | More readable code | Not in paper; could mask bugs |

**Recommendation: A** — the formula naturally handles this case. Add a code comment explaining the tie behavior.

---

### Q5. Component Placement: `joiners` vs. `retrievers`
**Context:** `DATDocumentJoiner` is structurally a joiner (it merges two document lists) but conceptually a hybrid retriever. Where should it live?

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| A. `haystack/components/joiners/` | Parallel to `DocumentJoiner`; clear it fuses results | Slightly obscures its retrieval semantics |
| B. `haystack/components/retrievers/` | Communicates intent (it's a hybrid retriever) | Existing retrievers wrap document stores; this takes pre-retrieved docs |

**Recommendation: A — `joiners/`** — it operates on already-retrieved document lists (like `DocumentJoiner`), not on a document store. This is structurally accurate and maintains consistency.

---

## 7. Design Decisions

### DD1. Standalone Component vs. Extending DocumentJoiner

**Context:** Two approaches exist for integrating DAT into Haystack.

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A. Add `DYNAMIC_ALPHA_TUNING` to `JoinMode` enum | Extend `DocumentJoiner.run()` | Reuse existing serialization | `run()` signature uses `Variadic` input — can't distinguish dense from BM25; can't inject LLM; requires `query` parameter not present in current interface |
| B. Standalone `DATDocumentJoiner` (CHOSEN) | New component in `joiners/` | Clean interface; named inputs; injectable LLM; full control | Minor code duplication (normalization utility) |

**Decision: B — Standalone component.**
DAT has fundamentally different requirements from `DocumentJoiner`: it needs (1) named inputs to identify dense vs. BM25 results, (2) the query string for LLM prompting, and (3) an injectable LLM. Retrofitting these into `DocumentJoiner`'s `Variadic`-input, query-unaware interface would require breaking changes to the existing API. A standalone component follows the Single Responsibility Principle and avoids polluting the existing, stable `DocumentJoiner`.

---

### DD2. Named vs. Variadic Document Inputs

**Context:** `DocumentJoiner` uses `Variadic[list[Document]]` to accept N document lists. DAT requires exactly 2 named inputs.

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A. `Variadic[list[Document]]` | Flexible N-way input | Works for any number of retrievers | Cannot distinguish dense from BM25 results; ambiguous ordering |
| B. Named `dense_documents`, `bm25_documents` (CHOSEN) | Explicit 2-way input | Unambiguous; self-documenting; directly maps to paper | Fixed to exactly 2 retrievers |

**Decision: B — Named inputs.**
The DAT algorithm is explicitly designed for a two-retriever (dense + sparse) hybrid. Named inputs make the component contract unambiguous and directly reflect the paper's mathematical formulation. Users combining 3+ retrievers should use standard `DocumentJoiner` upstream.

---

### DD3. LLM Integration: Injected ChatGenerator vs. Internal Construction

**Context:** The scoring LLM must be configurable.

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A. Accept any `ChatGenerator` in `__init__` (CHOSEN) | Dependency injection | Provider-agnostic; fully testable with mocks; consistent with `LLMEvaluator` | User must instantiate generator separately |
| B. Build LLM client internally (e.g., always OpenAI) | Simple setup | Fewer lines of user code | Tight coupling; untestable; violates provider-agnosticism |
| C. Accept a `PromptBuilder` + `ChatGenerator` pair | Allows prompt customization | Flexible | Overly complex for this use case |

**Decision: A — Injected `ChatGenerator`.**
Follows `LLMEvaluator`'s proven pattern. Users swap OpenAI for Anthropic, local models, or mocks without changing the component. Serialization via `component_to_dict` handles the generator's own `to_dict`/`from_dict`.

---

### DD4. Async Support

**Context:** Haystack 2.x components increasingly implement `run_async` alongside `run` for AsyncPipeline support.

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A. Sync only | Simpler code | Inconsistent with most retrievers | Cannot be used in `AsyncPipeline` |
| B. Both `run()` and `run_async()` (CHOSEN) | Full compatibility | Consistent with `InMemoryBM25Retriever`, `InMemoryEmbeddingRetriever` patterns | ~30% more code |

**Decision: B — Both sync and async.**
Most production Haystack pipelines leverage async for throughput. Shipping without async support would require users to work around the limitation. The async implementation follows the established pattern in existing retrievers (delegate to async client).

---

### DD5. Precision@K Evaluator Placement

**Context:** The paper uses Precision@1 as its primary metric. Haystack currently has MRR, nDCG, Recall, MAP, but no dedicated Precision@K evaluator.

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A. New `DocumentPrecisionEvaluator` in `evaluators/` (CHOSEN) | Standalone evaluator | Reusable; consistent with existing evaluators; fills a gap in the framework | New file needed |
| B. Use `DocumentMAPEvaluator` (which subsumes Precision@K) | No new file | MAP@1 ≡ Precision@1 | Not obvious to users; MAP@K ≠ Precision@K in general |

**Decision: A — New evaluator.**
Precision@K is a fundamental retrieval metric distinct from MAP. Having a dedicated `DocumentPrecisionEvaluator` makes the framework more complete and directly mirrors the paper's evaluation methodology. Consistent with the existing evaluator pattern: stateless `@component`, `run(ground_truth_documents, retrieved_documents)`.

---

## 8. Sprint Roadmap

### Sprint Overview

| Sprint | Title | Deliverables | Value |
|--------|-------|-------------|-------|
| S1 | Core Component | `DATDocumentJoiner`, unit tests | Working DAT component with full algorithm |
| S2 | Pipeline Integration | Integration tests, example pipeline | Verified pipeline wiring; usable end-to-end |
| S3 | Evaluation Framework | `DocumentPrecisionEvaluator`, evaluation pipeline, regression tests | Full evaluation capability comparing DAT vs. static hybrid |

---

### Sprint 1: Core DATDocumentJoiner Component

**Goal:** Implement the complete DAT algorithm as a production-quality Haystack component.

**Files Created:**
- `haystack/components/joiners/dat_document_joiner.py`
- `test/components/joiners/test_dat_document_joiner.py`

**Files Modified:**
- `haystack/components/joiners/__init__.py`

**Acceptance Criteria:**
- `DATDocumentJoiner` accepts `query: str`, `dense_documents: list[Document]`, `bm25_documents: list[Document]`
- Min-max score normalization handles edge cases (identical scores, empty lists)
- LLM prompt matches paper's Appendix A verbatim
- Dynamic alpha calculation implements all 4 cases from Eq. 6
- Alpha rounded to 1 decimal place
- Final fusion uses `R(q,d) = α·S̃_dense + (1-α)·S̃_BM25`
- Graceful handling: empty inputs, LLM parse failure (fallback α=0.5)
- `to_dict()` / `from_dict()` serialization complete
- `run()` and `run_async()` both implemented
- Unit tests: ≥90% branch coverage, all alpha cases covered, normalization edge cases, mock LLM fixtures

**Test Plan:**
```
Unit Tests:
- test_init_defaults
- test_init_custom_params
- test_to_dict / test_from_dict
- test_normalize_scores_standard
- test_normalize_scores_identical (delta=0 edge case)
- test_normalize_scores_empty
- test_alpha_both_zero
- test_alpha_dense_perfect
- test_alpha_bm25_perfect
- test_alpha_proportional
- test_alpha_rounding
- test_run_returns_fused_documents
- test_run_empty_dense_docs (α=0.0 fallback)
- test_run_empty_bm25_docs (α=1.0 fallback)
- test_run_both_empty
- test_run_llm_parse_failure (α=0.5 fallback)
- test_run_top_k_parameter
- test_run_async
- test_run_outputs_alpha_in_meta
```

---

### Sprint 2: Pipeline Integration

**Goal:** Verify `DATDocumentJoiner` works correctly wired into a full hybrid retrieval pipeline with real document stores and retrievers.

**Files Created:**
- `test/components/joiners/test_dat_document_joiner_integration.py`

**Acceptance Criteria:**
- Integration test: `InMemoryBM25Retriever` → `DATDocumentJoiner` ← `InMemoryEmbeddingRetriever` (via `TextEmbedder`) produces ranked documents
- Pipeline can be serialized and deserialized (`pipeline.to_dict()` / `Pipeline.from_dict()`)
- `DATDocumentJoiner` correctly exported from `haystack.components.joiners`
- Alpha value and individual retriever scores reflected in document metadata
- Test against known documents confirms alpha selects correct retriever

**Test Plan:**
```
Integration Tests:
- test_dat_joiner_in_pipeline_with_real_document_store
- test_pipeline_serialization_roundtrip
- test_dat_joiner_export_from_package
- test_alpha_reflected_in_document_meta
- test_full_pipeline_with_mocked_llm_known_alpha
```

---

### Sprint 3: Evaluation Framework

**Goal:** Provide a complete evaluation suite enabling measurement and comparison of retrieval effectiveness.

**Files Created:**
- `haystack/components/evaluators/document_precision.py`
- `test/components/evaluators/test_document_precision.py`

**Files Modified:**
- `haystack/components/evaluators/__init__.py`

**Acceptance Criteria:**
- `DocumentPrecisionEvaluator` implements Precision@K where K is configurable (default K=1)
- Precision@1 matches paper's evaluation metric
- Evaluator follows existing pattern: `run(ground_truth_documents, retrieved_documents)` → `{score, individual_scores}`
- Evaluation pipeline demonstrates DAT vs. Fixed Hybrid (α=0.5, α=0.6) comparison
- Regression test: DAT Precision@1 exceeds Fixed Hybrid Precision@1 on sample dataset

**Test Plan:**
```
Unit Tests (DocumentPrecisionEvaluator):
- test_precision_at_1_perfect
- test_precision_at_1_miss
- test_precision_at_k_partial
- test_precision_empty_retrieved
- test_precision_mismatched_lengths (raises ValueError)
- test_to_dict / test_from_dict
- test_export_from_package

Regression Tests:
- test_dat_vs_static_hybrid_on_sample_corpus (DAT P@1 ≥ fixed α P@1)
```

---

## Appendix A: Key File Sketches

### `DATDocumentJoiner` Interface (for review)

```python
@component
class DATDocumentJoiner:
    """
    Dynamic Alpha Tuning Document Joiner for hybrid retrieval.

    Implements the DAT algorithm (Hsu & Tzeng, 2025) which uses an LLM to
    dynamically determine the optimal weighting coefficient α for combining
    dense and BM25 retrieval results on a per-query basis.
    """

    def __init__(
        self,
        chat_generator: ChatGenerator,       # LLM for effectiveness scoring
        top_k: int = 10,                     # Final number of documents to return
        scoring_top_k: int = 1,              # Docs scored by LLM per retriever (paper: 1)
        sort_by_score: bool = True,
    ): ...

    @component.output_types(
        documents=list[Document],
        alpha=float,                          # Transparency: the computed α
    )
    def run(
        self,
        query: str,
        dense_documents: list[Document],
        bm25_documents: list[Document],
        top_k: int | None = None,
    ) -> dict: ...

    @component.output_types(
        documents=list[Document],
        alpha=float,
    )
    async def run_async(
        self,
        query: str,
        dense_documents: list[Document],
        bm25_documents: list[Document],
        top_k: int | None = None,
    ) -> dict: ...

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DATDocumentJoiner": ...
```

### `DocumentPrecisionEvaluator` Interface (for review)

```python
@component
class DocumentPrecisionEvaluator:
    """
    Evaluates Precision@K: the fraction of retrieved documents at position K
    that are relevant. Precision@1 is the primary metric used in the DAT paper.
    """

    def __init__(self, k: int = 1): ...

    @component.output_types(score=float, individual_scores=list[float])
    def run(
        self,
        ground_truth_documents: list[list[Document]],
        retrieved_documents: list[list[Document]],
    ) -> dict[str, Any]: ...

    def to_dict(self) -> dict[str, Any]: ...
```

---

## Appendix B: Usage Example

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

# Setup
document_store = InMemoryDocumentStore()
# ... index documents with embeddings ...

# DAT Hybrid Retrieval Pipeline
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

# Connections
pipeline.connect("text_embedder.embedding", "dense_retriever.query_embedding")
pipeline.connect("bm25_retriever.documents", "dat_joiner.bm25_documents")
pipeline.connect("dense_retriever.documents", "dat_joiner.dense_documents")

# Run
query = "What gun did the Royal Navy start using?"
result = pipeline.run({
    "text_embedder": {"text": query},
    "bm25_retriever": {"query": query},
    "dat_joiner": {"query": query},
})

print(f"Alpha: {result['dat_joiner']['alpha']}")
print(f"Top documents: {result['dat_joiner']['documents']}")
```
