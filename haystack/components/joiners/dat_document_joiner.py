# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from math import inf
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.errors import ComponentError
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage
from haystack.utils import deserialize_chatgenerator_inplace

logger = logging.getLogger(__name__)

DAT_SCORING_PROMPT = (
    "You are an evaluator assessing the retrieval effectiveness of dense\n"
    "retrieval (Cosine Distance) and BM25 retrieval for finding the correct answer.\n"
    "\n"
    "## Task:\n"
    "Given a question and two top1 search results (one from dense retrieval,\n"
    "one from BM25 retrieval), score each retrieval method from **0 to 5**\n"
    "based on whether the correct answer is likely to appear in top2, top3, etc.\n"
    "\n"
    "### **Scoring Criteria:**\n"
    "1. **Direct hit --> 5 points**\n"
    "   - If the retrieved document directly answers the question, assign **5 points**.\n"
    "2. **Good wrong result (High likelihood correct answer is nearby) --> 3-4 points**\n"
    "   - If the top1 result is **conceptually close** to the correct answer\n"
    "     (e.g., mentions relevant entities, related events, partial answer),\n"
    "     it indicates the search method is in the right direction.\n"
    "   - Give **4** if it's very close, **3** if somewhat close.\n"
    "3. **Bad wrong result (Low likelihood correct answer is nearby) --> 1-2 points**\n"
    "   - If the top1 result is **loosely related but misleading** (e.g.,\n"
    "     shares keywords but changes context), correct answers might not be in top2, top3.\n"
    "   - Give **2** if there's a small chance correct answers are nearby, **1** if unlikely.\n"
    "4. **Completely off-track --> 0 points**\n"
    "   - If the result is **totally unrelated**, it means the retrieval method is failing.\n"
    "\n"
    "---\n"
    "### **Given Data:**\n"
    '- **Question:** "{question}"\n'
    '- **dense retrieval Top1 Result:** "{vector_reference}"\n'
    '- **BM25 retrieval Top1 Result:** "{bm25_reference}"\n'
    "\n"
    "---\n"
    "### **Output Format:**\n"
    "Return two integers separated by a space:\n"
    "- **First number:** dense retrieval score.\n"
    "- **Second number:** BM25 retrieval score.\n"
    "- Example output: 3 4\n"
    "(Vector: 3, BM25: 4)\n"
    "**Do not output any other text.**"
)


@component
class DATDocumentJoiner:
    """
    Dynamically weights and fuses results from dense and BM25 retrievers using the
    DAT (Dynamic Alpha Tuning) algorithm (Hsu & Tzeng, 2025, arXiv:2503.23013).

    Instead of applying a fixed weighting coefficient α, this component leverages a
    large language model to evaluate the top-1 result from each retriever against the
    query. The resulting effectiveness scores (0–5) determine a per-query α that is
    applied to min-max-normalised scores from both retrievers.

    The component raises a :exc:`~haystack.core.errors.ComponentError` by default when
    the LLM returns output that cannot be parsed. Set ``raise_on_failure=False`` to fall
    back to α = 0.5 (equal weighting) with a warning instead.

    **Limitations:** DAT introduces one LLM call per query. For high-throughput
    production systems consider using the async interface and/or caching.

    ### Usage example

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
    pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
    )
    pipeline.add_component(
        "bm25_retriever",
        InMemoryBM25Retriever(document_store=document_store, top_k=10),
    )
    pipeline.add_component(
        "dense_retriever",
        InMemoryEmbeddingRetriever(document_store=document_store, top_k=10),
    )
    pipeline.add_component(
        "dat_joiner",
        DATDocumentJoiner(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            top_k=5,
        ),
    )
    pipeline.connect("text_embedder.embedding", "dense_retriever.query_embedding")
    pipeline.connect("bm25_retriever.documents", "dat_joiner.bm25_documents")
    pipeline.connect("dense_retriever.documents", "dat_joiner.dense_documents")

    query = "What is the capital of France?"
    result = pipeline.run({
        "text_embedder": {"text": query},
        "bm25_retriever": {"query": query},
        "dat_joiner": {"query": query},
    })
    print(result["dat_joiner"]["alpha"])
    print(result["dat_joiner"]["documents"])
    ```
    """

    def __init__(
        self,
        chat_generator: ChatGenerator,
        top_k: int = 10,
        scoring_top_k: int = 1,
        sort_by_score: bool = True,
        raise_on_failure: bool = True,
    ):
        """
        Create a DATDocumentJoiner component.

        :param chat_generator:
            A :class:`~haystack.components.generators.chat.types.ChatGenerator` instance
            used to score the top-``scoring_top_k`` results from each retriever.
            Any ChatGenerator compatible with the protocol works (OpenAI, Anthropic,
            local models, etc.).
        :param top_k:
            Maximum number of documents to return after fusion and ranking.
            Must be > 0.
        :param scoring_top_k:
            Number of top documents from each retriever passed to the LLM for
            effectiveness scoring.  The paper recommends ``1`` (the default) as it
            provides sufficient signal while minimising cost.  Must be > 0.
        :param sort_by_score:
            If ``True`` (default), return documents sorted by descending fused score.
        :param raise_on_failure:
            If ``True`` (default), raise a :exc:`~haystack.core.errors.ComponentError`
            when the LLM response cannot be parsed as two integers in [0, 5].
            If ``False``, log a warning and fall back to α = 0.5 (equal weighting).
        :raises ValueError:
            If ``top_k`` or ``scoring_top_k`` is not a positive integer.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")
        if scoring_top_k <= 0:
            raise ValueError(f"scoring_top_k must be greater than 0, got {scoring_top_k}")

        self.chat_generator = chat_generator
        self.top_k = top_k
        self.scoring_top_k = scoring_top_k
        self.sort_by_score = sort_by_score
        self.raise_on_failure = raise_on_failure
        self._is_warmed_up = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """
        Warm up the underlying chat generator, if it supports warm-up.

        Called automatically on the first :meth:`run` or :meth:`run_async` invocation.
        """
        if not self._is_warmed_up:
            if hasattr(self.chat_generator, "warm_up"):
                self.chat_generator.warm_up()
            self._is_warmed_up = True

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data sent to Posthog for usage analytics.
        """
        return {"chat_generator": type(self.chat_generator).__name__}

    # ------------------------------------------------------------------
    # Core algorithm helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(documents: list[Document]) -> list[Document]:
        """
        Apply min-max normalisation to document scores, returning new Document copies
        with scores in [0, 1].

        When all documents carry the same score (delta == 0) every normalised score is
        set to 0.0, consistent with the DBSF behaviour in
        :class:`~haystack.components.joiners.document_joiner.DocumentJoiner`.

        Original Document objects are **not mutated**.

        :param documents: Documents with raw retriever scores.
        :returns: New Document instances with normalised scores.
        """
        if not documents:
            return []

        raw_scores = [doc.score if doc.score is not None else 0.0 for doc in documents]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        delta = max_score - min_score

        normalised = []
        for doc, raw in zip(documents, raw_scores):
            norm_score = (raw - min_score) / delta if delta != 0.0 else 0.0
            doc_dict = doc.to_dict()
            doc_dict["score"] = norm_score
            normalised.append(Document.from_dict(doc_dict))

        return normalised

    @staticmethod
    def _compute_alpha(dense_score: int, bm25_score: int) -> float:
        """
        Compute the dynamic α weighting coefficient from LLM-assigned effectiveness
        scores, implementing Equation 6 from Hsu & Tzeng (2025).

        The result is rounded to one decimal place as specified in the paper.

        Rules:

        - Both zero  → 0.5 (equal weighting; both retrievers failed)
        - Dense == 5, BM25 ≠ 5 → 1.0 (pure dense; dense is a perfect hit)
        - BM25 == 5, Dense ≠ 5 → 0.0 (pure BM25; BM25 is a perfect hit)
        - Otherwise  → S_v / (S_v + S_b), rounded to 1 decimal place

        :param dense_score: LLM effectiveness score for the dense top-1 result (0–5).
        :param bm25_score:  LLM effectiveness score for the BM25 top-1 result (0–5).
        :returns: α ∈ {0.0, 0.1, …, 1.0}.
        """
        if dense_score == 0 and bm25_score == 0:
            # Both methods failed to retrieve anything relevant — equal weighting
            return 0.5
        if dense_score == 5 and bm25_score != 5:
            # Dense is a perfect hit; BM25 is not
            return 1.0
        if bm25_score == 5 and dense_score != 5:
            # BM25 is a perfect hit; dense is not
            return 0.0
        # Proportional weighting (handles ties naturally: equal scores → α = 0.5)
        alpha = dense_score / (dense_score + bm25_score)
        return round(alpha, 1)

    def _parse_scores(self, reply_text: str) -> tuple[int, int]:
        """
        Parse the LLM reply into a ``(dense_score, bm25_score)`` integer pair.

        The expected format is two space-separated integers in [0, 5] (e.g. ``"3 4"``).
        A regex search is used so that the LLM may include minor surrounding text
        without causing a parse failure.

        :param reply_text: Raw text returned by the LLM.
        :returns: ``(dense_score, bm25_score)`` integers.
        :raises ComponentError:
            If parsing fails and ``raise_on_failure`` is ``True``.
        """
        match = re.search(r"\b([0-5])\s+([0-5])\b", reply_text)
        if match:
            return int(match.group(1)), int(match.group(2))

        msg = (
            "DATDocumentJoiner could not parse the LLM scoring response. "
            f"Expected two space-separated integers in [0, 5], got: {reply_text!r}"
        )
        if self.raise_on_failure:
            raise ComponentError(msg)

        logger.warning(msg + " Falling back to α = 0.5 (equal weighting).")
        # _compute_alpha(0, 0) → 0.5
        return 0, 0

    def _call_llm(self, query: str, dense_documents: list[Document], bm25_documents: list[Document]) -> tuple[int, int]:
        """
        Build the scoring prompt, invoke the LLM synchronously, and return parsed scores.

        :param query: The user query string.
        :param dense_documents: Normalised dense retriever documents.
        :param bm25_documents:  Normalised BM25 retriever documents.
        :returns: ``(dense_score, bm25_score)`` integers.
        :raises ComponentError: On parse failure when ``raise_on_failure`` is ``True``.
        """
        dense_top = dense_documents[: self.scoring_top_k]
        bm25_top = bm25_documents[: self.scoring_top_k]

        vector_reference = " ".join(d.content or "" for d in dense_top).strip()
        bm25_reference = " ".join(d.content or "" for d in bm25_top).strip()

        prompt_text = DAT_SCORING_PROMPT.format(
            question=query,
            vector_reference=vector_reference,
            bm25_reference=bm25_reference,
        )
        messages = [ChatMessage.from_user(prompt_text)]
        result = self.chat_generator.run(messages=messages)
        reply_text: str = result["replies"][0].text.strip()
        return self._parse_scores(reply_text)

    async def _call_llm_async(
        self, query: str, dense_documents: list[Document], bm25_documents: list[Document]
    ) -> tuple[int, int]:
        """
        Async version of :meth:`_call_llm`.

        Uses ``chat_generator.run_async`` when available; otherwise falls back to the
        synchronous ``run`` method.

        :param query: The user query string.
        :param dense_documents: Normalised dense retriever documents.
        :param bm25_documents:  Normalised BM25 retriever documents.
        :returns: ``(dense_score, bm25_score)`` integers.
        :raises ComponentError: On parse failure when ``raise_on_failure`` is ``True``.
        """
        dense_top = dense_documents[: self.scoring_top_k]
        bm25_top = bm25_documents[: self.scoring_top_k]

        vector_reference = " ".join(d.content or "" for d in dense_top).strip()
        bm25_reference = " ".join(d.content or "" for d in bm25_top).strip()

        prompt_text = DAT_SCORING_PROMPT.format(
            question=query,
            vector_reference=vector_reference,
            bm25_reference=bm25_reference,
        )
        messages = [ChatMessage.from_user(prompt_text)]

        if hasattr(self.chat_generator, "run_async"):
            result = await self.chat_generator.run_async(messages=messages)
        else:
            result = self.chat_generator.run(messages=messages)

        reply_text: str = result["replies"][0].text.strip()
        return self._parse_scores(reply_text)

    @staticmethod
    def _fuse(alpha: float, dense_norm: list[Document], bm25_norm: list[Document]) -> list[Document]:
        """
        Compute ``R(q, d) = α · S̃_dense + (1 − α) · S̃_BM25`` for every document
        and merge the two result sets, keeping the highest fused score for duplicates.

        :param alpha:      Dynamic weighting coefficient for the dense retriever.
        :param dense_norm: Normalised dense retriever documents.
        :param bm25_norm:  Normalised BM25 retriever documents.
        :returns: Merged list of Documents with fused scores.
        """
        fused: dict[str, Document] = {}

        for doc in dense_norm:
            dense_contribution = alpha * (doc.score if doc.score is not None else 0.0)
            doc_dict = doc.to_dict()
            doc_dict["score"] = dense_contribution
            fused[doc.id] = Document.from_dict(doc_dict)

        for doc in bm25_norm:
            bm25_contribution = (1.0 - alpha) * (doc.score if doc.score is not None else 0.0)
            if doc.id in fused:
                existing_score = fused[doc.id].score or 0.0
                doc_dict = fused[doc.id].to_dict()
                doc_dict["score"] = existing_score + bm25_contribution
                fused[doc.id] = Document.from_dict(doc_dict)
            else:
                doc_dict = doc.to_dict()
                doc_dict["score"] = bm25_contribution
                fused[doc.id] = Document.from_dict(doc_dict)

        return list(fused.values())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @component.output_types(documents=list[Document], alpha=float)
    def run(
        self,
        query: str,
        dense_documents: list[Document],
        bm25_documents: list[Document],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Fuse dense and BM25 retrieval results using Dynamic Alpha Tuning.

        :param query:
            The user query string used to prompt the LLM for effectiveness scoring.
        :param dense_documents:
            Ranked documents returned by the dense (embedding) retriever.
        :param bm25_documents:
            Ranked documents returned by the BM25 (sparse) retriever.
        :param top_k:
            Maximum number of documents to return.  Overrides the instance-level
            ``top_k`` when provided.
        :returns:
            A dictionary with:

            - ``documents``: List of :class:`~haystack.dataclasses.Document` objects
              ranked by their fused score, truncated to ``top_k``.
            - ``alpha``: The computed dynamic weighting coefficient α ∈ [0.0, 1.0].
              A value of 1.0 means pure dense weighting; 0.0 means pure BM25 weighting.
        :raises ComponentError:
            If the LLM returns a response that cannot be parsed as two integers in
            [0, 5] and ``raise_on_failure`` is ``True``.
        """
        if not self._is_warmed_up:
            self.warm_up()

        effective_top_k = top_k if top_k is not None else self.top_k

        # --- Edge cases: one or both retrievers returned nothing ---
        if not dense_documents and not bm25_documents:
            return {"documents": [], "alpha": 0.5}

        if not dense_documents:
            norm_bm25 = self._normalize_scores(bm25_documents)
            docs = sorted(norm_bm25, key=lambda d: d.score if d.score is not None else -inf, reverse=True)
            return {"documents": docs[:effective_top_k], "alpha": 0.0}

        if not bm25_documents:
            norm_dense = self._normalize_scores(dense_documents)
            docs = sorted(norm_dense, key=lambda d: d.score if d.score is not None else -inf, reverse=True)
            return {"documents": docs[:effective_top_k], "alpha": 1.0}

        # --- Standard path ---
        norm_dense = self._normalize_scores(dense_documents)
        norm_bm25 = self._normalize_scores(bm25_documents)

        dense_score, bm25_score = self._call_llm(query, norm_dense, norm_bm25)
        alpha = self._compute_alpha(dense_score, bm25_score)

        fused = self._fuse(alpha, norm_dense, norm_bm25)

        if self.sort_by_score:
            fused = sorted(fused, key=lambda d: d.score if d.score is not None else -inf, reverse=True)

        return {"documents": fused[:effective_top_k], "alpha": alpha}

    @component.output_types(documents=list[Document], alpha=float)
    async def run_async(
        self,
        query: str,
        dense_documents: list[Document],
        bm25_documents: list[Document],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Async version of :meth:`run`.

        Uses ``chat_generator.run_async`` when available; otherwise falls back to the
        synchronous LLM call.

        :param query:
            The user query string.
        :param dense_documents:
            Ranked documents from the dense retriever.
        :param bm25_documents:
            Ranked documents from the BM25 retriever.
        :param top_k:
            Maximum number of documents to return.
        :returns:
            Same structure as :meth:`run`.
        :raises ComponentError:
            If the LLM returns an unparseable response and ``raise_on_failure`` is
            ``True``.
        """
        if not self._is_warmed_up:
            self.warm_up()

        effective_top_k = top_k if top_k is not None else self.top_k

        if not dense_documents and not bm25_documents:
            return {"documents": [], "alpha": 0.5}

        if not dense_documents:
            norm_bm25 = self._normalize_scores(bm25_documents)
            docs = sorted(norm_bm25, key=lambda d: d.score if d.score is not None else -inf, reverse=True)
            return {"documents": docs[:effective_top_k], "alpha": 0.0}

        if not bm25_documents:
            norm_dense = self._normalize_scores(dense_documents)
            docs = sorted(norm_dense, key=lambda d: d.score if d.score is not None else -inf, reverse=True)
            return {"documents": docs[:effective_top_k], "alpha": 1.0}

        norm_dense = self._normalize_scores(dense_documents)
        norm_bm25 = self._normalize_scores(bm25_documents)

        dense_score, bm25_score = await self._call_llm_async(query, norm_dense, norm_bm25)
        alpha = self._compute_alpha(dense_score, bm25_score)

        fused = self._fuse(alpha, norm_dense, norm_bm25)

        if self.sort_by_score:
            fused = sorted(fused, key=lambda d: d.score if d.score is not None else -inf, reverse=True)

        return {"documents": fused[:effective_top_k], "alpha": alpha}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise this component to a dictionary.

        :returns:
            Dictionary representation suitable for
            :func:`~haystack.core.serialization.default_from_dict`.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            top_k=self.top_k,
            scoring_top_k=self.scoring_top_k,
            sort_by_score=self.sort_by_score,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DATDocumentJoiner":
        """
        Deserialise a :class:`DATDocumentJoiner` from a dictionary.

        :param data:
            The serialised component dictionary (as produced by :meth:`to_dict`).
        :returns:
            A new :class:`DATDocumentJoiner` instance.
        :raises DeserializationError:
            If the ``type`` field is missing or does not match this class, or if the
            embedded ``chat_generator`` cannot be deserialised.
        """
        if data["init_parameters"].get("chat_generator"):
            deserialize_chatgenerator_inplace(data["init_parameters"], key="chat_generator")
        return default_from_dict(cls, data)
