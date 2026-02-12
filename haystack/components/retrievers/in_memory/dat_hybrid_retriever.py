# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.joiners.dat_document_joiner import DATDocumentJoiner
from haystack.core.serialization import component_to_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import FilterPolicy
from haystack.utils import deserialize_chatgenerator_inplace

logger = logging.getLogger(__name__)


@component
class InMemoryDATHybridRetriever:
    """
    Self-contained DAT hybrid retriever for :class:`~haystack.document_stores.in_memory.InMemoryDocumentStore`.

    Combines BM25 and embedding retrieval from a single ``InMemoryDocumentStore``, then applies
    Dynamic Alpha Tuning (Hsu & Tzeng, 2025 — arXiv:2503.23013) to optimally fuse the results.
    Reduces a three-component hybrid pipeline (``BM25Retriever + EmbeddingRetriever + DATDocumentJoiner``)
    to a single component.

    The component requires that documents stored in the document store have embeddings indexed
    (use a :class:`~haystack.components.embedders.DocumentEmbedder` at indexing time), and that
    query embeddings are provided at inference time via a
    :class:`~haystack.components.embedders.TextEmbedder`.

    **Limitations:** Adds one LLM call per query. For high-throughput systems consider the async
    interface and/or caching.

    ### Usage example

    ```python
    from haystack import Document, Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.retrievers.in_memory import InMemoryDATHybridRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    document_store = InMemoryDocumentStore()
    # ... write documents with embeddings via a DocumentEmbedder ...

    pipeline = Pipeline()
    pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
    )
    pipeline.add_component(
        "dat_retriever",
        InMemoryDATHybridRetriever(
            document_store=document_store,
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            top_k=5,
        ),
    )
    pipeline.connect("text_embedder.embedding", "dat_retriever.query_embedding")

    query = "What is the capital of France?"
    result = pipeline.run({
        "text_embedder": {"text": query},
        "dat_retriever": {"query": query},
    })
    print(result["dat_retriever"]["alpha"])
    print(result["dat_retriever"]["documents"])
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        document_store: InMemoryDocumentStore,
        chat_generator: ChatGenerator,
        top_k: int = 10,
        scoring_top_k: int = 1,
        scale_score: bool = False,
        filters: dict[str, Any] | None = None,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
        raise_on_failure: bool = True,
    ):
        """
        Create an InMemoryDATHybridRetriever component.

        :param document_store:
            An :class:`~haystack.document_stores.in_memory.InMemoryDocumentStore` instance.
            Both BM25 and embedding retrieval will be performed against this store.
        :param chat_generator:
            A :class:`~haystack.components.generators.chat.types.ChatGenerator` instance used to
            score the top-``scoring_top_k`` results from each retriever.
        :param top_k:
            Maximum number of documents to return after fusion. Must be > 0.
        :param scoring_top_k:
            Number of top documents from each retriever passed to the LLM for effectiveness
            scoring. The paper recommends ``1`` (the default). Must be > 0.
        :param scale_score:
            When ``True``, scales BM25 and embedding scores to [0, 1] before DAT normalisation.
            When ``False`` (default), uses raw retriever scores.

            .. note::
                DAT applies its own min-max normalisation regardless of this setting. Enabling
                ``scale_score=True`` therefore results in double normalisation (score→[0,1]→[0,1])
                which is redundant and not recommended. The default ``False`` is correct for
                almost all use cases.
        :param filters:
            Metadata filters applied to both BM25 and embedding retrieval.
        :param filter_policy:
            Controls how runtime filters are combined with init-time filters:
            ``REPLACE`` (default) overrides init filters; ``MERGE`` combines them.
        :param raise_on_failure:
            If ``True`` (default), raises :exc:`~haystack.core.errors.ComponentError` when the
            LLM response cannot be parsed. If ``False``, logs a warning and falls back to α = 0.5.
        :raises ValueError:
            If ``document_store`` is not an ``InMemoryDocumentStore``, or ``top_k`` / ``scoring_top_k``
            are not positive integers.
        """
        if not isinstance(document_store, InMemoryDocumentStore):
            raise ValueError("document_store must be an instance of InMemoryDocumentStore")
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")
        if scoring_top_k <= 0:
            raise ValueError(f"scoring_top_k must be greater than 0, got {scoring_top_k}")

        self.document_store = document_store
        self.chat_generator = chat_generator
        self.top_k = top_k
        self.scoring_top_k = scoring_top_k
        self.scale_score = scale_score
        self.filters = filters
        self.filter_policy = filter_policy
        self.raise_on_failure = raise_on_failure

        # Internal joiner handles normalisation, LLM scoring, α computation, and fusion.
        # It is not serialised directly — it is reconstructed from the top-level params.
        self._joiner = DATDocumentJoiner(
            chat_generator=chat_generator,
            top_k=top_k,
            scoring_top_k=scoring_top_k,
            raise_on_failure=raise_on_failure,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """
        Warm up the underlying chat generator, if it supports warm-up.

        Delegates to :meth:`~haystack.components.joiners.DATDocumentJoiner.warm_up`.
        """
        self._joiner.warm_up()

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data sent to Posthog for usage analytics.
        """
        return {
            "document_store": type(self.document_store).__name__,
            "chat_generator": type(self.chat_generator).__name__,
        }

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def _resolve_filters(self, runtime_filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Apply the configured :attr:`filter_policy` to combine init-time and runtime filters.

        :param runtime_filters: Filters supplied at call time.
        :returns: The effective filters to use for this retrieval.
        """
        if self.filter_policy == FilterPolicy.MERGE and runtime_filters:
            return {**(self.filters or {}), **runtime_filters}
        return runtime_filters or self.filters

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @component.output_types(documents=list[Document], alpha=float)
    def run(
        self,
        query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve and fuse documents using Dynamic Alpha Tuning.

        Performs BM25 and embedding retrieval from :attr:`document_store` concurrently,
        then delegates to :class:`~haystack.components.joiners.DATDocumentJoiner` for
        normalisation, LLM scoring, α computation, and score fusion.

        :param query:
            The user query string used for BM25 retrieval and LLM effectiveness scoring.
        :param query_embedding:
            Query embedding vector used for dense (embedding) retrieval.
            Must have the same dimensionality as the embeddings stored in
            :attr:`document_store`. A dimension mismatch will propagate as an error from
            the document store.
        :param filters:
            Runtime metadata filters. Combined with init-time filters according to
            :attr:`filter_policy`.
        :param top_k:
            Maximum number of documents to return. Overrides the instance-level ``top_k``
            when provided.
        :returns:
            A dictionary with:

            - ``documents``: List of :class:`~haystack.dataclasses.Document` objects ranked
              by fused score, truncated to ``top_k``.
            - ``alpha``: The computed dynamic weighting coefficient α ∈ [0.0, 1.0].
        :raises ComponentError:
            If the LLM returns an unparseable response and ``raise_on_failure`` is ``True``.
        """
        effective_filters = self._resolve_filters(filters)
        effective_top_k = top_k if top_k is not None else self.top_k

        bm25_docs = self.document_store.bm25_retrieval(
            query=query,
            filters=effective_filters,
            top_k=effective_top_k,
            scale_score=self.scale_score,
        )
        dense_docs = self.document_store.embedding_retrieval(
            query_embedding=query_embedding,
            filters=effective_filters,
            top_k=effective_top_k,
            scale_score=self.scale_score,
        )

        return self._joiner.run(
            query=query,
            dense_documents=dense_docs,
            bm25_documents=bm25_docs,
            top_k=effective_top_k,
        )

    @component.output_types(documents=list[Document], alpha=float)
    async def run_async(
        self,
        query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Async version of :meth:`run`.

        Performs BM25 and embedding retrieval concurrently using ``asyncio.gather``, then
        delegates to :meth:`~haystack.components.joiners.DATDocumentJoiner.run_async`.

        :param query:
            The user query string.
        :param query_embedding:
            Query embedding vector. Must have the same dimensionality as the embeddings
            stored in :attr:`document_store`.
        :param filters:
            Runtime metadata filters.
        :param top_k:
            Maximum number of documents to return.
        :returns:
            Same structure as :meth:`run`.
        :raises ComponentError:
            If the LLM returns an unparseable response and ``raise_on_failure`` is ``True``.
        """
        effective_filters = self._resolve_filters(filters)
        effective_top_k = top_k if top_k is not None else self.top_k

        bm25_docs, dense_docs = await asyncio.gather(
            self.document_store.bm25_retrieval_async(
                query=query,
                filters=effective_filters,
                top_k=effective_top_k,
                scale_score=self.scale_score,
            ),
            self.document_store.embedding_retrieval_async(
                query_embedding=query_embedding,
                filters=effective_filters,
                top_k=effective_top_k,
                scale_score=self.scale_score,
            ),
        )

        return await self._joiner.run_async(
            query=query,
            dense_documents=dense_docs,
            bm25_documents=bm25_docs,
            top_k=effective_top_k,
        )

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
            document_store=self.document_store,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            top_k=self.top_k,
            scoring_top_k=self.scoring_top_k,
            scale_score=self.scale_score,
            filters=self.filters,
            filter_policy=self.filter_policy.value,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InMemoryDATHybridRetriever":
        """
        Deserialise an :class:`InMemoryDATHybridRetriever` from a dictionary.

        :param data:
            The serialised component dictionary (as produced by :meth:`to_dict`).
        :returns:
            A new :class:`InMemoryDATHybridRetriever` instance.
        :raises DeserializationError:
            If the ``type`` field is missing or incompatible, or if any nested component
            cannot be deserialised.
        """
        init_params = data.get("init_parameters", {})
        if "filter_policy" in init_params:
            init_params["filter_policy"] = FilterPolicy.from_str(init_params["filter_policy"])
        if init_params.get("chat_generator"):
            deserialize_chatgenerator_inplace(init_params, key="chat_generator")
        return default_from_dict(cls, data)
