# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryDATHybridRetriever
from haystack.core.errors import ComponentError
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import FilterPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_reply(text: str) -> dict[str, Any]:
    msg = MagicMock(spec=ChatMessage)
    msg.text = text
    return {"replies": [msg]}


def _make_doc(content: str, score: float, id: str | None = None) -> Document:
    return Document(id=id, content=content, score=score)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_generator():
    gen = MagicMock()
    gen.run.return_value = _make_chat_reply("3 4")
    return gen


@pytest.fixture()
def async_mock_generator():
    gen = MagicMock()
    gen.run.return_value = _make_chat_reply("3 4")
    gen.run_async = AsyncMock(return_value=_make_chat_reply("3 4"))
    return gen


@pytest.fixture()
def empty_store():
    return InMemoryDocumentStore()


@pytest.fixture()
def populated_store():
    """Document store with 4 documents that have hand-crafted embeddings."""
    store = InMemoryDocumentStore()
    docs = [
        Document(id="d1", content="Paris is the capital of France.", embedding=[1.0, 0.0]),
        Document(id="d2", content="France has many cultural sites.", embedding=[0.8, 0.2]),
        Document(id="d3", content="Berlin is the capital of Germany.", embedding=[0.0, 1.0]),
        Document(id="d4", content="Germany borders France to the east.", embedding=[0.2, 0.8]),
    ]
    store.write_documents(docs)
    return store


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInMemoryDATHybridRetrieverInit:
    def test_init_defaults(self, mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store,
            chat_generator=mock_generator,
        )
        assert retriever.document_store is empty_store
        assert retriever.chat_generator is mock_generator
        assert retriever.top_k == 10
        assert retriever.scoring_top_k == 1
        assert retriever.scale_score is False
        assert retriever.filters is None
        assert retriever.filter_policy == FilterPolicy.REPLACE
        assert retriever.raise_on_failure is True

    def test_init_custom_params(self, mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store,
            chat_generator=mock_generator,
            top_k=5,
            scoring_top_k=2,
            scale_score=True,
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
            filter_policy=FilterPolicy.MERGE,
            raise_on_failure=False,
        )
        assert retriever.top_k == 5
        assert retriever.scoring_top_k == 2
        assert retriever.scale_score is True
        assert retriever.filters is not None
        assert retriever.filter_policy == FilterPolicy.MERGE
        assert retriever.raise_on_failure is False

    def test_init_invalid_document_store(self, mock_generator):
        with pytest.raises(ValueError, match="InMemoryDocumentStore"):
            InMemoryDATHybridRetriever(document_store="not_a_store", chat_generator=mock_generator)

    def test_init_invalid_top_k(self, mock_generator, empty_store):
        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            InMemoryDATHybridRetriever(document_store=empty_store, chat_generator=mock_generator, top_k=0)

    def test_init_invalid_scoring_top_k(self, mock_generator, empty_store):
        with pytest.raises(ValueError, match="scoring_top_k must be greater than 0"):
            InMemoryDATHybridRetriever(
                document_store=empty_store, chat_generator=mock_generator, scoring_top_k=-1
            )

    def test_internal_joiner_created(self, mock_generator, empty_store):
        from haystack.components.joiners.dat_document_joiner import DATDocumentJoiner

        retriever = InMemoryDATHybridRetriever(document_store=empty_store, chat_generator=mock_generator)
        assert isinstance(retriever._joiner, DATDocumentJoiner)
        assert retriever._joiner.chat_generator is mock_generator


# ---------------------------------------------------------------------------
# Filter resolution
# ---------------------------------------------------------------------------


class TestFilterResolution:
    def test_replace_policy_uses_runtime_filters(self, mock_generator, empty_store):
        init_filters = {"field": "meta.a", "operator": "==", "value": "x"}
        runtime_filters = {"field": "meta.b", "operator": "==", "value": "y"}
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store,
            chat_generator=mock_generator,
            filters=init_filters,
            filter_policy=FilterPolicy.REPLACE,
        )
        result = retriever._resolve_filters(runtime_filters)
        assert result == runtime_filters

    def test_replace_policy_falls_back_to_init_filters(self, mock_generator, empty_store):
        init_filters = {"field": "meta.a", "operator": "==", "value": "x"}
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store,
            chat_generator=mock_generator,
            filters=init_filters,
            filter_policy=FilterPolicy.REPLACE,
        )
        result = retriever._resolve_filters(None)
        assert result == init_filters

    def test_merge_policy_combines_filters(self, mock_generator, empty_store):
        init_filters = {"field": "meta.a", "operator": "==", "value": "x"}
        runtime_filters = {"field": "meta.b", "operator": "==", "value": "y"}
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store,
            chat_generator=mock_generator,
            filters=init_filters,
            filter_policy=FilterPolicy.MERGE,
        )
        result = retriever._resolve_filters(runtime_filters)
        assert "field" in result  # merged dict contains both keys
        # runtime keys override init keys for shared keys; here both unique keys present
        assert result.get("field") == "meta.b"  # runtime takes precedence

    def test_no_filters_returns_none(self, mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(document_store=empty_store, chat_generator=mock_generator)
        assert retriever._resolve_filters(None) is None


# ---------------------------------------------------------------------------
# run() — unit (mocked document store)
# ---------------------------------------------------------------------------


class TestRunUnit:
    def test_run_calls_both_retrieval_methods(self, mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=mock_generator
        )
        dense_result = [Document(id="d1", content="Paris.", score=0.9)]
        bm25_result = [Document(id="d2", content="France.", score=10.0)]

        empty_store.bm25_retrieval = MagicMock(return_value=bm25_result)
        empty_store.embedding_retrieval = MagicMock(return_value=dense_result)

        result = retriever.run(query="capital France", query_embedding=[0.1, 0.9])
        empty_store.bm25_retrieval.assert_called_once()
        empty_store.embedding_retrieval.assert_called_once()
        assert "documents" in result
        assert "alpha" in result

    def test_run_passes_top_k_to_store(self, mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=mock_generator, top_k=3
        )
        empty_store.bm25_retrieval = MagicMock(return_value=[])
        empty_store.embedding_retrieval = MagicMock(return_value=[])

        retriever.run(query="q", query_embedding=[0.5])
        call_kwargs = empty_store.bm25_retrieval.call_args[1]
        assert call_kwargs["top_k"] == 3

    def test_run_top_k_override(self, mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=mock_generator, top_k=10
        )
        empty_store.bm25_retrieval = MagicMock(return_value=[])
        empty_store.embedding_retrieval = MagicMock(return_value=[])

        retriever.run(query="q", query_embedding=[0.5], top_k=2)
        call_kwargs = empty_store.bm25_retrieval.call_args[1]
        assert call_kwargs["top_k"] == 2

    def test_run_passes_filters_to_store(self, mock_generator, empty_store):
        filters = {"field": "meta.lang", "operator": "==", "value": "en"}
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=mock_generator, filters=filters
        )
        empty_store.bm25_retrieval = MagicMock(return_value=[])
        empty_store.embedding_retrieval = MagicMock(return_value=[])

        retriever.run(query="q", query_embedding=[0.5])
        assert empty_store.bm25_retrieval.call_args[1]["filters"] == filters
        assert empty_store.embedding_retrieval.call_args[1]["filters"] == filters

    def test_run_empty_results_both_sides(self, mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=mock_generator
        )
        empty_store.bm25_retrieval = MagicMock(return_value=[])
        empty_store.embedding_retrieval = MagicMock(return_value=[])

        result = retriever.run(query="q", query_embedding=[0.5])
        assert result == {"documents": [], "alpha": 0.5}
        mock_generator.run.assert_not_called()

    def test_run_llm_parse_failure_raises(self, empty_store):
        gen = MagicMock()
        gen.run.return_value = _make_chat_reply("cannot determine scores")
        dense_docs = [Document(id="d1", content="a", score=0.9)]
        bm25_docs = [Document(id="d2", content="b", score=10.0)]

        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=gen, raise_on_failure=True
        )
        empty_store.bm25_retrieval = MagicMock(return_value=bm25_docs)
        empty_store.embedding_retrieval = MagicMock(return_value=dense_docs)

        with pytest.raises(ComponentError, match="could not parse"):
            retriever.run(query="q", query_embedding=[0.5])

    def test_run_alpha_in_output(self, mock_generator, empty_store):
        mock_generator.run.return_value = _make_chat_reply("5 2")
        dense_docs = [Document(id="d1", content="a", score=0.9)]
        bm25_docs = [Document(id="d2", content="b", score=10.0)]

        retriever = InMemoryDATHybridRetriever(document_store=empty_store, chat_generator=mock_generator)
        empty_store.bm25_retrieval = MagicMock(return_value=bm25_docs)
        empty_store.embedding_retrieval = MagicMock(return_value=dense_docs)

        result = retriever.run(query="q", query_embedding=[0.5])
        # S_v=5, S_b=2 → dense perfect, bm25≠5 → α=1.0
        assert result["alpha"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# run_async() — unit (mocked document store)
# ---------------------------------------------------------------------------


class TestRunAsyncUnit:
    @pytest.mark.asyncio
    async def test_run_async_both_empty(self, async_mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=async_mock_generator
        )
        empty_store.bm25_retrieval_async = AsyncMock(return_value=[])
        empty_store.embedding_retrieval_async = AsyncMock(return_value=[])

        result = await retriever.run_async(query="q", query_embedding=[0.5])
        assert result == {"documents": [], "alpha": 0.5}
        async_mock_generator.run_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_calls_both_async_methods(self, async_mock_generator, empty_store):
        dense_docs = [Document(id="d1", content="a", score=0.9)]
        bm25_docs = [Document(id="d2", content="b", score=10.0)]
        async_mock_generator.run_async.return_value = _make_chat_reply("3 2")

        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=async_mock_generator
        )
        empty_store.bm25_retrieval_async = AsyncMock(return_value=bm25_docs)
        empty_store.embedding_retrieval_async = AsyncMock(return_value=dense_docs)

        result = await retriever.run_async(query="q", query_embedding=[0.5])
        empty_store.bm25_retrieval_async.assert_awaited_once()
        empty_store.embedding_retrieval_async.assert_awaited_once()
        # 3/(3+2) = 0.6
        assert result["alpha"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# warm_up
# ---------------------------------------------------------------------------


class TestWarmUp:
    def test_warm_up_delegates_to_joiner(self, mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=mock_generator
        )
        mock_generator.warm_up = MagicMock()
        retriever.warm_up()
        mock_generator.warm_up.assert_called_once()

    def test_warm_up_called_once(self, mock_generator, empty_store):
        mock_generator.warm_up = MagicMock()
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=mock_generator
        )
        retriever.warm_up()
        retriever.warm_up()
        mock_generator.warm_up.assert_called_once()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    def test_get_telemetry_data(self, mock_generator, empty_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=mock_generator
        )
        data = retriever._get_telemetry_data()
        assert "document_store" in data
        assert "chat_generator" in data
        assert data["document_store"] == "InMemoryDocumentStore"
        assert data["chat_generator"] == type(mock_generator).__name__


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict(self, mock_generator, empty_store):
        mock_generator.to_dict = MagicMock(
            return_value={"type": "tests.mock.MockChatGenerator", "init_parameters": {}}
        )
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store,
            chat_generator=mock_generator,
            top_k=5,
            scoring_top_k=2,
            scale_score=True,
            filters=None,
            filter_policy=FilterPolicy.MERGE,
            raise_on_failure=False,
        )
        data = retriever.to_dict()
        assert (
            data["type"]
            == "haystack.components.retrievers.in_memory.dat_hybrid_retriever.InMemoryDATHybridRetriever"
        )
        ip = data["init_parameters"]
        assert ip["top_k"] == 5
        assert ip["scoring_top_k"] == 2
        assert ip["scale_score"] is True
        assert ip["filter_policy"] == FilterPolicy.MERGE.value
        assert ip["raise_on_failure"] is False
        assert ip["chat_generator"]["type"] == "tests.mock.MockChatGenerator"
        assert "document_store" in ip

    def test_to_dict_default_params(self, mock_generator, empty_store):
        mock_generator.to_dict = MagicMock(
            return_value={"type": "tests.mock.MockChatGenerator", "init_parameters": {}}
        )
        retriever = InMemoryDATHybridRetriever(
            document_store=empty_store, chat_generator=mock_generator
        )
        data = retriever.to_dict()
        ip = data["init_parameters"]
        assert ip["top_k"] == 10
        assert ip["scoring_top_k"] == 1
        assert ip["scale_score"] is False
        assert ip["filters"] is None
        assert ip["raise_on_failure"] is True

    def test_from_dict(self, monkeypatch, empty_store):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")
        from haystack.components.generators.chat import OpenAIChatGenerator

        store_dict = empty_store.to_dict()
        data = {
            "type": "haystack.components.retrievers.in_memory.dat_hybrid_retriever.InMemoryDATHybridRetriever",
            "init_parameters": {
                "document_store": store_dict,
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4o-mini",
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": False},
                    },
                },
                "top_k": 7,
                "scoring_top_k": 1,
                "scale_score": False,
                "filters": None,
                "filter_policy": FilterPolicy.REPLACE.value,
                "raise_on_failure": True,
            },
        }
        retriever = InMemoryDATHybridRetriever.from_dict(data)
        assert isinstance(retriever.chat_generator, OpenAIChatGenerator)
        assert isinstance(retriever.document_store, InMemoryDocumentStore)
        assert retriever.top_k == 7
        assert retriever.raise_on_failure is True


# ---------------------------------------------------------------------------
# Package export
# ---------------------------------------------------------------------------


class TestPackageExport:
    def test_importable_from_in_memory(self):
        from haystack.components.retrievers.in_memory import InMemoryDATHybridRetriever as _R

        assert _R is InMemoryDATHybridRetriever

    def test_importable_from_retrievers(self):
        from haystack.components.retrievers import InMemoryDATHybridRetriever as _R

        assert _R is InMemoryDATHybridRetriever


# ---------------------------------------------------------------------------
# Integration tests (real InMemoryDocumentStore, mocked LLM)
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_run_with_real_store_and_mocked_llm(self, mock_generator, populated_store):
        """Full retrieval path with real InMemoryDocumentStore and mocked LLM."""
        mock_generator.run.return_value = _make_chat_reply("3 4")
        retriever = InMemoryDATHybridRetriever(
            document_store=populated_store,
            chat_generator=mock_generator,
            top_k=3,
        )
        result = retriever.run(
            query="capital of France",
            query_embedding=[1.0, 0.0],  # close to d1 and d2 embeddings
        )
        assert "documents" in result
        assert "alpha" in result
        assert isinstance(result["documents"], list)
        assert 0.0 <= result["alpha"] <= 1.0
        assert len(result["documents"]) <= 3

    def test_run_alpha_value_reflected(self, populated_store):
        """Alpha computed from LLM scores should be reflected in output."""
        gen = MagicMock()
        gen.run.return_value = _make_chat_reply("5 0")  # dense perfect → α=1.0
        retriever = InMemoryDATHybridRetriever(
            document_store=populated_store,
            chat_generator=gen,
        )
        result = retriever.run(query="capital France", query_embedding=[1.0, 0.0])
        assert result["alpha"] == pytest.approx(1.0)

    def test_run_sorted_by_score(self, mock_generator, populated_store):
        retriever = InMemoryDATHybridRetriever(
            document_store=populated_store,
            chat_generator=mock_generator,
        )
        result = retriever.run(query="capital", query_embedding=[1.0, 0.0])
        scores = [d.score for d in result["documents"] if d.score is not None]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_run_async_with_real_store(self, async_mock_generator, populated_store):
        """Async retrieval with real document store."""
        async_mock_generator.run_async.return_value = _make_chat_reply("2 3")
        retriever = InMemoryDATHybridRetriever(
            document_store=populated_store,
            chat_generator=async_mock_generator,
            top_k=2,
        )
        result = await retriever.run_async(
            query="capital Germany",
            query_embedding=[0.0, 1.0],
        )
        assert "documents" in result
        assert "alpha" in result
        assert len(result["documents"]) <= 2
        # 2/(2+3) = 0.4
        assert result["alpha"] == pytest.approx(0.4)

    def test_run_raises_on_llm_failure_by_default(self, populated_store):
        gen = MagicMock()
        gen.run.return_value = _make_chat_reply("I cannot score this.")
        retriever = InMemoryDATHybridRetriever(
            document_store=populated_store, chat_generator=gen, raise_on_failure=True
        )
        with pytest.raises(ComponentError, match="could not parse"):
            retriever.run(query="capital", query_embedding=[1.0, 0.0])

    def test_run_fallback_on_llm_failure(self, populated_store):
        gen = MagicMock()
        gen.run.return_value = _make_chat_reply("I cannot score this.")
        retriever = InMemoryDATHybridRetriever(
            document_store=populated_store, chat_generator=gen, raise_on_failure=False
        )
        result = retriever.run(query="capital", query_embedding=[1.0, 0.0])
        # (0,0) → α=0.5
        assert result["alpha"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pipeline wiring tests  (I-01)
# ---------------------------------------------------------------------------


class TestPipelineWiring:
    def test_inmemorydathybrid_pipeline_wiring(self, populated_store, mock_generator):
        """InMemoryDATHybridRetriever wired into a Pipeline via add_component / run."""
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
                "query_embedding": [1.0, 0.0],
            },
        })
        assert "dat_retriever" in result
        assert "documents" in result["dat_retriever"]
        assert "alpha" in result["dat_retriever"]
        assert isinstance(result["dat_retriever"]["documents"], list)
        assert len(result["dat_retriever"]["documents"]) <= 3
        assert 0.0 <= result["dat_retriever"]["alpha"] <= 1.0

    def test_pipeline_alpha_propagated(self, populated_store):
        """Alpha computed inside the Pipeline is correctly surfaced in pipeline output."""
        gen = MagicMock()
        gen.run.return_value = _make_chat_reply("5 0")  # dense perfect → α=1.0
        pipeline = Pipeline()
        pipeline.add_component(
            "dat_retriever",
            InMemoryDATHybridRetriever(document_store=populated_store, chat_generator=gen),
        )
        result = pipeline.run({
            "dat_retriever": {"query": "capital France", "query_embedding": [1.0, 0.0]},
        })
        assert result["dat_retriever"]["alpha"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pipeline serialisation round-trip tests  (I-02)
# ---------------------------------------------------------------------------


class TestPipelineSerializationRoundTrip:
    def test_inmemorydathybrid_pipeline_round_trip(self, populated_store, monkeypatch):
        """InMemoryDATHybridRetriever pipeline survives to_dict → from_dict round-trip."""
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
                scale_score=True,
            ),
        )
        serialised = pipeline.to_dict()
        restored = Pipeline.from_dict(serialised)

        restored_retriever = restored.get_component("dat_retriever")
        assert isinstance(restored_retriever, InMemoryDATHybridRetriever)
        assert isinstance(restored_retriever.document_store, InMemoryDocumentStore)
        assert restored_retriever.top_k == 4
        assert restored_retriever.raise_on_failure is False
        assert restored_retriever.scale_score is True

    def test_inmemorydathybrid_to_dict_structure(self, populated_store, monkeypatch):
        """Pipeline to_dict contains all expected keys for InMemoryDATHybridRetriever."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")
        from haystack.components.generators.chat import OpenAIChatGenerator

        pipeline = Pipeline()
        pipeline.add_component(
            "dat_retriever",
            InMemoryDATHybridRetriever(
                document_store=populated_store,
                chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
                top_k=5,
            ),
        )
        serialised = pipeline.to_dict()
        comp = serialised["components"]["dat_retriever"]
        assert comp["type"] == (
            "haystack.components.retrievers.in_memory.dat_hybrid_retriever.InMemoryDATHybridRetriever"
        )
        ip = comp["init_parameters"]
        assert ip["top_k"] == 5
        assert "document_store" in ip
        assert "chat_generator" in ip
        assert ip["chat_generator"]["type"] == (
            "haystack.components.generators.chat.openai.OpenAIChatGenerator"
        )
