# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for DATDocumentJoiner inside a Haystack Pipeline.

These tests verify:
- Pipeline wiring (DATDocumentJoiner receives outputs of BM25 + embedding retrievers)
- Pipeline serialization round-trip
- Alpha value is propagated through the pipeline output
- Known-alpha selection (dense-perfect → α=1.0, BM25-perfect → α=0.0)
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from haystack import Pipeline
from haystack.components.joiners.dat_document_joiner import DATDocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_reply(text: str) -> dict[str, Any]:
    msg = MagicMock(spec=ChatMessage)
    msg.text = text
    return {"replies": [msg]}


def _make_generator(reply: str = "3 4") -> MagicMock:
    gen = MagicMock()
    gen.run.return_value = _make_chat_reply(reply)
    return gen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def document_store():
    """InMemoryDocumentStore with 4 documents and hand-crafted embeddings."""
    store = InMemoryDocumentStore()
    docs = [
        Document(id="d1", content="Paris is the capital of France.", embedding=[1.0, 0.0, 0.0]),
        Document(id="d2", content="France has many cultural sites.", embedding=[0.8, 0.2, 0.0]),
        Document(id="d3", content="Berlin is the capital of Germany.", embedding=[0.0, 1.0, 0.0]),
        Document(id="d4", content="The Eiffel Tower is in Paris.", embedding=[0.9, 0.1, 0.0]),
    ]
    store.write_documents(docs)
    return store


# ---------------------------------------------------------------------------
# Pipeline wiring tests
# ---------------------------------------------------------------------------


class TestDATJoinerPipelineWiring:
    def test_pipeline_runs_end_to_end(self, document_store):
        """DATDocumentJoiner correctly wired to BM25 + embedding retrievers."""
        gen = _make_generator("3 4")
        pipeline = Pipeline()
        pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=document_store, top_k=4))
        pipeline.add_component("dense", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4))
        pipeline.add_component("joiner", DATDocumentJoiner(chat_generator=gen, top_k=3))

        pipeline.connect("bm25.documents", "joiner.bm25_documents")
        pipeline.connect("dense.documents", "joiner.dense_documents")

        result = pipeline.run({
            "bm25": {"query": "capital of France"},
            "dense": {"query_embedding": [1.0, 0.0, 0.0]},
            "joiner": {"query": "capital of France"},
        })

        assert "joiner" in result
        assert "documents" in result["joiner"]
        assert "alpha" in result["joiner"]
        assert isinstance(result["joiner"]["documents"], list)
        assert len(result["joiner"]["documents"]) <= 3
        assert 0.0 <= result["joiner"]["alpha"] <= 1.0

    def test_pipeline_alpha_value_from_llm(self, document_store):
        """Alpha value reported in pipeline output matches LLM scores."""
        gen = _make_generator("3 2")  # 3/(3+2) = 0.6
        pipeline = Pipeline()
        pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=document_store, top_k=4))
        pipeline.add_component("dense", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4))
        pipeline.add_component("joiner", DATDocumentJoiner(chat_generator=gen, top_k=4))

        pipeline.connect("bm25.documents", "joiner.bm25_documents")
        pipeline.connect("dense.documents", "joiner.dense_documents")

        result = pipeline.run({
            "bm25": {"query": "capital"},
            "dense": {"query_embedding": [1.0, 0.0, 0.0]},
            "joiner": {"query": "capital"},
        })
        assert result["joiner"]["alpha"] == pytest.approx(0.6)

    def test_pipeline_dense_perfect_alpha(self, document_store):
        """When dense scores 5, BM25 scores 0: α should be 1.0."""
        gen = _make_generator("5 0")
        pipeline = Pipeline()
        pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=document_store, top_k=4))
        pipeline.add_component("dense", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4))
        pipeline.add_component("joiner", DATDocumentJoiner(chat_generator=gen, top_k=4))

        pipeline.connect("bm25.documents", "joiner.bm25_documents")
        pipeline.connect("dense.documents", "joiner.dense_documents")

        result = pipeline.run({
            "bm25": {"query": "capital France"},
            "dense": {"query_embedding": [1.0, 0.0, 0.0]},
            "joiner": {"query": "capital France"},
        })
        assert result["joiner"]["alpha"] == pytest.approx(1.0)

    def test_pipeline_bm25_perfect_alpha(self, document_store):
        """When BM25 scores 5, dense scores 0: α should be 0.0."""
        gen = _make_generator("0 5")
        pipeline = Pipeline()
        pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=document_store, top_k=4))
        pipeline.add_component("dense", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4))
        pipeline.add_component("joiner", DATDocumentJoiner(chat_generator=gen, top_k=4))

        pipeline.connect("bm25.documents", "joiner.bm25_documents")
        pipeline.connect("dense.documents", "joiner.dense_documents")

        result = pipeline.run({
            "bm25": {"query": "capital France"},
            "dense": {"query_embedding": [1.0, 0.0, 0.0]},
            "joiner": {"query": "capital France"},
        })
        assert result["joiner"]["alpha"] == pytest.approx(0.0)

    def test_pipeline_output_sorted_descending(self, document_store):
        """Pipeline output documents are sorted by descending fused score."""
        gen = _make_generator("3 3")
        pipeline = Pipeline()
        pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=document_store, top_k=4))
        pipeline.add_component("dense", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4))
        pipeline.add_component("joiner", DATDocumentJoiner(chat_generator=gen, top_k=4))

        pipeline.connect("bm25.documents", "joiner.bm25_documents")
        pipeline.connect("dense.documents", "joiner.dense_documents")

        result = pipeline.run({
            "bm25": {"query": "capital"},
            "dense": {"query_embedding": [1.0, 0.0, 0.0]},
            "joiner": {"query": "capital"},
        })
        scores = [d.score for d in result["joiner"]["documents"] if d.score is not None]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Pipeline serialisation round-trip
# ---------------------------------------------------------------------------


class TestPipelineSerializationRoundTrip:
    def test_dat_joiner_pipeline_to_dict(self, document_store, monkeypatch):
        """DATDocumentJoiner pipeline serialises to a dict without errors."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")
        from haystack.components.generators.chat import OpenAIChatGenerator

        pipeline = Pipeline()
        pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=document_store, top_k=4))
        pipeline.add_component("dense", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4))
        pipeline.add_component(
            "joiner",
            DATDocumentJoiner(
                chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
                top_k=5,
            ),
        )
        pipeline.connect("bm25.documents", "joiner.bm25_documents")
        pipeline.connect("dense.documents", "joiner.dense_documents")

        serialised = pipeline.to_dict()
        assert "joiner" in serialised["components"]
        joiner_dict = serialised["components"]["joiner"]
        assert joiner_dict["type"] == (
            "haystack.components.joiners.dat_document_joiner.DATDocumentJoiner"
        )
        assert joiner_dict["init_parameters"]["top_k"] == 5
        assert "chat_generator" in joiner_dict["init_parameters"]

    def test_dat_joiner_pipeline_round_trip(self, document_store, monkeypatch):
        """DATDocumentJoiner pipeline survives a to_dict → from_dict round-trip."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")
        from haystack.components.generators.chat import OpenAIChatGenerator

        pipeline = Pipeline()
        pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=document_store, top_k=4))
        pipeline.add_component("dense", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4))
        pipeline.add_component(
            "joiner",
            DATDocumentJoiner(
                chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
                top_k=5,
                raise_on_failure=False,
            ),
        )
        pipeline.connect("bm25.documents", "joiner.bm25_documents")
        pipeline.connect("dense.documents", "joiner.dense_documents")

        serialised = pipeline.to_dict()
        restored = Pipeline.from_dict(serialised)

        restored_joiner = restored.get_component("joiner")
        assert isinstance(restored_joiner, DATDocumentJoiner)
        assert restored_joiner.top_k == 5
        assert restored_joiner.raise_on_failure is False
