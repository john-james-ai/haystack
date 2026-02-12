# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack.components.joiners.dat_document_joiner import DATDocumentJoiner
from haystack.core.errors import ComponentError
from haystack.dataclasses import ChatMessage, Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chat_reply(text: str) -> dict[str, Any]:
    """Build a minimal ChatGenerator-style reply dict."""
    msg = MagicMock(spec=ChatMessage)
    msg.text = text
    return {"replies": [msg]}


@pytest.fixture()
def mock_generator():
    """Synchronous mock ChatGenerator that returns '3 4' by default."""
    gen = MagicMock()
    gen.run.return_value = _make_chat_reply("3 4")
    return gen


@pytest.fixture()
def async_mock_generator():
    """ChatGenerator that exposes both sync and async run methods."""
    gen = MagicMock()
    gen.run.return_value = _make_chat_reply("3 4")
    gen.run_async = AsyncMock(return_value=_make_chat_reply("3 4"))
    return gen


@pytest.fixture()
def dense_docs():
    return [
        Document(content="Paris is the capital of France.", score=0.9),
        Document(content="France has many cultural sites.", score=0.6),
        Document(content="The Eiffel Tower is in Paris.", score=0.3),
    ]


@pytest.fixture()
def bm25_docs():
    return [
        Document(content="The capital city of France is Paris.", score=15.0),
        Document(content="France borders Germany and Spain.", score=8.0),
        Document(content="French cuisine is world-renowned.", score=3.0),
    ]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestDATDocumentJoinerInit:
    def test_init_defaults(self, mock_generator):
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        assert joiner.chat_generator is mock_generator
        assert joiner.top_k == 10
        assert joiner.scoring_top_k == 1
        assert joiner.sort_by_score is True
        assert joiner.raise_on_failure is True
        assert joiner._is_warmed_up is False

    def test_init_custom_params(self, mock_generator):
        joiner = DATDocumentJoiner(
            chat_generator=mock_generator,
            top_k=5,
            scoring_top_k=2,
            sort_by_score=False,
            raise_on_failure=False,
        )
        assert joiner.top_k == 5
        assert joiner.scoring_top_k == 2
        assert joiner.sort_by_score is False
        assert joiner.raise_on_failure is False

    def test_init_invalid_top_k(self, mock_generator):
        with pytest.raises(ValueError, match="top_k must be greater than 0"):
            DATDocumentJoiner(chat_generator=mock_generator, top_k=0)

    def test_init_invalid_scoring_top_k(self, mock_generator):
        with pytest.raises(ValueError, match="scoring_top_k must be greater than 0"):
            DATDocumentJoiner(chat_generator=mock_generator, scoring_top_k=-1)


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_normalize_standard(self):
        docs = [
            Document(content="a", score=0.0),
            Document(content="b", score=0.5),
            Document(content="c", score=1.0),
        ]
        result = DATDocumentJoiner._normalize_scores(docs)
        assert result[0].score == pytest.approx(0.0)
        assert result[1].score == pytest.approx(0.5)
        assert result[2].score == pytest.approx(1.0)

    def test_normalize_arbitrary_range(self):
        docs = [
            Document(content="a", score=10.0),
            Document(content="b", score=20.0),
            Document(content="c", score=30.0),
        ]
        result = DATDocumentJoiner._normalize_scores(docs)
        assert result[0].score == pytest.approx(0.0)
        assert result[1].score == pytest.approx(0.5)
        assert result[2].score == pytest.approx(1.0)

    def test_normalize_all_identical_scores(self):
        """Delta == 0 → all normalised scores should be 0.0."""
        docs = [Document(content=str(i), score=5.0) for i in range(3)]
        result = DATDocumentJoiner._normalize_scores(docs)
        for doc in result:
            assert doc.score == pytest.approx(0.0)

    def test_normalize_empty_list(self):
        assert DATDocumentJoiner._normalize_scores([]) == []

    def test_normalize_does_not_mutate_originals(self):
        docs = [Document(content="a", score=3.0), Document(content="b", score=7.0)]
        original_scores = [d.score for d in docs]
        DATDocumentJoiner._normalize_scores(docs)
        for doc, orig in zip(docs, original_scores):
            assert doc.score == orig

    def test_normalize_none_score_treated_as_zero(self):
        docs = [Document(content="a", score=None), Document(content="b", score=10.0)]
        result = DATDocumentJoiner._normalize_scores(docs)
        assert result[0].score == pytest.approx(0.0)
        assert result[1].score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Alpha calculation
# ---------------------------------------------------------------------------


class TestComputeAlpha:
    def test_both_zero_returns_half(self):
        assert DATDocumentJoiner._compute_alpha(0, 0) == pytest.approx(0.5)

    def test_dense_perfect_bm25_not(self):
        assert DATDocumentJoiner._compute_alpha(5, 0) == pytest.approx(1.0)
        assert DATDocumentJoiner._compute_alpha(5, 3) == pytest.approx(1.0)
        assert DATDocumentJoiner._compute_alpha(5, 4) == pytest.approx(1.0)

    def test_bm25_perfect_dense_not(self):
        assert DATDocumentJoiner._compute_alpha(0, 5) == pytest.approx(0.0)
        assert DATDocumentJoiner._compute_alpha(3, 5) == pytest.approx(0.0)
        assert DATDocumentJoiner._compute_alpha(4, 5) == pytest.approx(0.0)

    def test_both_perfect_proportional(self):
        """When both score 5, proportional rule applies → 5/(5+5) = 0.5."""
        assert DATDocumentJoiner._compute_alpha(5, 5) == pytest.approx(0.5)

    def test_proportional_weighting(self):
        assert DATDocumentJoiner._compute_alpha(3, 2) == pytest.approx(0.6)
        assert DATDocumentJoiner._compute_alpha(3, 4) == pytest.approx(0.4)
        assert DATDocumentJoiner._compute_alpha(1, 1) == pytest.approx(0.5)

    def test_alpha_rounded_to_one_decimal(self):
        # 3/(3+4) ≈ 0.4285... → rounds to 0.4
        alpha = DATDocumentJoiner._compute_alpha(3, 4)
        assert alpha == pytest.approx(0.4)
        # Verify it is actually rounded
        assert str(alpha) in ("0.4", "0.4285714285714286") or round(alpha, 1) == 0.4

    def test_alpha_range(self):
        for sv in range(6):
            for sb in range(6):
                alpha = DATDocumentJoiner._compute_alpha(sv, sb)
                assert 0.0 <= alpha <= 1.0


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


class TestParseScores:
    def test_parse_valid_response(self, mock_generator):
        joiner = DATDocumentJoiner(chat_generator=mock_generator, raise_on_failure=True)
        assert joiner._parse_scores("3 4") == (3, 4)
        assert joiner._parse_scores("0 5") == (0, 5)
        assert joiner._parse_scores("5 0") == (5, 0)

    def test_parse_response_with_surrounding_text(self, mock_generator):
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        assert joiner._parse_scores("Vector: 3, BM25: 4\nScores: 3 4") == (3, 4)
        assert joiner._parse_scores("The scores are: 2 3.") == (2, 3)

    def test_parse_failure_raises_component_error(self, mock_generator):
        joiner = DATDocumentJoiner(chat_generator=mock_generator, raise_on_failure=True)
        with pytest.raises(ComponentError, match="could not parse"):
            joiner._parse_scores("I cannot score this.")

    def test_parse_failure_fallback_when_raise_disabled(self, mock_generator):
        joiner = DATDocumentJoiner(chat_generator=mock_generator, raise_on_failure=False)
        # Falls back to (0, 0) → α = 0.5
        dense_score, bm25_score = joiner._parse_scores("unparseable output")
        assert dense_score == 0
        assert bm25_score == 0

    def test_parse_out_of_range_not_matched(self, mock_generator):
        """Values outside [0,5] should not be matched by the regex."""
        joiner = DATDocumentJoiner(chat_generator=mock_generator, raise_on_failure=True)
        with pytest.raises(ComponentError):
            joiner._parse_scores("6 7")


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class TestFuse:
    def test_fuse_combines_scores(self):
        dense = [Document(id="d1", content="a", score=1.0), Document(id="d2", content="b", score=0.5)]
        bm25 = [Document(id="d3", content="c", score=1.0), Document(id="d2", content="b", score=0.8)]
        alpha = 0.6

        fused = DATDocumentJoiner._fuse(alpha, dense, bm25)
        by_id = {d.id: d for d in fused}

        # d1: dense-only = 0.6 * 1.0 = 0.6
        assert by_id["d1"].score == pytest.approx(0.6)
        # d3: bm25-only = 0.4 * 1.0 = 0.4
        assert by_id["d3"].score == pytest.approx(0.4)
        # d2: dense + bm25 = 0.6 * 0.5 + 0.4 * 0.8 = 0.30 + 0.32 = 0.62
        assert by_id["d2"].score == pytest.approx(0.62)

    def test_fuse_alpha_zero_pure_bm25(self):
        dense = [Document(id="d1", content="a", score=1.0)]
        bm25 = [Document(id="d2", content="b", score=1.0)]
        fused = DATDocumentJoiner._fuse(0.0, dense, bm25)
        by_id = {d.id: d for d in fused}
        assert by_id["d1"].score == pytest.approx(0.0)
        assert by_id["d2"].score == pytest.approx(1.0)

    def test_fuse_alpha_one_pure_dense(self):
        dense = [Document(id="d1", content="a", score=1.0)]
        bm25 = [Document(id="d2", content="b", score=1.0)]
        fused = DATDocumentJoiner._fuse(1.0, dense, bm25)
        by_id = {d.id: d for d in fused}
        assert by_id["d1"].score == pytest.approx(1.0)
        assert by_id["d2"].score == pytest.approx(0.0)

    def test_fuse_empty_inputs(self):
        assert DATDocumentJoiner._fuse(0.5, [], []) == []

    def test_fuse_does_not_mutate_inputs(self):
        dense = [Document(id="d1", content="a", score=0.8)]
        bm25 = [Document(id="d1", content="a", score=0.6)]
        DATDocumentJoiner._fuse(0.5, dense, bm25)
        assert dense[0].score == pytest.approx(0.8)
        assert bm25[0].score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# run() — edge cases
# ---------------------------------------------------------------------------


class TestRunEdgeCases:
    def test_run_both_empty(self, mock_generator):
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        result = joiner.run(query="q", dense_documents=[], bm25_documents=[])
        assert result == {"documents": [], "alpha": 0.5}
        mock_generator.run.assert_not_called()

    def test_run_empty_dense_returns_bm25_only(self, mock_generator, bm25_docs):
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        result = joiner.run(query="q", dense_documents=[], bm25_documents=bm25_docs)
        assert result["alpha"] == pytest.approx(0.0)
        assert len(result["documents"]) == len(bm25_docs)
        mock_generator.run.assert_not_called()

    def test_run_empty_bm25_returns_dense_only(self, mock_generator, dense_docs):
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        result = joiner.run(query="q", dense_documents=dense_docs, bm25_documents=[])
        assert result["alpha"] == pytest.approx(1.0)
        assert len(result["documents"]) == len(dense_docs)
        mock_generator.run.assert_not_called()

    def test_run_top_k_limits_results(self, mock_generator, dense_docs, bm25_docs):
        joiner = DATDocumentJoiner(chat_generator=mock_generator, top_k=2)
        result = joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        assert len(result["documents"]) <= 2

    def test_run_top_k_override(self, mock_generator, dense_docs, bm25_docs):
        joiner = DATDocumentJoiner(chat_generator=mock_generator, top_k=10)
        result = joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs, top_k=1)
        assert len(result["documents"]) <= 1


# ---------------------------------------------------------------------------
# run() — standard path
# ---------------------------------------------------------------------------


class TestRunStandard:
    def test_run_returns_documents_and_alpha(self, mock_generator, dense_docs, bm25_docs):
        mock_generator.run.return_value = _make_chat_reply("3 4")
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        result = joiner.run(query="What is the capital of France?", dense_documents=dense_docs, bm25_documents=bm25_docs)
        assert "documents" in result
        assert "alpha" in result
        assert isinstance(result["documents"], list)
        assert all(isinstance(d, Document) for d in result["documents"])

    def test_run_alpha_computed_correctly(self, mock_generator, dense_docs, bm25_docs):
        mock_generator.run.return_value = _make_chat_reply("3 2")
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        result = joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        # 3/(3+2) = 0.6
        assert result["alpha"] == pytest.approx(0.6)

    def test_run_sorted_by_score_descending(self, mock_generator, dense_docs, bm25_docs):
        joiner = DATDocumentJoiner(chat_generator=mock_generator, sort_by_score=True)
        result = joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        scores = [d.score for d in result["documents"] if d.score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_run_calls_llm_once(self, mock_generator, dense_docs, bm25_docs):
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        mock_generator.run.assert_called_once()

    def test_run_llm_prompt_contains_query(self, mock_generator, dense_docs, bm25_docs):
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        query = "What is the capital of France?"
        joiner.run(query=query, dense_documents=dense_docs, bm25_documents=bm25_docs)
        call_args = mock_generator.run.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        assert any(query in msg.text for msg in messages)

    def test_run_parse_failure_raises(self, dense_docs, bm25_docs):
        gen = MagicMock()
        gen.run.return_value = _make_chat_reply("I cannot determine the scores.")
        joiner = DATDocumentJoiner(chat_generator=gen, raise_on_failure=True)
        with pytest.raises(ComponentError, match="could not parse"):
            joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)

    def test_run_parse_failure_fallback(self, dense_docs, bm25_docs):
        gen = MagicMock()
        gen.run.return_value = _make_chat_reply("I cannot determine the scores.")
        joiner = DATDocumentJoiner(chat_generator=gen, raise_on_failure=False)
        result = joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        # (0,0) → α=0.5
        assert result["alpha"] == pytest.approx(0.5)
        assert isinstance(result["documents"], list)

    def test_run_dense_perfect_hit(self, dense_docs, bm25_docs):
        gen = MagicMock()
        gen.run.return_value = _make_chat_reply("5 3")
        joiner = DATDocumentJoiner(chat_generator=gen)
        result = joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        assert result["alpha"] == pytest.approx(1.0)

    def test_run_bm25_perfect_hit(self, dense_docs, bm25_docs):
        gen = MagicMock()
        gen.run.return_value = _make_chat_reply("2 5")
        joiner = DATDocumentJoiner(chat_generator=gen)
        result = joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        assert result["alpha"] == pytest.approx(0.0)

    def test_scoring_top_k_two_concatenates_both_docs(self, mock_generator):
        """With scoring_top_k=2, both documents' content should appear in the LLM prompt (I-06)."""
        dense = [
            Document(content="Paris is the capital.", score=0.9),
            Document(content="France is in Europe.", score=0.6),
        ]
        bm25 = [
            Document(content="BM25 result one.", score=12.0),
            Document(content="BM25 result two.", score=8.0),
        ]
        joiner = DATDocumentJoiner(chat_generator=mock_generator, scoring_top_k=2)
        joiner.run(query="capital", dense_documents=dense, bm25_documents=bm25)

        call_args = mock_generator.run.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0].text
        # Both dense docs and both BM25 docs should appear in the prompt
        assert "Paris is the capital." in prompt_text
        assert "France is in Europe." in prompt_text
        assert "BM25 result one." in prompt_text
        assert "BM25 result two." in prompt_text

    def test_scoring_top_k_none_content_sends_empty_string(self, mock_generator):
        """Documents with None content should produce an empty reference string, not raise TypeError (I-06)."""
        dense = [Document(content=None, score=0.9)]
        bm25 = [Document(content=None, score=10.0)]
        joiner = DATDocumentJoiner(chat_generator=mock_generator, raise_on_failure=False)
        # Should not raise — None content becomes "" and the LLM call proceeds normally
        result = joiner.run(query="q", dense_documents=dense, bm25_documents=bm25)
        assert "documents" in result
        # Verify the prompt contained empty references (not "None")
        call_args = mock_generator.run.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        assert "None" not in messages[0].text


# ---------------------------------------------------------------------------
# run_async()
# ---------------------------------------------------------------------------


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_run_async_both_empty(self, async_mock_generator):
        joiner = DATDocumentJoiner(chat_generator=async_mock_generator)
        result = await joiner.run_async(query="q", dense_documents=[], bm25_documents=[])
        assert result == {"documents": [], "alpha": 0.5}
        async_mock_generator.run_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_empty_dense(self, async_mock_generator, bm25_docs):
        joiner = DATDocumentJoiner(chat_generator=async_mock_generator)
        result = await joiner.run_async(query="q", dense_documents=[], bm25_documents=bm25_docs)
        assert result["alpha"] == pytest.approx(0.0)
        async_mock_generator.run_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_standard_path_uses_run_async(self, async_mock_generator, dense_docs, bm25_docs):
        async_mock_generator.run_async.return_value = _make_chat_reply("3 2")
        joiner = DATDocumentJoiner(chat_generator=async_mock_generator)
        result = await joiner.run_async(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        assert result["alpha"] == pytest.approx(0.6)
        async_mock_generator.run_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_fallback_to_sync_when_no_async(self, mock_generator, dense_docs, bm25_docs):
        """Generator without run_async should use sync run instead."""
        # MagicMock auto-creates any attribute, so explicitly remove run_async to simulate
        # a synchronous-only generator.
        del mock_generator.run_async
        mock_generator.run.return_value = _make_chat_reply("4 1")
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        result = await joiner.run_async(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        # 4/(4+1) = 0.8
        assert result["alpha"] == pytest.approx(0.8)
        mock_generator.run.assert_called()


# ---------------------------------------------------------------------------
# warm_up
# ---------------------------------------------------------------------------


class TestWarmUp:
    def test_warm_up_called_once(self, mock_generator):
        mock_generator.warm_up = MagicMock()
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        joiner.warm_up()
        joiner.warm_up()
        mock_generator.warm_up.assert_called_once()

    def test_warm_up_skipped_when_no_method(self, mock_generator):
        """Generator without warm_up method should not raise."""
        del mock_generator.warm_up
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        joiner.warm_up()  # Should not raise
        assert joiner._is_warmed_up is True

    def test_warm_up_triggered_by_run(self, mock_generator, dense_docs, bm25_docs):
        mock_generator.warm_up = MagicMock()
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        assert joiner._is_warmed_up is False
        joiner.run(query="q", dense_documents=dense_docs, bm25_documents=bm25_docs)
        assert joiner._is_warmed_up is True
        mock_generator.warm_up.assert_called_once()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    def test_get_telemetry_data(self, mock_generator):
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        data = joiner._get_telemetry_data()
        assert "chat_generator" in data
        assert data["chat_generator"] == type(mock_generator).__name__


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict(self, mock_generator):
        mock_generator.to_dict = MagicMock(
            return_value={
                "type": "tests.mock.MockChatGenerator",
                "init_parameters": {},
            }
        )
        joiner = DATDocumentJoiner(
            chat_generator=mock_generator,
            top_k=5,
            scoring_top_k=2,
            sort_by_score=False,
            raise_on_failure=False,
        )
        data = joiner.to_dict()
        assert data["type"] == "haystack.components.joiners.dat_document_joiner.DATDocumentJoiner"
        ip = data["init_parameters"]
        assert ip["top_k"] == 5
        assert ip["scoring_top_k"] == 2
        assert ip["sort_by_score"] is False
        assert ip["raise_on_failure"] is False
        assert ip["chat_generator"]["type"] == "tests.mock.MockChatGenerator"

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")
        from haystack.components.generators.chat import OpenAIChatGenerator

        data = {
            "type": "haystack.components.joiners.dat_document_joiner.DATDocumentJoiner",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4o-mini",
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": False},
                    },
                },
                "top_k": 7,
                "scoring_top_k": 1,
                "sort_by_score": True,
                "raise_on_failure": True,
            },
        }
        joiner = DATDocumentJoiner.from_dict(data)
        assert isinstance(joiner.chat_generator, OpenAIChatGenerator)
        assert joiner.top_k == 7
        assert joiner.raise_on_failure is True

    def test_to_dict_default_params(self, mock_generator):
        mock_generator.to_dict = MagicMock(
            return_value={"type": "tests.mock.MockChatGenerator", "init_parameters": {}}
        )
        joiner = DATDocumentJoiner(chat_generator=mock_generator)
        data = joiner.to_dict()
        ip = data["init_parameters"]
        assert ip["top_k"] == 10
        assert ip["scoring_top_k"] == 1
        assert ip["sort_by_score"] is True
        assert ip["raise_on_failure"] is True


# ---------------------------------------------------------------------------
# Export from package
# ---------------------------------------------------------------------------


class TestPackageExport:
    def test_importable_from_joiners(self):
        from haystack.components.joiners import DATDocumentJoiner as _DATDocumentJoiner

        assert _DATDocumentJoiner is DATDocumentJoiner
