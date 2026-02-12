# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock

import pytest

from haystack.components.evaluators.document_precision import DocumentPrecisionEvaluator
from haystack.dataclasses import ChatMessage, Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_reply(text: str) -> dict[str, Any]:
    """Build a minimal ChatGenerator-style reply dict with the given text."""
    msg = MagicMock(spec=ChatMessage)
    msg.text = text
    return {"replies": [msg]}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_default_k():
    evaluator = DocumentPrecisionEvaluator()
    assert evaluator.k == 1


def test_init_custom_k():
    evaluator = DocumentPrecisionEvaluator(k=5)
    assert evaluator.k == 5


def test_init_invalid_k_zero():
    with pytest.raises(ValueError, match="k must be greater than 0"):
        DocumentPrecisionEvaluator(k=0)


def test_init_invalid_k_negative():
    with pytest.raises(ValueError, match="k must be greater than 0"):
        DocumentPrecisionEvaluator(k=-3)


# ---------------------------------------------------------------------------
# Core precision computation
# ---------------------------------------------------------------------------


def test_precision_at_1_perfect_retrieval():
    evaluator = DocumentPrecisionEvaluator(k=1)
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
    )
    assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}


def test_precision_at_1_complete_miss():
    evaluator = DocumentPrecisionEvaluator(k=1)
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
    )
    assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}


def test_precision_at_1_partial():
    evaluator = DocumentPrecisionEvaluator(k=1)
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
    )
    assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}


def test_precision_at_k_partial():
    """2 relevant docs out of top-3 → Precision@3 = 2/3."""
    evaluator = DocumentPrecisionEvaluator(k=3)
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="France"), Document(content="Germany")],
        ],
        retrieved_documents=[
            [
                Document(content="France"),
                Document(content="UK"),
                Document(content="Germany"),
            ],
        ],
    )
    assert result["individual_scores"] == [pytest.approx(2 / 3)]
    assert result["score"] == pytest.approx(2 / 3)


def test_precision_only_top_k_considered():
    """Relevant document at rank k+1 must NOT improve the score."""
    evaluator = DocumentPrecisionEvaluator(k=1)
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")]],
        # Berlin appears at rank 2, outside top-1
        retrieved_documents=[[Document(content="Paris"), Document(content="Berlin")]],
    )
    assert result == {"individual_scores": [0.0], "score": 0.0}


def test_precision_multi_query_averaging():
    evaluator = DocumentPrecisionEvaluator(k=1)
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="Berlin")],
            [Document(content="Paris")],
            [Document(content="Rome")],
        ],
        retrieved_documents=[
            [Document(content="Berlin")],   # hit
            [Document(content="London")],   # miss
            [Document(content="Rome")],     # hit
        ],
    )
    assert result["individual_scores"] == [1.0, 0.0, 1.0]
    assert result["score"] == pytest.approx(2 / 3)


def test_precision_multiple_ground_truth_docs():
    """Any retrieved doc matching one of the ground truth docs counts as relevant."""
    evaluator = DocumentPrecisionEvaluator(k=1)
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="9th century"), Document(content="9th")],
        ],
        retrieved_documents=[[Document(content="9th")]],
    )
    assert result == {"individual_scores": [1.0], "score": 1.0}


def test_precision_empty_retrieved_docs():
    """Empty retrieved list → score 0.0, no error."""
    evaluator = DocumentPrecisionEvaluator(k=1)
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")]],
        retrieved_documents=[[]],
    )
    assert result == {"individual_scores": [0.0], "score": 0.0}


def test_precision_fewer_docs_than_k():
    """When retrieved has fewer than k docs, denominator is still k."""
    evaluator = DocumentPrecisionEvaluator(k=3)
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")]],
        # Only 1 doc, but k=3; Berlin is relevant → Precision@3 = 1/3
        retrieved_documents=[[Document(content="Berlin")]],
    )
    assert result["individual_scores"] == [pytest.approx(1 / 3)]
    assert result["score"] == pytest.approx(1 / 3)


def test_precision_mismatched_list_lengths():
    evaluator = DocumentPrecisionEvaluator(k=1)
    with pytest.raises(ValueError, match="must be the same"):
        evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")]],
            retrieved_documents=[
                [Document(content="Berlin")],
                [Document(content="Paris")],
            ],
        )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_to_dict():
    evaluator = DocumentPrecisionEvaluator(k=3)
    d = evaluator.to_dict()
    assert d["type"] == "haystack.components.evaluators.document_precision.DocumentPrecisionEvaluator"
    assert d["init_parameters"]["k"] == 3


def test_from_dict():
    data = {
        "type": "haystack.components.evaluators.document_precision.DocumentPrecisionEvaluator",
        "init_parameters": {"k": 5},
    }
    evaluator = DocumentPrecisionEvaluator.from_dict(data)
    assert isinstance(evaluator, DocumentPrecisionEvaluator)
    assert evaluator.k == 5


def test_round_trip_serialisation():
    original = DocumentPrecisionEvaluator(k=7)
    restored = DocumentPrecisionEvaluator.from_dict(original.to_dict())
    assert restored.k == original.k


# ---------------------------------------------------------------------------
# Package export
# ---------------------------------------------------------------------------


def test_export_from_package():
    from haystack.components.evaluators import DocumentPrecisionEvaluator as DPE

    assert DPE is DocumentPrecisionEvaluator


# ---------------------------------------------------------------------------
# Regression tests — DAT vs. fixed hybrid (Scenarios A and B from I-04)
# ---------------------------------------------------------------------------


class TestRegressionDATvsFixedAlpha:
    """
    Deterministic regression tests that validate the core claim of the DAT paper:
    per-query dynamic alpha achieves Precision@1 ≥ static alpha=0.5 hybrid.

    Both scenarios use hand-crafted document scores and a mocked LLM — no real
    embedding model or LLM is required.
    """

    @pytest.fixture()
    def joiner(self):
        from haystack.components.joiners.dat_document_joiner import DATDocumentJoiner

        mock_gen = MagicMock()
        mock_gen.run.return_value = _make_chat_reply("3 4")  # default; overridden per test
        return DATDocumentJoiner(chat_generator=mock_gen, top_k=3)

    def test_regression_dat_vs_fixed_alpha_dense_wins(self, joiner):
        """
        Scenario A — Dense retrieval is better.

        Design:
        - Correct answer has high dense score (1.0) but moderate BM25 score.
        - Decoy has moderate dense score (0.7) but very high BM25 score.
        - After DAT min-max normalization: dense[correct]=1.0, dense[decoy]=0.7;
          BM25[decoy]=1.0, BM25[correct]=0.1.

        Fixed hybrid (α=0.5):
          decoy  = 0.5*0.7 + 0.5*1.0 = 0.85  (ranked 1st → wrong)
          correct = 0.5*1.0 + 0.5*0.1 = 0.55
          → Precision@1 = 0.0

        DAT (α=1.0, LLM returns "5 0"):
          correct = 1.0*1.0 + 0.0*0.1 = 1.0  (ranked 1st → correct)
          → Precision@1 = 1.0
        """
        from haystack.components.evaluators.document_precision import DocumentPrecisionEvaluator

        correct_gt = Document(content="Paris")

        dense_docs = [
            Document(content="Paris", id="correct", score=1.0),
            Document(content="France capital France capital", id="decoy", score=0.7),
            Document(content="filler", id="noise", score=0.0),
        ]
        bm25_docs = [
            Document(content="France capital France capital", id="decoy", score=10.0),
            Document(content="Paris", id="correct", score=1.0),
            Document(content="filler", id="noise", score=0.0),
        ]

        # DAT run: dense wins
        joiner.chat_generator.run.return_value = _make_chat_reply("5 0")
        dat_result = joiner.run(query="capital of France", dense_documents=dense_docs, bm25_documents=bm25_docs)

        # Fixed hybrid run: α = 0.5 (LLM returns "0 0")
        joiner.chat_generator.run.return_value = _make_chat_reply("0 0")
        fixed_result = joiner.run(query="capital of France", dense_documents=dense_docs, bm25_documents=bm25_docs)

        evaluator = DocumentPrecisionEvaluator(k=1)
        dat_p1 = evaluator.run(
            ground_truth_documents=[[correct_gt]],
            retrieved_documents=[dat_result["documents"]],
        )["score"]
        fixed_p1 = evaluator.run(
            ground_truth_documents=[[correct_gt]],
            retrieved_documents=[fixed_result["documents"]],
        )["score"]

        assert dat_result["alpha"] == 1.0
        assert dat_p1 == 1.0
        assert dat_p1 >= fixed_p1

    def test_regression_dat_vs_fixed_alpha_bm25_wins(self, joiner):
        """
        Scenario B — BM25 retrieval is better.

        Design:
        - Decoy has very high dense score (1.0); correct has low dense score (0.2).
        - Correct has high BM25 score (3.0); decoy has lower BM25 score (1.0).
        - After DAT min-max normalization: dense[decoy]=1.0, dense[correct]=0.2;
          BM25[correct]=1.0, BM25[decoy]≈0.33.

        Fixed hybrid (α=0.5):
          decoy  = 0.5*1.0 + 0.5*0.33 ≈ 0.667  (ranked 1st → wrong)
          correct = 0.5*0.2 + 0.5*1.0  = 0.600
          → Precision@1 = 0.0

        DAT (α=0.0, LLM returns "0 5"):
          correct = 0.0*0.2 + 1.0*1.0 = 1.0  (ranked 1st → correct)
          → Precision@1 = 1.0
        """
        from haystack.components.evaluators.document_precision import DocumentPrecisionEvaluator

        correct_gt = Document(content="Paris")

        dense_docs = [
            Document(content="France capital France capital", id="decoy", score=1.0),
            Document(content="Paris", id="correct", score=0.2),
            Document(content="filler", id="noise", score=0.0),
        ]
        bm25_docs = [
            Document(content="Paris", id="correct", score=3.0),
            Document(content="France capital France capital", id="decoy", score=1.0),
            Document(content="filler", id="noise", score=0.0),
        ]

        # DAT run: BM25 wins
        joiner.chat_generator.run.return_value = _make_chat_reply("0 5")
        dat_result = joiner.run(query="capital of France", dense_documents=dense_docs, bm25_documents=bm25_docs)

        # Fixed hybrid run: α = 0.5 (LLM returns "0 0")
        joiner.chat_generator.run.return_value = _make_chat_reply("0 0")
        fixed_result = joiner.run(query="capital of France", dense_documents=dense_docs, bm25_documents=bm25_docs)

        evaluator = DocumentPrecisionEvaluator(k=1)
        dat_p1 = evaluator.run(
            ground_truth_documents=[[correct_gt]],
            retrieved_documents=[dat_result["documents"]],
        )["score"]
        fixed_p1 = evaluator.run(
            ground_truth_documents=[[correct_gt]],
            retrieved_documents=[fixed_result["documents"]],
        )["score"]

        assert dat_result["alpha"] == 0.0
        assert dat_p1 == 1.0
        assert dat_p1 >= fixed_p1
