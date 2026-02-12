# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict


@component
class DocumentPrecisionEvaluator:
    """
    Evaluator that calculates Precision@K for retrieved documents.

    Precision@K measures the fraction of the top-K retrieved documents that are
    relevant (i.e., appear in the ground truth set for that query):

    ```
    Precision@K = |{relevant docs} ∩ {top-K retrieved docs}| / K
    ```

    With the default ``k=1``, this computes **Precision@1** — the primary metric used
    in the DAT paper (Hsu & Tzeng, 2025, arXiv:2503.23013) to evaluate hybrid retrieval
    effectiveness.

    **Distinction from** :class:`~haystack.components.evaluators.DocumentMAPEvaluator`:
    MAP (Mean Average Precision) averages the precision at every rank where a relevant
    document appears and is therefore sensitive to the ordering of *all* relevant documents
    across the full retrieved list.  Precision@K considers *only* the top-K retrieved
    documents and does not weight by rank position within those K.  At K=1,
    Precision@1 equals Hit Rate@1, which cannot be derived from MAP.

    ``DocumentPrecisionEvaluator`` doesn't normalize its inputs; use ``DocumentCleaner``
    to clean and normalize documents before passing them to this evaluator.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentPrecisionEvaluator

    evaluator = DocumentPrecisionEvaluator(k=1)
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="France")],
            [Document(content="9th century"), Document(content="9th")],
        ],
        retrieved_documents=[
            [Document(content="France")],
            [Document(content="10th century"), Document(content="9th century"), Document(content="9th")],
        ],
    )
    print(result["individual_scores"])
    # [1.0, 0.0]
    print(result["score"])
    # 0.5
    ```
    """

    def __init__(self, k: int = 1):
        """
        Create a DocumentPrecisionEvaluator component.

        :param k:
            Number of top retrieved documents to consider when computing precision.
            Must be > 0.  Defaults to ``1``, which directly matches the Precision@1
            metric used as the primary evaluation criterion in the DAT paper.
        :raises ValueError:
            If ``k`` is not a positive integer.
        """
        if k <= 0:
            raise ValueError(f"k must be greater than 0, got {k}")
        self.k = k

    @component.output_types(score=float, individual_scores=list[float])
    def run(
        self,
        ground_truth_documents: list[list[Document]],
        retrieved_documents: list[list[Document]],
    ) -> dict[str, Any]:
        """
        Run the DocumentPrecisionEvaluator on the given inputs.

        ``ground_truth_documents`` and ``retrieved_documents`` must have the same
        length (one inner list per query).

        :param ground_truth_documents:
            A list of expected documents for each query.
        :param retrieved_documents:
            A list of retrieved documents for each query.  Only the first ``k``
            documents in each inner list are evaluated; documents beyond rank ``k``
            are ignored.
        :returns:
            A dictionary with the following outputs:

            - ``score``: Mean Precision@K averaged over all queries.
            - ``individual_scores``: Per-query Precision@K values in [0.0, 1.0].
        :raises ValueError:
            If ``ground_truth_documents`` and ``retrieved_documents`` have different
            lengths.
        """
        if len(ground_truth_documents) != len(retrieved_documents):
            msg = "The length of ground_truth_documents and retrieved_documents must be the same."
            raise ValueError(msg)

        individual_scores = []

        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            top_k = retrieved[: self.k]
            if not top_k:
                individual_scores.append(0.0)
                continue

            ground_truth_contents = {doc.content for doc in ground_truth if doc.content is not None}
            relevant = sum(
                1 for doc in top_k if doc.content is not None and doc.content in ground_truth_contents
            )
            individual_scores.append(relevant / self.k)

        score = sum(individual_scores) / len(ground_truth_documents)
        return {"score": score, "individual_scores": individual_scores}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the component to a dictionary.

        :returns:
            Dictionary with serialised data.
        """
        return default_to_dict(self, k=self.k)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentPrecisionEvaluator":
        """
        Deserialise a :class:`DocumentPrecisionEvaluator` from a dictionary.

        :param data:
            The serialised component dictionary (as produced by :meth:`to_dict`).
        :returns:
            A new :class:`DocumentPrecisionEvaluator` instance.
        """
        return default_from_dict(cls, data)
