#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
benchmarks/dat_vs_fixed_alpha.py

Benchmark: DAT vs. Fixed-Alpha Hybrid Retrieval

Runs both DAT (Dynamic Alpha Tuning) and fixed-alpha hybrid retrievers on a
30-document world-facts corpus with 15 queries (8 BM25-friendly, 7 semantic).
Measures Precision@1 and MRR@10 across alpha values 0.0 → 1.0 and visualises
the comparison in a 3-panel matplotlib figure.

Usage::

    python benchmarks/dat_vs_fixed_alpha.py
    python benchmarks/dat_vs_fixed_alpha.py --scorer openai
    python benchmarks/dat_vs_fixed_alpha.py --top-k 5
    python benchmarks/dat_vs_fixed_alpha.py --output results.png
    python benchmarks/dat_vs_fixed_alpha.py --model sentence-transformers/all-mpnet-base-v2
"""
from __future__ import annotations

import argparse
import sys
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Dependency guards (friendly messages before the heavier Haystack imports)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit(
        "Missing dependency. Install with:\n"
        "    pip install sentence-transformers matplotlib"
    )

try:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
except ImportError:
    sys.exit(
        "Missing dependency. Install with:\n"
        "    pip install sentence-transformers matplotlib"
    )

import numpy as np

from haystack import Document
from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.components.evaluators.document_precision import DocumentPrecisionEvaluator
from haystack.components.joiners.dat_document_joiner import DATDocumentJoiner
from haystack.document_stores.in_memory import InMemoryDocumentStore

# ---------------------------------------------------------------------------
# Corpus — 30 world-facts documents (15 capital cities + 15 landmarks)
# ---------------------------------------------------------------------------
# Each entry is (doc_id, content).  The mix ensures BM25 and embedding excel
# on different query types so the fixed-alpha curve has a clear peak.
CORPUS_DOCS: list[tuple[str, str]] = [
    # --- Capital cities ---
    (
        "cap_paris",
        "Paris is the capital and largest city of France, situated on the River Seine. "
        "It is home to numerous historic monuments.",
    ),
    (
        "cap_berlin",
        "Berlin is the capital of Germany and its largest city. "
        "Berlin was reunified after the fall of the Wall in 1989.",
    ),
    (
        "cap_rome",
        "Rome is the capital of Italy, known as the Eternal City. "
        "It is the seat of the Catholic Church at Vatican City nearby.",
    ),
    (
        "cap_tokyo",
        "Tokyo is the capital of Japan and the world's most populous metropolitan area, "
        "with approximately 37 million residents.",
    ),
    (
        "cap_beijing",
        "Beijing is the capital of China and a major cultural, political, and economic "
        "center with over 21 million inhabitants.",
    ),
    (
        "cap_london",
        "London is the capital of England and the United Kingdom, located on the River "
        "Thames in southeastern England.",
    ),
    (
        "cap_madrid",
        "Madrid is the capital and most populous city of Spain, situated on the Iberian Peninsula.",
    ),
    (
        "cap_moscow",
        "Moscow is the capital of Russia and its most populous city, located in western "
        "Russia on the Moskva River.",
    ),
    (
        "cap_washington",
        "Washington, D.C. is the capital of the United States of America, established "
        "as the federal capital in 1790.",
    ),
    (
        "cap_canberra",
        "Canberra is the capital of Australia, chosen as a compromise between Sydney "
        "and Melbourne, purpose-built as a national capital.",
    ),
    (
        "cap_cairo",
        "Cairo is the capital of Egypt and the Arab world's largest city, situated "
        "along the Nile River.",
    ),
    (
        "cap_athens",
        "Athens is the capital of Greece, regarded as the cradle of Western civilization "
        "and birthplace of democracy.",
    ),
    (
        "cap_buenosaires",
        "Buenos Aires is the capital of Argentina, nicknamed the 'Paris of South America' "
        "for its European-style boulevards.",
    ),
    (
        "cap_ottawa",
        "Ottawa is the capital of Canada, located in Ontario on the south bank of the "
        "Ottawa River.",
    ),
    (
        "cap_newdelhi",
        "New Delhi is the capital of India, designed by British architects Edwin Lutyens "
        "and Herbert Baker in the early 20th century.",
    ),
    # --- Landmarks ---
    (
        "ldmk_eiffel",
        "The Eiffel Tower was constructed between 1887 and 1889 in Paris as the entrance "
        "arch for the 1889 World's Fair. It stands 330 meters tall.",
    ),
    (
        "ldmk_brandgate",
        "The Brandenburg Gate is an 18th-century neoclassical triumphal arch in Berlin, "
        "one of the most recognizable symbols of Germany.",
    ),
    (
        "ldmk_colosseum",
        "The Colosseum is an ancient elliptical amphitheater in Rome, completed in 80 AD, "
        "with a seating capacity of up to 80,000 spectators.",
    ),
    (
        "ldmk_fuji",
        "Mount Fuji is an active stratovolcano located about 100 km southwest of Tokyo. "
        "At 3,776 meters, it is the highest mountain in Japan.",
    ),
    (
        "ldmk_greatwall",
        "The Great Wall of China is a series of walls built along China's historical "
        "northern borders, spanning over 21,000 km to protect against invasion.",
    ),
    (
        "ldmk_bigben",
        "Big Ben refers to the Great Bell inside the Elizabeth Tower at the Palace of "
        "Westminster in London, standing 96 meters tall.",
    ),
    (
        "ldmk_prado",
        "The Prado Museum in Madrid is one of the world's premier art galleries, housing "
        "masterpieces by Velázquez, Goya, and El Greco.",
    ),
    (
        "ldmk_kremlin",
        "The Kremlin is a fortified complex in central Moscow serving as the official "
        "residence of the President of Russia.",
    ),
    (
        "ldmk_whitehouse",
        "The White House is the official residence and office of the United States "
        "president, located at 1600 Pennsylvania Avenue in Washington, D.C.",
    ),
    (
        "ldmk_operahouse",
        "The Sydney Opera House is an iconic performing arts venue on Sydney Harbour, "
        "completed in 1973, designed by Jorn Utzon.",
    ),
    (
        "ldmk_pyramid",
        "The Great Pyramid of Giza is the oldest of the Seven Wonders of the Ancient "
        "World, located near Cairo, constructed around 2560 BCE.",
    ),
    (
        "ldmk_parthenon",
        "The Parthenon is a classical temple on the Athenian Acropolis in Athens, "
        "dedicated to the goddess Athena, completed in 432 BCE.",
    ),
    (
        "ldmk_iguazu",
        "The Iguazu Falls on the border of Argentina and Brazil is one of the world's "
        "largest and most spectacular waterfall systems.",
    ),
    (
        "ldmk_parliament",
        "The Parliament of Canada is housed in buildings on Parliament Hill in Ottawa, "
        "overlooking the Ottawa River.",
    ),
    (
        "ldmk_tajmahal",
        "The Taj Mahal in Agra, India, is an ivory-white marble mausoleum built in the "
        "17th century by Mughal emperor Shah Jahan for his wife Mumtaz Mahal.",
    ),
]

# Build a content look-up once (used by get_relevant_doc and oracle)
_CONTENT_BY_ID: dict[str, str] = {doc_id: content for doc_id, content in CORPUS_DOCS}

# ---------------------------------------------------------------------------
# Queries — 8 BM25-friendly + 7 embedding-friendly
# ---------------------------------------------------------------------------


class QueryItem(NamedTuple):
    query: str
    relevant_doc_id: str
    friendly: str  # "bm25" or "embedding"


QUERIES: list[QueryItem] = [
    # --- BM25-friendly: multiple exact keywords from the answer doc ---
    QueryItem(
        "Eiffel Tower 1887 1889 World's Fair Paris 330 meters",
        "ldmk_eiffel",
        "bm25",
    ),
    QueryItem(
        "Colosseum elliptical amphitheater Rome 80 AD 80000 spectators capacity",
        "ldmk_colosseum",
        "bm25",
    ),
    QueryItem(
        "Mount Fuji active stratovolcano 3776 meters highest mountain Japan",
        "ldmk_fuji",
        "bm25",
    ),
    QueryItem(
        "Great Wall China northern borders 21000 km invasion fortification",
        "ldmk_greatwall",
        "bm25",
    ),
    QueryItem(
        "Big Ben Great Bell Elizabeth Tower Palace Westminster London 96 meters",
        "ldmk_bigben",
        "bm25",
    ),
    QueryItem(
        "White House official residence president 1600 Pennsylvania Avenue Washington",
        "ldmk_whitehouse",
        "bm25",
    ),
    QueryItem(
        "Parthenon classical temple Athenian Acropolis Athens goddess Athena 432 BCE",
        "ldmk_parthenon",
        "bm25",
    ),
    QueryItem(
        "Kremlin fortified complex central Moscow official residence President Russia",
        "ldmk_kremlin",
        "bm25",
    ),
    # --- Embedding-friendly: semantic paraphrases, keywords differ from answer ---
    QueryItem(
        "principal city governing western Europe's largest country by land area",
        "cap_paris",
        "embedding",
    ),
    QueryItem(
        "densely packed capital of the island nation east of the Korean Peninsula",
        "cap_tokyo",
        "embedding",
    ),
    QueryItem(
        "German-speaking country's biggest metropolis partitioned during the Cold War",
        "cap_berlin",
        "embedding",
    ),
    QueryItem(
        "specially constructed seat of government chosen to avoid rivalry "
        "between two Australian coastal cities",
        "cap_canberra",
        "embedding",
    ),
    QueryItem(
        "South American capital famed for Parisian atmosphere and tree-lined avenues",
        "cap_buenosaires",
        "embedding",
    ),
    QueryItem(
        "monumental triangular burial structure near a Nile delta city, "
        "completed before 2500 BCE",
        "ldmk_pyramid",
        "embedding",
    ),
    QueryItem(
        "gleaming white Indian tomb erected by a 17th-century ruler "
        "to honor his late wife",
        "ldmk_tajmahal",
        "embedding",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_document_store(
    model_name: str,
) -> tuple[InMemoryDocumentStore, SentenceTransformer]:
    """Embed the corpus and write it into a fresh InMemoryDocumentStore."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    contents = [content for _, content in CORPUS_DOCS]
    print(f"Encoding {len(contents)} documents ...")
    embeddings = model.encode(contents, show_progress_bar=True)

    docs = [
        Document(id=doc_id, content=content, embedding=emb.tolist())
        for (doc_id, content), emb in zip(CORPUS_DOCS, embeddings)
    ]

    store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    store.write_documents(docs)
    print(f"Indexed {len(docs)} documents.\n")
    return store, model


def get_relevant_doc(doc_id: str) -> Document:
    """Return a content-only Document suitable for evaluator ground truth."""
    return Document(content=_CONTENT_BY_ID[doc_id])


def oracle_alpha(
    relevant_doc_id: str,
    dense_docs: list[Document],
    bm25_docs: list[Document],
    top_k: int,
) -> float:
    """Per-query oracle: sweep α ∈ {0.0, 0.1, …, 1.0} and pick the value that
    gives the best rank for the ground-truth document.  Ties are broken by
    preferring α closest to 0.5 (equal weighting).  This is the theoretical
    upper bound of DAT when the LLM has perfect information.
    """
    norm_dense = DATDocumentJoiner._normalize_scores(dense_docs)
    norm_bm25 = DATDocumentJoiner._normalize_scores(bm25_docs)

    best_rank: float = float("inf")
    best_alpha = 0.5

    for alpha_tenth in range(0, 11):  # 0, 1, …, 10  → 0.0, 0.1, …, 1.0
        alpha = alpha_tenth / 10.0
        fused = DATDocumentJoiner._fuse(alpha, norm_dense, norm_bm25)
        fused_sorted = sorted(
            fused,
            key=lambda d: d.score if d.score is not None else -float("inf"),
            reverse=True,
        )

        rank: float = top_k + 1  # sentinel: not found
        for i, doc in enumerate(fused_sorted[:top_k]):
            if doc.id == relevant_doc_id:
                rank = i + 1
                break

        if rank < best_rank or (
            rank == best_rank and abs(alpha - 0.5) < abs(best_alpha - 0.5)
        ):
            best_rank = rank
            best_alpha = alpha

    return best_alpha


def fuse_with_alpha(
    alpha: float,
    dense_docs: list[Document],
    bm25_docs: list[Document],
    top_k: int,
) -> list[Document]:
    """Fuse dense + BM25 docs with a fixed alpha; return top_k sorted docs."""
    norm_dense = DATDocumentJoiner._normalize_scores(dense_docs)
    norm_bm25 = DATDocumentJoiner._normalize_scores(bm25_docs)
    fused = DATDocumentJoiner._fuse(alpha, norm_dense, norm_bm25)
    return sorted(
        fused,
        key=lambda d: d.score if d.score is not None else -float("inf"),
        reverse=True,
    )[:top_k]


def compute_metrics(
    retrieved_lists: list[list[Document]],
    ground_truth_lists: list[list[Document]],
) -> tuple[float, float]:
    """Return (Precision@1, MRR@10) over all queries."""
    p1_result = DocumentPrecisionEvaluator(k=1).run(
        ground_truth_documents=ground_truth_lists,
        retrieved_documents=retrieved_lists,
    )
    mrr_result = DocumentMRREvaluator().run(
        ground_truth_documents=ground_truth_lists,
        retrieved_documents=retrieved_lists,
    )
    return p1_result["score"], mrr_result["score"]


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary(
    queries: list[QueryItem],
    dat_alphas: list[float],
    dat_retrieved: list[list[Document]],
    fixed_alphas: np.ndarray,
    fixed_p1: list[float],
    fixed_mrr: list[float],
    dat_p1: float,
    dat_mrr: float,
    scorer_mode: str,
) -> None:
    best_p1_alpha = fixed_alphas[int(np.argmax(fixed_p1))]
    best_p1 = max(fixed_p1)
    best_mrr_alpha = fixed_alphas[int(np.argmax(fixed_mrr))]
    best_mrr = max(fixed_mrr)

    print()
    print("=" * 76)
    print(f" Per-query results  (scorer={scorer_mode})")
    print("=" * 76)
    print(f"  {'Query':<48} {'Type':<10} {'alpha':>6}  {'Hit@1'}")
    print("-" * 76)

    for qi, alpha, retrieved in zip(queries, dat_alphas, dat_retrieved):
        relevant_content = _CONTENT_BY_ID[qi.relevant_doc_id]
        hit = retrieved and retrieved[0].content == relevant_content
        hit_str = "[HIT]" if hit else "     "
        print(f"  {qi.query[:47]:<48} {qi.friendly:<10} {alpha:>6.1f}  {hit_str}")

    print("=" * 76)
    print()
    print(f"  {'Metric':<20} {'Best fixed-alpha':<24} {'DAT (' + scorer_mode + ')'}")
    print("-" * 60)
    print(
        f"  {'Precision@1':<20} {best_p1:.3f}  (alpha={best_p1_alpha:.2f})       {dat_p1:.3f}"
    )
    print(
        f"  {'MRR@10':<20} {best_mrr:.3f}  (alpha={best_mrr_alpha:.2f})       {dat_mrr:.3f}"
    )
    print()
    unique_alphas = sorted(set(round(a, 1) for a in dat_alphas))
    mean_alpha = float(np.mean(dat_alphas))
    print(
        f"  DAT alpha distribution: "
        f"min={min(dat_alphas):.1f}  max={max(dat_alphas):.1f}  "
        f"mean={mean_alpha:.2f}  unique={unique_alphas}"
    )
    if scorer_mode == "oracle":
        print(
            "  Note: Oracle DAT is the per-query upper bound "
            "(sweeps all discrete alpha values)."
        )
        if dat_p1 >= best_p1 - 1e-9:
            print("  [OK] Oracle P@1 >= best fixed-alpha P@1 (as expected).")
        else:
            print(
                f"  [WARN] Oracle P@1 ({dat_p1:.3f}) < best fixed P@1 ({best_p1:.3f}) "
                "— check corpus/query design."
            )
    print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_results(
    fixed_alphas: np.ndarray,
    fixed_p1: list[float],
    fixed_mrr: list[float],
    dat_p1: float,
    dat_mrr: float,
    dat_alphas_per_query: list[float],
    scorer_mode: str,
    output_path: str | None,
) -> None:
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    fig.suptitle(
        f"DAT vs. Fixed-Alpha Hybrid Retrieval  |  scorer={scorer_mode}",
        fontsize=13,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Shared annotation transform: x in data coords, y in axes-fraction [0,1]
    for ax, fixed_vals, dat_val, metric_name in [
        (ax1, fixed_p1, dat_p1, "Precision@1"),
        (ax2, fixed_mrr, dat_mrr, "MRR@10"),
    ]:
        best_idx = int(np.argmax(fixed_vals))

        # Blue curve — fixed-alpha sweep
        ax.plot(
            fixed_alphas,
            fixed_vals,
            "b-o",
            markersize=4,
            linewidth=1.5,
            label="Fixed-α sweep",
        )
        # Red dashed line — DAT
        ax.axhline(
            dat_val,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"DAT ({scorer_mode}) = {dat_val:.3f}",
        )
        # Green triangle — best fixed alpha
        ax.plot(
            fixed_alphas[best_idx],
            fixed_vals[best_idx],
            "g^",
            markersize=10,
            label=f"Best α={fixed_alphas[best_idx]:.2f} ({fixed_vals[best_idx]:.3f})",
        )

        # Vertical reference lines at α = 0, 0.5, 1
        for x_ref in (0.0, 0.5, 1.0):
            ax.axvline(x_ref, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

        # Text annotations slightly above the top edge
        xdata_yaxes = blended_transform_factory(ax.transData, ax.transAxes)
        for x_ref, label in ((0.0, "BM25\nonly"), (0.5, "Equal\nweight"), (1.0, "Dense\nonly")):
            ax.text(
                x_ref,
                1.02,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
                transform=xdata_yaxes,
            )

        ax.set_xlabel("α  (dense weight)")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs. Fixed α")
        ax.legend(fontsize=8, loc="lower center")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

    # Panel 3 — histogram of per-query DAT alphas
    bins = np.linspace(-0.05, 1.05, 13)
    ax3.hist(
        dat_alphas_per_query,
        bins=bins,
        color="steelblue",
        edgecolor="white",
        alpha=0.85,
    )
    mean_alpha = float(np.mean(dat_alphas_per_query))
    ax3.axvline(
        mean_alpha,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean α = {mean_alpha:.2f}",
    )
    ax3.set_xlabel("α per query")
    ax3.set_ylabel("Number of queries")
    ax3.set_title(f"DAT α Distribution  ({scorer_mode})")
    ax3.legend(fontsize=8)
    ax3.set_xlim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DAT vs. fixed-alpha hybrid retrieval on a world-facts corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scorer",
        choices=["oracle", "openai"],
        default="oracle",
        help=(
            "'oracle': per-query sweep over α (theoretical upper bound, no API key). "
            "'openai': real OpenAIChatGenerator(model='gpt-4o-mini') — requires OPENAI_API_KEY."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K docs to retrieve per query.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save the plot to this path (e.g. results.png) instead of displaying it.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model for document and query embeddings.",
    )
    args = parser.parse_args()

    # --- Build document store ---
    store, emb_model = build_document_store(args.model)

    # --- Retrieve BM25 + dense results for every query ---
    print("Retrieving documents for each query ...")
    all_bm25_docs: list[list[Document]] = []
    all_dense_docs: list[list[Document]] = []

    for qi in QUERIES:
        q_emb = emb_model.encode(qi.query).tolist()
        bm25_docs = store.bm25_retrieval(query=qi.query, top_k=args.top_k)
        dense_docs = store.embedding_retrieval(query_embedding=q_emb, top_k=args.top_k)
        all_bm25_docs.append(bm25_docs)
        all_dense_docs.append(dense_docs)

    ground_truth_lists = [[get_relevant_doc(qi.relevant_doc_id)] for qi in QUERIES]

    # --- Fixed-alpha sweep (21 values: 0.00, 0.05, …, 1.00) ---
    fixed_alphas = np.linspace(0.0, 1.0, 21)
    fixed_p1_list: list[float] = []
    fixed_mrr_list: list[float] = []

    print("Running fixed-alpha sweep (21 values) ...")
    for alpha in fixed_alphas:
        retrieved = [
            fuse_with_alpha(alpha, dense, bm25, args.top_k)
            for dense, bm25 in zip(all_dense_docs, all_bm25_docs)
        ]
        p1, mrr = compute_metrics(retrieved, ground_truth_lists)
        fixed_p1_list.append(p1)
        fixed_mrr_list.append(mrr)

    # --- DAT scoring ---
    dat_alphas_per_query: list[float] = []

    if args.scorer == "oracle":
        print("Computing per-query oracle alphas ...")
        for qi, dense_docs, bm25_docs in zip(QUERIES, all_dense_docs, all_bm25_docs):
            alpha = oracle_alpha(qi.relevant_doc_id, dense_docs, bm25_docs, args.top_k)
            dat_alphas_per_query.append(alpha)

    else:  # openai
        import os

        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY environment variable is not set.")
        try:
            from haystack.components.generators.chat import OpenAIChatGenerator
        except ImportError:
            sys.exit("OpenAI generator not found — install with: pip install haystack-ai[openai]")

        print("Scoring with OpenAI gpt-4o-mini ...")
        scorer = DATDocumentJoiner(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            raise_on_failure=False,
        )
        scorer.warm_up()
        for qi, dense_docs, bm25_docs in zip(QUERIES, all_dense_docs, all_bm25_docs):
            result = scorer.run(
                query=qi.query,
                dense_documents=dense_docs,
                bm25_documents=bm25_docs,
                top_k=args.top_k,
            )
            dat_alphas_per_query.append(result["alpha"])

    # --- DAT retrieval and metrics ---
    dat_retrieved_lists = [
        fuse_with_alpha(alpha, dense, bm25, args.top_k)
        for alpha, dense, bm25 in zip(dat_alphas_per_query, all_dense_docs, all_bm25_docs)
    ]
    dat_p1, dat_mrr = compute_metrics(dat_retrieved_lists, ground_truth_lists)

    # --- Console summary ---
    print_summary(
        queries=QUERIES,
        dat_alphas=dat_alphas_per_query,
        dat_retrieved=dat_retrieved_lists,
        fixed_alphas=fixed_alphas,
        fixed_p1=fixed_p1_list,
        fixed_mrr=fixed_mrr_list,
        dat_p1=dat_p1,
        dat_mrr=dat_mrr,
        scorer_mode=args.scorer,
    )

    # --- Plot ---
    plot_results(
        fixed_alphas=fixed_alphas,
        fixed_p1=fixed_p1_list,
        fixed_mrr=fixed_mrr_list,
        dat_p1=dat_p1,
        dat_mrr=dat_mrr,
        dat_alphas_per_query=dat_alphas_per_query,
        scorer_mode=args.scorer,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
