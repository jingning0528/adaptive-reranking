# Support both `python -m scripts.NAME` and `python scripts/NAME.py`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from adaptive_reranking.reranking.features import tokens
from adaptive_reranking.utils.io import load_json, write_json

BUDGET_RESULTS = load_json("results/metrics/method4/budget_curve_metrics.json")
CONFIG = BUDGET_RESULTS["config"]
BUDGETS = CONFIG["budgets"]
PER_QUERY = BUDGET_RESULTS["per_query"]
CANDIDATES = joblib.load("results/cache/method1/candidates.joblib")
SPLIT = load_json("results/manifests/method1/split.json")
OUTPUT = Path("results/metrics/method5")
OUTPUT.mkdir(parents=True, exist_ok=True)


def normalized_entropy(values):
    scores = np.asarray(values, dtype=float)
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-15))
    return float(entropy / math.log(len(scores)))


def mean_pairwise_jaccard_distance(token_sets):
    similarities = []
    for left, right in combinations(token_sets, 2):
        union = left | right
        similarities.append(len(left & right) / len(union) if union else 1.0)
    return float(1.0 - np.mean(similarities)) if similarities else 0.0


def query_signals(query, ranked_candidates, corpus):
    scores = np.asarray([score for _, score in ranked_candidates], dtype=float)
    query_tokens = set(tokens(query))
    top20_sets = []
    overlaps = []
    for doc_id, _ in ranked_candidates[:20]:
        document = corpus[doc_id]
        document_tokens = set(
            tokens((document.get("title", "") + " " + document.get("text", "")).strip())
        )
        top20_sets.append(document_tokens)
        overlaps.append(
            len(query_tokens & document_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
    return {
        "query_token_count": len(query_tokens),
        "retrieval_margin_1_2": float(scores[0] - scores[1]),
        "retrieval_margin_1_10": float(scores[0] - scores[9]),
        "retrieval_score_std_top10": float(np.std(scores[:10])),
        "retrieval_score_std_top100": float(np.std(scores)),
        "retrieval_score_entropy_top100": normalized_entropy(scores),
        "query_overlap_mean_top20": float(np.mean(overlaps)),
        "query_overlap_max_top20": float(np.max(overlaps)),
        "candidate_lexical_diversity_top20": mean_pairwise_jaccard_distance(top20_sets),
    }


def signal_relationship(values, oracle_budgets, seed):
    signal = np.asarray(values, dtype=float)
    budgets = np.asarray(oracle_budgets, dtype=float)
    needs_more_than_5 = budgets > 5
    correlation = spearmanr(signal, budgets)
    raw_auc = roc_auc_score(needs_more_than_5, signal)
    positive = signal[needs_more_than_5]
    negative = signal[~needs_more_than_5]

    def auc_from_groups(positive_values, negative_values):
        comparisons = positive_values[:, None] - negative_values[None, :]
        return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))

    rng = np.random.default_rng(seed)
    bootstrap_auc = np.empty(10_000)
    for index in range(len(bootstrap_auc)):
        sampled_positive = positive[rng.integers(0, len(positive), len(positive))]
        sampled_negative = negative[rng.integers(0, len(negative), len(negative))]
        value = auc_from_groups(sampled_positive, sampled_negative)
        bootstrap_auc[index] = value if raw_auc >= 0.5 else 1.0 - value
    return {
        "spearman_rho_with_oracle_k": float(correlation.statistic),
        "spearman_p_value": float(correlation.pvalue),
        "mean_when_oracle_k_5": float(signal[~needs_more_than_5].mean()),
        "mean_when_oracle_k_gt_5": float(signal[needs_more_than_5].mean()),
        "raw_auc_for_oracle_k_gt_5": float(raw_auc),
        "best_direction_auc_for_oracle_k_gt_5": float(max(raw_auc, 1.0 - raw_auc)),
        "best_direction_auc_ci_95_percentile": [
            float(np.quantile(bootstrap_auc, 0.025)),
            float(np.quantile(bootstrap_auc, 0.975)),
        ],
        "higher_values_indicate_k_gt_5": bool(raw_auc >= 0.5),
    }


def write_histogram_svg(distribution, destination):
    width, height = 760, 440
    left, right, top, bottom = 80, 30, 45, 75
    chart_width = width - left - right
    chart_height = height - top - bottom
    maximum = max(50, math.ceil(max(distribution.values()) / 50) * 50)
    bar_slot = chart_width / len(BUDGETS)
    bar_width = bar_slot * 0.58
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="25" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">Oracle budget distribution</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333"/>',
    ]
    for tick in range(0, maximum + 1, 50):
        y = top + chart_height - (tick / maximum) * chart_height
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#ddd"/>',
                f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick}</text>',
            ]
        )
    for index, budget in enumerate(BUDGETS):
        count = distribution.get(str(budget), 0)
        x = left + index * bar_slot + (bar_slot - bar_width) / 2
        bar_height = (count / maximum) * chart_height
        y = top + chart_height - bar_height
        elements.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13">{count}</text>',
                f'<text x="{x + bar_width / 2:.1f}" y="{top + chart_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="13">{budget}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left + chart_width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">Oracle K</text>',
            f'<text x="18" y="{top + chart_height / 2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 18 {top + chart_height / 2})">Queries</text>',
            '</svg>',
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(elements), encoding="utf-8")


corpus, queries, _ = GenericDataLoader("results/datasets/scifact").load(split="train")
validation_ids = sorted(SPLIT["validation_query_ids"])
rows = []
for query_id in validation_ids:
    ndcg_by_budget = {
        budget: PER_QUERY[str(budget)]["ndcg_at_10"][query_id]
        for budget in BUDGETS
    }
    best_ndcg = max(ndcg_by_budget.values())
    best_budgets = [
        budget for budget in BUDGETS if ndcg_by_budget[budget] >= best_ndcg - 1e-12
    ]
    reference_ndcg = ndcg_by_budget[CONFIG["candidate_top_k"]]
    matching_budgets = [
        budget
        for budget in BUDGETS
        if ndcg_by_budget[budget] >= reference_ndcg - 1e-12
    ]
    ranked_candidates = sorted(
        CANDIDATES[query_id].items(), key=lambda item: (-item[1], item[0])
    )
    row = {
        "query_id": query_id,
        "query": queries[query_id],
        "oracle_k": min(best_budgets),
        "best_budget_tie_count": len(best_budgets),
        "smallest_k_matching_or_exceeding_100": min(matching_budgets),
        "oracle_ndcg_at_10": best_ndcg,
        "ndcg_at_100_budget": reference_ndcg,
        **query_signals(queries[query_id], ranked_candidates, corpus),
    }
    for budget in BUDGETS:
        row[f"ndcg_at_10_k_{budget}"] = ndcg_by_budget[budget]
    rows.append(row)

distribution = Counter(row["oracle_k"] for row in rows)
signal_names = (
    "query_token_count",
    "retrieval_margin_1_2",
    "retrieval_margin_1_10",
    "retrieval_score_std_top10",
    "retrieval_score_std_top100",
    "retrieval_score_entropy_top100",
    "query_overlap_mean_top20",
    "query_overlap_max_top20",
    "candidate_lexical_diversity_top20",
)
relationships = {
    signal: signal_relationship(
        [row[signal] for row in rows],
        [row["oracle_k"] for row in rows],
        20261001 + index,
    )
    for index, signal in enumerate(signal_names)
}
ordered_signals = sorted(
    relationships,
    key=lambda name: relationships[name]["best_direction_auc_for_oracle_k_gt_5"],
    reverse=True,
)

analysis = {
    "definition": (
        "oracle_k is the smallest tested K attaining the maximum per-query validation NDCG@10"
    ),
    "warning": (
        "Exploratory qrels-based validation analysis; do not tune a future policy on these labels "
        "and report the same queries as untouched final validation."
    ),
    "counts": {
        "queries": len(rows),
        "oracle_k_5": sum(row["oracle_k"] == 5 for row in rows),
        "oracle_k_gt_5": sum(row["oracle_k"] > 5 for row in rows),
        "all_budgets_tied": sum(row["best_budget_tie_count"] == len(BUDGETS) for row in rows),
        "multiple_best_budgets": sum(row["best_budget_tie_count"] > 1 for row in rows),
    },
    "oracle_budget_distribution": {
        str(budget): distribution.get(budget, 0) for budget in BUDGETS
    },
    "mean_oracle_k": float(np.mean([row["oracle_k"] for row in rows])),
    "signal_relationships": relationships,
    "signals_ranked_by_exploratory_auc": ordered_signals,
}
write_json(OUTPUT / "oracle_budget_analysis.json", analysis)
with (OUTPUT / "oracle_budget_by_query.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
write_histogram_svg(
    analysis["oracle_budget_distribution"],
    Path("results/figures/method5/oracle_budget_histogram.svg"),
)
print(json.dumps(analysis, indent=2))
