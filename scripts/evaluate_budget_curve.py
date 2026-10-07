# Support both `python -m scripts.NAME` and `python scripts/NAME.py`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from adaptive_reranking.utils.io import load_json, write_json
from adaptive_reranking.evaluation.validation import K_VALUES, query_ndcg_at_k

CONFIG = load_json("configs/budget_curve.json")
CANDIDATES = joblib.load("results/cache/method1/candidates.joblib")
TEACHER_SCORES = load_json("results/cache/method1/teacher_scores_checkpoint.json")
SPLIT = load_json("results/manifests/method1/split.json")
OUTPUT = Path("results/metrics/method4/budget_curve_metrics.json")
CSV_OUTPUT = Path("results/metrics/method4/budget_curve.csv")


def per_query_recall_at_k(qrels, results, k=10):
    scores = {}
    for query_id, relevant_docs in qrels.items():
        ranked = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:k]
        retrieved = sum(relevant_docs.get(doc_id, 0) > 0 for doc_id, _ in ranked)
        total = sum(relevance > 0 for relevance in relevant_docs.values())
        scores[query_id] = retrieved / total if total else 0.0
    return scores


def paired_bootstrap(left, right, query_ids):
    differences = np.asarray([left[query_id] - right[query_id] for query_id in query_ids])
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    indices = rng.integers(
        0,
        len(differences),
        size=(CONFIG["bootstrap_samples"], len(differences)),
    )
    sample_means = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci_95_percentile": [
            float(np.quantile(sample_means, 0.025)),
            float(np.quantile(sample_means, 0.975)),
        ],
        "improved_queries": int(np.sum(differences > 1e-12)),
        "unchanged_queries": int(np.sum(np.abs(differences) <= 1e-12)),
        "worsened_queries": int(np.sum(differences < -1e-12)),
        "bootstrap_samples": CONFIG["bootstrap_samples"],
    }


def rerank_at_budget(query_ids, budget):
    results = {}
    for query_id in query_ids:
        retrieval_order = [
            doc_id
            for doc_id, _ in sorted(
                CANDIDATES[query_id].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        scored = retrieval_order[:budget]
        reranked = sorted(
            scored,
            key=lambda doc_id: (-TEACHER_SCORES[query_id][doc_id], doc_id),
        )
        final_order = reranked + retrieval_order[budget:]
        count = len(final_order)
        results[query_id] = {
            doc_id: float(count - rank)
            for rank, doc_id in enumerate(final_order)
        }
    return results


_, _, all_qrels = GenericDataLoader("results/datasets/scifact").load(split="train")
validation_ids = sorted(SPLIT["validation_query_ids"])
qrels = {query_id: all_qrels[query_id] for query_id in validation_ids}
if set(validation_ids) - set(CANDIDATES):
    raise RuntimeError("Candidate cache is missing frozen validation queries")
if set(validation_ids) - set(TEACHER_SCORES):
    raise RuntimeError("Teacher cache is missing frozen validation queries")
if any(len(CANDIDATES[query_id]) != CONFIG["candidate_top_k"] for query_id in validation_ids):
    raise RuntimeError("Every validation query must have exactly candidate_top_k candidates")
if any(
    set(CANDIDATES[query_id]) != set(TEACHER_SCORES[query_id])
    for query_id in validation_ids
):
    raise RuntimeError("Teacher scores do not match the frozen candidate pool")

by_budget = {}
per_query = {}
for budget in CONFIG["budgets"]:
    results = rerank_at_budget(validation_ids, budget)
    ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, K_VALUES
    )
    key = str(budget)
    by_budget[key] = {
        "teacher_scores_per_query": budget,
        "relative_teacher_score_budget": budget / CONFIG["candidate_top_k"],
        "ndcg": ndcg,
        "map": map_scores,
        "recall": recall,
        "precision": precision,
    }
    per_query[key] = {
        "ndcg_at_10": query_ndcg_at_k(qrels, results),
        "recall_at_10": per_query_recall_at_k(qrels, results),
    }

reference_key = str(CONFIG["candidate_top_k"])
comparisons = {}
for budget in CONFIG["budgets"]:
    key = str(budget)
    comparisons[f"budget_{budget}_minus_100"] = {
        metric: paired_bootstrap(
            per_query[key][metric],
            per_query[reference_key][metric],
            validation_ids,
        )
        for metric in ("ndcg_at_10", "recall_at_10")
    }

curve = [
    {
        "budget": budget,
        "relative_teacher_score_budget": budget / CONFIG["candidate_top_k"],
        "NDCG@10": by_budget[str(budget)]["ndcg"]["NDCG@10"],
        "Recall@10": by_budget[str(budget)]["recall"]["Recall@10"],
        "NDCG@10_fraction_of_100": (
            by_budget[str(budget)]["ndcg"]["NDCG@10"]
            / by_budget[reference_key]["ndcg"]["NDCG@10"]
        ),
        "Recall@10_fraction_of_100": (
            by_budget[str(budget)]["recall"]["Recall@10"]
            / by_budget[reference_key]["recall"]["Recall@10"]
        ),
        "NDCG@10_difference_from_100": comparisons[f"budget_{budget}_minus_100"][
            "ndcg_at_10"
        ]["mean_difference"],
        "Recall@10_difference_from_100": comparisons[f"budget_{budget}_minus_100"][
            "recall_at_10"
        ]["mean_difference"],
    }
    for budget in CONFIG["budgets"]
]

# Diagnostic only: qrels choose a budget after observing per-query NDCG.
# This upper bound measures adaptive headroom and is not an implementable policy.
oracle_budget_by_query = {}
smallest_budget_matching_100 = {}
for query_id in validation_ids:
    best_ndcg = max(
        per_query[str(budget)]["ndcg_at_10"][query_id]
        for budget in CONFIG["budgets"]
    )
    oracle_budget_by_query[query_id] = next(
        budget
        for budget in CONFIG["budgets"]
        if per_query[str(budget)]["ndcg_at_10"][query_id] >= best_ndcg - 1e-12
    )
    target = per_query[reference_key]["ndcg_at_10"][query_id]
    smallest_budget_matching_100[query_id] = next(
        budget
        for budget in CONFIG["budgets"]
        if per_query[str(budget)]["ndcg_at_10"][query_id] >= target - 1e-12
    )

oracle_ndcg = {
    query_id: per_query[str(oracle_budget_by_query[query_id])]["ndcg_at_10"][query_id]
    for query_id in validation_ids
}
oracle_recall = {
    query_id: per_query[str(oracle_budget_by_query[query_id])]["recall_at_10"][query_id]
    for query_id in validation_ids
}
adaptive_oracle = {
    "warning": "Diagnostic only: qrels select the best budget per validation query.",
    "tie_break": "smallest budget among equal best per-query NDCG@10 values",
    "mean_budget": float(np.mean(list(oracle_budget_by_query.values()))),
    "budget_distribution": {
        str(budget): count
        for budget, count in sorted(Counter(oracle_budget_by_query.values()).items())
    },
    "mean_NDCG@10": float(np.mean(list(oracle_ndcg.values()))),
    "mean_Recall@10": float(np.mean(list(oracle_recall.values()))),
    "smallest_budget_matching_or_exceeding_100": {
        "mean_budget": float(np.mean(list(smallest_budget_matching_100.values()))),
        "budget_distribution": {
            str(budget): count
            for budget, count in sorted(Counter(smallest_budget_matching_100.values()).items())
        },
    },
}

metrics = {
    "experiment": "fixed reranking budget-quality curve from cached teacher scores",
    "simulation": "quality only; no online latency or speedup claim",
    "config": CONFIG,
    "counts": {
        "validation_queries": len(validation_ids),
        "candidates_per_query": CONFIG["candidate_top_k"],
    },
    "curve": curve,
    "metrics_by_budget": by_budget,
    "paired_comparisons_to_budget_100": comparisons,
    "adaptive_budget_oracle": adaptive_oracle,
    "per_query": per_query,
}
write_json(OUTPUT, metrics)
CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=curve[0].keys())
    writer.writeheader()
    writer.writerows(curve)
print(json.dumps(curve, indent=2))
