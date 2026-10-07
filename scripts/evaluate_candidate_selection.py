# Support both `python -m scripts.NAME` and `python scripts/NAME.py`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
from collections import defaultdict
import math

import joblib
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from adaptive_reranking.reranking.features import feature_matrix
from adaptive_reranking.utils.io import load_json, write_json

CONFIG = load_json("configs/teacher.json")
DATA = joblib.load("results/cache/method1/teacher_dataset.joblib")
BUNDLE = joblib.load("results/cache/method1/score_predictor.joblib")
ROWS = DATA["rows"]
BUDGET = CONFIG["selection_budget"]
OUTPUT_TOP_K = CONFIG["output_top_k"]
SPLIT = load_json("results/manifests/method1/split.json")

_, _, all_qrels = GenericDataLoader("results/datasets/scifact").load(split="train")
rows_by_query = defaultdict(list)
for row in ROWS:
    rows_by_query[row["query_id"]].append(row)


def per_query_quality(query_ids, results, k=10):
    ndcg, recall = {}, {}
    for query_id in query_ids:
        relevant = all_qrels[query_id]
        ranked = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:k]
        gains = [relevant.get(doc_id, 0) for doc_id, _ in ranked]
        ideal = sorted(relevant.values(), reverse=True)[:k]
        dcg = sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(gains))
        idcg = sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal))
        ndcg[query_id] = dcg / idcg if idcg else 0.0
        recall[query_id] = sum(rel > 0 for rel in gains) / sum(rel > 0 for rel in relevant.values())
    return {"ndcg_at_10": ndcg, "recall_at_10": recall}


def paired_bootstrap(left, right, query_ids, seed, samples=10_000):
    differences = np.asarray([left[query_id] - right[query_id] for query_id in query_ids])
    rng = np.random.default_rng(seed)
    sample_means = np.empty(samples)
    for index in range(samples):
        sample_means[index] = differences[rng.integers(0, len(differences), len(differences))].mean()
    return {
        "mean_difference": float(differences.mean()),
        "ci_95_percentile": [
            float(np.quantile(sample_means, 0.025)),
            float(np.quantile(sample_means, 0.975)),
        ],
        "improved_queries": int(np.sum(differences > 1e-12)),
        "unchanged_queries": int(np.sum(np.abs(differences) <= 1e-12)),
        "worsened_queries": int(np.sum(differences < -1e-12)),
        "bootstrap_samples": samples,
    }

start = time.perf_counter()
features = feature_matrix(ROWS)
ensemble_predictions = np.vstack([model.predict(features) for model in BUNDLE["models"]])
means = ensemble_predictions.mean(axis=0)
uncertainties = ensemble_predictions.std(axis=0)
prediction_seconds = time.perf_counter() - start
for row, mean, uncertainty in zip(ROWS, means, uncertainties):
    row["prediction_mean"] = float(mean)
    row["prediction_uncertainty"] = float(uncertainty)


def teacher_results(query_ids, policy: str, uncertainty_lambda: float = 0.0):
    results, selected_ids = {}, {}
    for query_id in query_ids:
        query_rows = rows_by_query[query_id]
        if policy == "fixed_top20":
            selected = sorted(query_rows, key=lambda row: row["retrieval_rank"])[:BUDGET]
        elif policy == "predicted_top20":
            selected = sorted(query_rows, key=lambda row: row["prediction_mean"], reverse=True)[:BUDGET]
        elif policy == "uncertainty_top20":
            selected = sorted(
                query_rows,
                key=lambda row: row["prediction_mean"]
                + uncertainty_lambda * row["prediction_uncertainty"],
                reverse=True,
            )[:BUDGET]
        elif policy == "random_top20":
            rng = np.random.default_rng(CONFIG["split_seed"] + int(query_id))
            indexes = rng.choice(len(query_rows), size=BUDGET, replace=False)
            selected = [query_rows[index] for index in indexes]
        elif policy == "teacher_top100":
            selected = query_rows
        else:
            raise ValueError(policy)
        selected_ids[query_id] = {row["doc_id"] for row in selected}
        results[query_id] = {row["doc_id"]: row["teacher_score"] for row in selected}
    return results, selected_ids


def evaluate(query_ids, policy: str, uncertainty_lambda: float = 0.0):
    results, selected = teacher_results(query_ids, policy, uncertainty_lambda)
    qrels = {query_id: all_qrels[query_id] for query_id in query_ids}
    ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [1, 3, 5, 10]
    )
    teacher_top10_coverage = []
    for query_id in query_ids:
        teacher_top10 = {
            row["doc_id"]
            for row in sorted(
                rows_by_query[query_id], key=lambda row: row["teacher_score"], reverse=True
            )[:OUTPUT_TOP_K]
        }
        teacher_top10_coverage.append(len(teacher_top10 & selected[query_id]) / OUTPUT_TOP_K)
    return {
        "ndcg": ndcg,
        "map": map_scores,
        "recall": recall,
        "precision": precision,
        "mean_teacher_top10_candidate_coverage": float(np.mean(teacher_top10_coverage)),
        "strong_model_calls_per_query": len(rows_by_query[query_ids[0]])
        if policy == "teacher_top100"
        else BUDGET,
    }, selected, per_query_quality(query_ids, results)


calibration_ids = sorted(
    query_id for query_id, rows in rows_by_query.items() if rows[0]["split"] == "calibration"
)
validation_ids = sorted(
    query_id for query_id, rows in rows_by_query.items() if rows[0]["split"] == "validation"
)
if calibration_ids != sorted(SPLIT["calibration_query_ids"]):
    raise RuntimeError("Calibration rows do not match the frozen calibration split")
if validation_ids != sorted(SPLIT["validation_query_ids"]):
    raise RuntimeError("Validation rows do not match the frozen validation split")
if set(BUNDLE["fit_query_ids"]) != set(SPLIT["train_query_ids"]):
    raise RuntimeError("Predictor was not fit exclusively on the frozen training split")
lambda_scores = {}
for value in CONFIG["lambda_grid"]:
    metrics, _, _ = evaluate(calibration_ids, "uncertainty_top20", value)
    lambda_scores[str(value)] = metrics["ndcg"]["NDCG@10"]
best_lambda = max(CONFIG["lambda_grid"], key=lambda value: (lambda_scores[str(value)], -value))

validation_metrics = {}
validation_selections = {}
validation_per_query = {}
for policy in (
    "fixed_top20",
    "predicted_top20",
    "uncertainty_top20",
    "random_top20",
    "teacher_top100",
):
    metrics, selected, per_query = evaluate(validation_ids, policy, best_lambda)
    validation_metrics[policy] = metrics
    validation_selections[policy] = selected
    validation_per_query[policy] = per_query

comparisons = {}
for left, right in (
    ("predicted_top20", "fixed_top20"),
    ("uncertainty_top20", "predicted_top20"),
    ("uncertainty_top20", "fixed_top20"),
    ("uncertainty_top20", "teacher_top100"),
):
    comparison_name = f"{left}_minus_{right}"
    comparisons[comparison_name] = {
        metric: paired_bootstrap(
            validation_per_query[left][metric],
            validation_per_query[right][metric],
            validation_ids,
            CONFIG["split_seed"],
        )
        for metric in ("ndcg_at_10", "recall_at_10")
    }

changed_candidates = np.mean(
    [
        len(
            validation_selections["uncertainty_top20"][query_id]
            - validation_selections["predicted_top20"][query_id]
        )
        for query_id in validation_ids
    ]
)
write_json(
    "results/metrics/method1/candidate_selection_metrics.json",
    {
        "simulation": "cached teacher scores; no online teacher calls were timed",
        "config": CONFIG,
        "calibration": {
            "query_count": len(calibration_ids),
            "lambda_ndcg_at_10": lambda_scores,
            "selected_lambda": best_lambda,
        },
        "validation": {
            "query_count": len(validation_ids),
            "methods": validation_metrics,
            "paired_comparisons": comparisons,
            "mean_candidates_changed_by_uncertainty_vs_prediction": float(changed_candidates),
        },
        "cached_simulation_timings_seconds": {
            "ensemble_prediction_all_candidates": prediction_seconds,
        },
    },
)
print(f"Selected lambda={best_lambda}; wrote results/metrics/method1/candidate_selection_metrics.json")
