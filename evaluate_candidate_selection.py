import time
from collections import defaultdict

import joblib
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from adaptive_reranking.features import feature_matrix
from adaptive_reranking.io import load_json, write_json

CONFIG = load_json("configs/teacher.json")
DATA = joblib.load("artifacts/method1/teacher_dataset.joblib")
BUNDLE = joblib.load("artifacts/method1/score_predictor.joblib")
ROWS = DATA["rows"]
BUDGET = CONFIG["selection_budget"]
OUTPUT_TOP_K = CONFIG["output_top_k"]

_, _, all_qrels = GenericDataLoader("datasets/scifact").load(split="train")
rows_by_query = defaultdict(list)
for row in ROWS:
    rows_by_query[row["query_id"]].append(row)

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
    }, selected


calibration_ids = sorted(
    query_id for query_id, rows in rows_by_query.items() if rows[0]["split"] == "calibration"
)
validation_ids = sorted(
    query_id for query_id, rows in rows_by_query.items() if rows[0]["split"] == "validation"
)
lambda_scores = {}
for value in CONFIG["lambda_grid"]:
    metrics, _ = evaluate(calibration_ids, "uncertainty_top20", value)
    lambda_scores[str(value)] = metrics["ndcg"]["NDCG@10"]
best_lambda = max(CONFIG["lambda_grid"], key=lambda value: (lambda_scores[str(value)], -value))

validation_metrics = {}
validation_selections = {}
for policy in (
    "fixed_top20",
    "predicted_top20",
    "uncertainty_top20",
    "random_top20",
    "teacher_top100",
):
    metrics, selected = evaluate(validation_ids, policy, best_lambda)
    validation_metrics[policy] = metrics
    validation_selections[policy] = selected

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
    "results/method1/candidate_selection_metrics.json",
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
            "mean_candidates_changed_by_uncertainty_vs_prediction": float(changed_candidates),
        },
        "cached_simulation_timings_seconds": {
            "ensemble_prediction_all_candidates": prediction_seconds,
        },
    },
)
print(f"Selected lambda={best_lambda}; wrote results/method1/candidate_selection_metrics.json")
