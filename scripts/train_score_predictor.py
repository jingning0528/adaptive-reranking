# Support both `python -m scripts.NAME` and `python scripts/NAME.py`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from adaptive_reranking.reranking.features import feature_matrix
from adaptive_reranking.utils.io import load_json, write_json

CONFIG = load_json("configs/teacher.json")
DATA = joblib.load("results/cache/method1/teacher_dataset.joblib")
ROWS = DATA["rows"]
OUTPUT = Path("results/cache/method1/score_predictor.joblib")
SPLIT = load_json("results/manifests/method1/split.json")

train_rows = [row for row in ROWS if row["split"] == "train"]
calibration_rows = [row for row in ROWS if row["split"] == "calibration"]
validation_rows = [row for row in ROWS if row["split"] == "validation"]
if {row["query_id"] for row in train_rows} != set(SPLIT["train_query_ids"]):
    raise RuntimeError("Training rows do not match the frozen training split")
if {row["query_id"] for row in calibration_rows} != set(SPLIT["calibration_query_ids"]):
    raise RuntimeError("Calibration rows do not match the frozen calibration split")
if {row["query_id"] for row in validation_rows} != set(SPLIT["validation_query_ids"]):
    raise RuntimeError("Validation rows do not match the frozen validation split")
train_by_query = defaultdict(list)
for row in train_rows:
    train_by_query[row["query_id"]].append(row)
train_query_ids = sorted(train_by_query)

models = []
fit_seconds = 0.0
sample_size = round(len(train_query_ids) * CONFIG["ensemble_query_fraction"])
for model_index in range(CONFIG["ensemble_size"]):
    rng = np.random.default_rng(CONFIG["split_seed"] + model_index)
    sampled_ids = rng.choice(train_query_ids, size=sample_size, replace=False)
    sampled_rows = [row for query_id in sampled_ids for row in train_by_query[query_id]]
    features = feature_matrix(sampled_rows)
    targets = np.asarray([row["teacher_score"] for row in sampled_rows], dtype=np.float32)
    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=CONFIG["split_seed"] + model_index,
    )
    start = time.perf_counter()
    model.fit(features, targets)
    fit_seconds += time.perf_counter() - start
    models.append(model)

calibration_features = feature_matrix(calibration_rows)
targets = np.asarray([row["teacher_score"] for row in calibration_rows])
predictions = np.vstack([model.predict(calibration_features) for model in models])
means = predictions.mean(axis=0)
disagreement = predictions.std(axis=0)
absolute_errors = np.abs(means - targets)

rank_errors = {}
for lower, upper in ((1, 10), (11, 20), (21, 50), (51, 100)):
    indexes = [
        index
        for index, row in enumerate(calibration_rows)
        if lower <= row["retrieval_rank"] <= upper
    ]
    rank_errors[f"{lower}-{upper}"] = {
        "count": len(indexes),
        "mae": float(mean_absolute_error(targets[indexes], means[indexes])),
        "rmse": float(mean_squared_error(targets[indexes], means[indexes]) ** 0.5),
    }

bundle = {
    "models": models,
    "feature_names": DATA["feature_names"],
    "config": CONFIG,
    "fit_query_ids": train_query_ids,
}
joblib.dump(bundle, OUTPUT, compress=3)
write_json(
    "results/metrics/method1/predictor_metrics.json",
    {
        "model": "ensemble of HistGradientBoostingRegressor",
        "ensemble_size": len(models),
        "query_sampling_fraction": CONFIG["ensemble_query_fraction"],
        "counts": {
            "training_queries": len(train_query_ids),
            "training_rows": len(train_rows),
            "calibration_rows": len(calibration_rows),
        },
        "fit_seconds_total": fit_seconds,
        "calibration": {
            "rmse": float(mean_squared_error(targets, means) ** 0.5),
            "mae": float(mean_absolute_error(targets, means)),
            "r2": float(r2_score(targets, means)),
            "spearman": float(spearmanr(targets, means).statistic),
            "disagreement_absolute_error_pearson": float(
                pearsonr(disagreement, absolute_errors).statistic
            ),
            "mean_disagreement": float(disagreement.mean()),
            "error_by_retrieval_rank": rank_errors,
        },
        "artifact": str(OUTPUT),
    },
)
print(f"Saved {len(models)} predictors to {OUTPUT}")
