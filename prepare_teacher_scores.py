import json
import logging
import time
from pathlib import Path

import joblib
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import CrossEncoder

from adaptive_reranking.features import FEATURE_NAMES, candidate_features
from adaptive_reranking.io import load_json, write_json
from adaptive_reranking.splits import group_sizes, train_calibration_split

logging.basicConfig(level=logging.INFO)

CONFIG = load_json("configs/teacher.json")
DATA_PATH = Path("datasets") / CONFIG["dataset"]
ARTIFACTS = Path("artifacts/method1")
RESULTS = Path("results/method1")
CANDIDATES_PATH = ARTIFACTS / "candidates.joblib"
CHECKPOINT_PATH = ARTIFACTS / "teacher_scores_checkpoint.json"
DATASET_PATH = ARTIFACTS / "teacher_dataset.joblib"

ARTIFACTS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
corpus, queries, qrels = GenericDataLoader(str(DATA_PATH)).load(split=CONFIG["source_split"])
validation_ids = set(load_json("results/validation/split.json")["query_ids"])
remaining_queries = {query_id: text for query_id, text in queries.items() if query_id not in validation_ids}
train_ids, calibration_ids, groups = train_calibration_split(
    remaining_queries,
    seed=CONFIG["split_seed"],
    calibration_fraction=CONFIG["calibration_fraction"],
    threshold=CONFIG["near_duplicate_threshold"],
)
split_by_query = {
    **{query_id: "train" for query_id in train_ids},
    **{query_id: "calibration" for query_id in calibration_ids},
    **{query_id: "validation" for query_id in validation_ids},
}
split_manifest = {
    "seed": CONFIG["split_seed"],
    "near_duplicate_threshold": CONFIG["near_duplicate_threshold"],
    "grouping": "connected components of char 3-5 gram TF-IDF cosine similarity",
    "counts": {
        "train": len(train_ids),
        "calibration": len(calibration_ids),
        "validation": len(validation_ids),
        "near_duplicate_groups": len(set(groups.values())),
        "multi_query_groups": sum(size > 1 for size in group_sizes(groups).values()),
    },
    "train_query_ids": train_ids,
    "calibration_query_ids": calibration_ids,
    "validation_query_ids": sorted(validation_ids),
    "group_by_query_id": groups,
}
write_json(RESULTS / "split.json", split_manifest)

if CANDIDATES_PATH.exists():
    candidates = joblib.load(CANDIDATES_PATH)
else:
    validation_candidates_path = Path("results/validation/baseline_retrieval.json")
    candidates = (
        json.loads(validation_candidates_path.read_text(encoding="utf-8"))
        if validation_candidates_path.exists()
        else {}
    )
    missing_queries = {query_id: text for query_id, text in queries.items() if query_id not in candidates}
    if missing_queries:
        retriever_model = models.SentenceBERT(CONFIG["retriever_model"], device=CONFIG["device"])
        retriever = EvaluateRetrieval(
            DenseRetrievalExactSearch(
                retriever_model,
                batch_size=CONFIG["retriever_batch_size"],
            ),
            score_function="cos_sim",
            k_values=[CONFIG["candidate_top_k"]],
        )
        candidates.update(retriever.retrieve(corpus, missing_queries))
    joblib.dump(candidates, CANDIDATES_PATH, compress=3)

teacher_scores = load_json(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else {}
validation_teacher_path = Path("results/validation/minilm_l6_retrieval.json")
if (
    validation_teacher_path.exists()
    and CONFIG.get("teacher_cache_device", CONFIG["device"]) == CONFIG["device"]
):
    cached_validation = json.loads(validation_teacher_path.read_text(encoding="utf-8"))
    for query_id in validation_ids:
        if query_id in cached_validation and set(cached_validation[query_id]) == set(candidates[query_id]):
            teacher_scores[query_id] = cached_validation[query_id]
    write_json(CHECKPOINT_PATH, teacher_scores)

pending_ids = [query_id for query_id in sorted(queries) if query_id not in teacher_scores]
teacher_load_seconds = 0.0
teacher_score_seconds = 0.0
if pending_ids:
    start = time.perf_counter()
    teacher = CrossEncoder(
        CONFIG["teacher_model"],
        device=CONFIG.get("teacher_cache_device", CONFIG["device"]),
        max_length=CONFIG["max_length"],
    )
    teacher_load_seconds = time.perf_counter() - start
    chunk_size = CONFIG["teacher_query_chunk_size"]
    for offset in range(0, len(pending_ids), chunk_size):
        chunk_ids = pending_ids[offset : offset + chunk_size]
        pairs, pair_ids = [], []
        for query_id in chunk_ids:
            ranked = sorted(candidates[query_id].items(), key=lambda item: item[1], reverse=True)
            for doc_id, _ in ranked[: CONFIG["candidate_top_k"]]:
                document = corpus[doc_id]
                text = (document.get("title", "") + " " + document.get("text", "")).strip()
                pairs.append([queries[query_id], text])
                pair_ids.append((query_id, doc_id))
        start = time.perf_counter()
        scores = teacher.predict(
            pairs,
            batch_size=CONFIG["teacher_batch_size"],
            show_progress_bar=True,
        )
        teacher_score_seconds += time.perf_counter() - start
        for (query_id, doc_id), score in zip(pair_ids, scores):
            teacher_scores.setdefault(query_id, {})[doc_id] = float(score)
        write_json(CHECKPOINT_PATH, teacher_scores)
        logging.info("Teacher checkpoint: %d/%d pending queries", min(offset + chunk_size, len(pending_ids)), len(pending_ids))

rows = []
for query_id in sorted(queries):
    ranked = sorted(candidates[query_id].items(), key=lambda item: item[1], reverse=True)[
        : CONFIG["candidate_top_k"]
    ]
    top_score = ranked[0][1]
    for rank, (doc_id, retrieval_score) in enumerate(ranked, start=1):
        rows.append(
            {
                "query_id": query_id,
                "doc_id": doc_id,
                "split": split_by_query[query_id],
                "retrieval_rank": rank,
                "retrieval_score": retrieval_score,
                "teacher_score": teacher_scores[query_id][doc_id],
                "features": candidate_features(
                    queries[query_id], corpus[doc_id], retrieval_score, rank, top_score
                ),
            }
        )
joblib.dump({"feature_names": FEATURE_NAMES, "rows": rows}, DATASET_PATH, compress=3)
write_json(
    RESULTS / "teacher_manifest.json",
    {
        "config": CONFIG,
        "counts": {
            "queries": len(queries),
            "candidate_rows": len(rows),
            "newly_scored_queries": len(pending_ids),
        },
        "timings_seconds": {
            "teacher_model_load": teacher_load_seconds,
            "teacher_scoring_new_queries": teacher_score_seconds,
        },
        "feature_names": FEATURE_NAMES,
        "artifacts": {
            "candidate_cache": str(CANDIDATES_PATH),
            "teacher_checkpoint": str(CHECKPOINT_PATH),
            "training_dataset": str(DATASET_PATH),
        },
    },
)
print(f"Prepared {len(rows)} candidate rows at {DATASET_PATH}")
