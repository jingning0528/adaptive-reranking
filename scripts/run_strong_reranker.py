# Support both `python -m scripts.NAME` and `python scripts/NAME.py`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from adaptive_reranking.data import load_dataset
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import CrossEncoder

from adaptive_reranking.utils.io import load_json, write_json
from adaptive_reranking.evaluation.validation import K_VALUES, query_ndcg_at_k

logging.basicConfig(level=logging.INFO)

CONFIG = load_json("configs/strong_reranker.json")
CANDIDATES_PATH = Path("results/cache/method1/candidates.joblib")
CHEAP_SCORES_PATH = Path("results/cache/method1/teacher_scores_checkpoint.json")
SPLIT_PATH = Path("results/manifests/method1/split.json")
CHECKPOINT_PATH = Path("results/cache/method2/strong_scores_checkpoint.json")
METADATA_PATH = Path("results/cache/method2/strong_scores_metadata.json")
RESULTS_PATH = Path("results/metrics/method2/strong_reranker_metrics.json")
RETRIEVAL_PATH = Path("results/metrics/method2/strong_reranker_retrieval.json")


def per_query_recall_at_k(qrels, results, k=10):
    scores = {}
    for query_id, relevant_docs in qrels.items():
        ranked = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:k]
        retrieved_relevant = sum(relevant_docs.get(doc_id, 0) > 0 for doc_id, _ in ranked)
        total_relevant = sum(relevance > 0 for relevance in relevant_docs.values())
        scores[query_id] = retrieved_relevant / total_relevant if total_relevant else 0.0
    return scores


def paired_bootstrap(left, right, query_ids, seed=20261001, samples=10_000):
    differences = np.asarray([left[query_id] - right[query_id] for query_id in query_ids])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
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
        "bootstrap_samples": samples,
    }


contract = {
    **CONFIG,
    "candidate_cache": str(CANDIDATES_PATH),
    "validation_split": str(SPLIT_PATH),
}
fingerprint = hashlib.sha256(
    json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
if CHECKPOINT_PATH.exists():
    if not METADATA_PATH.exists():
        raise RuntimeError("Strong-score checkpoint exists without metadata; refusing unsafe resume")
    metadata = load_json(METADATA_PATH)
    if metadata["fingerprint"] != fingerprint:
        raise RuntimeError("Strong-score checkpoint configuration does not match the current contract")
else:
    metadata = {"fingerprint": fingerprint, "contract": contract, "timings_seconds": {}}
    write_json(METADATA_PATH, metadata)

dataset = load_dataset("scifact", split="train", download=False)
corpus = dataset.corpus
all_queries = dataset.queries
all_qrels = dataset.qrels
split = load_json(SPLIT_PATH)
validation_ids = sorted(split["validation_query_ids"])
queries = {query_id: all_queries[query_id] for query_id in validation_ids}
qrels = {query_id: all_qrels[query_id] for query_id in validation_ids}
all_candidates = joblib.load(CANDIDATES_PATH)
candidates = {query_id: all_candidates[query_id] for query_id in validation_ids}
if any(len(scores) != CONFIG["candidate_top_k"] for scores in candidates.values()):
    raise RuntimeError("Validation candidate cache is not exactly candidate_top_k per query")

cheap_all = load_json(CHEAP_SCORES_PATH)
cheap_results = {
    query_id: {doc_id: cheap_all[query_id][doc_id] for doc_id in candidates[query_id]}
    for query_id in validation_ids
}
strong_scores = load_json(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else {}
pending_ids = [query_id for query_id in validation_ids if query_id not in strong_scores]

load_seconds = 0.0
scoring_seconds = 0.0
if pending_ids:
    load_start = time.perf_counter()
    model = CrossEncoder(
        CONFIG["model"],
        revision=CONFIG["model_revision"],
        device=CONFIG["device"],
        max_length=CONFIG["max_length"],
    )
    load_seconds = time.perf_counter() - load_start
    scoring_start = time.perf_counter()
    chunk_size = CONFIG["query_chunk_size"]
    for offset in range(0, len(pending_ids), chunk_size):
        chunk_ids = pending_ids[offset : offset + chunk_size]
        pairs = []
        locations = []
        for query_id in chunk_ids:
            for doc_id in candidates[query_id]:
                document = corpus[doc_id]
                text = (document.get("title", "") + " " + document.get("text", "")).strip()
                pairs.append([queries[query_id], text])
                locations.append((query_id, doc_id))
        scores = model.predict(
            pairs,
            batch_size=CONFIG["batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        for (query_id, doc_id), score in zip(locations, scores, strict=True):
            strong_scores.setdefault(query_id, {})[doc_id] = float(score)
        write_json(CHECKPOINT_PATH, strong_scores)
        logging.info("Strong reranker checkpoint: %d/%d queries", min(offset + chunk_size, len(pending_ids)), len(pending_ids))
    scoring_seconds = time.perf_counter() - scoring_start
    previous_timings = metadata.get("timings_seconds", {})
    metadata["timings_seconds"] = {
        "model_load": previous_timings.get("model_load", 0.0) + load_seconds,
        "strong_scoring": previous_timings.get("strong_scoring", 0.0) + scoring_seconds,
    }
    write_json(METADATA_PATH, metadata)

if set(strong_scores) != set(validation_ids):
    raise RuntimeError("Strong-score cache does not contain exactly the frozen validation queries")
if any(set(strong_scores[query_id]) != set(candidates[query_id]) for query_id in validation_ids):
    raise RuntimeError("Strong-score cache does not match the frozen Top-100 candidates")

strong_results = {query_id: strong_scores[query_id] for query_id in validation_ids}
cheap_ndcg, cheap_map, cheap_recall, cheap_precision = EvaluateRetrieval.evaluate(
    qrels, cheap_results, K_VALUES
)
strong_ndcg, strong_map, strong_recall, strong_precision = EvaluateRetrieval.evaluate(
    qrels, strong_results, K_VALUES
)
cheap_query_ndcg = query_ndcg_at_k(qrels, cheap_results)
strong_query_ndcg = query_ndcg_at_k(qrels, strong_results)
cheap_query_recall = per_query_recall_at_k(qrels, cheap_results)
strong_query_recall = per_query_recall_at_k(qrels, strong_results)

metrics = {
    "experiment": "strong reranker feasibility; no routing",
    "config": CONFIG,
    "cache_fingerprint": fingerprint,
    "counts": {
        "validation_queries": len(validation_ids),
        "candidate_pairs": sum(len(scores) for scores in candidates.values()),
        "newly_scored_queries": len(pending_ids),
    },
    "timings_seconds": {
        "model_load": metadata.get("timings_seconds", {}).get("model_load", 0.0),
        "strong_scoring": metadata.get("timings_seconds", {}).get("strong_scoring", 0.0),
    },
    "cheap_minilm_l6": {
        "ndcg": cheap_ndcg,
        "map": cheap_map,
        "recall": cheap_recall,
        "precision": cheap_precision,
        "per_query_ndcg_at_10": cheap_query_ndcg,
        "per_query_recall_at_10": cheap_query_recall,
    },
    "strong_reranker": {
        "ndcg": strong_ndcg,
        "map": strong_map,
        "recall": strong_recall,
        "precision": strong_precision,
        "per_query_ndcg_at_10": strong_query_ndcg,
        "per_query_recall_at_10": strong_query_recall,
    },
    "strong_minus_cheap": {
        "ndcg_at_10": paired_bootstrap(strong_query_ndcg, cheap_query_ndcg, validation_ids),
        "recall_at_10": paired_bootstrap(strong_query_recall, cheap_query_recall, validation_ids),
    },
}
write_json(RESULTS_PATH, metrics)
write_json(RETRIEVAL_PATH, strong_results)
print(
    json.dumps(
        {
            "cheap_NDCG@10": cheap_ndcg["NDCG@10"],
            "strong_NDCG@10": strong_ndcg["NDCG@10"],
            "cheap_Recall@10": cheap_recall["Recall@10"],
            "strong_Recall@10": strong_recall["Recall@10"],
            "strong_minus_cheap": metrics["strong_minus_cheap"],
        },
        indent=2,
    )
)
