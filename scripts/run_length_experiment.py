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
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

from adaptive_reranking.utils.io import load_json, write_json
from adaptive_reranking.evaluation.validation import K_VALUES, query_ndcg_at_k

logging.basicConfig(level=logging.INFO)

CONFIG = load_json("configs/length_experiment.json")
CANDIDATES_PATH = Path("results/cache/method1/candidates.joblib")
SCORES_256_PATH = Path("results/cache/method1/teacher_scores_checkpoint.json")
SPLIT_PATH = Path("results/manifests/method1/split.json")
CHECKPOINT_PATH = Path("results/cache/method3/minilm_512_scores_checkpoint.json")
METADATA_PATH = Path("results/cache/method3/minilm_512_metadata.json")
METRICS_PATH = Path("results/metrics/method3/length_experiment_metrics.json")
RETRIEVAL_PATH = Path("results/metrics/method3/minilm_512_retrieval.json")


def per_query_recall_at_k(qrels, results, k=10):
    scores = {}
    for query_id, relevant_docs in qrels.items():
        ranked = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:k]
        retrieved = sum(relevant_docs.get(doc_id, 0) > 0 for doc_id, _ in ranked)
        total = sum(relevance > 0 for relevance in relevant_docs.values())
        scores[query_id] = retrieved / total if total else 0.0
    return scores


def paired_summary(left, right, query_ids, seed=20261001, samples=10_000):
    differences = np.asarray([left[query_id] - right[query_id] for query_id in query_ids])
    if len(differences) == 0:
        return {"query_count": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    sample_means = differences[indices].mean(axis=1)
    return {
        "query_count": len(query_ids),
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


def subgroup_contrast(left, right, first_ids, second_ids, seed=20261001, samples=10_000):
    first = np.asarray([left[query_id] - right[query_id] for query_id in first_ids])
    second = np.asarray([left[query_id] - right[query_id] for query_id in second_ids])
    rng = np.random.default_rng(seed)
    sample_differences = np.empty(samples)
    for index in range(samples):
        sample_differences[index] = (
            first[rng.integers(0, len(first), len(first))].mean()
            - second[rng.integers(0, len(second), len(second))].mean()
        )
    return {
        "first_query_count": len(first_ids),
        "second_query_count": len(second_ids),
        "difference_between_mean_changes": float(first.mean() - second.mean()),
        "ci_95_percentile": [
            float(np.quantile(sample_differences, 0.025)),
            float(np.quantile(sample_differences, 0.975)),
        ],
        "bootstrap_samples": samples,
    }


def evaluate(qrels, results):
    ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(qrels, results, K_VALUES)
    return {"ndcg": ndcg, "map": map_scores, "recall": recall, "precision": precision}


contract = {
    **CONFIG,
    "candidate_cache": str(CANDIDATES_PATH),
    "baseline_score_cache": str(SCORES_256_PATH),
    "validation_split": str(SPLIT_PATH),
}
fingerprint = hashlib.sha256(
    json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
if CHECKPOINT_PATH.exists():
    if not METADATA_PATH.exists():
        raise RuntimeError("512-token checkpoint exists without metadata; refusing unsafe resume")
    metadata = load_json(METADATA_PATH)
    if metadata["fingerprint"] != fingerprint:
        raise RuntimeError("512-token checkpoint configuration does not match the current contract")
else:
    metadata = {"fingerprint": fingerprint, "contract": contract, "timings_seconds": {}}
    write_json(METADATA_PATH, metadata)

corpus, all_queries, all_qrels = GenericDataLoader("results/datasets/scifact").load(split="train")
validation_ids = sorted(load_json(SPLIT_PATH)["validation_query_ids"])
queries = {query_id: all_queries[query_id] for query_id in validation_ids}
qrels = {query_id: all_qrels[query_id] for query_id in validation_ids}
all_candidates = joblib.load(CANDIDATES_PATH)
candidates = {query_id: all_candidates[query_id] for query_id in validation_ids}
if any(len(scores) != CONFIG["candidate_top_k"] for scores in candidates.values()):
    raise RuntimeError("Validation candidates are not exactly candidate_top_k per query")

all_scores_256 = load_json(SCORES_256_PATH)
results_256 = {
    query_id: {doc_id: all_scores_256[query_id][doc_id] for doc_id in candidates[query_id]}
    for query_id in validation_ids
}
scores_512 = load_json(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else {}
pending_ids = [query_id for query_id in validation_ids if query_id not in scores_512]
if pending_ids:
    load_start = time.perf_counter()
    model = CrossEncoder(
        CONFIG["model"],
        revision=CONFIG["model_revision"],
        device=CONFIG["device"],
        max_length=CONFIG["experimental_max_length"],
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
        values = model.predict(
            pairs,
            batch_size=CONFIG["batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        for (query_id, doc_id), value in zip(locations, values, strict=True):
            scores_512.setdefault(query_id, {})[doc_id] = float(value)
        write_json(CHECKPOINT_PATH, scores_512)
        logging.info("512-token checkpoint: %d/%d queries", min(offset + chunk_size, len(pending_ids)), len(pending_ids))
    scoring_seconds = time.perf_counter() - scoring_start
    previous = metadata.get("timings_seconds", {})
    metadata["timings_seconds"] = {
        "model_load": previous.get("model_load", 0.0) + load_seconds,
        "scoring": previous.get("scoring", 0.0) + scoring_seconds,
    }
    write_json(METADATA_PATH, metadata)

if set(scores_512) != set(validation_ids):
    raise RuntimeError("512-token cache does not contain exactly the validation queries")
if any(set(scores_512[query_id]) != set(candidates[query_id]) for query_id in validation_ids):
    raise RuntimeError("512-token cache does not match the frozen Top-100 candidates")
results_512 = {query_id: scores_512[query_id] for query_id in validation_ids}

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model"], revision=CONFIG["model_revision"]
)
lengths = {}
for query_id in validation_ids:
    lengths[query_id] = {}
    for doc_id in candidates[query_id]:
        document = corpus[doc_id]
        text = (document.get("title", "") + " " + document.get("text", "")).strip()
        lengths[query_id][doc_id] = len(
            tokenizer(queries[query_id], text, truncation=False)["input_ids"]
        )

ndcg_256 = query_ndcg_at_k(qrels, results_256)
ndcg_512 = query_ndcg_at_k(qrels, results_512)
recall_256 = per_query_recall_at_k(qrels, results_256)
recall_512 = per_query_recall_at_k(qrels, results_512)
query_truncated_fraction = {
    query_id: sum(length > CONFIG["baseline_max_length"] for length in lengths[query_id].values())
    / len(lengths[query_id])
    for query_id in validation_ids
}

groups = {
    "no_truncated_candidates": [
        query_id for query_id in validation_ids if query_truncated_fraction[query_id] == 0
    ],
    "any_truncated_candidate": [
        query_id for query_id in validation_ids if query_truncated_fraction[query_id] > 0
    ],
    "no_truncated_relevant_candidate": [
        query_id
        for query_id in validation_ids
        if not any(
            relevance > 0 and lengths[query_id].get(doc_id, 0) > CONFIG["baseline_max_length"]
            for doc_id, relevance in qrels[query_id].items()
        )
    ],
    "any_truncated_relevant_candidate": [
        query_id
        for query_id in validation_ids
        if any(
            relevance > 0 and lengths[query_id].get(doc_id, 0) > CONFIG["baseline_max_length"]
            for doc_id, relevance in qrels[query_id].items()
        )
    ],
}
for lower, upper in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
    groups[f"truncated_fraction_{lower:.2f}_to_{min(upper, 1.0):.2f}"] = [
        query_id
        for query_id in validation_ids
        if lower <= query_truncated_fraction[query_id] < upper
    ]

rank_256 = {
    query_id: {
        doc_id: rank
        for rank, (doc_id, _) in enumerate(
            sorted(results_256[query_id].items(), key=lambda item: item[1], reverse=True), start=1
        )
    }
    for query_id in validation_ids
}
rank_512 = {
    query_id: {
        doc_id: rank
        for rank, (doc_id, _) in enumerate(
            sorted(results_512[query_id].items(), key=lambda item: item[1], reverse=True), start=1
        )
    }
    for query_id in validation_ids
}
relevant_rank_analysis = {}
for label, is_truncated in (("not_truncated", False), ("truncated_at_256", True)):
    rows = []
    for query_id in validation_ids:
        for doc_id, relevance in qrels[query_id].items():
            if relevance <= 0 or doc_id not in candidates[query_id]:
                continue
            if (lengths[query_id][doc_id] > CONFIG["baseline_max_length"]) == is_truncated:
                rows.append((rank_256[query_id][doc_id], rank_512[query_id][doc_id]))
    relevant_rank_analysis[label] = {
        "relevant_candidate_count": len(rows),
        "mean_rank_256": float(np.mean([row[0] for row in rows])) if rows else None,
        "mean_rank_512": float(np.mean([row[1] for row in rows])) if rows else None,
        "moved_up": sum(new < old for old, new in rows),
        "unchanged": sum(new == old for old, new in rows),
        "moved_down": sum(new > old for old, new in rows),
        "entered_top10": sum(old > 10 and new <= 10 for old, new in rows),
        "left_top10": sum(old <= 10 and new > 10 for old, new in rows),
    }

metrics = {
    "experiment": "controlled MiniLM max-length comparison; no adaptive policy",
    "config": CONFIG,
    "cache_fingerprint": fingerprint,
    "counts": {
        "validation_queries": len(validation_ids),
        "candidate_pairs": sum(len(scores) for scores in candidates.values()),
        "pairs_over_256": sum(
            length > CONFIG["baseline_max_length"]
            for query_lengths in lengths.values()
            for length in query_lengths.values()
        ),
        "pairs_over_512": sum(
            length > CONFIG["experimental_max_length"]
            for query_lengths in lengths.values()
            for length in query_lengths.values()
        ),
        "newly_scored_queries": len(pending_ids),
    },
    "timings_seconds": metadata.get("timings_seconds", {}),
    "max_length_256": {
        **evaluate(qrels, results_256),
        "per_query_ndcg_at_10": ndcg_256,
        "per_query_recall_at_10": recall_256,
    },
    "max_length_512": {
        **evaluate(qrels, results_512),
        "per_query_ndcg_at_10": ndcg_512,
        "per_query_recall_at_10": recall_512,
    },
    "max_length_512_minus_256": {
        "ndcg_at_10": paired_summary(ndcg_512, ndcg_256, validation_ids),
        "recall_at_10": paired_summary(recall_512, recall_256, validation_ids),
    },
    "predefined_query_strata": {
        name: {
            "ndcg_at_10": paired_summary(ndcg_512, ndcg_256, query_ids),
            "recall_at_10": paired_summary(recall_512, recall_256, query_ids),
        }
        for name, query_ids in groups.items()
    },
    "truncated_relevant_vs_not_subgroup_contrast": {
        "definition": (
            "mean 512-minus-256 change for queries with a truncated relevant Top-100 candidate "
            "minus the mean change for queries without one"
        ),
        "ndcg_at_10": subgroup_contrast(
            ndcg_512,
            ndcg_256,
            groups["any_truncated_relevant_candidate"],
            groups["no_truncated_relevant_candidate"],
        ),
        "recall_at_10": subgroup_contrast(
            recall_512,
            recall_256,
            groups["any_truncated_relevant_candidate"],
            groups["no_truncated_relevant_candidate"],
        ),
    },
    "query_truncated_candidate_fraction": query_truncated_fraction,
    "relevant_candidate_rank_analysis": relevant_rank_analysis,
}
write_json(METRICS_PATH, metrics)
write_json(RETRIEVAL_PATH, results_512)
print(
    json.dumps(
        {
            "NDCG@10_256": metrics["max_length_256"]["ndcg"]["NDCG@10"],
            "NDCG@10_512": metrics["max_length_512"]["ndcg"]["NDCG@10"],
            "Recall@10_256": metrics["max_length_256"]["recall"]["Recall@10"],
            "Recall@10_512": metrics["max_length_512"]["recall"]["Recall@10"],
            "paired_difference": metrics["max_length_512_minus_256"],
            "relevant_candidate_rank_analysis": relevant_rank_analysis,
        },
        indent=2,
    )
)
