import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import CrossEncoder, SentenceTransformer

from adaptive_reranking.io import load_json, write_json

POLICY = load_json("configs/frozen_adaptive_policy.json")
TRIALS = 2
INDEX_PATH = Path("artifacts/method7/corpus_index.joblib")
OUTPUT = Path("results/method7/end_to_end_latency.json")

corpus, queries, _ = GenericDataLoader("datasets/scifact").load(split="test")
query_ids = sorted(queries)
doc_ids = sorted(corpus)
documents = [
    (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
    for doc_id in doc_ids
]

retriever = SentenceTransformer(POLICY["retriever_model"], device="cpu")
if INDEX_PATH.exists():
    index = joblib.load(INDEX_PATH)
    if index["doc_ids"] != doc_ids or index["model"] != POLICY["retriever_model"]:
        raise RuntimeError("Corpus index does not match the frozen benchmark contract")
    corpus_embeddings = index["embeddings"]
    index_build_seconds = index["build_seconds"]
else:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    corpus_embeddings = retriever.encode(
        documents,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    index_build_seconds = time.perf_counter() - start
    joblib.dump(
        {
            "model": POLICY["retriever_model"],
            "doc_ids": doc_ids,
            "embeddings": corpus_embeddings,
            "build_seconds": index_build_seconds,
        },
        INDEX_PATH,
        compress=3,
    )

reranker = CrossEncoder(
    POLICY["reranker_model"],
    revision=POLICY["reranker_revision"],
    device="mps",
    max_length=POLICY["max_length"],
)
reranker.predict(
    [[queries[query_ids[0]], documents[0]]],
    batch_size=1,
    show_progress_bar=False,
)


def retrieve():
    start = time.perf_counter()
    query_embeddings = retriever.encode(
        [queries[query_id] for query_id in query_ids],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    encode_seconds = time.perf_counter() - start
    search_start = time.perf_counter()
    similarities = query_embeddings @ corpus_embeddings.T
    top_indices = np.argpartition(
        similarities, -POLICY["candidate_top_k"], axis=1
    )[:, -POLICY["candidate_top_k"] :]
    ranked = []
    for row, indices in enumerate(top_indices):
        order = indices[np.argsort(-similarities[row, indices], kind="stable")]
        ranked.append(
            [(doc_ids[index], float(similarities[row, index])) for index in order]
        )
    search_seconds = time.perf_counter() - search_start
    return ranked, encode_seconds, search_seconds


def run_strategy(name, fixed_budget=None):
    total_start = time.perf_counter()
    ranked, encode_seconds, search_seconds = retrieve()
    route_start = time.perf_counter()
    if fixed_budget is not None:
        budgets = [fixed_budget] * len(query_ids)
    else:
        budgets = [
            POLICY["hard_budget"]
            if scores[0][1] - scores[9][1] <= POLICY["margin_threshold"]
            else POLICY["easy_budget"]
            for scores in ranked
        ]
    routing_seconds = time.perf_counter() - route_start
    pair_start = time.perf_counter()
    pairs = []
    locations = []
    for query_id, candidates, budget in zip(query_ids, ranked, budgets, strict=True):
        for doc_id, _ in candidates[:budget]:
            document = corpus[doc_id]
            text = (document.get("title", "") + " " + document.get("text", "")).strip()
            pairs.append([queries[query_id], text])
            locations.append((query_id, doc_id))
    pair_seconds = time.perf_counter() - pair_start
    rerank_start = time.perf_counter()
    with torch.inference_mode():
        values = reranker.predict(
            pairs,
            batch_size=POLICY["reranker_batch_size"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    rerank_seconds = time.perf_counter() - rerank_start
    ranking_start = time.perf_counter()
    scores_by_query = {query_id: [] for query_id in query_ids}
    for (query_id, doc_id), value in zip(locations, values, strict=True):
        scores_by_query[query_id].append((doc_id, float(value)))
    for query_id in query_ids:
        sorted(scores_by_query[query_id], key=lambda item: (-item[1], item[0]))[:10]
    ranking_seconds = time.perf_counter() - ranking_start
    total_seconds = time.perf_counter() - total_start
    return {
        "strategy": name,
        "queries": len(query_ids),
        "pairs_scored": len(pairs),
        "mean_budget": float(np.mean(budgets)),
        "easy_queries": sum(b == POLICY["easy_budget"] for b in budgets),
        "hard_queries": sum(b == POLICY["hard_budget"] for b in budgets)
        if fixed_budget is None
        else None,
        "timings_seconds": {
            "query_encoding": encode_seconds,
            "top100_search": search_seconds,
            "routing": routing_seconds,
            "pair_construction": pair_seconds,
            "cross_encoder": rerank_seconds,
            "final_ranking": ranking_seconds,
            "online_total": total_seconds,
        },
        "throughput_queries_per_second": len(query_ids) / total_seconds,
    }


results = {"full_100": [], "adaptive": []}
for trial in range(TRIALS):
    order = ("full_100", "adaptive") if trial % 2 == 0 else ("adaptive", "full_100")
    for strategy in order:
        if strategy == "full_100":
            results[strategy].append(run_strategy(strategy, fixed_budget=100))
        else:
            results[strategy].append(run_strategy(strategy))

summary = {}
for strategy, trials in results.items():
    summary[strategy] = {
        "trials": len(trials),
        "mean_budget": float(np.mean([trial["mean_budget"] for trial in trials])),
        "median_online_total_seconds": float(
            np.median([trial["timings_seconds"]["online_total"] for trial in trials])
        ),
        "median_cross_encoder_seconds": float(
            np.median([trial["timings_seconds"]["cross_encoder"] for trial in trials])
        ),
        "median_throughput_queries_per_second": float(
            np.median([trial["throughput_queries_per_second"] for trial in trials])
        ),
    }
summary["adaptive"]["online_speedup_vs_full"] = (
    summary["full_100"]["median_online_total_seconds"]
    / summary["adaptive"]["median_online_total_seconds"]
)
summary["adaptive"]["online_latency_reduction_fraction"] = 1.0 - (
    summary["adaptive"]["median_online_total_seconds"]
    / summary["full_100"]["median_online_total_seconds"]
)

contract = {
    "policy": POLICY,
    "trials": TRIALS,
    "corpus_index_excluded_from_online_timing": True,
    "model_loading_excluded_from_online_timing": True,
    "hardware": platform.platform(),
}
write_json(
    OUTPUT,
    {
        "contract": contract,
        "contract_fingerprint": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "offline_index_build_seconds": index_build_seconds,
        "raw_trials": results,
        "summary": summary,
    },
)
print(json.dumps(summary, indent=2))
