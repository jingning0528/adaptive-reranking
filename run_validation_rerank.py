import argparse
import json
import time
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from beir.retrieval.evaluation import EvaluateRetrieval

from validation_utils import K_VALUES, query_ndcg_at_k

PRESETS = {
    "tinybert_l2": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "minilm_l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}
TOP_K = 100
BATCH_SIZE = 128
MAX_LENGTH = 256
OUTPUT = Path("results/validation")

parser = argparse.ArgumentParser()
parser.add_argument("preset", choices=PRESETS)
args = parser.parse_args()
model_name = PRESETS[args.preset]

corpus, train_queries, train_qrels = GenericDataLoader("datasets/scifact").load(split="train")
split = json.loads((OUTPUT / "split.json").read_text(encoding="utf-8"))
validation_ids = split["query_ids"]
queries = {query_id: train_queries[query_id] for query_id in validation_ids}
qrels = {query_id: train_qrels[query_id] for query_id in validation_ids}
baseline_results = json.loads(
    (OUTPUT / "baseline_retrieval.json").read_text(encoding="utf-8")
)

model_load_start = time.perf_counter()
model = CrossEncoder(model_name, device="cpu", max_length=MAX_LENGTH)
model_load_seconds = time.perf_counter() - model_load_start
reranker = Rerank(model, batch_size=BATCH_SIZE)
rerank_start = time.perf_counter()
results = reranker.rerank(corpus, queries, baseline_results, top_k=TOP_K)
rerank_seconds = time.perf_counter() - rerank_start

evaluation_start = time.perf_counter()
ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(qrels, results, K_VALUES)
evaluation_seconds = time.perf_counter() - evaluation_start
baseline_per_query = query_ndcg_at_k(qrels, baseline_results)
rerank_per_query = query_ndcg_at_k(qrels, results)
deltas = {
    query_id: rerank_per_query[query_id] - baseline_per_query[query_id]
    for query_id in validation_ids
}

metrics = {
    "dataset": "scifact",
    "split": "held-out validation from train",
    "model": model_name,
    "device": "cpu",
    "candidate_top_k": TOP_K,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "input_format": "[query, title + space + abstract]",
    "counts": {
        "queries": len(validation_ids),
        "candidate_pairs": sum(len(scores) for scores in results.values()),
    },
    "timings_seconds": {
        "model_load": model_load_seconds,
        "rerank_total": rerank_seconds,
        "rerank_per_query": rerank_seconds / len(validation_ids),
        "evaluation": evaluation_seconds,
    },
    "throughput_candidates_per_second": sum(len(scores) for scores in results.values()) / rerank_seconds,
    "query_changes_at_10": {
        "improved": sum(delta > 1e-12 for delta in deltas.values()),
        "unchanged": sum(abs(delta) <= 1e-12 for delta in deltas.values()),
        "worsened": sum(delta < -1e-12 for delta in deltas.values()),
    },
    "ndcg": ndcg,
    "map": map_scores,
    "recall": recall,
    "precision": precision,
    "per_query_ndcg_at_10": rerank_per_query,
    "per_query_delta_ndcg_at_10": deltas,
}
(OUTPUT / f"{args.preset}_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
(OUTPUT / f"{args.preset}_retrieval.json").write_text(
    json.dumps(results), encoding="utf-8"
)
print(json.dumps({key: value for key, value in metrics.items() if not key.startswith("per_query_")}, indent=2))
