import json
import time
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from transformers import AutoTokenizer

from validation_utils import K_VALUES, oracle_ranking, validation_query_ids

DATA_PATH = "datasets/scifact"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 100
OUTPUT = Path("results/validation")

corpus, train_queries, train_qrels = GenericDataLoader(DATA_PATH).load(split="train")
validation_ids = validation_query_ids(train_qrels)
queries = {query_id: train_queries[query_id] for query_id in validation_ids}
qrels = {query_id: train_qrels[query_id] for query_id in validation_ids}

split = {
    "source_split": "train",
    "method": "lowest 20% by SHA-256(query_id), rounded to nearest integer",
    "total_train_queries": len(train_qrels),
    "validation_queries": len(validation_ids),
    "query_ids": validation_ids,
}
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

model_load_start = time.perf_counter()
encoder = models.SentenceBERT(BASE_MODEL, device="cpu")
model_load_seconds = time.perf_counter() - model_load_start
retriever = EvaluateRetrieval(
    DenseRetrievalExactSearch(encoder, batch_size=16),
    score_function="cos_sim",
    k_values=K_VALUES,
)
retrieval_start = time.perf_counter()
baseline_results = retriever.retrieve(corpus, queries)
retrieval_seconds = time.perf_counter() - retrieval_start
baseline_metrics = retriever.evaluate(qrels, baseline_results, K_VALUES)
(OUTPUT / "baseline_retrieval.json").write_text(
    json.dumps(baseline_results), encoding="utf-8"
)

oracle_results = oracle_ranking(qrels, baseline_results)
oracle_metrics = EvaluateRetrieval.evaluate(qrels, oracle_results, K_VALUES)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
token_lengths = []
for query_id, scores in baseline_results.items():
    for doc_id in scores:
        document = corpus[doc_id]
        text = (document.get("title", "") + " " + document.get("text", "")).strip()
        token_lengths.append(
            len(tokenizer(queries[query_id], text, truncation=False)["input_ids"])
        )
token_lengths.sort()


def percentile(percent: float) -> int:
    index = round((len(token_lengths) - 1) * percent)
    return token_lengths[index]


def metric_dict(values):
    ndcg, map_scores, recall, precision = values
    return {"ndcg": ndcg, "map": map_scores, "recall": recall, "precision": precision}


metrics = {
    "dataset": "scifact",
    "split": "held-out validation from train",
    "base_model": BASE_MODEL,
    "device": "cpu",
    "candidate_top_k": TOP_K,
    "input_format": "[query, title + space + abstract]",
    "counts": {
        "documents": len(corpus),
        "train_queries": len(train_qrels),
        "validation_queries": len(validation_ids),
        "candidate_pairs": len(token_lengths),
    },
    "timings_seconds": {
        "model_load": model_load_seconds,
        "validation_retrieval": retrieval_seconds,
    },
    "token_lengths": {
        "tokenizer": TOKENIZER_MODEL,
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(token_lengths),
        "over_256_count": sum(length > 256 for length in token_lengths),
        "over_256_fraction": sum(length > 256 for length in token_lengths) / len(token_lengths),
        "over_512_count": sum(length > 512 for length in token_lengths),
        "over_512_fraction": sum(length > 512 for length in token_lengths) / len(token_lengths),
    },
    "baseline": metric_dict(baseline_metrics),
    "oracle_top_100": metric_dict(oracle_metrics),
    "oracle_note": "Diagnostic only: judged relevant Top-100 candidates are placed first.",
}
(OUTPUT / "prepare_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
print(json.dumps(metrics, indent=2))
