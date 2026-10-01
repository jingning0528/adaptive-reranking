import json
import logging
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from beir.retrieval.evaluation import EvaluateRetrieval

logging.basicConfig(level=logging.INFO)

DATASET = "scifact"
MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
TOP_K = 100
BATCH_SIZE = 128
MAX_LENGTH = 256
K_VALUES = [1, 3, 5, 10, 100, 1000]


def cpu_model() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


data_path = Path("datasets") / DATASET
baseline_path = Path("results/baseline/retrieval.json")
if not data_path.is_dir() or not baseline_path.is_file():
    raise FileNotFoundError("Run run_baseline.py before reranking")

corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split="test")
baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))
candidate_count = sum(min(TOP_K, len(scores)) for scores in baseline_results.values())

model_load_start = time.perf_counter()
cross_encoder = CrossEncoder(MODEL, device="cpu", max_length=MAX_LENGTH)
model_load_seconds = time.perf_counter() - model_load_start
reranker = Rerank(cross_encoder, batch_size=BATCH_SIZE)

rerank_start = time.perf_counter()
rerank_results = reranker.rerank(
    corpus, queries, baseline_results, top_k=TOP_K
)
rerank_seconds = time.perf_counter() - rerank_start

evaluation_start = time.perf_counter()
ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(
    qrels, rerank_results, K_VALUES
)
evaluation_seconds = time.perf_counter() - evaluation_start

baseline_metrics = json.loads(
    Path("results/baseline/metrics.json").read_text(encoding="utf-8")
)
output = Path("results/rerank")
output.mkdir(parents=True, exist_ok=True)

metrics = {
    "dataset": DATASET,
    "model": MODEL,
    "device": "cpu",
    "candidate_source": "results/baseline/retrieval.json",
    "candidate_top_k": TOP_K,
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "git_commit": git_commit(),
    "environment": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "dependencies": {
            package: version(package)
            for package in ("beir", "sentence-transformers", "torch", "transformers")
        },
    },
    "counts": {
        "documents": len(corpus),
        "queries": len(queries),
        "candidates_scored": candidate_count,
    },
    "timings_seconds": {
        "model_load": model_load_seconds,
        "rerank_total": rerank_seconds,
        "rerank_per_query": rerank_seconds / len(queries),
        "evaluation": evaluation_seconds,
    },
    "throughput_candidates_per_second": candidate_count / rerank_seconds,
    "ndcg": ndcg,
    "map": map_scores,
    "recall": recall,
    "precision": precision,
    "comparison_to_baseline": {
        "NDCG@10_absolute": ndcg["NDCG@10"] - baseline_metrics["ndcg"]["NDCG@10"],
        "NDCG@10_relative_percent": (
            ndcg["NDCG@10"] / baseline_metrics["ndcg"]["NDCG@10"] - 1
        ) * 100,
        "Recall@10_absolute": recall["Recall@10"] - baseline_metrics["recall"]["Recall@10"],
    },
}
(output / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
(output / "retrieval.json").write_text(
    json.dumps(rerank_results), encoding="utf-8"
)

print(json.dumps(metrics, indent=2))
