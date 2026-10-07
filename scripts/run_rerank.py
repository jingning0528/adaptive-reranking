# Support both `python -m scripts.NAME` and `python scripts/NAME.py`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import logging
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

from beir.retrieval.evaluation import EvaluateRetrieval

from adaptive_reranking.data import load_dataset
from adaptive_reranking.utils.environment import cpu_model, git_commit
from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from scripts._dataset_options import parse_dataset_args, result_directory

logging.basicConfig(level=logging.INFO)

MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
TOP_K = 100
BATCH_SIZE = 128
MAX_LENGTH = 256
K_VALUES = [1, 3, 5, 10, 100, 1000]


def main(argv=None):
    args = parse_dataset_args("rerank", argv)
    baseline_dir = args.baseline_dir or result_directory("baseline", args.dataset, args.split)
    baseline_path = baseline_dir / "retrieval.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Missing {baseline_path}; run scripts.run_baseline for this dataset and split first")
    dataset = load_dataset(args.dataset, split=args.split, data_dir=args.data_dir, download=False)
    corpus, queries, qrels = dataset.corpus, dataset.queries, dataset.qrels
    baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_metrics = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    if baseline_metrics["dataset"] != dataset.name or baseline_metrics.get("split", "test") != dataset.split:
        raise ValueError("Baseline dataset/split does not match the requested reranking dataset/split")
    candidate_count = sum(min(TOP_K, len(scores)) for scores in baseline_results.values())

    model_load_start = time.perf_counter()
    cross_encoder = CrossEncoder(MODEL, device="cpu", max_length=MAX_LENGTH)
    model_load_seconds = time.perf_counter() - model_load_start
    reranker = Rerank(cross_encoder, batch_size=BATCH_SIZE)

    rerank_start = time.perf_counter()
    rerank_results = reranker.rerank(corpus, queries, baseline_results, top_k=TOP_K)
    rerank_seconds = time.perf_counter() - rerank_start

    evaluation_start = time.perf_counter()
    ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, K_VALUES)
    evaluation_seconds = time.perf_counter() - evaluation_start

    output = args.output_dir or result_directory("rerank", args.dataset, args.split)
    output.mkdir(parents=True, exist_ok=True)

    metrics = {
        "dataset": dataset.name,
        "model": MODEL,
        "device": "cpu",
        "candidate_source": str(baseline_path),
        "candidate_top_k": TOP_K,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "git_commit": git_commit(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu": cpu_model(),
            "dependencies": {
                package: version(package) for package in ("beir", "sentence-transformers", "torch", "transformers")
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
            "NDCG@10_relative_percent": (ndcg["NDCG@10"] / baseline_metrics["ndcg"]["NDCG@10"] - 1) * 100,
            "Recall@10_absolute": recall["Recall@10"] - baseline_metrics["recall"]["Recall@10"],
        },
    }
    if dataset.split != "test":
        metrics["split"] = dataset.split
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "retrieval.json").write_text(json.dumps(rerank_results), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
