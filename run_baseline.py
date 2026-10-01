import json
import logging
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

logging.basicConfig(level=logging.INFO)


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


class TimedEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.query_encoding_seconds = 0.0
        self.corpus_encoding_seconds = 0.0

    def encode_queries(self, *args, **kwargs):
        start = time.perf_counter()
        encoded = self.encoder.encode_queries(*args, **kwargs)
        self.query_encoding_seconds += time.perf_counter() - start
        return encoded

    def encode_corpus(self, *args, **kwargs):
        start = time.perf_counter()
        encoded = self.encoder.encode_corpus(*args, **kwargs)
        self.corpus_encoding_seconds += time.perf_counter() - start
        return encoded

# 1. 下载数据：文档、查询和相关性标注
dataset = "scifact"
url = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    f"thakur/BEIR/datasets/{dataset}.zip"
)
data_path = util.download_and_unzip(url, "datasets")

corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(split="test")

print(f"Documents: {len(corpus)}, queries: {len(queries)}")

# 2. 加载预训练模型；本次不训练
model_load_start = time.perf_counter()
encoder = TimedEncoder(models.SentenceBERT(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
))
model_load_seconds = time.perf_counter() - model_load_start
search = DenseRetrievalExactSearch(encoder, batch_size=16)
evaluator = EvaluateRetrieval(search, score_function="cos_sim")

# 3. 检索并评估
retrieval_start = time.perf_counter()
results = evaluator.retrieve(corpus, queries)
retrieval_seconds = time.perf_counter() - retrieval_start
evaluation_start = time.perf_counter()
ndcg, map_scores, recall, precision = evaluator.evaluate(
    qrels, results, evaluator.k_values
)
evaluation_seconds = time.perf_counter() - evaluation_start

# 4. 保存指标和检索结果，供后续重排使用
output = Path("results/baseline")
output.mkdir(parents=True, exist_ok=True)

metrics = {
    "dataset": dataset,
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
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
    "counts": {"documents": len(corpus), "queries": len(queries)},
    "timings_seconds": {
        "model_load": model_load_seconds,
        "query_encoding": encoder.query_encoding_seconds,
        "corpus_encoding": encoder.corpus_encoding_seconds,
        "retrieval_total": retrieval_seconds,
        "evaluation": evaluation_seconds,
    },
    "ndcg": ndcg,
    "map": map_scores,
    "recall": recall,
    "precision": precision,
}
(output / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
(output / "retrieval.json").write_text(
    json.dumps(results), encoding="utf-8"
)

print(json.dumps(metrics, indent=2))
