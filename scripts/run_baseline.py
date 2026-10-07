# Support both `python -m scripts.NAME` and `python scripts/NAME.py`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

from adaptive_reranking.utils.environment import cpu_model, git_commit

from adaptive_reranking.retrieval.timing import TimedEncoder

logging.basicConfig(level=logging.INFO)







# 1. 下载数据：文档、查询和相关性标注
dataset = "scifact"
url = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    f"thakur/BEIR/datasets/{dataset}.zip"
)
data_path = util.download_and_unzip(url, "results/datasets")

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
output = Path("results/metrics/baseline")
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
