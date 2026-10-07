# Support both `python -m scripts.NAME` and `python scripts/NAME.py`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

from adaptive_reranking.data import load_dataset

from adaptive_reranking.evaluation.validation import query_ndcg_at_k

OUTPUT = Path("results/metrics/validation")
dataset = load_dataset("scifact", split="train", download=False)
corpus = dataset.corpus
train_queries = dataset.queries
train_qrels = dataset.qrels
split = json.loads((Path("results/manifests/validation/split.json")).read_text(encoding="utf-8"))
validation_ids = split["query_ids"]
qrels = {query_id: train_qrels[query_id] for query_id in validation_ids}
baseline = json.loads((OUTPUT / "baseline_retrieval.json").read_text(encoding="utf-8"))
baseline_ndcg = query_ndcg_at_k(qrels, baseline)


def top_documents(results, query_id, limit=5):
    ranked = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:limit]
    return [
        {
            "rank": rank,
            "doc_id": doc_id,
            "relevance": qrels[query_id].get(doc_id, 0),
            "title": corpus[doc_id].get("title", ""),
            "text_preview": corpus[doc_id].get("text", "")[:400],
        }
        for rank, (doc_id, _) in enumerate(ranked, start=1)
    ]


report = {"input_format": "[query, title + space + abstract]", "models": {}}
for preset in ("tinybert_l2", "minilm_l6"):
    metrics = json.loads((OUTPUT / f"{preset}_metrics.json").read_text(encoding="utf-8"))
    results = json.loads((OUTPUT / f"{preset}_retrieval.json").read_text(encoding="utf-8"))
    deltas = metrics["per_query_delta_ndcg_at_10"]
    ordered = sorted(deltas, key=deltas.get)
    selected = ordered[:5] + list(reversed(ordered[-5:]))
    cases = []
    for query_id in selected:
        cases.append(
            {
                "direction": "worsened" if deltas[query_id] < 0 else "improved",
                "query_id": query_id,
                "query": train_queries[query_id],
                "baseline_ndcg_at_10": baseline_ndcg[query_id],
                "rerank_ndcg_at_10": metrics["per_query_ndcg_at_10"][query_id],
                "delta": deltas[query_id],
                "relevant_documents": [
                    {
                        "doc_id": doc_id,
                        "title": corpus[doc_id].get("title", ""),
                        "text_preview": corpus[doc_id].get("text", "")[:400],
                    }
                    for doc_id in qrels[query_id]
                ],
                "baseline_top_5": top_documents(baseline, query_id),
                "rerank_top_5": top_documents(results, query_id),
            }
        )
    report["models"][preset] = {"extreme_cases": cases}

(OUTPUT / "error_cases.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Wrote {OUTPUT / 'error_cases.json'}")
