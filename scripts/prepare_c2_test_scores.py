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

from adaptive_reranking.data import load_dataset
from sentence_transformers import CrossEncoder

from adaptive_reranking.utils.io import load_json, write_json

logging.basicConfig(level=logging.INFO)

CONFIG = load_json("configs/adaptive_budget.json")
RETRIEVAL_PATH = Path("results/metrics/baseline/retrieval.json")
CHECKPOINT_PATH = Path("results/cache/method6/test_teacher_scores_checkpoint.json")
METADATA_PATH = Path("results/cache/method6/test_teacher_scores_metadata.json")
MANIFEST_PATH = Path("results/manifests/method6/test_teacher_manifest.json")

contract = {
    "dataset": CONFIG["dataset"],
    "split": "test",
    "model": CONFIG["model"],
    "model_revision": CONFIG["model_revision"],
    "device": CONFIG["device"],
    "max_length": CONFIG["max_length"],
    "batch_size": CONFIG["batch_size"],
    "candidate_top_k": CONFIG["candidate_top_k"],
    "input_format": "[query, title + space + abstract]",
    "candidate_source": str(RETRIEVAL_PATH),
    "candidate_order": "retrieval score descending; document id breaks ties",
}
fingerprint = hashlib.sha256(
    json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
if CHECKPOINT_PATH.exists():
    if not METADATA_PATH.exists():
        raise RuntimeError("Test teacher checkpoint exists without metadata")
    metadata = load_json(METADATA_PATH)
    if metadata["fingerprint"] != fingerprint:
        raise RuntimeError("Test teacher checkpoint configuration mismatch")
else:
    metadata = {"fingerprint": fingerprint, "contract": contract, "timings_seconds": {}}
    write_json(METADATA_PATH, metadata)

dataset = load_dataset("scifact", split="test", download=False)
corpus = dataset.corpus
queries = dataset.queries
retrieval = load_json(RETRIEVAL_PATH)
candidates = {
    query_id: dict(
        sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            : CONFIG["candidate_top_k"]
        ]
    )
    for query_id, scores in retrieval.items()
}
if set(candidates) != set(queries):
    raise RuntimeError("Test candidate cache does not match test queries")
if any(len(scores) != CONFIG["candidate_top_k"] for scores in candidates.values()):
    raise RuntimeError("Every test query must have exactly 100 candidates")

teacher_scores = load_json(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else {}
pending_ids = sorted(set(queries) - set(teacher_scores))
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
        values = model.predict(
            pairs,
            batch_size=CONFIG["batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        for (query_id, doc_id), value in zip(locations, values, strict=True):
            teacher_scores.setdefault(query_id, {})[doc_id] = float(value)
        write_json(CHECKPOINT_PATH, teacher_scores)
        logging.info(
            "C2 test teacher checkpoint: %d/%d queries",
            min(offset + chunk_size, len(pending_ids)),
            len(pending_ids),
        )
    scoring_seconds = time.perf_counter() - scoring_start
    previous = metadata.get("timings_seconds", {})
    metadata["timings_seconds"] = {
        "model_load": previous.get("model_load", 0.0) + load_seconds,
        "scoring": previous.get("scoring", 0.0) + scoring_seconds,
    }
    write_json(METADATA_PATH, metadata)

if set(teacher_scores) != set(queries):
    raise RuntimeError("Test teacher cache is incomplete")
if any(set(teacher_scores[q]) != set(candidates[q]) for q in queries):
    raise RuntimeError("Test teacher scores do not match frozen candidates")

write_json(
    MANIFEST_PATH,
    {
        "fingerprint": fingerprint,
        "contract": contract,
        "counts": {
            "queries": len(queries),
            "candidate_pairs": sum(len(scores) for scores in candidates.values()),
            "newly_scored_queries": len(pending_ids),
        },
        "timings_seconds": metadata.get("timings_seconds", {}),
        "artifacts": {
            "teacher_checkpoint": str(CHECKPOINT_PATH),
        },
    },
)
print(f"Prepared {sum(len(scores) for scores in teacher_scores.values())} test teacher scores")
