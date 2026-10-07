"""Encode a query with the saved model contract and search a persistent index."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from dataclasses import replace
from pathlib import Path

from adaptive_reranking.retrieval.embeddings import EmbeddingConfig
from adaptive_reranking.retrieval.faiss_backend import FaissRetriever
from adaptive_reranking.retrieval.faiss_index import _positive_integer, load_faiss_index


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search an exact FAISS cosine index")
    parser.add_argument("--dataset", required=True, help="Dataset name recorded in the index bundle")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--device", default="cpu", help="Query encoder device; vector search remains on CPU")
    args = parser.parse_args(argv)
    _positive_integer(args.top_k, "top_k")
    if not args.query.strip():
        raise ValueError("Query text must be nonempty")
    directory = args.index_dir or Path("results/cache/faiss") / args.dataset
    bundle = load_faiss_index(directory)
    if bundle.metadata["dataset"] != args.dataset:
        raise ValueError("Index dataset does not match --dataset")
    config = replace(EmbeddingConfig(**bundle.config), device=args.device)
    config.validate()
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(config.model_name, revision=config.model_revision, device=config.device)
    if config.max_length is not None:
        encoder.max_seq_length = config.max_length
    results = FaissRetriever(bundle, encoder, config).search(args.query, args.top_k)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "query": args.query,
                "results": [{"doc_id": doc_id, "score": score} for doc_id, score in results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
