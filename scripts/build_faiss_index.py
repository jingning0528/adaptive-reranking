"""Build an exact FAISS index using existing offline embedding artifacts."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from adaptive_reranking.retrieval.faiss_index import build_faiss_index


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build an exact IndexFlatIP from saved embeddings")
    parser.add_argument("--dataset", required=True, help="Dataset name recorded in the embedding bundle")
    parser.add_argument("--embeddings-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    source = args.embeddings_dir or Path("results/cache/embeddings") / args.dataset
    output = args.output_dir or Path("results/cache/faiss") / args.dataset
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Index directory already exists: {output}; choose a new --output-dir")
    bundle = build_faiss_index(source, output, expected_dataset=args.dataset)
    print(json.dumps({"output_dir": str(output), **bundle.metadata}, indent=2))


if __name__ == "__main__":
    main()
