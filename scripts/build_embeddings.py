"""Build document vectors offline; no retrieval, reranking, or ANN indexing."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from adaptive_reranking.data import load_dataset
from adaptive_reranking.data.loader import SUPPORTED_DATASETS, SUPPORTED_SPLITS
from adaptive_reranking.retrieval.artifacts import save_embedding_artifacts
from adaptive_reranking.retrieval.embeddings import EmbeddingConfig, build_document_embeddings


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build reusable corpus embedding artifacts")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="test")
    parser.add_argument("--data-dir", type=Path, default=Path("results/datasets"))
    parser.add_argument("--no-download", action="store_true", help="Require an existing local dataset")
    parser.add_argument("--output-dir", type=Path, help="Default: results/cache/embeddings/<dataset>")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-revision", help="Optional pinned Hugging Face model revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, help="Override the encoder's default maximum sequence length")
    args = parser.parse_args(argv)
    config = EmbeddingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize,
        model_revision=args.model_revision,
        device=args.device,
        max_length=args.max_length,
    )
    config.validate()
    output = args.output_dir or Path("results/cache/embeddings") / args.dataset
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Artifact directory already exists: {output}; choose a new --output-dir")
    dataset = load_dataset(args.dataset, split=args.split, data_dir=args.data_dir, download=not args.no_download)
    # Keep the core pipeline and artifact loader independent of Transformer loading.
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(config.model_name, revision=config.model_revision, device=config.device)
    if config.max_length is not None:
        encoder.max_seq_length = config.max_length
    # Record the effective truncation length even when the CLI did not override it.
    config = EmbeddingConfig(**{**vars(config), "max_length": encoder.max_seq_length})
    artifacts = build_document_embeddings(dataset, encoder, config)
    save_embedding_artifacts(artifacts, output)
    print(json.dumps({"output_dir": str(output), **artifacts.metadata}, indent=2))


if __name__ == "__main__":
    main()
