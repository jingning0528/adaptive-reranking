"""Shared dataset options for the baseline and reranking entry points."""

import argparse
from pathlib import Path

from adaptive_reranking.data.loader import SUPPORTED_DATASETS, SUPPORTED_SPLITS


def result_directory(stage: str, dataset: str, split: str) -> Path:
    root = Path("results/metrics") / stage
    # Keep V1 commands and their cached SciFact test paths compatible.
    return root if (dataset, split) == ("scifact", "test") else root / dataset / split


def parse_dataset_args(stage: str, argv=None):
    parser = argparse.ArgumentParser(description=f"Run {stage} on a common-format dataset")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="scifact")
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="test")
    parser.add_argument("--data-dir", type=Path, default=Path("results/datasets"))
    parser.add_argument("--no-download", action="store_true", help="Require existing local dataset files")
    if stage in ("baseline", "rerank"):
        parser.add_argument("--output-dir", type=Path, help="Override the dataset-specific result directory")
    if stage == "rerank":
        parser.add_argument(
            "--baseline-dir", type=Path, help="Read retrieval.json and metrics.json from this directory"
        )
    return parser.parse_args(argv)
