"""Single entry point for supported retrieval datasets."""

from pathlib import Path

from adaptive_reranking.data.base import Dataset
from adaptive_reranking.data.beir import BEIRDatasetLoader

SUPPORTED_DATASETS = ("scifact", "nfcorpus")
SUPPORTED_SPLITS = ("train", "dev", "test")


def load_dataset(
    name: str,
    *,
    split: str = "test",
    data_dir: str | Path = "results/datasets",
    download: bool = True,
) -> Dataset:
    """Load a supported dataset's split, downloading an absent dataset if allowed.

    ``data_dir`` is the cache parent: files live in ``data_dir / name``.
    Pass ``download=False`` for offline use and deterministic tests.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; supported datasets: {', '.join(SUPPORTED_DATASETS)}")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unknown split {split!r}; supported splits: {', '.join(SUPPORTED_SPLITS)}")
    return BEIRDatasetLoader().load(name, split, Path(data_dir), download=download)
