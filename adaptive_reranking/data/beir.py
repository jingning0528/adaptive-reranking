"""BEIR filesystem adapter; dataset-specific loading stays in this module."""

from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader

from adaptive_reranking.data.base import Dataset
from beir import util

BEIR_DOWNLOAD_ROOT = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


class BEIRDatasetLoader:
    def load(self, name: str, split: str, data_dir: Path, *, download: bool) -> Dataset:
        folder = data_dir / name
        # Do not redownload a populated dataset just because the requested split
        # is missing. A missing split should produce a clear local-file error.
        if not folder.exists() and download:
            folder = Path(util.download_and_unzip(f"{BEIR_DOWNLOAD_ROOT}/{name}.zip", str(data_dir)))
        for relative in ("corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv"):
            if not (folder / relative).is_file():
                raise FileNotFoundError(f"Dataset {name!r}, split {split!r}: missing {folder / relative}")
        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=str(folder)).load(split=split)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid BEIR data for dataset {name!r}, split {split!r}: {error}") from error
        # BEIR permits omitted titles. Keep the common schema usable by encoders
        # that concatenate title and text; never alter nonempty source text.
        for document in corpus.values():
            if document["title"] is None:
                document["title"] = ""
        dataset = Dataset(name=name, split=split, corpus=corpus, queries=queries, qrels=qrels)
        dataset.validate()
        return dataset
