"""Persistent exact vector indexes built solely from saved embedding artifacts."""

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from adaptive_reranking.retrieval.artifacts import load_embedding_artifacts
from adaptive_reranking.retrieval.embeddings import EmbeddingConfig

INDEX_SCHEMA_VERSION = 1
INDEX_FILES = ("faiss.index", "doc_ids.json", "config.json")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class FaissIndex:
    """Vector storage/search only; no document or query encoder is constructed."""

    index: faiss.IndexFlatIP
    doc_ids: list[str]
    metadata: dict
    config: dict

    def validate(self) -> None:
        if not isinstance(self.index, faiss.IndexFlatIP) or self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Only an exact IndexFlatIP index is supported")
        if self.index.d <= 0 or self.index.ntotal <= 0 or not self.index.is_trained:
            raise ValueError("FAISS index must contain nonempty trained vectors")
        if not isinstance(self.doc_ids, list) or any(
            not isinstance(value, str) or not value for value in self.doc_ids
        ):
            raise ValueError("Document IDs must be nonempty strings")
        if len(self.doc_ids) != self.index.ntotal or len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("Document IDs must be unique and aligned with the FAISS index rows")
        if self.doc_ids != sorted(self.doc_ids):
            raise ValueError("Document IDs must be sorted lexicographically")
        if not isinstance(self.metadata, dict) or not isinstance(self.config, dict):
            raise ValueError("Index metadata and configuration must be objects")
        try:
            config = EmbeddingConfig(**self.config)
        except TypeError as error:
            raise ValueError("Invalid index encoder configuration") from error
        config.validate()
        if not config.normalize_embeddings:
            raise ValueError("Cosine IndexFlatIP requires normalized document embeddings")
        expected = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "index_type": "IndexFlatIP",
            "metric": "inner_product",
            "similarity": "cosine",
            "embedding_dimension": self.index.d,
            "num_documents": self.index.ntotal,
            "normalize_embeddings": True,
            "model_name": config.model_name,
            "model_revision": config.model_revision,
        }
        if any(key not in self.metadata or self.metadata[key] != value for key, value in expected.items()):
            raise ValueError("Index metadata does not match the index or encoder configuration")
        for key in ("schema_version", "embedding_dimension", "num_documents"):
            if type(self.metadata[key]) is not int:
                raise ValueError(f"Index metadata {key} must be an integer")
        if self.metadata["normalize_embeddings"] is not True:
            raise ValueError("Index metadata must record normalized document embeddings")
        for key in ("dataset", "split", "created_at", "corpus_sha256", "faiss_version"):
            if not isinstance(self.metadata.get(key), str) or not self.metadata[key]:
                raise ValueError(f"Index metadata must contain {key}")
        source_checksums = self.metadata.get("source_files_sha256")
        if not isinstance(source_checksums, dict) or any(
            not isinstance(source_checksums.get(name), str) or len(source_checksums[name]) != 64
            for name in ("corpus_embeddings.npy", "doc_ids.json", "config.json")
        ):
            raise ValueError("Index metadata must record the source embedding bundle checksums")

    def search_vectors(self, query_vectors: np.ndarray, top_k: int = 100) -> list[list[tuple[str, float]]]:
        """Normalize query vectors and return cosine-ranked IDs with stable ties.

        A tied cutoff is expanded to include all tied candidates before sorting
        by document ID. Queries with zero vectors return zero-score ID order.
        """
        _positive_integer(top_k, "top_k")
        raw = np.asarray(query_vectors)
        if raw.ndim != 2 or raw.shape[1] != self.index.d:
            raise ValueError(f"Query embeddings must have shape (queries, {self.index.d})")
        if raw.dtype.kind not in "fiu":
            raise ValueError("Query embeddings must be real numeric vectors")
        vectors = np.ascontiguousarray(raw, dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Query embeddings must contain only finite values")
        if len(vectors) == 0:
            return []
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
        limit = min(top_k, self.index.ntotal)
        probe = min(limit + 1, self.index.ntotal)
        all_scores, all_positions = self.index.search(vectors, probe)
        results = []
        for row in range(len(vectors)):
            scores, positions = all_scores[row], all_positions[row]
            current_probe = probe
            while current_probe < self.index.ntotal and scores[-1] == scores[limit - 1]:
                current_probe = min(current_probe * 2, self.index.ntotal)
                expanded_scores, expanded_positions = self.index.search(vectors[row : row + 1], current_probe)
                scores, positions = expanded_scores[0], expanded_positions[0]
            candidates = [
                (self.doc_ids[int(position)], float(score))
                for position, score in zip(positions, scores, strict=True)
                if position >= 0
            ]
            results.append(sorted(candidates, key=lambda item: (-item[1], item[0]))[:limit])
        return results


def build_faiss_index(
    embedding_directory: str | Path, output_directory: str | Path, *, expected_dataset: str | None = None
) -> FaissIndex:
    """Build and persist an exact cosine index without recomputing embeddings."""
    output = Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Index directory already exists: {output}; choose a new output directory")
    artifacts = load_embedding_artifacts(embedding_directory)
    if expected_dataset is not None and artifacts.metadata["dataset"] != expected_dataset:
        raise ValueError("Embedding bundle dataset does not match the requested dataset")
    if not artifacts.config["normalize_embeddings"]:
        raise ValueError("Cosine IndexFlatIP requires normalized document embeddings; rebuild with normalization")
    index = faiss.IndexFlatIP(artifacts.embeddings.shape[1])
    index.add(np.ascontiguousarray(artifacts.embeddings))
    bundle = FaissIndex(
        index=index,
        doc_ids=list(artifacts.doc_ids),
        config=dict(artifacts.config),
        metadata={
            "schema_version": INDEX_SCHEMA_VERSION,
            "index_type": "IndexFlatIP",
            "metric": "inner_product",
            "similarity": "cosine",
            "dataset": artifacts.metadata["dataset"],
            "split": artifacts.metadata["split"],
            "embedding_dimension": index.d,
            "num_documents": index.ntotal,
            "normalize_embeddings": True,
            "model_name": artifacts.config["model_name"],
            "model_revision": artifacts.config["model_revision"],
            "corpus_sha256": artifacts.metadata["corpus_sha256"],
            "source_files_sha256": dict(artifacts.metadata["files_sha256"]),
            "created_at": datetime.now(UTC).isoformat(),
            "faiss_version": faiss.__version__,
        },
    )
    bundle.validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        faiss.write_index(index, str(staging / "faiss.index"))
        (staging / "doc_ids.json").write_text(json.dumps(bundle.doc_ids, indent=2), encoding="utf-8")
        (staging / "config.json").write_text(json.dumps(bundle.config, indent=2), encoding="utf-8")
        bundle.metadata["files_sha256"] = {name: _sha256(staging / name) for name in INDEX_FILES}
        (staging / "metadata.json").write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Index directory already exists: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return bundle


def load_faiss_index(directory: str | Path) -> FaissIndex:
    """Load a standalone index bundle; the source embedding bundle is not needed."""
    directory = Path(directory)
    for name in (*INDEX_FILES, "metadata.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Incomplete FAISS index bundle: missing {directory / name}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("files_sha256"), dict):
        raise ValueError("FAISS index metadata must contain file checksums")
    # Check file integrity before entering FAISS's native deserializer.
    for name in INDEX_FILES:
        if metadata["files_sha256"].get(name) != _sha256(directory / name):
            raise ValueError(f"FAISS index checksum mismatch: {name}")
    try:
        index = faiss.read_index(str(directory / "faiss.index"))
    except RuntimeError as error:
        raise ValueError("Invalid FAISS index file") from error
    bundle = FaissIndex(
        index=index,
        doc_ids=json.loads((directory / "doc_ids.json").read_text(encoding="utf-8")),
        config=json.loads((directory / "config.json").read_text(encoding="utf-8")),
        metadata=metadata,
    )
    bundle.validate()
    return bundle
