"""Batch document encoding independent of storage and vector-search backends."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from adaptive_reranking.data.base import Dataset, Document

PREPROCESSING = "(title + ' ' + text).strip()"
ARTIFACT_SCHEMA_VERSION = 1


class DocumentEncoder(Protocol):
    """The small SentenceTransformer-compatible surface required by the pipeline."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    batch_size: int = 32
    normalize_embeddings: bool = True
    model_revision: str | None = None
    device: str = "cpu"
    max_length: int | None = None

    def validate(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(self.normalize_embeddings) is not bool:
            raise ValueError("normalize_embeddings must be a boolean")
        if self.max_length is not None and (type(self.max_length) is not int or self.max_length <= 0):
            raise ValueError("max_length must be a positive integer or None")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str) or not self.model_revision.strip()
        ):
            raise ValueError("model_revision must be a nonempty string or None")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a nonempty string")


@dataclass
class EmbeddingArtifacts:
    """Row i in embeddings always belongs to doc_ids[i]."""

    embeddings: np.ndarray
    doc_ids: list[str]
    metadata: dict
    config: dict

    def validate(self) -> None:
        if not isinstance(self.embeddings, np.ndarray) or self.embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional NumPy array")
        rows, dimension = self.embeddings.shape
        if rows == 0 or dimension == 0 or self.embeddings.dtype != np.float32:
            raise ValueError("Embeddings must be nonempty float32 vectors")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("Embeddings must contain only finite values")
        if not isinstance(self.doc_ids, list) or any(
            not isinstance(doc_id, str) or not doc_id for doc_id in self.doc_ids
        ):
            raise ValueError("Document IDs must be a list of nonempty strings")
        if len(self.doc_ids) != rows or len(set(self.doc_ids)) != rows:
            raise ValueError("Document IDs must be unique and match the number of embedding rows")
        if not isinstance(self.metadata, dict) or not isinstance(self.config, dict):
            raise ValueError("Metadata and config must be JSON objects")
        try:
            config = EmbeddingConfig(**self.config)
        except TypeError as error:
            raise ValueError("Invalid embedding configuration") from error
        config.validate()
        expected = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "num_documents": rows,
            "embedding_dimension": dimension,
            "dtype": "float32",
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "normalize_embeddings": config.normalize_embeddings,
            "preprocessing": PREPROCESSING,
            "doc_id_order": "lexicographic",
        }
        if any(self.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Metadata does not match embedding shape, schema, or configuration")
        if self.doc_ids != sorted(self.doc_ids):
            raise ValueError("Document IDs must be in lexicographic order")
        for key in ("dataset", "split", "created_at", "corpus_sha256"):
            if not isinstance(self.metadata.get(key), str) or not self.metadata[key]:
                raise ValueError(f"Metadata must contain {key}")
        if config.normalize_embeddings:
            norms = np.linalg.norm(self.embeddings.astype(np.float64), axis=1)
            if not np.all((norms == 0) | np.isclose(norms, 1.0, atol=1e-5)):
                raise ValueError("Normalized artifacts must contain unit-length or zero vectors")


def document_text(document: Document) -> str:
    """Use the same title/body concatenation as the existing V1 benchmark."""
    return (document["title"] + " " + document["text"]).strip()


def build_document_embeddings(
    dataset: Dataset, encoder: DocumentEncoder, config: EmbeddingConfig
) -> EmbeddingArtifacts:
    """Encode bounded batches; normalization and row ordering belong to this layer.

    The caller constructs the encoder using the recorded model/device/revision
    settings. No models, datasets, or search indexes are loaded here.
    Zero vectors remain zero when normalization is requested.
    """
    config.validate()
    dataset.validate()
    doc_ids = sorted(dataset.corpus)
    corpus_hash = hashlib.sha256()
    embeddings = None
    for start in range(0, len(doc_ids), config.batch_size):
        batch_ids = doc_ids[start : start + config.batch_size]
        texts = [document_text(dataset.corpus[doc_id]) for doc_id in batch_ids]
        for doc_id, text in zip(batch_ids, texts, strict=True):
            corpus_hash.update(json.dumps([doc_id, text], ensure_ascii=False).encode("utf-8") + b"\n")
        raw = np.asarray(
            encoder.encode(
                texts,
                batch_size=config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        )
        if raw.ndim != 2 or raw.shape[0] != len(batch_ids) or raw.shape[1] == 0:
            raise ValueError("Encoder must return one nonempty vector per document")
        if raw.dtype.kind not in "fiu":
            raise ValueError("Encoder must return real numeric vectors")
        values = raw.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Encoder returned nonfinite embeddings")
        if config.normalize_embeddings:
            norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
            values = np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)
        if embeddings is None:
            embeddings = np.empty((len(doc_ids), values.shape[1]), dtype=np.float32)
        elif values.shape[1] != embeddings.shape[1]:
            raise ValueError("Encoder embedding dimension changed between batches")
        embeddings[start : start + len(batch_ids)] = values
    artifacts = EmbeddingArtifacts(
        embeddings=embeddings,
        doc_ids=doc_ids,
        config=asdict(config),
        metadata={
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "dataset": dataset.name,
            "split": dataset.split,
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "embedding_dimension": embeddings.shape[1],
            "num_documents": len(doc_ids),
            "normalize_embeddings": config.normalize_embeddings,
            "dtype": "float32",
            "created_at": datetime.now(UTC).isoformat(),
            "corpus_sha256": corpus_hash.hexdigest(),
            "preprocessing": PREPROCESSING,
            "doc_id_order": "lexicographic",
            "creation_info": {"python_version": platform.python_version(), "numpy_version": np.__version__},
        },
    )
    artifacts.validate()
    return artifacts
