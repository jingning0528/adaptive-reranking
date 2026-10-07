"""Persist backend-independent embedding bundles without loading an encoder."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from adaptive_reranking.retrieval.embeddings import EmbeddingArtifacts

DATA_FILES = ("corpus_embeddings.npy", "doc_ids.json", "config.json")


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_embedding_artifacts(artifacts: EmbeddingArtifacts, directory: str | Path) -> Path:
    """Stage a complete bundle before publication; refuse an existing destination."""
    artifacts.validate()
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Artifact directory already exists: {directory}; choose a new output directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        np.save(staging / "corpus_embeddings.npy", artifacts.embeddings, allow_pickle=False)
        (staging / "doc_ids.json").write_text(json.dumps(artifacts.doc_ids, indent=2), encoding="utf-8")
        (staging / "config.json").write_text(json.dumps(artifacts.config, indent=2), encoding="utf-8")
        metadata = dict(artifacts.metadata)
        metadata["files_sha256"] = {name: _file_sha256(staging / name) for name in DATA_FILES}
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Artifact directory already exists: {directory}")
        staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return directory


def load_embedding_artifacts(directory: str | Path) -> EmbeddingArtifacts:
    """Verify bundle integrity and row alignment, without downloading or encoding."""
    directory = Path(directory)
    for name in (*DATA_FILES, "metadata.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Incomplete embedding artifact bundle: missing {directory / name}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("files_sha256"), dict):
        raise ValueError("Embedding metadata must contain file checksums")
    for name in DATA_FILES:
        if metadata["files_sha256"].get(name) != _file_sha256(directory / name):
            raise ValueError(f"Embedding artifact checksum mismatch: {name}")
    artifacts = EmbeddingArtifacts(
        embeddings=np.load(directory / "corpus_embeddings.npy", allow_pickle=False),
        doc_ids=json.loads((directory / "doc_ids.json").read_text(encoding="utf-8")),
        config=json.loads((directory / "config.json").read_text(encoding="utf-8")),
        metadata=metadata,
    )
    artifacts.validate()
    return artifacts
