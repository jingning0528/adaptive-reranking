"""Offline embedding pipeline tests; no Transformer weights or network access."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from adaptive_reranking.data import Dataset
from adaptive_reranking.retrieval.artifacts import load_embedding_artifacts, save_embedding_artifacts
from adaptive_reranking.retrieval.embeddings import EmbeddingConfig, build_document_embeddings, document_text


@pytest.fixture
def dataset():
    return Dataset(
        name="scifact",
        split="test",
        corpus={
            "c": {"title": "Gamma", "text": "gamma"},
            "a": {"title": "Alpha", "text": "alpha body"},
            "e": {"title": "Omega", "text": "omega"},
            "b": {"title": "", "text": "beta body"},
            "d": {"title": "", "text": " delta "},
        },
        queries={"q": "alpha"},
        qrels={"q": {"a": 1}},
    )


class FakeEncoder:
    max_seq_length = 256
    vectors = {
        "Alpha alpha body": [3, 4, 0],
        "beta body": [0, 2, 0],
        "Gamma gamma": [1, 2, 2],
        "delta": [0, 0, 0],
        "Omega omega": [-3, 0, 4],
    }

    def __init__(self, *args, **kwargs):
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        return np.asarray([self.vectors[text] for text in sentences], dtype=np.float64)


def test_sorted_document_ids_align_with_bounded_batches(dataset):
    encoder = FakeEncoder()
    artifacts = build_document_embeddings(dataset, encoder, EmbeddingConfig("fake", batch_size=2))
    assert artifacts.doc_ids == ["a", "b", "c", "d", "e"]
    assert [len(texts) for texts, _ in encoder.calls] == [2, 2, 1]
    assert [text for texts, _ in encoder.calls for text in texts] == list(FakeEncoder.vectors)
    for _, kwargs in encoder.calls:
        assert kwargs == {
            "batch_size": 2,
            "convert_to_numpy": True,
            "normalize_embeddings": False,
            "show_progress_bar": False,
        }
    np.testing.assert_allclose(artifacts.embeddings[0], [0.6, 0.8, 0])
    assert artifacts.embeddings.dtype == np.float32
    assert artifacts.embeddings.shape == (5, 3)


@pytest.mark.parametrize("normalize", [True, False])
def test_normalization_setting_and_zero_vectors(dataset, normalize):
    artifacts = build_document_embeddings(
        dataset, FakeEncoder(), EmbeddingConfig("fake", normalize_embeddings=normalize)
    )
    raw = np.asarray(list(FakeEncoder.vectors.values()), dtype=np.float32)
    expected = raw.copy()
    if normalize:
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        expected = np.divide(raw, norms, out=np.zeros_like(raw), where=norms != 0)
    np.testing.assert_allclose(artifacts.embeddings, expected)
    assert artifacts.metadata["normalize_embeddings"] is normalize
    assert np.isfinite(artifacts.embeddings).all()


def test_preprocessing_preserves_internal_text_whitespace():
    assert document_text({"title": " A ", "text": " B  C "}) == "A   B  C"
    assert document_text({"title": "", "text": "body"}) == "body"


def test_metadata_and_effective_encoder_config(dataset):
    config = EmbeddingConfig("fake/model", batch_size=3, model_revision="pinned", device="cpu", max_length=256)
    artifacts = build_document_embeddings(dataset, FakeEncoder(), config)
    metadata = artifacts.metadata
    assert metadata["dataset"] == "scifact" and metadata["split"] == "test"
    assert metadata["model_name"] == "fake/model" and metadata["model_revision"] == "pinned"
    assert metadata["embedding_dimension"] == 3 and metadata["num_documents"] == 5
    assert metadata["dtype"] == "float32"
    assert metadata["created_at"].endswith("+00:00")
    assert len(metadata["corpus_sha256"]) == 64
    assert metadata["creation_info"]["numpy_version"] == np.__version__
    assert artifacts.config == {
        "model_name": "fake/model",
        "batch_size": 3,
        "normalize_embeddings": True,
        "model_revision": "pinned",
        "device": "cpu",
        "max_length": 256,
    }


def test_reordered_corpus_keeps_rows_and_hash_stable(dataset):
    first = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake", batch_size=2))
    dataset.corpus = dict(reversed(list(dataset.corpus.items())))
    second = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake", batch_size=4))
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    assert first.doc_ids == second.doc_ids
    assert first.metadata["corpus_sha256"] == second.metadata["corpus_sha256"]


def test_corpus_hash_tracks_changed_document_text(dataset):
    first = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake"))
    dataset.corpus["a"]["text"] = "alpha body changed"
    encoder = FakeEncoder()
    encoder.vectors = {**encoder.vectors, "Alpha alpha body changed": [3, 4, 0]}
    second = build_document_embeddings(dataset, encoder, EmbeddingConfig("fake"))
    assert first.metadata["corpus_sha256"] != second.metadata["corpus_sha256"]


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_save_load_round_trip_without_reencoding(dataset, name, tmp_path):
    dataset.name = name
    encoder = FakeEncoder()
    artifacts = build_document_embeddings(dataset, encoder, EmbeddingConfig("fake", batch_size=2))
    directory = save_embedding_artifacts(artifacts, tmp_path / name)
    loaded = load_embedding_artifacts(directory)
    np.testing.assert_array_equal(loaded.embeddings, artifacts.embeddings)
    assert loaded.doc_ids == artifacts.doc_ids and loaded.config == artifacts.config
    assert loaded.metadata["dataset"] == name
    assert {key: value for key, value in loaded.metadata.items() if key != "files_sha256"} == artifacts.metadata
    assert len(encoder.calls) == 3
    assert {path.name for path in directory.iterdir()} == {
        "corpus_embeddings.npy",
        "doc_ids.json",
        "metadata.json",
        "config.json",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"model_name": ""},
        {"normalize_embeddings": "yes"},
        {"max_length": 0},
        {"model_revision": ""},
        {"device": ""},
    ],
)
def test_invalid_config_rejected_before_encoding(dataset, kwargs):
    encoder = FakeEncoder()
    with pytest.raises(ValueError):
        build_document_embeddings(dataset, encoder, EmbeddingConfig(**{"model_name": "fake", **kwargs}))
    assert not encoder.calls


@pytest.mark.parametrize(
    "output",
    [
        np.ones((1, 3)),
        np.ones((2, 0)),
        np.ones(2),
        np.full((2, 3), np.nan),
        np.full((2, 3), np.inf),
        np.full((2, 3), "string"),
        np.ones((2, 3), dtype=complex),
    ],
)
def test_invalid_encoder_output_rejected(dataset, output):
    encoder = FakeEncoder()
    with patch.object(encoder, "encode", return_value=output), pytest.raises(ValueError):
        build_document_embeddings(dataset, encoder, EmbeddingConfig("fake", batch_size=2))


def test_changing_dimension_between_batches_rejected(dataset):
    encoder = FakeEncoder()
    with patch.object(encoder, "encode", side_effect=[np.ones((2, 3)), np.ones((2, 4))]):
        with pytest.raises(ValueError, match="dimension changed"):
            build_document_embeddings(dataset, encoder, EmbeddingConfig("fake", batch_size=2))


def test_empty_corpus_rejected_before_encoding(dataset):
    dataset.corpus = {}
    encoder = FakeEncoder()
    with pytest.raises(ValueError, match="nonempty"):
        build_document_embeddings(dataset, encoder, EmbeddingConfig("fake"))
    assert not encoder.calls


def test_existing_bundle_is_not_overwritten(dataset, tmp_path):
    artifacts = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake"))
    directory = save_embedding_artifacts(artifacts, tmp_path / "bundle")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(FileExistsError):
        save_embedding_artifacts(artifacts, directory)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_failed_write_does_not_publish_partial_bundle(dataset, tmp_path):
    artifacts = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake"))
    with patch("adaptive_reranking.retrieval.artifacts.np.save", side_effect=OSError("write failed")):
        with pytest.raises(OSError, match="write failed"):
            save_embedding_artifacts(artifacts, tmp_path / "bundle")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("filename", ["corpus_embeddings.npy", "doc_ids.json", "config.json", "metadata.json"])
def test_incomplete_bundle_rejected(dataset, tmp_path, filename):
    artifacts = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake"))
    directory = save_embedding_artifacts(artifacts, tmp_path / "bundle")
    (directory / filename).unlink()
    with pytest.raises(FileNotFoundError, match=filename):
        load_embedding_artifacts(directory)


@pytest.mark.parametrize("filename", ["corpus_embeddings.npy", "doc_ids.json", "config.json"])
def test_tampered_data_file_rejected(dataset, tmp_path, filename):
    artifacts = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake"))
    directory = save_embedding_artifacts(artifacts, tmp_path / "bundle")
    with (directory / filename).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_embedding_artifacts(directory)


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_dimension", 4),
        ("num_documents", 6),
        ("schema_version", 999),
        ("model_name", "different"),
        ("normalize_embeddings", False),
    ],
)
def test_inconsistent_metadata_rejected(dataset, tmp_path, field, value):
    artifacts = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake"))
    directory = save_embedding_artifacts(artifacts, tmp_path / "bundle")
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Metadata does not match"):
        load_embedding_artifacts(directory)


@pytest.mark.parametrize("ids", [["a"], ["a", "a", "c", "d", "e"], ["b", "a", "c", "d", "e"]])
def test_invalid_document_alignment_rejected_even_with_updated_checksum(dataset, tmp_path, ids):
    artifacts = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake"))
    directory = save_embedding_artifacts(artifacts, tmp_path / "bundle")
    path = directory / "doc_ids.json"
    path.write_text(json.dumps(ids))
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["files_sha256"]["doc_ids.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Document IDs"):
        load_embedding_artifacts(directory)


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_cli_loads_dataset_encodes_and_saves_common_bundle(dataset, name, tmp_path, monkeypatch, capsys):
    from scripts import build_embeddings

    monkeypatch.chdir(tmp_path)
    dataset.name = name
    with patch.object(build_embeddings, "load_dataset", return_value=dataset) as loader:
        with patch("sentence_transformers.SentenceTransformer", side_effect=FakeEncoder) as model:
            build_embeddings.main(
                [
                    "--dataset",
                    name,
                    "--model",
                    "fake/model",
                    "--batch-size",
                    "2",
                    "--no-download",
                    "--model-revision",
                    "pinned",
                    "--max-length",
                    "128",
                    "--no-normalize",
                ]
            )
    loader.assert_called_once_with(name, split="test", data_dir=Path("results/datasets"), download=False)
    model.assert_called_once_with("fake/model", revision="pinned", device="cpu")
    loaded = load_embedding_artifacts(tmp_path / "results/cache/embeddings" / name)
    assert loaded.config["max_length"] == 128
    assert loaded.config["batch_size"] == 2 and loaded.config["normalize_embeddings"] is False
    np.testing.assert_array_equal(loaded.embeddings, np.asarray(list(FakeEncoder.vectors.values()), dtype=np.float32))
    assert json.loads(capsys.readouterr().out)["dataset"] == name


def test_cli_records_default_model_length_and_custom_output(dataset, tmp_path):
    from scripts import build_embeddings

    output = tmp_path / "custom"
    with patch.object(build_embeddings, "load_dataset", return_value=dataset):
        with patch("sentence_transformers.SentenceTransformer", side_effect=FakeEncoder):
            build_embeddings.main(["--dataset", "scifact", "--model", "fake", "--output-dir", str(output)])
    loaded = load_embedding_artifacts(output)
    assert loaded.config["max_length"] == 256


def test_cli_existing_output_fails_before_loading_dataset_or_model(tmp_path):
    from scripts import build_embeddings

    with patch.object(build_embeddings, "load_dataset") as loader:
        with pytest.raises(FileExistsError):
            build_embeddings.main(["--dataset", "scifact", "--model", "fake", "--output-dir", str(tmp_path)])
        loader.assert_not_called()


def test_broken_destination_symlink_is_not_replaced(dataset, tmp_path):
    destination = tmp_path / "bundle"
    destination.symlink_to(tmp_path / "missing")
    artifacts = build_document_embeddings(dataset, FakeEncoder(), EmbeddingConfig("fake"))
    with pytest.raises(FileExistsError):
        save_embedding_artifacts(artifacts, destination)
    assert destination.is_symlink()
