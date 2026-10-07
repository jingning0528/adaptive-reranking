"""Use real CPU FAISS with tiny fake document/query encoders; no model downloads."""

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pytest

from adaptive_reranking.data import Dataset
from adaptive_reranking.retrieval.artifacts import load_embedding_artifacts, save_embedding_artifacts
from adaptive_reranking.retrieval.embeddings import EmbeddingConfig, build_document_embeddings, document_text
from adaptive_reranking.retrieval.faiss_backend import FaissRetriever
from adaptive_reranking.retrieval.faiss_index import build_faiss_index, load_faiss_index


class FakeEncoder:
    max_seq_length = 128
    vectors = {
        "A alpha": [1, 0, 0],
        "B beta": [0.8, 0.6, 0],
        "C gamma": [0.3, 0.4, np.sqrt(0.75)],
        "D delta": [-1, 0, 0],
        "E epsilon": [0, -1, 0],
        "query one": [1, 0.2, 0],
        "query two": [0.12, 1, 0.5],
    }

    def __init__(self, *args, **kwargs):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    def encode_queries(self, queries, **kwargs):
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus, **kwargs):
        return self.encode([document_text(document) for document in corpus], **kwargs)


@pytest.fixture
def source(tmp_path):
    corpus = {
        "doc-c": {"title": "C", "text": "gamma"},
        "doc-a": {"title": "A", "text": "alpha"},
        "doc-e": {"title": "E", "text": "epsilon"},
        "doc-b": {"title": "B", "text": "beta"},
        "doc-d": {"title": "D", "text": "delta"},
    }
    dataset = Dataset(
        "scifact",
        "test",
        corpus,
        {"q1": "query one", "q2": "query two"},
        {
            "q1": {"doc-a": 1},
            "q2": {"doc-c": 1},
        },
    )
    config = EmbeddingConfig("fake", batch_size=2, model_revision="pinned", max_length=128)
    artifacts = build_document_embeddings(dataset, FakeEncoder(), config)
    directory = save_embedding_artifacts(artifacts, tmp_path / "embeddings")
    return dataset, config, directory, artifacts


@pytest.fixture
def indexed(source, tmp_path):
    return build_faiss_index(source[2], tmp_path / "index")


def brute_force(vectors, ids, queries, top_k):
    queries = np.asarray(queries, dtype=np.float32)
    norms = np.linalg.norm(queries.astype(np.float64), axis=1, keepdims=True)
    queries = np.divide(queries, norms, out=np.zeros_like(queries), where=norms != 0)
    scores = queries @ vectors.T
    return [
        sorted(zip(ids, row.tolist(), strict=True), key=lambda item: (-item[1], item[0]))[:top_k] for row in scores
    ]


def assert_rankings_equal(actual, expected):
    assert [doc_id for doc_id, _ in actual] == [doc_id for doc_id, _ in expected]
    np.testing.assert_allclose([score for _, score in actual], [score for _, score in expected], atol=1e-6)


def test_build_load_index_preserves_document_row_alignment(source, tmp_path):
    _, _, directory, artifacts = source
    built = build_faiss_index(directory, tmp_path / "index")
    loaded = load_faiss_index(tmp_path / "index")
    assert isinstance(loaded.index, faiss.IndexFlatIP)
    assert loaded.doc_ids == artifacts.doc_ids == ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"]
    np.testing.assert_array_equal(loaded.index.reconstruct_n(0, loaded.index.ntotal), artifacts.embeddings)
    assert loaded.metadata == built.metadata
    assert loaded.config == artifacts.config
    assert loaded.metadata["embedding_dimension"] == 3
    assert loaded.metadata["num_documents"] == 5
    assert loaded.metadata["dataset"] == "scifact"
    assert loaded.metadata["source_files_sha256"] == load_embedding_artifacts(directory).metadata["files_sha256"]
    assert {file.name for file in (tmp_path / "index").iterdir()} == {
        "faiss.index",
        "doc_ids.json",
        "config.json",
        "metadata.json",
    }


def test_reload_search_needs_no_embedding_bundle_or_document_encoding(source, tmp_path):
    build_faiss_index(source[2], tmp_path / "index")
    shutil.rmtree(source[2])
    loaded = load_faiss_index(tmp_path / "index")
    encoder = FakeEncoder()
    results = FaissRetriever(loaded, encoder, source[1]).search("query one", 2)
    assert [doc_id for doc_id, _ in results] == ["doc-a", "doc-b"]
    assert [texts for texts, _ in encoder.calls] == [["query one"]]


def test_index_load_and_vector_search_do_not_import_model_libraries(indexed, tmp_path):
    code = """
import sys
import numpy as np
from adaptive_reranking.retrieval.faiss_index import load_faiss_index
index = load_faiss_index(sys.argv[1])
assert index.search_vectors(np.ones((2, 3), dtype=np.float32), 2)
assert 'torch' not in sys.modules
assert 'sentence_transformers' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "index")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("top_k", [1, 2, 4, 5, 100])
def test_flat_faiss_matches_numpy_brute_force(source, indexed, top_k):
    vectors = FakeEncoder().encode(["query one", "query two"])
    actual = indexed.search_vectors(vectors, top_k)
    expected = brute_force(source[3].embeddings, source[3].doc_ids, vectors, top_k)
    for left, right in zip(actual, expected, strict=True):
        assert_rankings_equal(left, right)


def test_faiss_matches_current_beir_exact_dense_retrieval(source, indexed):
    from beir.retrieval.evaluation import EvaluateRetrieval

    from beir.retrieval.search.dense import DenseRetrievalExactSearch

    dataset, config, _, _ = source
    reference = EvaluateRetrieval(DenseRetrievalExactSearch(FakeEncoder()), score_function="cos_sim").retrieve(
        dataset.corpus, dataset.queries
    )
    actual = FaissRetriever(indexed, FakeEncoder(), config).retrieve(dataset.queries, top_k=3)
    for query_id in dataset.queries:
        expected = sorted(reference[query_id].items(), key=lambda item: (-item[1], item[0]))[:3]
        assert_rankings_equal(list(actual[query_id].items()), expected)


def test_query_encoding_uses_bounded_batches(source, indexed):
    encoder = FakeEncoder()
    retriever = FaissRetriever(indexed, encoder, source[1])
    queries = {f"q{i}": "query one" for i in range(5)}
    results = retriever.retrieve(queries, top_k=1, batch_size=2)
    assert [len(texts) for texts, _ in encoder.calls] == [2, 2, 1]
    assert set(results) == set(queries)
    for result in results.values():
        assert list(result) == ["doc-a"]


def test_self_match_exclusion_matches_beir_semantics(source, indexed):
    results = FaissRetriever(indexed, FakeEncoder(), source[1]).retrieve({"doc-a": "query one"}, top_k=2)
    assert list(results["doc-a"]) == ["doc-b", "doc-c"]


def test_ties_at_top_k_boundary_are_deterministic(source, tmp_path):
    artifacts = source[3]
    artifacts.embeddings[:] = np.array([1, 0, 0], dtype=np.float32)
    directory = save_embedding_artifacts(artifacts, tmp_path / "ties")
    index = build_faiss_index(directory, tmp_path / "tie-index")
    for top_k in [1, 2, 3]:
        results = index.search_vectors(np.array([[1, 0, 0]], dtype=np.float32), top_k)[0]
        assert [doc_id for doc_id, _ in results] == artifacts.doc_ids[:top_k]


def test_zero_query_and_empty_query_matrix(indexed):
    results = indexed.search_vectors(np.zeros((1, 3), dtype=np.float32), 2)[0]
    assert results == [("doc-a", 0.0), ("doc-b", 0.0)]
    assert indexed.search_vectors(np.zeros((0, 3), dtype=np.float32)) == []


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((1, 2)),
        np.ones(3),
        np.full((1, 3), np.nan),
        np.full((1, 3), np.inf),
        np.ones((1, 3), dtype=complex),
        np.full((1, 3), "text"),
    ],
)
def test_invalid_query_vectors_rejected(indexed, vectors):
    with pytest.raises(ValueError):
        indexed.search_vectors(vectors)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_fails_before_query_encoding(source, indexed, top_k):
    encoder = FakeEncoder()
    with pytest.raises(ValueError, match="top_k"):
        FaissRetriever(indexed, encoder, source[1]).search("query one", top_k)
    assert not encoder.calls


@pytest.mark.parametrize("query", ["", "   ", None, 42])
def test_invalid_query_text(source, indexed, query):
    with pytest.raises(ValueError, match="Query text"):
        FaissRetriever(indexed, FakeEncoder(), source[1]).search(query)


def test_query_encoder_shape_and_dimension_errors(source, indexed):
    encoder = FakeEncoder()
    retriever = FaissRetriever(indexed, encoder, source[1])
    for output in [np.ones((1, 2)), np.ones(3), np.ones((2, 3))]:
        with patch.object(encoder, "encode", return_value=output), pytest.raises(ValueError):
            retriever.search("query one")


@pytest.mark.parametrize(
    "change",
    [
        {"model_name": "another"},
        {"model_revision": "another"},
        {"max_length": 256},
        {"normalize_embeddings": False},
    ],
)
def test_incompatible_query_encoder_config_rejected(source, indexed, change):
    with pytest.raises(ValueError, match="does not match"):
        FaissRetriever(indexed, FakeEncoder(), replace(source[1], **change))


def test_query_device_and_batch_size_can_differ_from_document_encoding(source, indexed):
    FaissRetriever(indexed, FakeEncoder(), replace(source[1], device="mps", batch_size=4))


def test_empty_queries_do_not_encode(source, indexed):
    encoder = FakeEncoder()
    assert FaissRetriever(indexed, encoder, source[1]).retrieve({}) == {}
    assert not encoder.calls


def test_unnormalized_embedding_bundle_rejected(source, tmp_path):
    dataset, config, _, _ = source
    artifacts = build_document_embeddings(dataset, FakeEncoder(), replace(config, normalize_embeddings=False))
    directory = save_embedding_artifacts(artifacts, tmp_path / "raw")
    with pytest.raises(ValueError, match="normalized document embeddings"):
        build_faiss_index(directory, tmp_path / "index")
    assert not (tmp_path / "index").exists()


def test_source_bundle_corruption_rejected_before_index_build(source, tmp_path):
    (source[2] / "corpus_embeddings.npy").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_faiss_index(source[2], tmp_path / "index")
    assert not (tmp_path / "index").exists()


@pytest.mark.parametrize("filename", ["faiss.index", "doc_ids.json", "config.json", "metadata.json"])
def test_missing_index_bundle_file(source, indexed, tmp_path, filename):
    (tmp_path / "index" / filename).unlink()
    with pytest.raises(FileNotFoundError, match=filename):
        load_faiss_index(tmp_path / "index")


@pytest.mark.parametrize("filename", ["faiss.index", "doc_ids.json", "config.json"])
def test_corrupted_index_bundle_rejected_before_native_read(indexed, tmp_path, filename):
    with (tmp_path / "index" / filename).open("ab") as handle:
        handle.write(b"corrupt")
    with patch("adaptive_reranking.retrieval.faiss_index.faiss.read_index") as native_read:
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_faiss_index(tmp_path / "index")
        native_read.assert_not_called()


def update_checksum(directory, filename):
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["files_sha256"][filename] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata))


def test_malformed_native_index_with_updated_checksum(indexed, tmp_path):
    directory = tmp_path / "index"
    (directory / "faiss.index").write_bytes(b"not a faiss index")
    update_checksum(directory, "faiss.index")
    with pytest.raises(ValueError, match="Invalid FAISS index file"):
        load_faiss_index(directory)


@pytest.mark.parametrize("kind", ["wrong-dimension", "wrong-count", "wrong-metric"])
def test_incompatible_native_index_rejected(indexed, tmp_path, kind):
    directory = tmp_path / "index"
    dimension = 2 if kind == "wrong-dimension" else 3
    index = faiss.IndexFlatL2(dimension) if kind == "wrong-metric" else faiss.IndexFlatIP(dimension)
    index.add(np.ones((4 if kind == "wrong-count" else 5, dimension), dtype=np.float32))
    faiss.write_index(index, str(directory / "faiss.index"))
    update_checksum(directory, "faiss.index")
    with pytest.raises(ValueError):
        load_faiss_index(directory)


@pytest.mark.parametrize("ids", [["doc-a"], ["doc-a"] * 5, ["doc-b", "doc-a", "doc-c", "doc-d", "doc-e"]])
def test_invalid_document_id_alignment(indexed, tmp_path, ids):
    directory = tmp_path / "index"
    (directory / "doc_ids.json").write_text(json.dumps(ids))
    update_checksum(directory, "doc_ids.json")
    with pytest.raises(ValueError, match="Document IDs"):
        load_faiss_index(directory)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 999),
        ("embedding_dimension", 4),
        ("num_documents", 10),
        ("model_name", "another"),
        ("normalize_embeddings", False),
    ],
)
def test_invalid_index_metadata(indexed, tmp_path, field, value):
    path = tmp_path / "index/metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata"):
        load_faiss_index(tmp_path / "index")


def test_existing_index_not_overwritten(source, indexed, tmp_path):
    directory = tmp_path / "index"
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(FileExistsError):
        build_faiss_index(source[2], directory)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_failed_index_write_cleans_staging(source, tmp_path):
    with patch("adaptive_reranking.retrieval.faiss_index.faiss.write_index", side_effect=RuntimeError("write failed")):
        with pytest.raises(RuntimeError, match="write failed"):
            build_faiss_index(source[2], tmp_path / "index")
    assert not (tmp_path / "index").exists()
    assert not list(tmp_path.glob(".index-*"))


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_build_and_search_cli_from_saved_step2_bundle(source, tmp_path, monkeypatch, capsys, name):
    from scripts import build_faiss_index as build_cli
    from scripts import search_faiss

    monkeypatch.chdir(tmp_path)
    artifacts = source[3]
    artifacts.metadata["dataset"] = name
    save_embedding_artifacts(artifacts, tmp_path / "results/cache/embeddings" / name)
    with patch("sentence_transformers.SentenceTransformer", side_effect=AssertionError("builder must not encode")):
        build_cli.main(["--dataset", name])
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset"] == name and summary["index_type"] == "IndexFlatIP"
    shutil.rmtree(tmp_path / "results/cache/embeddings")
    encoder = FakeEncoder()
    with patch("sentence_transformers.SentenceTransformer", return_value=encoder) as model:
        search_faiss.main(["--dataset", name, "--query", "query one", "--top-k", "2"])
    model.assert_called_once_with("fake", revision="pinned", device="cpu")
    result = json.loads(capsys.readouterr().out)
    assert [item["doc_id"] for item in result["results"]] == ["doc-a", "doc-b"]
    assert [texts for texts, _ in encoder.calls] == [["query one"]]
    assert encoder.max_seq_length == 128


def test_cli_wrong_dataset_rejected_before_build_or_model_encoding(source, tmp_path, capsys):
    from scripts import build_faiss_index as build_cli
    from scripts import search_faiss

    with pytest.raises(ValueError, match="dataset"):
        build_cli.main(
            ["--dataset", "nfcorpus", "--embeddings-dir", str(source[2]), "--output-dir", str(tmp_path / "bad")]
        )
    assert not (tmp_path / "bad").exists()
    build_faiss_index(source[2], tmp_path / "index")
    with patch("sentence_transformers.SentenceTransformer") as model:
        with pytest.raises(ValueError, match="dataset"):
            search_faiss.main(
                ["--dataset", "nfcorpus", "--query", "query one", "--index-dir", str(tmp_path / "index")]
            )
        model.assert_not_called()
