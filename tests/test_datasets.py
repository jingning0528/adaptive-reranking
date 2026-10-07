"""Offline adapter and end-to-end CLI tests; no model or dataset downloads."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from adaptive_reranking.data import Dataset, load_dataset
from adaptive_reranking.data.base import Document
from scripts._dataset_options import result_directory


@pytest.fixture
def data_dir(tmp_path):
    """Two deliberately different BEIR-format fixtures, including omitted titles."""
    for name, prefix in (("scifact", "s"), ("nfcorpus", "n")):
        folder = tmp_path / name
        (folder / "qrels").mkdir(parents=True)
        documents = [
            {"_id": f"{prefix}-a", "title": "Alpha", "text": "alpha evidence"},
            {"_id": f"{prefix}-b", "text": "beta evidence"},
        ]
        if name == "nfcorpus":
            documents[1]["title"] = None
        queries = [
            {"_id": f"{prefix}-q1", "text": "alpha"},
            {"_id": f"{prefix}-q2", "text": "beta"},
            {"_id": f"{prefix}-unjudged", "text": "excluded by the BEIR split"},
        ]
        for filename, rows in (("corpus.jsonl", documents), ("queries.jsonl", queries)):
            (folder / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
        (folder / "qrels/test.tsv").write_text(
            f"query-id\tcorpus-id\tscore\n{prefix}-q1\t{prefix}-a\t1\n{prefix}-q2\t{prefix}-b\t2\n"
        )
        (folder / "qrels/dev.tsv").write_text(f"query-id\tcorpus-id\tscore\n{prefix}-q2\t{prefix}-b\t2\n")
    return tmp_path


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_corpus_loads(name, data_dir):
    dataset = load_dataset(name, data_dir=data_dir, download=False)
    assert len(dataset.corpus) == 2
    assert any(document["text"] == "alpha evidence" for document in dataset.corpus.values())


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_queries_load_for_requested_split(name, data_dir):
    test = load_dataset(name, data_dir=data_dir, download=False)
    dev = load_dataset(name, split="dev", data_dir=data_dir, download=False)
    assert list(test.queries.values()) == ["alpha", "beta"]
    assert list(dev.queries.values()) == ["beta"]
    assert test.split == "test" and dev.split == "dev"


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_qrels_load_without_changing_relevance(name, data_dir):
    dataset = load_dataset(name, data_dir=data_dir, download=False)
    assert sorted(score for judgments in dataset.qrels.values() for score in judgments.values()) == [1, 2]
    assert set(dataset.qrels) == set(dataset.queries)


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_common_schema_and_missing_title(name, data_dir):
    dataset = load_dataset(name, data_dir=data_dir, download=False)
    assert isinstance(dataset, Dataset)
    assert dataset.name == name
    dataset.validate()
    for doc_id, document in dataset.corpus.items():
        assert isinstance(doc_id, str)
        assert set(document) == set(Document.__annotations__)
        assert all(isinstance(value, str) for value in document.values())
    assert any(document["title"] == "" for document in dataset.corpus.values())


@pytest.mark.parametrize("name", ["unknown", "../scifact", "", "SciFact"])
def test_invalid_dataset_fails_before_download(name, tmp_path):
    with patch("adaptive_reranking.data.beir.util.download_and_unzip") as download:
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset(name, data_dir=tmp_path)
        download.assert_not_called()


def test_invalid_split_fails_before_download(tmp_path):
    with patch("adaptive_reranking.data.beir.util.download_and_unzip") as download:
        with pytest.raises(ValueError, match="Unknown split"):
            load_dataset("scifact", split="../test", data_dir=tmp_path)
        download.assert_not_called()


def test_missing_local_dataset_does_not_download(tmp_path):
    with patch("adaptive_reranking.data.beir.util.download_and_unzip") as download:
        with pytest.raises(FileNotFoundError, match="corpus.jsonl"):
            load_dataset("nfcorpus", data_dir=tmp_path, download=False)
        download.assert_not_called()


def test_download_absent_dataset_and_reuse_cache(data_dir, tmp_path):
    with patch(
        "adaptive_reranking.data.beir.util.download_and_unzip", return_value=str(data_dir / "nfcorpus")
    ) as download:
        dataset = load_dataset("nfcorpus", data_dir=tmp_path / "fresh")
        assert dataset.name == "nfcorpus"
        assert download.call_args.args[0].endswith("/nfcorpus.zip")
    with patch("adaptive_reranking.data.beir.util.download_and_unzip") as download:
        load_dataset("nfcorpus", data_dir=data_dir)
        download.assert_not_called()


def test_missing_split_does_not_redownload_populated_dataset(data_dir):
    with patch("adaptive_reranking.data.beir.util.download_and_unzip") as download:
        with pytest.raises(FileNotFoundError, match="train.tsv"):
            load_dataset("scifact", split="train", data_dir=data_dir)
        download.assert_not_called()


@pytest.mark.parametrize("problem", ["empty-corpus", "unknown-query", "unknown-document", "noninteger-score"])
def test_malformed_beir_files_raise_clear_error(problem, data_dir):
    folder = data_dir / "scifact"
    if problem == "empty-corpus":
        (folder / "corpus.jsonl").write_text("")
    else:
        query = "unknown" if problem == "unknown-query" else "s-q1"
        document = "unknown" if problem == "unknown-document" else "s-a"
        score = "invalid" if problem == "noninteger-score" else "1"
        (folder / "qrels/test.tsv").write_text(f"query-id\tcorpus-id\tscore\n{query}\t{document}\t{score}\n")
    with pytest.raises(ValueError):
        load_dataset("scifact", data_dir=data_dir, download=False)


@pytest.mark.parametrize("problem", ["empty", "text", "query", "score", "document-reference", "query-reference"])
def test_malformed_common_schema_is_rejected(problem):
    dataset = Dataset("example", "test", {"d": {"title": "", "text": "alpha"}}, {"q": "alpha"}, {"q": {"d": 1}})
    if problem == "empty":
        dataset.corpus = {}
    elif problem == "text":
        dataset.corpus["d"]["text"] = None
    elif problem == "query":
        dataset.queries["q"] = None
    elif problem == "score":
        dataset.qrels["q"]["d"] = "1"
    elif problem == "document-reference":
        dataset.qrels["q"] = {"missing": 1}
    else:
        dataset.qrels = {"missing": {"d": 1}}
    with pytest.raises(ValueError):
        dataset.validate()


class TinyEncoder:
    """Deterministic model stand-in; real BEIR exact retrieval stays unchanged."""

    def __init__(self, *args, **kwargs):
        pass

    def encode_queries(self, queries, **kwargs):
        return np.asarray([[text.count("alpha"), text.count("beta")] for text in queries], dtype=np.float32)

    def encode_corpus(self, corpus, **kwargs):
        return self.encode_queries([document["text"] for document in corpus])


class TinyCrossEncoder:
    """Stand-in for neural weights, not for the reranking or evaluation code."""

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs, **kwargs):
        return [float(query in document) for query, document in pairs]


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_two_datasets_run_same_retrieval_and_reranking_entry_points(name, data_dir, tmp_path, monkeypatch, capsys):
    from scripts import run_baseline, run_rerank

    monkeypatch.setattr(run_baseline.models, "SentenceBERT", TinyEncoder)
    monkeypatch.setattr(run_rerank, "CrossEncoder", TinyCrossEncoder)
    baseline_dir = tmp_path / "outputs" / name / "baseline"
    rerank_dir = tmp_path / "outputs" / name / "rerank"
    common = ["--dataset", name, "--data-dir", str(data_dir), "--no-download"]
    run_baseline.main(common + ["--output-dir", str(baseline_dir)])
    run_rerank.main(common + ["--baseline-dir", str(baseline_dir), "--output-dir", str(rerank_dir)])
    for folder in (baseline_dir, rerank_dir):
        metrics = json.loads((folder / "metrics.json").read_text())
        results = json.loads((folder / "retrieval.json").read_text())
        assert metrics["dataset"] == name
        assert metrics["ndcg"]["NDCG@10"] == 1.0
        assert set(results) == set(load_dataset(name, data_dir=data_dir, download=False).queries)


def test_reranking_rejects_other_dataset_cache(data_dir, tmp_path, monkeypatch):
    from scripts import run_rerank

    (tmp_path / "retrieval.json").write_text("{}")
    (tmp_path / "metrics.json").write_text(json.dumps({"dataset": "scifact"}))
    with patch.object(run_rerank, "CrossEncoder") as model:
        with pytest.raises(ValueError, match="does not match"):
            run_rerank.main(["--dataset", "nfcorpus", "--data-dir", str(data_dir), "--baseline-dir", str(tmp_path)])
        model.assert_not_called()


def test_default_v1_paths_and_other_dataset_split_isolation():
    assert result_directory("baseline", "scifact", "test") == Path("results/metrics/baseline")
    assert result_directory("rerank", "nfcorpus", "test") == Path("results/metrics/rerank/nfcorpus/test")
    assert result_directory("baseline", "scifact", "train") == Path("results/metrics/baseline/scifact/train")


@pytest.mark.parametrize("name", ["scifact", "nfcorpus"])
def test_dataset_check_cli(name, data_dir, capsys):
    from scripts.check_dataset import main

    main(["--dataset", name, "--data-dir", str(data_dir), "--no-download"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset"] == name
    assert summary["documents"] == 2 and summary["queries"] == 2 and summary["schema_valid"]
