# Dataset interface — V2 Step 1

```python
from adaptive_reranking.data import Dataset, load_dataset

dataset = load_dataset("scifact")
dataset = load_dataset("nfcorpus", split="test")
```

Both return `Dataset` with the same fields:

| Field | Schema |
| --- | --- |
| `name` | Registered dataset name |
| `split` | Requested `train`, `dev`, or `test` split |
| `corpus` | `dict[str, {"title": str, "text": str}]` |
| `queries` | `dict[str, str]` for the selected split's judged queries |
| `qrels` | `dict[str, dict[str, int]]`, query ID → document ID → relevance |

Retrieval and reranking consume these dictionaries without knowing the source
dataset. The BEIR adapter preserves IDs, text, graded relevance, split filtering,
and ordering; omitted or null titles become empty strings. It validates the
schema and judgment references before handing data to a model.

`base.py` defines the common format; `beir.py` handles BEIR files and downloads;
`loader.py` selects supported datasets. Future BEIR datasets can be added to
`SUPPORTED_DATASETS` after checking their schema and splits. Sources with other
formats should have an adapter returning the same `Dataset`.

## Files, downloads, and errors

`data_dir` is a cache parent. For example,
`load_dataset("nfcorpus", data_dir="results/datasets")` reads
`results/datasets/nfcorpus/corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`.
Absent datasets are downloaded with BEIR's existing download utility.

For an offline cache:

```python
dataset = load_dataset("nfcorpus", split="dev", data_dir="/path/to/cache", download=False)
```

Unknown dataset names or split names raise `ValueError` before network access.
Missing files raise `FileNotFoundError`. A populated cache with a missing split
is not silently downloaded again. Invalid schema or judgment references raise
an error instead of proceeding to inference.

## CLI and output isolation

```bash
python -m scripts.check_dataset --dataset nfcorpus --no-download
python -m scripts.run_baseline --dataset nfcorpus --split test
python -m scripts.run_rerank --dataset nfcorpus --split test
```

`check_dataset` loads and validates the real files without model inference.
Baseline and reranking accept `--data-dir` and `--output-dir`; reranking also
accepts `--baseline-dir` and requires a matching dataset/split baseline.
SciFact test defaults retain V1 paths. Other datasets and splits have separate
directories under `results/metrics/{baseline,rerank}/{dataset}/{split}/`.
Reranking uses already downloaded local data; run its baseline first.

Frozen V1 analysis and benchmark scripts still select SciFact explicitly while
delegating all data loading to `load_dataset`. Their results and settings are
unchanged.

## Validation

Run `python -m pytest -q` after installing `python -m pip install -e ".[dev]"`.
Tests use temporary BEIR-format fixtures and do not require downloads. They
cover both datasets' corpus, queries, qrels, common schema, missing titles,
split selection, invalid names, offline errors, cache reuse, and schema errors.
Integration tests exercise the real BEIR exact retrieval, reranking, and
evaluation code through both CLI entry points with deterministic model
stand-ins. They test data compatibility without training or benchmarking new
models. Real dataset loading can be checked separately with `check_dataset`.
