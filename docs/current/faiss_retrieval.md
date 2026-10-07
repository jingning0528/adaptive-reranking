# Persistent FAISS retrieval — V2 Step 3

```text
Saved Step 2 vectors + ordered document IDs
                  ↓
            IndexFlatIP builder
                  ↓
     standalone persistent index bundle
                  ↓
Query → query encoder → normalized vector → FAISS search → IDs + scores
```

The document encoder, saved vector artifacts, and search backend have separate
responsibilities. `faiss_index.py` builds, persists, validates, and searches
vectors. It never constructs a model or loads a dataset. `faiss_backend.py`
orchestrates an injected query encoder and returns a reusable result format.
The CLI constructs the query model from the saved contract.

## Exact cosine search

Only CPU `IndexFlatIP` is supported in this step. It performs exhaustive inner
product search; with normalized document and query vectors this corresponds
to cosine similarity. See the [FAISS index reference](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
and [metric guidance](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

The builder requires a normalized [Step 2 bundle](embedding_pipeline.md).
Unnormalized bundles are refused rather than interpreted as cosine vectors.
Queries are converted to contiguous float32 and normalized before search.
Zero vectors remain zero, as in Step 2. Top-K is capped at corpus size.
Equal scores are ordered by document ID; a tied cutoff expands the candidate
set to include all boundary ties before selecting K. This may require another
Flat search when ties occur. Float scores are compared with numerical tolerance
in the tests; arbitrary near-tie bitwise equality across implementations is
not claimed.

## Build and load

```python
from adaptive_reranking.retrieval.faiss_index import build_faiss_index, load_faiss_index

build_faiss_index(
    "results/cache/embeddings/scifact",
    "results/cache/faiss/scifact",
)
index = load_faiss_index("results/cache/faiss/scifact")
```

The source bundle is validated before constructing the index. Its saved row
order is retained exactly; FAISS row `i` maps to `doc_ids[i]`. No document
encoding is repeated. The output contains:

| File | Content |
| --- | --- |
| `faiss.index` | Serialized CPU `IndexFlatIP` with all document vectors |
| `doc_ids.json` | Unique sorted document IDs aligned to index rows |
| `config.json` | Copy of the document encoder configuration |
| `metadata.json` | Index schema/type/metric, dataset/split, dimensions/count, model/revision, normalization, source checksums/corpus fingerprint, creation time, FAISS version, and index-bundle checksums |

The builder stages a complete bundle before publication and refuses an existing
destination. Loading verifies file checksums before native deserialization,
then checks index type, metric, dimension, document count, ID alignment, and
metadata/configuration consistency. Missing files raise `FileNotFoundError`;
corrupt or inconsistent bundles raise `ValueError`.

The index bundle is self-contained. Loading and vector search require neither
the original embedding files nor a dataset or Transformer library import.
The retrieval layer treats published indexes as immutable; rebuilding produces
a new bundle with its own validated ID mapping.

## Query encoding and retrieval

```python
from sentence_transformers import SentenceTransformer

from adaptive_reranking.retrieval.embeddings import EmbeddingConfig
from adaptive_reranking.retrieval.faiss_backend import FaissRetriever

config = EmbeddingConfig(**index.config)
encoder = SentenceTransformer(config.model_name, revision=config.model_revision, device=config.device)
if config.max_length is not None:
    encoder.max_seq_length = config.max_length
retriever = FaissRetriever(index, encoder, config)
ranked = retriever.search("Does exercise reduce cardiovascular risk?", top_k=100)
results = retriever.retrieve({"q1": "query text"}, top_k=100, batch_size=32)
```

`search` returns a ranked list of `(document_id, score)` pairs. `retrieve` returns
`dict[query_id, dict[document_id, score]]`, accepted by the existing reranking
and evaluation interfaces. It encodes bounded query batches and excludes
query/document ID self-matches, matching BEIR retrieval semantics.

The supplied encoder must use the recorded model, revision, and truncation
contract. The retriever rejects a mismatched configuration; it cannot inspect
an arbitrary injected encoder's weights. Query encoding device and batch size
may differ from document encoding. Vector search stays on CPU.

## CLI

```bash
python -m scripts.build_faiss_index --dataset scifact
python -m scripts.search_faiss --dataset scifact --query "query text" --top-k 100
```

The build CLI defaults to source `results/cache/embeddings/<dataset>/` and
destination `results/cache/faiss/<dataset>/`. Override them with
`--embeddings-dir` and `--output-dir`. Search accepts `--index-dir` and
`--device`; the model/revision/length comes from the persisted configuration.
Dataset names are checked against bundle provenance without loading a dataset.
Output is JSON document IDs and scores. Neither CLI recomputes corpus vectors.

## Validation and next step

Tests use real CPU FAISS with tiny fake embeddings/query encoders. They cover
build/load, row reconstruction/alignment, Top-K, deterministic ties, dimensions,
normalization, invalid inputs/configuration, missing/corrupt indexes, failed
writes, CLI operation for both datasets, and loading without source embeddings
or model library imports. Controlled non-tied rankings match NumPy brute force
and the current BEIR exact dense retrieval, with score tolerance of `1e-6`.

V1 results, settings, and retrieval scripts remain unchanged. This step contains
no IVF/HNSW, approximate search, or new latency/scalability claims. Step 4 is
the exact-versus-FAISS benchmark on a larger dataset, followed by ANN tradeoffs.
