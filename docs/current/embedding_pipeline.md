# Offline embedding pipeline — V2 Step 2

```text
Dataset → corpus text → batched document encoder → vectors → saved bundle
                                                           ↓
                                             future exact/ANN index builder
```

The pipeline accepts the [common dataset interface](dataset_interface.md).
It encodes documents only. Index building, querying, reranking, and relevance
evaluation stay outside this layer. Existing V1 scripts retain their original
behavior and recorded results.

## Reusable API

```python
from sentence_transformers import SentenceTransformer

from adaptive_reranking.data import load_dataset
from adaptive_reranking.retrieval.artifacts import (
    load_embedding_artifacts,
    save_embedding_artifacts,
)
from adaptive_reranking.retrieval.embeddings import (
    EmbeddingConfig,
    build_document_embeddings,
)

dataset = load_dataset("scifact")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
config = EmbeddingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=32,
    normalize_embeddings=True,
    device="cpu",
    max_length=encoder.max_seq_length,
)
bundle = build_document_embeddings(dataset, encoder, config)
save_embedding_artifacts(bundle, "results/cache/embeddings/scifact")
loaded = load_embedding_artifacts("results/cache/embeddings/scifact")
```

`DocumentEncoder` is a small protocol with a SentenceTransformer-compatible
`encode` method, so tests or future encoders can use the same pipeline. The
caller constructs the encoder with the recorded model/revision/device/length
configuration; the CLI does this automatically. The core builder never loads
a model or dataset.

Document IDs are sorted lexicographically. Preprocessing is exactly
`(title + " " + text).strip()`, matching the V1 benchmark. Each call encodes at
most `batch_size` documents, including a final partial batch. The matrix is
allocated once and filled batch by batch; the completed matrix stays in memory.
Rows must be real, finite, consistently dimensioned vectors. The stored dtype
is float32. The pipeline performs optional L2 normalization itself, retaining
zero vectors as zeros without division errors.

## Artifact contract

| File | Content |
| --- | --- |
| `corpus_embeddings.npy` | Nonempty float32 matrix of shape `(num_documents, embedding_dimension)` |
| `doc_ids.json` | Unique sorted string IDs; row `i` belongs to ID `i` |
| `metadata.json` | Schema version, dataset/split, model/revision, dimension/count, dtype, normalization, preprocessing, ID ordering, UTC creation time, runtime info, corpus SHA-256, and data-file SHA-256 checksums |
| `config.json` | Model name/revision, batch size, normalization, device, effective maximum sequence length |

The corpus fingerprint hashes the ordered IDs and preprocessed document text.
The CLI records the model's effective sequence length, including its default
when no override was requested. `--model-revision` can pin weights; when omitted,
the requested revision is recorded as null and the model's default is used.

Saving stages all four files in a sibling temporary directory before publishing
the bundle. A failed write is cleaned up; existing destinations are refused.
Use a new output directory to keep multiple model/configuration variants.
Loading verifies file checksums, metadata/config consistency, dimensions,
document counts, unique IDs, ordering, dtype, finite values, and normalization.
It uses `allow_pickle=False` and needs neither a dataset loader nor an encoder.

An index backend should consume `loaded.embeddings`, `loaded.doc_ids`, and the
recorded contract. Downloading data, constructing a Transformer, preprocessing
documents, and deciding artifact storage paths belong to upstream layers.

## CLI

```bash
python -m scripts.build_embeddings \
  --dataset nfcorpus \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 32 \
  --no-download \
  --output-dir results/cache/embeddings/nfcorpus
```

`--dataset` and `--model` are required. Defaults are split `test`, batch size
32, device `cpu`, normalized vectors, and output
`results/cache/embeddings/<dataset>/`. Other options are `--data-dir`,
`--model-revision`, `--max-length`, and `--no-normalize`. `--no-download`
requires cached dataset files; model loading follows the encoder library's
usual cache behavior. The CLI refuses an existing output before loading data
or models. No index is built.

## Tests

`python -m pytest -q` exercises fake-encoder batching, row alignment,
preprocessing, normalization and zero vectors, metadata, corpus fingerprints,
save/load round trips, invalid encoder output, corrupt/incomplete bundles,
non-overwrite behavior, failed-write cleanup, and both dataset CLI options.
The test suite needs no model weights or network access. The local `readline`
startup workaround remains external to the repository.
