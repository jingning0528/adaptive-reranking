"""Query encoding orchestration separate from persisted vector storage/search."""

import numpy as np

from adaptive_reranking.retrieval.embeddings import DocumentEncoder, EmbeddingConfig
from adaptive_reranking.retrieval.faiss_index import FaissIndex, _positive_integer


class FaissRetriever:
    def __init__(self, index: FaissIndex, query_encoder: DocumentEncoder, encoder_config: EmbeddingConfig):
        index.validate()
        encoder_config.validate()
        for key in ("model_name", "model_revision", "max_length", "normalize_embeddings"):
            if getattr(encoder_config, key) != index.config[key]:
                raise ValueError(f"Query encoder {key} does not match the indexed document encoder")
        self.index = index
        self.query_encoder = query_encoder

    def _encode(self, queries: list[str], batch_size: int) -> np.ndarray:
        vectors = np.asarray(
            self.query_encoder.encode(
                queries,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        )
        if vectors.ndim != 2 or len(vectors) != len(queries):
            raise ValueError("Query encoder must return one vector per query")
        return vectors

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return ranked document IDs and cosine scores for one query."""
        _positive_integer(top_k, "top_k")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query text must be a nonempty string")
        return self.index.search_vectors(self._encode([query], batch_size=1), top_k)[0]

    def retrieve(self, queries: dict[str, str], top_k: int = 100, batch_size: int = 32) -> dict[str, dict[str, float]]:
        """Return BEIR-compatible results; omit query/document ID self-matches."""
        _positive_integer(top_k, "top_k")
        _positive_integer(batch_size, "batch_size")
        if not isinstance(queries, dict) or any(
            not isinstance(query_id, str) or not query_id or not isinstance(text, str) or not text.strip()
            for query_id, text in queries.items()
        ):
            raise ValueError("Queries must map nonempty string IDs to nonempty text")
        query_ids = list(queries)
        results = {}
        for start in range(0, len(query_ids), batch_size):
            batch_ids = query_ids[start : start + batch_size]
            vectors = self._encode([queries[query_id] for query_id in batch_ids], batch_size)
            ranked = self.index.search_vectors(vectors, top_k + 1)
            for query_id, candidates in zip(batch_ids, ranked, strict=True):
                results[query_id] = dict([item for item in candidates if item[0] != query_id][:top_k])
        return results
