"""Dataset-independent inputs consumed by retrieval and reranking."""

from dataclasses import dataclass
from typing import TypedDict


class Document(TypedDict):
    title: str
    text: str


Corpus = dict[str, Document]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


@dataclass
class Dataset:
    """A named split in the common BEIR-compatible schema.

    Queries contain the selected split's judged queries, just as in BEIR.
    Loading performs validation without changing text, scores, or ordering.
    """

    name: str
    split: str
    corpus: Corpus
    queries: Queries
    qrels: Qrels

    def validate(self) -> None:
        """Reject malformed inputs before they reach model inference."""
        if any(not isinstance(value, dict) for value in (self.corpus, self.queries, self.qrels)):
            raise ValueError("Dataset corpus, queries, and qrels must be dictionaries")
        if not self.corpus or not self.queries or not self.qrels:
            raise ValueError("Dataset corpus, queries, and qrels must be nonempty")
        for doc_id, document in self.corpus.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("Corpus IDs must be nonempty strings")
            if not isinstance(document, dict) or any(
                not isinstance(document.get(field), str) for field in ("title", "text")
            ):
                raise ValueError(f"Document {doc_id!r} must have string title and text fields")
        for query_id, text in self.queries.items():
            if not isinstance(query_id, str) or not query_id or not isinstance(text, str):
                raise ValueError("Queries must map nonempty string IDs to string text")
        for query_id, judgments in self.qrels.items():
            if query_id not in self.queries:
                raise ValueError(f"Qrels refer to unknown query {query_id!r}")
            if not isinstance(judgments, dict) or not judgments:
                raise ValueError(f"Qrels for {query_id!r} must be a nonempty mapping")
            for doc_id, relevance in judgments.items():
                if doc_id not in self.corpus:
                    raise ValueError(f"Qrels refer to unknown document {doc_id!r}")
                if type(relevance) is not int:
                    raise ValueError("Qrels relevance scores must be integers")
