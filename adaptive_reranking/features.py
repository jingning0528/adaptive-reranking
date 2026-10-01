import re

import numpy as np

FEATURE_NAMES = [
    "retrieval_score",
    "rank",
    "reciprocal_rank",
    "score_gap_top1",
    "query_document_jaccard",
    "query_document_recall",
    "query_title_recall",
    "query_token_count",
    "title_token_count",
    "document_token_count",
    "query_number_count",
    "query_acronym_count",
    "all_numbers_present",
    "all_acronyms_present",
    "number_match_fraction",
    "acronym_match_fraction",
]

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
ACRONYM_PATTERN = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9-]*|[A-Za-z]+\d+[A-Za-z0-9/-]*)\b")


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _fraction_present(items: set[str], text: str) -> float:
    if not items:
        return 1.0
    lowered = text.lower()
    return sum(item.lower() in lowered for item in items) / len(items)


def candidate_features(
    query: str,
    document: dict[str, str],
    retrieval_score: float,
    rank: int,
    top_score: float,
) -> list[float]:
    title = document.get("title", "")
    body = document.get("text", "")
    full_text = (title + " " + body).strip()
    query_tokens = tokens(query)
    title_tokens = tokens(title)
    document_tokens = tokens(full_text)
    query_set = set(query_tokens)
    title_set = set(title_tokens)
    document_set = set(document_tokens)
    overlap = query_set & document_set
    union = query_set | document_set
    numbers = set(NUMBER_PATTERN.findall(query))
    acronyms = set(ACRONYM_PATTERN.findall(query))
    number_fraction = _fraction_present(numbers, full_text)
    acronym_fraction = _fraction_present(acronyms, full_text)
    return [
        retrieval_score,
        float(rank),
        1.0 / rank,
        top_score - retrieval_score,
        len(overlap) / len(union) if union else 0.0,
        len(overlap) / len(query_set) if query_set else 0.0,
        len(query_set & title_set) / len(query_set) if query_set else 0.0,
        float(len(query_tokens)),
        float(len(title_tokens)),
        float(len(document_tokens)),
        float(len(numbers)),
        float(len(acronyms)),
        float(number_fraction == 1.0),
        float(acronym_fraction == 1.0),
        number_fraction,
        acronym_fraction,
    ]


def feature_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["features"] for row in rows], dtype=np.float32)
