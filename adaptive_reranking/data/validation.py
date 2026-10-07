import hashlib

def validation_query_ids(qrels: dict[str, dict[str, int]], fraction: float = 0.2) -> list[str]:
    """Select a deterministic held-out subset without inspecting relevance labels."""
    ordered = sorted(
        qrels,
        key=lambda query_id: hashlib.sha256(query_id.encode("utf-8")).hexdigest(),
    )
    return sorted(ordered[: round(len(ordered) * fraction)])
