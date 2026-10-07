import hashlib
import math


from adaptive_reranking.data.validation import validation_query_ids

K_VALUES = [1, 3, 5, 10, 100]




def query_ndcg_at_k(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k: int = 10,
) -> dict[str, float]:
    scores = {}
    for query_id, relevant_docs in qrels.items():
        ranked = sorted(
            results.get(query_id, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        gains = [relevant_docs.get(doc_id, 0) for doc_id, _ in ranked]
        ideal = sorted(relevant_docs.values(), reverse=True)[:k]
        dcg = sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(gains))
        idcg = sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal))
        scores[query_id] = dcg / idcg if idcg else 0.0
    return scores


def oracle_ranking(
    qrels: dict[str, dict[str, int]],
    candidates: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Rank judged relevant candidates first, retaining baseline order within ties."""
    oracle = {}
    for query_id, candidate_scores in candidates.items():
        baseline_ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        count = len(baseline_ranked)
        oracle[query_id] = {
            doc_id: qrels[query_id].get(doc_id, 0) * 1_000_000 + count - rank
            for rank, (doc_id, _) in enumerate(baseline_ranked)
        }
    return oracle
