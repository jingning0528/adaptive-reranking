from adaptive_reranking.retrieval.candidates import ordered_candidates

def rerank(candidates, teacher_scores, budgets_by_query):
    results = {}
    for query_id, budget in budgets_by_query.items():
        retrieval_order = ordered_candidates(candidates[query_id])
        scored = retrieval_order[:budget]
        reranked = sorted(
            scored,
            key=lambda doc_id: (-teacher_scores[query_id][doc_id], doc_id),
        )
        final_order = reranked + retrieval_order[budget:]
        results[query_id] = {
            doc_id: float(len(final_order) - rank)
            for rank, doc_id in enumerate(final_order)
        }
    return results
