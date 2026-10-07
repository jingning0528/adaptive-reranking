from adaptive_reranking.retrieval.candidates import ordered_candidates

def margins(candidates):
    values = {}
    for query_id, scores in candidates.items():
        ranked_scores = [scores[doc_id] for doc_id in ordered_candidates(scores)]
        values[query_id] = float(ranked_scores[0] - ranked_scores[9])
    return values
