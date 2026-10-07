

def per_query_recall(qrels, results, k=10):
    output = {}
    for query_id, relevant in qrels.items():
        ranked = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:k]
        output[query_id] = sum(relevant.get(doc_id, 0) > 0 for doc_id, _ in ranked) / sum(
            value > 0 for value in relevant.values()
        )
    return output
