

def ordered_candidates(scores, limit=100):
    return [
        doc_id
        for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
