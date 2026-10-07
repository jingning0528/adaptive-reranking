# Method 2: strong-reranker feasibility test

## Question

Before implementing adaptive model routing, test whether a preselected larger
reranker actually improves over the cheap reranker on the same candidates.
This experiment contains no router, threshold, uncertainty rule, or training.

## Frozen comparison

- Validation data: the same 162 frozen SciFact queries.
- Candidates: the same 100 documents per query from the Method 1 cache.
- Cheap reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, reconstructed from
  the complete MPS teacher-score cache.
- Larger reranker: `BAAI/bge-reranker-base`, revision
  `2cfc18c9415c912f9d8155881c133215df768a70`, on MPS.
- Both rank `[query, title + space + abstract]` inputs truncated to 256 tokens.
- The larger reranker, model revision, and all settings were fixed before
  validation qrels were evaluated. No alternative strong model was tried on
  these 162 queries.

## Result

| Reranker | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| MiniLM-L6 | 0.73644 | 0.85041 |
| BGE reranker base | 0.68621 | 0.81337 |

The paired query-level BGE-minus-MiniLM NDCG@10 difference is `-0.05023`.
The 95% percentile bootstrap interval from 10,000 samples is
`[-0.09056, -0.01094]`: it excludes zero in the harmful direction. BGE
improves 22 queries, leaves 101 unchanged, and worsens 39.

For Recall@10, the difference is `-0.03704` with 95% CI
`[-0.07922, 0.00309]`; 4 queries improve, 147 are unchanged, and 11 worsen.
All 162 per-query NDCG@10 and Recall@10 values are stored in
`results/metrics/method2/strong_reranker_metrics.json`.

Scoring 16,200 pairs took 1,408.1 seconds on MPS after model loading. This is
an observed offline run time, not a controlled latency benchmark against
MiniLM, whose scores were read from cache.

## Decision

This model pair fails the routing prerequisite: the larger reranker is both
more expensive and significantly worse in NDCG@10 under the frozen SciFact
setup. A router between these two models is therefore not justified.

The result does not prove that model routing in general is unpromising. It
shows that model size and a generic reranker label are not enough; a future
strong model would need independent justification for scientific retrieval
and must be selected without repeatedly optimizing against the frozen
validation set. Until such a model is defined with a clean selection protocol,
Steps 2–6 of the routing proposal should remain unimplemented.
