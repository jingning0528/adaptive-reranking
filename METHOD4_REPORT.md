# Direction D: fixed budget-quality curve

## Question

How many MiniLM cross-encoder scores are actually needed to retain the quality
of reranking all 100 retrieved candidates?

This experiment does not train or introduce an adaptive method. It measures
the fixed-budget reference curve needed before developing Direction C,
budget-aware reranking.

## Policy and experimental contract

For each budget `K` in `{5, 10, 20, 30, 50, 100}`:

1. take the first K documents from the frozen bi-encoder ranking;
2. reorder those K documents using cached MiniLM-L6 scores;
3. append unscored candidates in their original retrieval order.

The third step matters for K=5: it preserves a complete Top-10 while paying
for only five cross-encoder evaluations. All budgets use the same 162 frozen
SciFact validation queries, candidate pool, MiniLM revision, 256-token input,
and MPS teacher cache.

The previously reported `0.73501` at K=20 belongs to learned predicted
selection. The no-learning fixed Top-20 reference used here is `0.73184`.

## Fixed-budget results

| K | Relative score budget | NDCG@10 | % of K=100 NDCG | Recall@10 | % of K=100 Recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 5% | 0.72479 | 98.42% | 0.81749 | 96.13% |
| 10 | 10% | 0.72785 | 98.83% | 0.81749 | 96.13% |
| 20 | 20% | 0.73184 | 99.38% | 0.83601 | 98.31% |
| 30 | 30% | 0.72964 | 99.08% | 0.83189 | 97.82% |
| 50 | 50% | 0.72978 | 99.10% | 0.83807 | 98.55% |
| 100 | 100% | 0.73644 | 100.00% | 0.85041 | 100.00% |

K=20 is the best tested limited fixed budget for NDCG@10. It uses 20% of the
teacher scores and retains 99.38% of full-reranking NDCG@10 and 98.31% of
Recall@10. Its NDCG difference from K=100 is `-0.00460`, with paired bootstrap
95% CI `[-0.03048, 0.01754]`; 17 queries improve, 140 are unchanged, and 5
worsen.

The confidence interval including zero is not an equivalence test. The result
supports strong descriptive retention, not a statistical claim that K=20 and
K=100 are identical.

## Non-monotonicity

The empirical curve is not monotonic: K=30 and K=50 score slightly below K=20.
Adding candidates gives the reranker more opportunities to recover relevant
documents, but also lets additional topically similar non-relevant documents
displace relevant documents in the final Top-10. Therefore, a larger scoring
budget does not guarantee higher per-query or mean NDCG for this reranker.

All tested limited budgets have NDCG confidence intervals versus K=100 that
include zero. Recall is more sensitive to very small budgets: K=5 and K=10
both trail K=100 by 0.03292 on average.

## Diagnostic adaptive-budget oracle

As a Direction C headroom diagnostic, qrels were allowed to choose the best K
after observing each validation query's NDCG@10. This is not deployable and
must not be reported as a method result.

| Oracle diagnostic | Value |
| --- | ---: |
| Mean selected K | 7.62 |
| NDCG@10 | 0.78229 |
| Recall@10 | 0.88128 |

The oracle selects K=5 for 143 of 162 queries, K=10 for 7, K=20 for 7, K=30
for 2, K=50 for 1, and K=100 for 2. The smallest budget that matches or exceeds
each query's K=100 NDCG has a mean K of 7.5.

This large oracle gain is partly possible because different fixed budgets can
outperform K=100 on individual queries. It demonstrates heterogeneous budget
requirements, but it is also an optimistic upper bound selected with validation
qrels. A real adaptive policy must predict budget without relevance labels and
must be tuned only on training/calibration data.

## Decision and implication for Direction C

Direction D establishes two facts:

1. full Top-100 reranking is heavily over-provisioned for many SciFact queries;
2. the quality response to additional candidates is query-dependent and
   non-monotonic.

The next task should not immediately train a complex router. First define a
qrels-free stopping target—whether scoring more candidates is likely to alter
the final Top-10 beneficially—and evaluate simple signals against fixed K at
the same average score budget.

This remains a cached quality simulation. Score-count reduction is not an
end-to-end speedup claim; controlled online timing is still required.
