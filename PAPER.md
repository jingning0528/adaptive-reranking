# Budget-Aware Adaptive Neural Reranking for Efficient Retrieval Systems

## Abstract

Cross-encoder reranking improves retrieval but scales linearly with the number
of query-document pairs. We study whether retrieval confidence can allocate
reranking computation without an auxiliary router. On SciFact, fixed-budget
quality is non-monotonic and Top-20 retains 99.38% of Top-100 validation
NDCG@10. A simple policy routes queries using the bi-encoder Top-1 minus
Top-10 score margin: easy queries rerank 5 candidates and hard queries rerank
50. The frozen policy averages 32.45 scores per test query, improves NDCG@10
from 0.68165 to 0.69065 (paired 95% CI [0.00150, 0.01721]), and reduces measured
online time from 368.36 to 109.14 seconds for 300 queries. This is a 3.38x
speedup with no additional neural model.

## 1. Research question

How can neural reranking computation be dynamically allocated under a limited
inference budget while preserving retrieval quality?

## 2. Experimental protocol

- Dataset: SciFact.
- Retriever: `sentence-transformers/all-MiniLM-L6-v2`, Top-100 candidates.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`, 256 tokens.
- Signal: bi-encoder Top-1 minus Top-10 cosine-score margin.
- Development: 522 training queries, 125 calibration queries, 162 exploration
  queries.
- Final evaluation: 300 official SciFact test queries, evaluated once after
  policy freeze.

Training queries select the hard budget. Calibration margins set the numeric
threshold required for a target scoring rate. Exploration queries motivated
the signal but are excluded from policy selection and final evaluation.

## 3. Evidence leading to the method

Three alternatives did not justify additional complexity:

1. Learned candidate selection did not reliably beat fixed Top-20; ensemble
   disagreement added no value.
2. BGE reranker base underperformed MiniLM-L6 by 0.05023 NDCG@10.
3. Increasing MiniLM context from 256 to 512 produced an uncertain +0.00416
   NDCG change and lower Recall@10.

Fixed budgets revealed the useful structure. On validation, K=20 reached
0.73184 NDCG@10 versus 0.73644 at K=100 using one fifth of the scores. Quality
was non-monotonic in K, showing that additional hard negatives can harm the
final ranking. A qrels-based budget oracle then showed heterogeneous budget
requirements. The Top-1/Top-10 margin separated queries requiring more than
K=5 with exploratory AUC 0.788 (95% CI [0.697, 0.868]).

## 4. Method

The deployed policy is deliberately minimal:

```text
margin(q) = retriever_score_1 - retriever_score_10

if margin(q) <= 0.18132317066192627:
    K(q) = 50
else:
    K(q) = 5
```

The rule requires one subtraction after retrieval. It adds no model parameters
and no neural inference. The policy is frozen in
`configs/frozen_adaptive_policy.json` with its originating commit and quality
result.

## 5. Quality result

| Method | Mean K | NDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Fixed K=30 | 30.00 | 0.68458 | 0.81122 |
| Full K=100 | 100.00 | 0.68165 | 0.80956 |
| Adaptive K | 32.45 | **0.69065** | **0.81167** |

Adaptive minus Full-100 is +0.00900 NDCG@10 with paired bootstrap 95% CI
[0.00150, 0.01721]. The policy uses 9,735 rather than 30,000 cross-encoder
scores, a 67.55% reduction. It outperforms Full-100 because the reranker's
quality is not monotonic in candidate count: restricting easy queries prevents
mis-scored hard negatives from entering the final Top-10.

## 6. End-to-end latency

The benchmark uses a prebuilt, normalized corpus embedding index. Model loading
and index construction are offline and excluded. Each online trial includes:

1. CPU query encoding;
2. exact cosine Top-100 search;
3. margin calculation and routing;
4. text-pair construction;
5. MPS cross-encoder inference;
6. final Top-10 score aggregation and sorting.

Both strategies run twice in alternating order after warm-up. Results are
medians over the two full 300-query trials on `macOS-14.3-arm64-arm-64bit`.

| Metric | Full-100 | Adaptive |
| --- | ---: | ---: |
| Pairs scored | 30,000 | 9,735 |
| Mean K | 100.00 | 32.45 |
| Cross-encoder time | 367.38 s | 108.38 s |
| Online total time | 368.36 s | 109.14 s |
| Throughput | 0.81 queries/s | 2.75 queries/s |

Adaptive reranking provides a 3.38x online speedup and reduces total online
batch time by 70.37%. Routing itself takes roughly 0.0001 seconds per 300-query
trial; cross-encoder inference remains the dominant cost.

The offline corpus-index build took 133.90 seconds and is amortized across
queries. Raw trial timings and the benchmark contract are stored in
`results/method7/end_to_end_latency.json`.

## 7. Contributions

1. An empirical budget-quality characterization showing diminishing and
   non-monotonic returns from reranking more candidates.
2. An interpretable confidence-aware policy requiring no learned router.
3. Held-out evidence that the policy improves ranking quality while reducing
   score count by 67.5%.
4. End-to-end measurement showing that score savings translate to a 3.38x
   online speedup rather than disappearing into routing overhead.

## 8. Limitations

- Results cover one small scientific retrieval dataset and one hardware stack.
- SciFact has sparse judgments and many per-query metric ties.
- The benchmark measures batch throughput over 300 queries, not interactive
  p50/p95 single-query latency.
- Exact dense search is used; production ANN systems may change the cost mix.
- The official test split is closed for this policy. Further policy changes
  require a new dataset or untouched split.

## 9. Reproduction

```bash
python prepare_teacher_scores.py
python evaluate_budget_curve.py
python analyze_oracle_budgets.py
python prepare_c2_test_scores.py
python evaluate_adaptive_budget.py
python benchmark_end_to_end.py
```

Large datasets, model caches, corpus embeddings, and teacher-score caches are
ignored by Git. Small metrics, frozen configurations, reports, and plot-ready
curves are versioned.
