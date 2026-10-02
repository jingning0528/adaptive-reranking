# Research Findings: Efficient Neural Reranking

## 1. Research Motivation

Cross-encoder reranking can improve retrieval quality, but it introduces
substantial inference cost because every query-document pair must be scored
independently.

This project investigates whether reranking computation can be reduced or
allocated more effectively while preserving retrieval quality. The work so
far tests three possible bottlenecks:

1. too many candidates are sent to the reranker;
2. the baseline reranker is too weak;
3. relevant evidence is lost through input truncation.

Each direction is treated as a controlled feasibility test. Negative results
are retained because they narrow the research question and prevent additional
complexity from being built on unsupported assumptions.

## 2. Experimental Setup

- Dataset: SciFact.
- Retrieval model: `sentence-transformers/all-MiniLM-L6-v2` bi-encoder.
- Candidate pool: the same retrieved Top-100 documents per query.
- Baseline reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Default input: `[query, title + space + abstract]`, truncated to 256 tokens.
- Primary metrics: NDCG@10 and Recall@10.
- Development split:
  - predictor training: 522 queries;
  - calibration: 125 queries;
  - frozen validation: 162 queries.

Model revisions, input contracts, candidate pools, and scoring devices are
recorded in configuration fingerprints. Predictor fitting uses only the 522
training queries, hyperparameter selection uses only the 125 calibration
queries, and the 162 validation queries are reserved for evaluation.

The MiniLM Top-100 validation reference is:

| Metric | Score |
| --- | ---: |
| NDCG@10 | 0.73644 |
| Recall@10 | 0.85041 |

The diagnostic Top-100 oracle reaches NDCG@10 = 0.94206 by placing judged
relevant candidates first. This oracle is not an implementable method; it only
shows that relevant documents are usually present in the candidate pool.

## 3. Direction 1 — Candidate Selection

### Hypothesis

Can a cheap predictor identify a small subset of candidates worth scoring with
the cross-encoder, preserving Top-100 quality with only 20 teacher evaluations?

### Method

Four policies were compared from cached MiniLM teacher scores:

- fixed Top-20 from the original retrieval ranking;
- predicted Top-20 using the mean score of seven tree predictors;
- uncertainty-aware Top-20 using `mu + lambda * u`, where `u` is predictor
  disagreement and `lambda` is selected on calibration queries;
- full Top-100 teacher reranking as the quality reference.

The predictors use inexpensive retrieval, rank, lexical-overlap, length,
number, and acronym features. They predict teacher scores rather than qrels.

### Results

| Method | Teacher scores/query | NDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Fixed Top-20 | 20 | 0.73184 | 0.83601 |
| Predicted Top-20 | 20 | 0.73501 | 0.83807 |
| Uncertainty Top-20 (`lambda = 1.0`) | 20 | 0.73501 | 0.83807 |
| Full Top-100 | 100 | 0.73644 | 0.85041 |

Predicted Top-20 minus fixed Top-20 has an NDCG@10 difference of `+0.00317`
with a paired bootstrap 95% confidence interval of
`[-0.01879, 0.02837]`. It improves 6 queries, leaves 143 unchanged, and
worsens 13.

Although calibration selected `lambda = 1.0`, uncertainty-aware selection
produces exactly the same final Top-10 rankings as predicted Top-20 on all 162
validation queries. Predictor disagreement therefore adds no observed ranking
value in this experiment.

### Finding

Using 20% of the teacher-score budget retains nearly all Top-100 NDCG@10:
predicted Top-20 is only 0.00143 below the full Top-100 reference. However,
learned candidate selection does not demonstrate a reliable advantage over
fixed Top-20, and the uncertainty heuristic provides no additional benefit.

This is a cached-score simulation. An 80% reduction in teacher evaluations is
not equivalent to an 80% end-to-end latency reduction.

### Decision

Do not add further complexity to the current uncertainty-selection rule. Keep
the strong budget-efficiency observation and measure it systematically across
multiple candidate budgets.

## 4. Direction 2 — Strong-Model Routing

### Hypothesis

Can difficult queries be routed from MiniLM-L6 to a larger, more expensive
reranker that provides higher quality?

Before implementing a router, the required feasibility test is whether the
larger model is actually better on the identical validation candidates.

### Method

`BAAI/bge-reranker-base` was preselected as the larger reranker. It was
evaluated on the same 162 queries and Top-100 candidates, using the same input
format, 256-token limit, evaluation pipeline, and MPS scoring device. No router
or routing signal was implemented.

### Results

| Reranker | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| MiniLM-L6 | 0.73644 | 0.85041 |
| BGE reranker base | 0.68621 | 0.81337 |

The BGE-minus-MiniLM NDCG@10 difference is `-0.05023`, with a paired
bootstrap 95% confidence interval of `[-0.09056, -0.01094]`. BGE improves 22
queries, leaves 101 unchanged, and worsens 39.

### Finding

BGE-base is larger and more expensive but significantly worse under this
SciFact setup. Model size alone does not solve the gap to the oracle. The
22 improved queries are useful for error analysis, but they are insufficient
to justify a router when 39 queries become worse and average quality declines.

This result applies to the tested model pair and does not establish that every
possible stronger or domain-specific reranker would fail.

### Decision

Do not implement model routing with MiniLM-L6 and BGE-base. A future routing
experiment first requires an independently justified model with a demonstrated
quality advantage under a clean selection protocol.

## 5. Direction 3 — Adaptive Context Length

### Hypothesis

Is reranking quality limited by truncating query-document inputs to 256
tokens? If so, does increasing the limit to 512 recover missing evidence?

### Method

MiniLM-L6 was evaluated at 256 and 512 tokens while holding the validation
queries, Top-100 candidates, model revision, device, input format, and
evaluation pipeline fixed.

Tokenizer analysis shows:

- 12,682 of 16,200 pairs (78.28%) exceed 256 tokens;
- 1,750 pairs (10.80%) exceed 512 tokens;
- every validation query has more than half of its candidates exceed 256.

### Results

| Maximum length | NDCG@10 | Recall@10 |
| ---: | ---: | ---: |
| 256 | 0.73644 | 0.85041 |
| 512 | 0.74060 | 0.84115 |

The NDCG@10 difference is `+0.00416`, with a paired bootstrap 95% confidence
interval of `[-0.01716, 0.02551]`. Thirteen queries improve, 131 are
unchanged, and 18 worsen. Recall@10 changes by `-0.00926`, with confidence
interval `[-0.03395, 0.01235]`.

### Diagnostic Analysis

Predefined qrels-based groups show heterogeneous effects:

| Query group | Queries | Mean NDCG@10 change | 95% CI |
| --- | ---: | ---: | ---: |
| Has a truncated relevant candidate | 129 | +0.01519 | [-0.00780, 0.04056] |
| No truncated relevant candidate | 33 | -0.03893 | [-0.08358, -0.00384] |

The difference between these subgroup changes is `+0.05412`, with bootstrap
confidence interval `[0.01185, 0.10576]`. This indicates heterogeneous length
effects: extra context is relatively more useful when it reaches a relevant
candidate that was previously truncated.

However, the first subgroup's absolute confidence interval still includes
zero. The grouping also uses qrels and is therefore diagnostic rather than an
inference-time routing signal. The overall fraction of long candidates does
not show a useful quality gradient.

Among 149 relevant candidates longer than 256 tokens, 20 move up, 105 remain
at the same rank, and 24 move down at 512. Only one enters the Top-10 while two
leave it. Recovering additional text is therefore not uniformly beneficial.

### Finding

Longer context can affect individual queries, particularly when relevant
evidence was truncated, but 512 tokens do not provide a reliable global
improvement. More queries worsen than improve, and Recall@10 declines.

### Decision

Do not globally increase the context limit to 512. Adaptive text budgets remain
a possible research direction only if a qrels-free signal can predict when the
unseen document tail contains useful evidence.

## 6. What We Learned

Three simple explanations for the gap between MiniLM and the oracle were
tested:

1. smarter candidate selection;
2. a larger reranker;
3. longer input context.

None produced a sufficiently reliable quality improvement to justify the
proposed additional complexity:

- learned Top-20 selection did not reliably beat fixed Top-20;
- BGE-base was significantly worse than MiniLM-L6;
- 512 tokens produced an uncertain NDCG gain and lower Recall@10.

Direction 1 nevertheless reveals the clearest efficiency result: scoring only
20 of 100 candidates retains nearly all NDCG@10 in the cached simulation. The
important next question is therefore not yet how to maximize reranking quality,
but how much reranking computation is required to preserve it.

## 7. Direction D — Fixed Budget-Quality Characterization

### Hypothesis

Before inventing an adaptive policy, how much full Top-100 reranking quality
can be retained at fixed budgets `K = {5, 10, 20, 30, 50, 100}`?

The no-learning policy reranks the retrieval Top-K with cached MiniLM scores
and preserves all remaining candidates in their original retrieval order.

### Results

| K | Score budget | NDCG@10 | Recall@10 |
| ---: | ---: | ---: | ---: |
| 5 | 5% | 0.72479 | 0.81749 |
| 10 | 10% | 0.72785 | 0.81749 |
| 20 | 20% | 0.73184 | 0.83601 |
| 30 | 30% | 0.72964 | 0.83189 |
| 50 | 50% | 0.72978 | 0.83807 |
| 100 | 100% | 0.73644 | 0.85041 |

K=20 retains 99.38% of K=100 NDCG@10 with 20% of the teacher evaluations.
Its paired NDCG difference from K=100 is `-0.00460`, with 95% CI
`[-0.03048, 0.01754]`. This is strong descriptive retention, not proof of
statistical equivalence.

The curve is non-monotonic: K=30 and K=50 score below K=20. Scoring more
candidates can expose useful documents, but it can also let additional
non-relevant candidates displace relevant documents. Budget requirements are
therefore query-dependent rather than a simple monotonic saturation process.

A diagnostic qrels-based budget oracle selects a mean K of 7.62 and reaches
NDCG@10 = 0.78229. This is an optimistic, non-deployable upper bound, but it
shows substantial headroom for Direction C if budget can be predicted without
relevance labels.

### Decision

Use this fixed curve as the required baseline for Direction C. Any adaptive
method must outperform fixed K at the same mean teacher-score budget and must
separate score-count savings from measured end-to-end speedup.

## 8. New Research Direction — Budget-Aware Reranking

### Research Question

How should a fixed inference budget be allocated to maximize retrieval quality,
and what is the smallest budget that preserves the full-reranking result?

### Phase 1 — Budget-Quality Characterization (complete)

Evaluate fixed candidate budgets:

```text
K = {5, 10, 20, 30, 50, 100}
```

For every budget, measure:

- NDCG@10;
- Recall@10;
- paired query-level change relative to Top-100;
- number and fraction of cross-encoder evaluations;
- eventually, controlled wall-clock latency and throughput.

The empirical fixed-budget curve is now established. Cached scores characterize
ranking quality, but real inference runs are still required for latency claims
because batching, feature computation, and framework overhead matter.

### Phase 2 — Budget-Aware Allocation

Only after establishing the fixed-budget curve, investigate whether different
queries require different budgets:

```text
K(q) in {5, 10, 20, 30, 50, 100}
```

Additional computation should be allocated only when it is likely to change
the final Top-10. Candidate signals must be available without qrels, calibrated
on the calibration split, and evaluated once on frozen validation data.

Required comparisons include:

- adaptive allocation versus fixed K at the same average score budget;
- quality loss versus Top-100 at matched latency;
- score-count savings versus actual end-to-end speedup.

### C1 oracle-budget analysis

The first analysis of the Direction D oracle labels finds K=5 for 143 of 162
queries, K=10 for 7, K=20 for 7, K=30 for 2, K=50 for 1, and K=100 for 2.
However, 120 queries tie at every tested budget and 142 have more than one
best budget. The label is therefore dominated by budget-insensitive queries.

Among qrels-free signals, a smaller bi-encoder Top-1/Top-10 score margin is the
strongest indicator that a query's oracle K exceeds 5. Its exploratory AUC is
0.788 with bootstrap CI [0.697, 0.868]. Top-10 score dispersion provides nearly
the same separation; lexical candidate diversity and query length do not.

This supports testing a simple margin policy before any learned router. Because
validation qrels were used to inspect signals and select the margin hypothesis,
the existing 162-query split is now exploratory for Direction C and cannot
serve as untouched final evaluation for that policy.

### C2 frozen adaptive-budget evaluation

A two-tier policy was fixed before official-test evaluation. It uses the
bi-encoder Top-1/Top-10 margin, routes easy queries to K=5, and routes hard
queries to a higher budget selected on 522 training queries. Thresholds are
set from 125 calibration-query margin quantiles; the 162 exploration queries
are excluded.

All four tested operating points improve mean test NDCG over their target
fixed-K baselines. At mean K=32.45, adaptive reranking reaches NDCG@10 =
0.69065 versus 0.68165 for full K=100. The paired difference is +0.00900 with
95% CI [0.00150, 0.01721], while the score count is 67.5% lower. At mean
K=54.72, the difference from full K=100 is +0.01232 with CI
[0.00558, 0.01966].

This is the first out-of-sample evidence that retrieval confidence can move
the quality-cost frontier without an additional learned router. It also
confirms that adaptive budgeting can avoid harmful over-reranking, since the
fixed quality curve is non-monotonic.

The official test split is now closed for this policy. The next experiment is
controlled end-to-end timing of the frozen operating points; any policy change
requires a new evaluation dataset or split.

## 9. Current Research Position

The evidence suggests that the primary opportunity is not yet improving the
maximum ranking quality. It is reducing the computation required to retain the
quality already achieved by MiniLM-L6.

The fixed budget-quality curve is the reference for adaptive allocation, and
the simple margin policy now has positive official-test evidence. The next
defensible experiment is controlled end-to-end timing of the frozen policies,
including dynamic batching and routing overhead. Score-count reductions must
not be reported as latency reductions until that measurement is complete.
