# Direction C2: retrieval-confidence adaptive budgets

## Research question

Can a simple retrieval-confidence signal allocate cross-encoder scoring budget
per query, reducing inference work while preserving or improving ranking
quality?

C1 identified the bi-encoder Top-1 minus Top-10 score margin as the strongest
cheap signal. C2 tests only that signal with a two-tier policy:

```text
if margin <= calibrated threshold:
    K = hard budget
else:
    K = 5
```

No neural router or additional learned model is used.

## Data protocol

- 522 training queries select the hard budget for each target average cost.
- 125 calibration queries determine the margin threshold by the required hard
  query fraction. Qrels are not used to fit the threshold.
- The 162-query exploration split used for C1 is excluded from C2 selection,
  calibration, and evaluation.
- The frozen policies are evaluated once on all 300 official SciFact test
  queries.
- Test candidates are the original bi-encoder Top-100. MiniLM-L6 revision,
  256-token input contract, MPS device, and candidate order are fingerprinted.

The official test split had previously been used for the initial bi-encoder
baseline, but it was not used to select the C2 signal, hard budgets, or
thresholds. No C2 policy was changed after observing the results below.

## Frozen test result

| Target budget | Selected easy/hard K | Actual mean K | Adaptive NDCG@10 | Fixed K NDCG@10 | Difference | 95% CI |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 5 / 50 | 10.10 | 0.68958 | 0.67700 | +0.01258 | [-0.00201, 0.02819] |
| 20 | 5 / 100 | 17.03 | 0.69185 | 0.68037 | +0.01148 | [-0.00505, 0.02822] |
| 30 | 5 / 50 | 32.45 | 0.69065 | 0.68458 | +0.00607 | [-0.00428, 0.01749] |
| 50 | 5 / 100 | 54.72 | 0.69397 | 0.68340 | +0.01058 | [0.00214, 0.01954] |

The target-50 comparison is statistically positive, but its actual test cost
is 54.72 rather than exactly 50 because a numeric threshold calibrated on a
different margin distribution does not guarantee an identical routing rate.
It should not be described as a perfectly cost-matched comparison.

At the low-cost operating point, the mean budget is 10.10: 266 test queries
use K=5 and 34 use K=50. It improves NDCG@10 by 0.01258 over fixed K=10; the
interval narrowly includes zero. Recall@10 improves from 0.78333 to 0.79833,
with a paired interval of [0.00000, 0.03167].

## Comparison with full reranking

Full K=100 obtains NDCG@10 = 0.68165 and Recall@10 = 0.80956 on the same test
cache. The fixed test curve is non-monotonic, as it was on the exploration
split.

| Adaptive mean K | NDCG@10 | Difference from K=100 | 95% CI | Score reduction |
| ---: | ---: | ---: | ---: | ---: |
| 10.10 | 0.68958 | +0.00793 | [-0.01077, 0.02599] | 89.9% |
| 17.03 | 0.69185 | +0.01020 | [-0.00695, 0.02721] | 83.0% |
| 32.45 | 0.69065 | +0.00900 | [0.00150, 0.01721] | 67.5% |
| 54.72 | 0.69397 | +0.01232 | [0.00558, 0.01966] | 45.3% |

The mean-K=32.45 and mean-K=54.72 policies significantly outperform full
K=100 while using fewer scores. This is possible because MiniLM reranking is
not monotonic in K: additional hard negatives can displace relevant documents.
Adaptive budgeting acts both as compute allocation and as protection against
harmful over-reranking.

![C2 quality-cost curve](../../results/figures/method6/quality_cost_curve.svg)

## Finding

C2 supplies the first out-of-sample support for the central project claim:
retrieval confidence can guide a very simple adaptive scoring budget that
moves the quality-cost frontier. The method has no learned router and uses no
additional neural inference.

The strongest practical operating point depends on the service objective:

- mean K=10.10 gives the largest score reduction and improves both mean NDCG
  and Recall over fixed K=10, although its NDCG interval includes zero;
- mean K=17.03 descriptively exceeds fixed K=20 while using fewer scores;
- mean K=32.45 provides the cleanest significant comparison with full K=100,
  reducing score count by 67.5% while improving NDCG@10.

## Limitations and next step

- Score-count savings are not measured end-to-end latency savings.
- Target and actual average budgets can differ under distribution shift.
- SciFact has sparse judgments, producing many per-query metric ties.
- The official test set should now remain closed; no further C2 threshold or
  policy modification should be justified from these results.

The next step is controlled online inference timing for the already frozen
operating points, including batching overhead. Any later policy improvement
requires a new evaluation dataset or split.
