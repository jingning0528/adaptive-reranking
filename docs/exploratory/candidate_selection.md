# Method 1: cached candidate-selection result

## Experimental contract

- Dataset: SciFact train queries, with 522 predictor-training queries, 125
  calibration queries, and 162 frozen validation queries.
- Teacher: `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`, maximum length 256, scored on
  MPS from the same cache for every policy and the Top-100 reference.
- Input: `[query, title + space + abstract]`.
- Candidate pool: the retriever's ordered Top-100; selection budget: 20;
  output depth: 10.
- `lambda` was selected on calibration only. Validation was not used to fit
  predictors or select hyperparameters.

The cache fingerprint is
`05777b1272e4fd73f3c60f0a82cb9b98df1974b7503073ef0022218c110d4b12`.

## Validation result

| Policy | Teacher scores/query | NDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Retrieval Top-20 | 20 | 0.73184 | 0.83601 |
| Predicted Top-20 | 20 | 0.73501 | 0.83807 |
| Uncertainty Top-20 (`lambda=1.0`) | 20 | 0.73501 | 0.83807 |
| Full teacher Top-100 | 100 | 0.73644 | 0.85041 |

Paired query-level comparisons use 10,000 bootstrap samples and percentile
95% confidence intervals.

| Difference | Mean NDCG@10 difference | 95% CI | Improved / unchanged / worsened |
| --- | ---: | ---: | ---: |
| Predicted Top-20 - retrieval Top-20 | +0.00317 | [-0.01879, 0.02837] | 6 / 143 / 13 |
| Uncertainty Top-20 - predicted Top-20 | 0.00000 | [0.00000, 0.00000] | 0 / 162 / 0 |
| Uncertainty Top-20 - full Top-100 | -0.00142 | [-0.00853, 0.00401] | 10 / 150 / 2 |

For Recall@10, predicted Top-20 minus retrieval Top-20 is +0.00206 with a
95% CI of [-0.02881, 0.03292] (4 / 155 / 3 queries). Uncertainty Top-20 and
predicted Top-20 are identical at output depth 10.

## Interpretation

The cheap predictor has promising average quality but no statistically clear
advantage over fixed Top-20 on this validation set: the NDCG interval includes
zero and more queries worsened than improved. More importantly, the current
ensemble-disagreement heuristic adds no validation quality at all. Calibration
selected `lambda=1.0`, but uncertainty changed only 0.58 selected candidates
per query on average and never changed a final validation Top-10 ranking.

Scoring 20 rather than 100 candidates represents an 80% reduction in teacher
score count, not an 80% end-to-end speedup. These results were simulated from
cached teacher scores. They do not include online feature extraction, seven
predictor calls, selection, batching, or teacher inference, so no latency or
speedup claim is made.

The current evidence does not support adding complexity to this uncertainty
rule. The next diagnostic should test whether disagreement identifies large
prediction errors specifically near the selection boundary; a redesigned rule
should earn a validation improvement before online latency measurement.
