# Current evaluation

The adaptive policy is selected on training and calibration queries, frozen, and evaluated on the official SciFact test split. Validation queries are held out from predictor fitting and policy tuning. No policy tuning is performed on the official test split.

Run from the repository root:

```bash
python -m scripts.prepare_c2_test_scores
python -m scripts.evaluate_adaptive_budget
python -m scripts.benchmark_end_to_end
```

Metrics are in [`results/metrics/method6/`](../../results/metrics/method6/) and [`results/metrics/method7/`](../../results/metrics/method7/), manifests in [`results/manifests/`](../../results/manifests/), and figures in [`results/figures/`](../../results/figures/).

See the [adaptive-budget report](adaptive_budget.md) and [paper](../paper.md) for results and statistical comparisons, and the [validation report](../validation_report.md) for the controlled reranker comparison.
