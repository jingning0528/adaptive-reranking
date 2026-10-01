# Adaptive Reranking for Efficient Search

This project studies how to allocate a limited neural reranking budget. The
first method selects 20 candidates from a retrieved Top-100, scores those 20
once in a single cross-encoder batch, and returns their strongest Top-10.

The project is independent from BEIR. It uses
[BEIR](https://github.com/beir-cellar/beir) as an installed dependency and
evaluation framework, with SciFact as the initial benchmark.

## Current evidence

- CPU bi-encoder baseline: `NDCG@10 = 0.64508` on the SciFact test split.
- Frozen validation set: 162 queries held out from SciFact train.
- MiniLM-L6 Top-100 reference: `NDCG@10 = 0.73826` on validation.
- Top-100 oracle: `NDCG@10 = 0.94206`, indicating substantial ranking headroom.
- 78.28% of validation query-document inputs exceed 256 tokens.
- Method 1 cached simulation: predicted Top-20 reaches `NDCG@10 = 0.73501`
  versus `0.73184` for retrieval Top-20, but the paired 95% bootstrap CI
  `[-0.01879, 0.02837]` includes zero.
- Although calibration selected `lambda = 1.0`, uncertainty-aware selection
  produced exactly the same validation Top-10 rankings as predicted Top-20.
  The current disagreement heuristic therefore has no supported added value.
- Method 2 feasibility test: `BAAI/bge-reranker-base` scored `0.68621`
  NDCG@10 versus MiniLM-L6's `0.73644` on the identical validation Top-100.
  The paired difference is `-0.05023` with 95% CI
  `[-0.09056, -0.01094]`, so this model pair is a routing no-go.

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the controlled reranker
comparison and error analysis, and [METHOD1_REPORT.md](METHOD1_REPORT.md) for
the first candidate-selection result. [METHOD2_REPORT.md](METHOD2_REPORT.md)
records the strong-reranker feasibility test.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Reproduce the baselines

```bash
python run_baseline.py
python run_rerank.py
python run_validation_prepare.py
python run_validation_rerank.py tinybert_l2
python run_validation_rerank.py minilm_l6
python analyze_validation.py
```

Large datasets, model caches, retrieval runs, and teacher-score caches are
ignored by Git. Small metrics and frozen split manifests are versioned.

## Method 1: candidate-budget allocation

The quality reference is frozen in
[`configs/teacher.json`](configs/teacher.json): MiniLM-L6, Top-100 candidates,
256-token inputs. Method 1 compares three policies with exactly 20 teacher
scores per query:

1. retrieval Top-20;
2. predicted Top-20 using the ensemble mean `mu`;
3. uncertainty-aware Top-20 using `mu + lambda * u`, where `u` is ensemble
   disagreement and `lambda` is selected only on calibration queries.

Run the cached simulation in order:

```bash
python prepare_teacher_scores.py
python train_score_predictor.py
python evaluate_candidate_selection.py
```

`prepare_teacher_scores.py` is resumable because teacher scoring is the
expensive stage. A configuration fingerprint prevents resuming with a
different model revision, input contract, candidate order, or scoring device.
On Apple Silicon, the checked-in configuration uses MPS to generate the
offline teacher cache. The cached simulation does not report online latency.
This first version performs one candidate-selection decision
and one batched reranking call; iterative selection and stopping are reserved
for later work.

## Data discipline

- The 162 frozen validation queries are never used to fit predictors or tune
  `lambda`.
- The remaining 647 training queries are grouped for near-duplicates, then
  divided into predictor-training and calibration queries with a fixed seed.
- Teacher-score prediction is distinct from relevance prediction. Final
  quality is always evaluated with qrels.
- Test data is reserved for reporting fixed methods, not model selection.

## Method 2: model-routing feasibility

The first routing prerequisite test intentionally implements no router. It
scores the frozen validation Top-100 with one preselected, larger reranker and
compares it with the existing MiniLM-L6 cache under the same 256-token input
contract:

```bash
python run_strong_reranker.py
```

The tested BGE-base reranker is worse than MiniLM-L6 on this validation set,
so the current pair does not justify routing work. The script is resumable and
pins the model revision, candidate pool, split, device, and preprocessing in a
cache fingerprint.
