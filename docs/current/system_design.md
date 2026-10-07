# Current system design

The frozen policy in [`configs/frozen_adaptive_policy.json`](../../configs/frozen_adaptive_policy.json) retrieves Top-100 candidates with the bi-encoder, computes the Top-1/Top-10 retrieval score margin, and chooses K=5 or K=50 using the frozen threshold. MiniLM scores the selected candidates in a batch and produces the final Top-10 ranking.

The online benchmark is `python -m scripts.benchmark_end_to_end`, run from the repository root. It includes query encoding, exact search, routing, pair construction, cross-encoder inference, and final ranking. Model loading and offline corpus indexing are excluded.

Reusable code is organized by function under `adaptive_reranking/`; experiment orchestration remains under `scripts/`. The benchmark retains its original timing boundaries and inline routing operations.

See [adaptive budget](adaptive_budget.md), [evaluation](evaluation.md), and the [paper](../paper.md) for the frozen protocol and evidence. Future directions remain in [the roadmap](../roadmap.md).
