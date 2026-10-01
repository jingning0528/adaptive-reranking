# Method 3: controlled 256-versus-512 experiment

## Experimental contract

- Same 162 frozen SciFact validation queries and same Top-100 candidates.
- Same `cross-encoder/ms-marco-MiniLM-L-6-v2` model at revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a` on MPS.
- Same `[query, title + space + abstract]` input and evaluation pipeline.
- The only model-input variable is `max_length`: 256 versus 512.
- The 256 result comes from the complete Method 1 cache; all 16,200 pairs were
  rescored at 512.

Untruncated tokenizer lengths confirm that 12,682 pairs (78.28%) exceed 256
tokens and 1,750 (10.80%) exceed 512 tokens. Every validation query has at
least one truncated candidate; in fact, every query has more than half of its
Top-100 candidates over 256 tokens.

## Overall result

| Maximum length | NDCG@10 | Recall@10 |
| ---: | ---: | ---: |
| 256 | 0.73644 | 0.85041 |
| 512 | 0.74060 | 0.84115 |

For NDCG@10, the paired 512-minus-256 difference is `+0.00416` with a 95%
bootstrap CI of `[-0.01716, 0.02551]`; 13 queries improve, 131 are unchanged,
and 18 worsen. For Recall@10, the difference is `-0.00926` with CI
`[-0.03395, 0.01235]`; 1 query improves, 158 are unchanged, and 3 worsen.

The evidence therefore does not support replacing 256 with 512 for every
candidate. The small average NDCG increase is uncertain, more queries worsen
than improve, and Recall@10 moves in the wrong direction.

## Predefined truncation diagnostics

The query groups below were defined before inspecting the results. “Truncated
relevant” means that qrels identify at least one relevant Top-100 candidate
whose pair length exceeds 256. This is a diagnostic that uses ground truth,
not an available inference-time routing signal.

| Query group | Queries | Mean NDCG@10 change | 95% CI | Better / same / worse |
| --- | ---: | ---: | ---: | ---: |
| Has truncated relevant candidate | 129 | +0.01519 | [-0.00780, 0.04056] | 11 / 106 / 12 |
| No truncated relevant candidate | 33 | -0.03893 | [-0.08358, -0.00384] | 2 / 25 / 6 |

The difference between these two mean changes is `+0.05412`, with bootstrap
CI `[0.01185, 0.10576]`. This is evidence of heterogeneous length effects:
queries whose relevant candidates were actually truncated fare relatively
better from extra context. It is not evidence that the first group has a
reliably positive absolute gain—the group's own interval still includes zero.

Splitting by the overall fraction of truncated candidates does not show a
useful gradient. Queries with 50–75% truncated candidates change by +0.00489;
queries with 75–100% change by +0.00383, and both intervals include zero.
Simply observing that a query has many long candidates is therefore not a
supported routing signal.

At the relevant-document level, 149 relevant candidates exceed 256 tokens.
Their mean rank changes from 5.59 to 5.38, but only one enters the Top-10 while
two leave it; 20 move up, 105 stay fixed, and 24 move down. The recovered tail
content is not uniformly useful evidence.

## Decision

The truncation hypothesis is not established as the main explanation for the
Oracle gap. A global 512-token policy is not justified. The subgroup contrast
is a useful clue for a future adaptive text-budget method, but converting it
into a method requires a qrels-free signal that predicts when the missing tail
contains relevant evidence, followed by calibration and fresh validation.

The 512-token scoring run took 451.7 seconds on MPS after model loading. This
is not a controlled latency ratio against 256 because the 256 scores were read
from cache.
