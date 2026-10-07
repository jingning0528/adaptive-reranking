# SciFact validation reranking report

## Protocol

- Source: SciFact `train` split (809 judged queries).
- Validation set: 162 queries selected deterministically as the lowest 20% by
  `SHA-256(query_id)`. The IDs are frozen in `results/manifests/validation/split.json`.
- Retrieval: `sentence-transformers/all-MiniLM-L6-v2`, cosine similarity,
  fixed Top-100 candidates, CPU.
- Reranking input: `[query, title + " " + abstract]`.
- Controlled comparison: Top-100, input format, maximum length 256, and batch
  size 128 are identical; only the cross-encoder changes.
- This is a held-out subset constructed from train, not an official SciFact
  development split. Future training must exclude these 162 queries.

## Quality and cost

| Method | NDCG@10 | Recall@10 | Improved / unchanged / worsened queries | Rerank time | Time/query | Candidates/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bi-encoder baseline | 0.71294 | 0.81749 | - | - | - | - |
| TinyBERT-L2 | 0.70809 | 0.83673 | 36 / 92 / 34 | 34.22 s | 0.211 s | 473.43 |
| MiniLM-L6 | 0.73826 | 0.85041 | 35 / 97 / 30 | 417.58 s | 2.578 s | 38.80 |
| Oracle over Top-100 | 0.94206 | 0.94136 | diagnostic only | - | - | - |

Relative to the bi-encoder, MiniLM-L6 improves NDCG@10 by 0.02532 absolute
(3.55% relative) and Recall@10 by 0.03292. TinyBERT-L2 lowers NDCG@10 by
0.00485 on validation despite its small positive test-set result. MiniLM-L6
is about 12.2 times slower per query than TinyBERT-L2 in these runs.

The oracle gap from the baseline is 0.22912 NDCG@10. MiniLM-L6 closes about
11.0% of that gap, so the fixed candidate set leaves substantial ranking
headroom. The oracle is not an executable retrieval method and is used only
as a diagnostic upper bound under the available judgments.

## Input truncation

Token lengths were measured on all 16,200 validation candidate pairs using
the MiniLM cross-encoder tokenizer before truncation.

| Statistic | Value |
| --- | ---: |
| Median | 336 |
| P90 | 521 |
| P95 | 593 |
| P99 | 821 |
| Maximum | 1,963 |
| Longer than 256 | 12,682 (78.28%) |
| Longer than 512 | 1,750 (10.80%) |

Truncation is therefore a plausible quality constraint, not a rare edge
case. This run does not establish causality because length was intentionally
held fixed while comparing model capability. The next controlled experiment
should compare 256 versus 512 with MiniLM-L6 on this same frozen validation
set.

As a descriptive check, 137 of 162 queries have at least one judged relevant
document longer than 256 tokens. MiniLM-L6's mean per-query NDCG@10 delta was
+0.02089 for those queries and +0.04956 for the remaining 25. This association
is consistent with a truncation effect but is not a controlled length result.

## Error inspection

The complete top-five before/after rankings and text previews for the five
largest improvements and regressions per model are stored in
`results/metrics/validation/error_cases.json`.

Observed patterns:

- Negation remains fragile. Both rerankers regress on “The G34R/V mutation
  does not create a hypomethylated phenotype…”, preferring a related histone
  mutation paper over the judged evidence.
- Shared terminology can dominate the required relation. For the AMPK/lung
  fibrosis claim, both models promote an AMPK/JAK1 anti-inflammatory paper
  instead of the judged metformin/bleomycin fibrosis study.
- Broad topic similarity is sometimes mistaken for direct support, including
  obesity side effects and exercise/quality-of-life examples.
- Metadata quality can matter: MiniLM-L6 ranks a document titled
  “Corresponding author:” first for one metastasis/genomics claim.
- Large improvements commonly occur when the relevant paper expresses the
  same specific mechanism, such as HOXB4/HSC expansion, p75 NTR signaling,
  conserved enhancer function, and pyridostatin checkpoint activation.

These examples support retaining MiniLM-L6 as the current quality reference,
while treating length handling and scientific relation/negation understanding
as separate hypotheses to test next.
