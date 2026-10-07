# Future roadmap

V1 is complete. The following milestones and system sketch describe future work.

CURRENT
SciFact Adaptive Reranking
        ↓

V2
1 Dataset abstraction ★
2 Offline data pipeline
3 FAISS
4 Add medium BEIR dataset ★
5 Exact vs ANN benchmark
6 FastAPI
7 Tests
8 p50 / p95 / QPS
9 Logging
10 Docker
11 GitHub Actions
12 Load testing

        ↓

V3
13 Add QA dataset ★
14 Basic RAG
15 RAG evaluation
16 Adaptive context budget
17 Caching
18 Quantization
19 KV-cache experiments
20 Model routing
21 End-to-end quality / latency / token analysis


## Future system sketch

```
Efficient Neural Search System
│
├── 1. Dense Retrieval
│      └── BEIR + bi-encoder
│
├── 2. ANN Index
│      └── FAISS
│
├── 3. Adaptive Reranking
│      └── dynamic compute budget
│
├── 4. Efficient Inference
│      ├── batching
│      └── FP16 / INT8
│
├── 5. Caching
│      └── query / embedding cache
│
├── 6. Serving
│      └── FastAPI
│
└── 7. Benchmark
       ├── NDCG@10
       ├── Recall@10
       ├── p50 / p95 latency
       ├── QPS
       ├── GPU memory
       └── cost / 1K queries

```
