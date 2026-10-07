# Future roadmap

V1 is complete. The following milestones and system sketch describe future work.

V2 Step 1 establishes a shared dataset interface for SciFact and NFCorpus.
Step 2 adds backend-independent offline embedding artifacts. Next: FAISS index
and ANN retrieval, then an exact-versus-FAISS benchmark, then FastAPI. See
[the dataset interface](current/dataset_interface.md) and
[the embedding pipeline](current/embedding_pipeline.md) for the current contracts.

CURRENT
SciFact Adaptive Reranking
        ↓

V2
1 Dataset abstraction — shared SciFact/NFCorpus interface
2 Offline embedding artifacts
3 FAISS index + ANN retrieval
4 Exact vs FAISS benchmark
5 FastAPI
6 Add medium BEIR dataset ★
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
