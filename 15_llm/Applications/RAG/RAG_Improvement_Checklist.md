```python
# RAG Improvement — Action List (Prioritized)

## 1) Index / Chunking
- 200–400 tokens, **10–20% overlap** (Korean/Chinese: **2–4 sentences**).
- Preserve structure: prepend `section_path`; keep tables/code as **block chunks** with captions.
- De-dup: **5-gram shingling + MinHash/SimHash** before indexing.
- Metadata: `{doc_id, section_path, chunk_id, lang, version, date, source, topic, pii_flag, index_version, embed_model}`

## 2) Retriever (Recall ↑)
- **Hybrid**: BM25 **0.4** + Dense **0.6** (RRF/weighted).
- **Multi-Query (3–5)** → **BM25 prefilter k≈200** → RRF merge.
- **MMR** for diversity (k=50 → m=8, λ=0.3).
- hyDE with **vague-text filter** (length, proper-noun/number density).
- Multilingual: detect → **original + translated** queries, RRF.

## 3) Reranker (Precision ↑)
- Cross-encoder on **top-50** → keep **top-m=8**.
- **Dynamic m** via **score entropy** (5–10).
- Timeout → **BM25-weighted** fallback.

## 4) Generator (Hallucination ↓)
- Prompt guardrails: **require citations**, forbid out-of-citation content, allow “I don’t know.”
- **Context budgeting** to target ~**200 tokens** answer.
- **Auto re-search** if `citation_coverage < τ` or numeric/date question lacks evidence.

## 5) Efficiency / Cost
- ANN tuning: IVF (`nlist≈√N`, `nprobe 8→64`), HNSW (`M=32`, `efC=200`, `efS 64→256`).
- Cache key: `hash(norm_query + filters + index_version + reranker_version)`.
- Batch/pipeline; optimize reranker/LLM with **INT8/ONNX/VLLM**.

## 6) Training Data
- Maintain **hard-negative pool** (recent false-positives).
- **Contrastive fine-tuning** (InfoNCE/Triplet, in-batch negatives) with small data.
- **Distill** heavy reranker labels into a lighter reranker/dual-encoder.

## 7) Evaluation / Monitoring
- Retrieval: **Recall@20, nDCG@20, MRR**
- Generation: **EM/F1 (QA), ROUGE-L (summarization)**
- Grounding: **citation coverage, false-cite rate, unique sources**
- Ops: **p50/p95 latency, $/1000 queries, source age_days**
- Online guardrails: if coverage < Y% or false-cite > X% → **increase m / re-search**.

## 8) Quick Defaults (Starter Combo)
- BM25+Dense (RRF 0.4/0.6) + **MQ×3** + **MMR**
- Rerank **top-50 → m=8**, **dynamic m**
- Prompt **citation-required + self-check**
- ANN tuning + batching + caching

```
