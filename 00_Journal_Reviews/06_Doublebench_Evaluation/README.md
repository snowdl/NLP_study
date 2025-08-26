# DOUBLE-BENCH — Summary & Notes

**Paper**: *Are We on the Right Way for Assessing Document Retrieval-Augmented Generation?* (arXiv:2508.03644)  
**PDF file in this folder**: `DOUBLE-BENCH.pdf`

---

## TL;DR
DOUBLE-BENCH is a large-scale, multilingual, and multimodal benchmark for **Document Retrieval-Augmented Generation (RAG)** systems. Unlike benchmarks that focus only on final scores, DOUBLE-BENCH evaluates each stage of the RAG pipeline—retrieval, evidence selection, reasoning, and generation—for **fine-grained error analysis**. It also addresses **data contamination** and supports diverse document types like scanned images and tables.

## Key Contributions
- **Fine-grained evaluation** across multiple pipeline stages.
- **Multilingual & multimodal**: six languages, various document formats.
- **Dynamic dataset updates** to mitigate contamination.
- **Comprehensive scope**: 3,276 documents (~72,880 pages), 5,168 queries (single-hop & multi-hop).

## Evaluation Pipeline
1. **Retrieval** – finding relevant document candidates.
2. **Selection / Reading** – extracting evidence spans.
3. **Reasoning / Synthesis** – combining evidence to answer queries.
4. **Generation** – producing grounded, accurate responses.

## Considerations
- OCR quality can be a performance bottleneck.
- LLM-as-judge requires bias mitigation and consistency checks.
- Language-specific difficulty and dataset balance need attention.

## Reference
- [ArXiv Paper](https://arxiv.org/abs/2508.03644)  
- [Hugging Face Dataset](https://huggingface.co/datasets/Episoode/Double-Bench)

