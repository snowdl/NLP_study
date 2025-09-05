# RAPTOR: Hierarchical Summarization

📌 **Overview**  
This project demonstrates hierarchical summarization (RAPTOR method) using  
the book *Harry Potter and the Sorcerer’s Stone* as input.

The pipeline:

1. **Chunking** – Split the raw text into sentences and chunks.  
2. **Chunk Summarization** – Use PEGASUS (XSum / MultiNews) to summarize each chunk.  
3. **Tree Building** – Build a bottom-up summary tree combining chunk-level summaries.  
4. **Root Summary** – Generate a concise, high-level summary of the entire document.  
5. **Retrieval & QA (Day 2)** – Perform semantic search over node/chunk summaries and answer queries with evidence.  

Full step-by-step process is documented in:  
- [01_day1_tree_build_EN-2.md](./01_day1_tree_build_EN-2.md)  
- [02_day2_retrieval_EN.md](./02_day2_retrieval_EN.md)  

---

⚙️ **Tech Stack**  
- Python 3.10  
- PyTorch (MPS / CUDA / CPU supported)  
- Hugging Face Transformers  
  - `google/pegasus-xsum`  
  - `google/pegasus-multi_news`  
- `sentence-transformers` (for SBERT embeddings, fallback TF-IDF)  
- `tqdm` – progress bars  
- `pandas` / `json` – data handling  

---

🆕 **Day 2 Highlights (Retrieval & QA)**  
- Added semantic search over hierarchical summaries (nodes + leaves).  
- Implemented hybrid backend: **SBERT** (all-MiniLM-L6-v2) with fallback to TF-IDF.  
- Support for RAPTOR-style query answering with evidence snippets.  
- Key code in: [02_day2_retrieval_EN.md](./02_day2_retrieval_EN.md).  
