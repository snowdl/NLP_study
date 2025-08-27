# RAG_Chunking

This folder contains a small lab for experimenting with **text chunking**  
using Harry Potter (Book 1) as example data.

---

## What I did
- **Loaded text** from a `.txt` file (Harry Potter and the Sorcerer's Stone).
- **Cleaned text**:
  - Normalize unicode
  - Remove extra spaces and newlines
- **Chunked text** by words:
  - Sizes: 256, 512, 1024 words
  - Overlap: 50 words
- **Saved summary** to CSV (`rag_lab_outputs/chunking_summary.csv`).
- **Plotted stats**:
  - Number of chunks per size
  - Average words per chunk

---

## Files
- `RAG_Chunking.ipynb` → Jupyter notebook with all steps
- `RAG_Chunking.md` → Notes / summary of results
- `output_16_0.png`, `output_16_1.png` → Chunking plots (bar charts)
- `corpus/` → Text data (Harry Potter book)

---

## Next steps
- Add retrieval experiments:
  - BM25 (keyword search)
  - Embedding-based search
  - Fusion (BM25 + Embedding)
- Build a tiny `qa.csv` for evaluation
- Save results into `retrieval_results.csv`

---

*This is a beginner-friendly mini project for learning RAG basics.*

