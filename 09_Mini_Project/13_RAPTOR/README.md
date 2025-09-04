# RAPTOR: Hierarchical Summarization (Day 1)

## 📌 Overview
This project demonstrates **hierarchical summarization** (RAPTOR method) using  
the book *Harry Potter and the Sorcerer’s Stone* as input.  

The pipeline:
1. **Chunking** – Split the raw text into sentences and chunks.  
2. **Chunk Summarization** – Use **PEGASUS** (XSum / MultiNews) to summarize each chunk.  
3. **Tree Building** – Build a bottom-up summary tree combining chunk-level summaries.  
4. **Root Summary** – Generate a concise, high-level summary of the entire document.  

Full step-by-step process is documented in  
[`01_day1_tree_build_EN-2.md`](./01_day1_tree_build_EN-2.md).

---

## ⚙️ Tech Stack
- **Python 3.10**
- **PyTorch** (MPS / CUDA / CPU supported)
- **Hugging Face Transformers**  
  - `google/pegasus-xsum`  
  - `google/pegasus-multi_news`
- **tqdm** – progress bars  
- **pandas / json** – data handling  

