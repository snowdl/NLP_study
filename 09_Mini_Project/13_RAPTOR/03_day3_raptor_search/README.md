# Day 3 — RAPTOR Search

This folder contains two simple baseline implementations of RAPTOR search, designed for beginners.

## Versions

### 1. Ultra-Lite: Keyword Overlap Search
- **File:** [raptor_search_ultralite_keyword.md](./raptor_search_ultralite_keyword.md)
- **Description:**  
  Simplest version. Ranks chunks by counting overlapping words between query and text.  
  No extra libraries required (pure Python).  
- **Pros:** Very easy to understand, minimal code.  
- **Cons:** Only exact word matches, no semantic similarity.  

---

### 2. Lite: Simple TF-IDF-like Search
- **File:** [raptor_search_lite_tfidf.md](./raptor_search_lite_tfidf.md)
- **Description:**  
  Slightly more advanced. Still no sklearn, but adds log-length normalization to mimic TF-IDF.  
- **Pros:** More robust than keyword overlap. Reduces bias toward long texts.  
- **Cons:** Still lexical only, not semantic.  

---

## Usage
Both versions:
1. Load `chunks.jsonl` from `outputs/`.  
2. Run search with a query (e.g., `"What strange events happened on Privet Drive?"`).  
3. Check top-k results and generated simple answers.

---

✍️ Next steps: You can extend these baselines with real TF-IDF (using `sklearn`) or semantic search (using embeddings).
