# Day 3 — RAPTOR Search

This folder contains different versions of RAPTOR search implementations.

## Versions

### 1. Ultra-Lite: Keyword Overlap Search
- **File:** [raptor_search_ultralite_keyword.md](./raptor_search_ultralite_keyword.md)
- **Description:**  
  Simplest version. Ranks chunks by counting overlapping words between query and text.  
  No extra libraries required.  

---

### 2. Lite: Simple TF-IDF-like Search
- **File:** [raptor_search_lite_tfidf.md](./raptor_search_lite_tfidf.md)
- **Description:**  
  Slightly more advanced. Mimics TF-IDF with basic normalization.  
  No sklearn dependency.  

---

### 3. Full RAPTOR Search (Standardized Pipeline)
- **File:** [raptor_search_full.md](./raptor_search_full.md)
- **Description:**  
  Full node-based RAPTOR search pipeline using TF-IDF over node summaries.  
  Includes:
  - Step 5: Run Standardization  
  - Step 6: RAPTOR Search Function  
  - Step 7: Run & Show Results  
