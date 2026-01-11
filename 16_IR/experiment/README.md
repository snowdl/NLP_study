# IR Experiments

This folder contains experimental notebooks for Information Retrieval (IR),
focusing on indexing structures, retrieval models, and efficiency trade-offs.
The experiments are implemented using toy corpora to clearly illustrate
core IR concepts.

---

## Ad-hoc Retrieval Experiment (Inverted Index + TF-IDF / BM25)  🔥 *Latest*
- Construction of an inverted index with document frequency and document length
- Candidate generation using postings list union (OR)
- Optional candidate reduction using rare query terms
- Ranking with TF-IDF (baseline) and BM25
- Top-k document retrieval
- Analysis of candidate set size and runtime

Goals:
- Demonstrate why inverted indexes are fundamental to efficient search
- Observe how candidate set size affects retrieval runtime
- Compare ranking behavior between TF-IDF and BM25
- Highlight structural differences between ad-hoc retrieval and kNN-style methods

Notebook:
- `010626_champion_list_ir_experiment.ipynb`

---

## Champion List Experiment
- TF-IDF vectorization
- Inverted index construction
- Champion lists for candidate pruning
- Comparison between full search and champion-based search

Notebook:
- `010626_champion_list_ir_experiment.ipynb`

---

## Tiered Index Search with TF-IDF
- TF-IDF-based scoring over a tiered inverted index
- Top-K retrieval using tier-by-tier search
- Early stopping based on score thresholds
- Experiments aligned with IR 538 coursework

Notebook:
- `20260108_tiered_index_search_tfidf.ipynb`

---

## Tiered Inverted Index Construction
- TF-IDF weighted inverted index
- Tiered postings lists (high-score vs low-score tiers)
- Toy examples for understanding index organization

Notebook:
- `20260107_tiered_index.ipynb`

---

## Summary

These experiments collectively explore:
- How different inverted index variants improve retrieval efficiency
- The trade-off between retrieval accuracy and computational cost
- Practical implementations of classic IR techniques taught in coursework
- Structural foundations for scalable search systems
