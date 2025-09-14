# 🧩 Mini Projects in NLP_Study

A collection of **machine learning & NLP mini projects**.  
Each project demonstrates **end-to-end experimentation**: preprocessing, training, evaluation, and documentation.  

👉 Navigate into each folder for details.

---

## 📂 Projects Overview

| No | Project | Task | Tech |
|----|---------|------|------|
| 01 | Breast_Cancer_Binary_Classification | Binary classification on a medical dataset | TensorFlow/Keras, scikit-learn |
| 02 | Recommendation_Systems | Collaborative filtering & content-based recommenders | Python, Pandas, Surprise |
| 03 | IMDB_Movie_Review_Sentiment_Analysis | Sentiment classification on IMDB with BiLSTM | TensorFlow/Keras |
| 04 | IMDB_Sentiment_Analysis_Advanced | Full pipeline: tokenization, embeddings, sequence models<br>+ ROC-AUC, PR curves, calibration curves | TensorFlow/Keras, scikit-learn, matplotlib |
| 05 | News_Summarization | Abstractive summarization with BART<br>+ evaluation & visualization | Hugging Face Transformers, PyTorch |
| 06 | Intent_Classification | Intent classification on BANKING77<br>Baselines: Logistic Regression, SVM<br>Transformers: DistilBERT, XLM-R | scikit-learn, Hugging Face |
| 07 | Text_Classification | Classical ML & DL experiments for text classification | scikit-learn, TensorFlow/PyTorch (varies) |
| 08 | BART_Text_Classification | Fine-tuning BART for text classification<br>+ reports & metrics | Hugging Face Transformers, PyTorch |
| 09 | COVID19_Smoking_MetaAnalysis | Statistical analysis of COVID-19 & smoking research | Python, stats libraries |

---

<details>
<summary><b>10_Decoding</b> (click to expand)</summary>

- **Task:** Beginner-friendly decoding strategies (Greedy, Sampling, Speculative, Medusa)  
- **Subfolders:**  
  - `00_decoding_basics`: basic greedy & sampling decoding  
  - `01_speculative_decoding`: drafter-verifier setup, prefix-accept, n-gram variants  
  - `02_medusa`: Medusa experiments (ultra-min, tiny, lite, prefix-accept)  
- **Tech:** Hugging Face Transformers, PyTorch  

</details>

<details>
<summary><b>11_ReAct_vs_Non_ReAct_on_Iris</b> (click to expand)</summary>

- **Task:** ReAct (Reasoning + Acting) framework vs baseline on Iris dataset  
- **Goal:** Compare reasoning-augmented acting vs plain baseline classification  
- **Files:**  
  - `execution/exp00_ReAct_vs_Non_ReAct_on_Iris.ipynb`  
  - `docs/ReAct_Reasoning_Acting_in_LLM.pdf`  
  - `docs/ReAct_VS_Non_ReAct_on_iris_v1.md`  
- **Tech:** Python, scikit-learn  

</details>

<details>
<summary><b>12_CLIP_Multimodal_Demo</b> (click to expand)</summary>

- **Task:** Text ↔ Image alignment demo (CLIP)  
- **Goal:** Explore multimodal understanding: text-based image retrieval, image captioning  
- **Tech:** Hugging Face Transformers, OpenAI CLIP  

</details>

<details>
<summary><b>13_IntentRouter</b> (click to expand)</summary>

- **Task:** Rules-based intent router with minimal self-improvement  
- **Flow:** detect → fallback → call handler  
- **Self-improvement:** logs gold labels, suggests new keywords  
- **File:** `exp00_fc_self_improve_min.md`  
- **Tech:** Pure Python (no ML libs)  

</details>

<details>
<summary><b>14_RAPTOR</b> (click to expand)</summary>

- **Task:** Tree-based retrieval with hierarchical summaries  
- **Pipeline:**  
  - **Day 1 — Tree Build:** `01_day1_tree_build.ipynb`, `01_day1_tree_build_EN.ipynb`  
  - **Day 2 — Retrieval:** `02_day2_retrieval.ipynb`, `02_day2_retrieval_EN.ipynb`  
  - **Day 3 — RAPTOR Search (Beginner-Friendly):** `03_day3_raptor_search/`  
    - Ultra-Lite: `raptor_search_ultralite_keyword.md`  
    - Lite: `raptor_search_lite_tfidf.md`  
    - Full Pipeline: `raptor_search_full.md`  
- **Tech:** Python, scikit-learn (TF-IDF) + pure-Python baselines  

</details>

---

## ✍️ Contributions
Feedback and contributions are always welcome!
