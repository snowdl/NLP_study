# 🧩 Mini Projects in NLP_Study

This folder contains a collection of machine learning and NLP mini projects.  
Each project demonstrates end-to-end experimentation: data preprocessing, model training, evaluation, and documentation.  
👉 Navigate into each folder for details.

---

## 📂 Projects Overview

1. **01_Breast_Cancer_Binary_Classification**  
   - Binary classification on a medical dataset.  
   - **Tech:** TensorFlow/Keras, scikit-learn

2. **02_Recommendation_Systems**  
   - Collaborative filtering & content-based recommenders.  
   - **Tech:** Python, Pandas, Surprise

3. **03_IMDB_Movie_Review_Sentiment_Analysis**  
   - Sentiment classification on IMDB with BiLSTM.  
   - **Tech:** TensorFlow/Keras

4. **04_IMDB_Sentiment_Analysis_Advanced**  
   - Full pipeline: tokenization, embeddings, sequence models.  
   - Advanced evaluation (ROC-AUC, PR curves, calibration curves).  
   - **Tech:** TensorFlow/Keras, scikit-learn, matplotlib

5. **05_News_Summarization**  
   - Abstractive summarization with BART; includes evaluation & visualization.  
   - **Tech:** Hugging Face Transformers, PyTorch

6. **06_Intent_Classification**  
   - Intent classification on BANKING77.  
   - Baselines: Logistic Regression, SVM; Transformers: DistilBERT, XLM-R.  
   - **Tech:** scikit-learn, Hugging Face

7. **07_Text_Classification**  
   - Classical ML & DL experiments for general text classification.  
   - **Tech:** scikit-learn, TensorFlow/PyTorch (varies)

8. **08_BART_Text_Classification**  
   - Fine-tuning BART for text classification with reports & metrics.  
   - **Tech:** Hugging Face Transformers, PyTorch

9. **09_COVID19_Smoking_MetaAnalysis**  
   - Statistical analysis of COVID-19 & smoking-related research.  
   - **Tech:** Python, stats libraries

10. **10_ReAct_vs_Non_ReAct_on_Iris**  
    - ReAct (Reasoning + Acting) framework vs baseline on Iris.  
    - **Tech:** Python, scikit-learn

11. **11_CLIP_Multimodal_Demo**  
    - Image-text alignment demo using CLIP.  
    - **Tech:** Hugging Face, OpenAI CLIP

12. **12_IntentRouter**  
    - Rules-based intent router with minimal self-improvement.  
    - **Flow:** detect → fallback → call handler  
    - **Self-improvement:** logs gold labels, suggests new keywords  
    - **File:** [exp00_fc_self_improve_min.md](12_IntentRouter/exp00_fc_self_improve_min.md)  
    - **Tech:** Pure Python (no ML libs)

13. **13_RAPTOR**  
    - RAPTOR: tree-based retrieval with hierarchical summaries (build → retrieve → search).  
    - **Day 1 — Tree Build:** `01_day1_tree_build.ipynb`, `01_day1_tree_build_EN.ipynb`  
    - **Day 2 — Retrieval:** `02_day2_retrieval.ipynb`, `02_day2_retrieval_EN.ipynb`  
    - **Day 3 — RAPTOR Search (Beginner-Friendly):** `03_day3_raptor_search/`  
      - Ultra-Lite (Keyword Overlap): [raptor_search_ultralite_keyword.md](13_RAPTOR/03_day3_raptor_search/raptor_search_ultralite_keyword.md)  
      - Lite (Simple TF-IDF-like): [raptor_search_lite_tfidf.md](13_RAPTOR/03_day3_raptor_search/raptor_search_lite_tfidf.md)  
      - Full Pipeline (Standardized): [raptor_search_full.md](13_RAPTOR/03_day3_raptor_search/raptor_search_full.md)  
    - **Tech:** Python, scikit-learn (TF-IDF) + pure-Python baselines

---

## 🔎 How to Use
- Dependencies are listed per project (in notebooks or READMEs).  
- Most projects include preprocessing, training, evaluation, and visualization.

---

✍️ Contributions and feedback welcome!
