# Speculative Decoding with N-gram + Prefix-Accept

## 📌 Overview
This mini-project demonstrates a **beginner-friendly prototype** of *speculative decoding*.  
The idea:  
- A **small drafter** proposes a short draft of tokens.  
- A **larger verifier** checks the draft and only accepts a **verified prefix**.  
- At the first mismatch, the verifier replaces the token and stops (**prefix-accept rule**).  

We implement simple **n-gram frequency models** to simulate the drafter–verifier interaction.

---

## ⚙️ Components

### 1. Drafter (Unigram)
- Samples `k` tokens based on unigram frequencies.  
- Includes an `alpha` parameter to sharpen (α>1) or flatten (α<1) the distribution.  

### 2. Verifiers
- **(A) Bigram verifier**  
  Predicts the most frequent next token given the previous token (`P(next|prev1)`).
- **(B) Backoff verifier (tri→bi→uni)**  
  Uses trigram if available, otherwise bigram, otherwise unigram.

### 3. Prefix-Accept Decoding
- Iterate over drafter tokens left→right.  
- Accept if drafter token matches verifier prediction.  
- On the first mismatch → replace with verifier token and stop.  

---

## 🗂️ Implementation Notes
- Language model = simple frequency counts from a toy corpus.  
- No external libraries (beyond `collections.Counter`, `defaultdict`, `random`).  
- Designed for **clarity, not performance**.  

---

## ▶️ Example Demo

Corpus:  
```text
"the wolf ran into the forest"

---

## 🧪 Experiment Results: Baseline vs Speculative

- **Prompt:** `["the", "wolf", "ran"]`  
- **Draft length (k):** 5  
- **Drafter temperature:** 0.9  
- **Verifier:** trigram → bigram → unigram (argmax)

### Sample Run



