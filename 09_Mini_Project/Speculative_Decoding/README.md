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

## 🧪 Experiment Results: Baseline vs Speculative

- **Prompt:** `["the", "wolf", "ran"]`  
- **Draft length (k):** 5  
- **Drafter temperature:** 0.9  
- **Verifier:** trigram → bigram → unigram (argmax)

### Sample Run
---- Baseline (T=0.7), next 5 tokens ----
Baseline: the wolf ran into the forest

---- Speculative (k=5, T_draft=0.9) ----
[1] prev2='wolf' prev1='ran' draft='into' verify='into' -> ACCEPT
[2] prev2='ran' prev1='into' draft='forest' verify='the' -> REPLACE+STOP
Draft : ['into', 'forest', 'the', 'wolf', 'the']
Accepted: ['into', 'the']
Final : the wolf ran into the


---

## 🔍 Takeaways

1. Out of 5 drafted tokens, on average **3.7 tokens** are directly accepted by the verifier.  
2. The rest are replaced at the **first mismatch** and decoding stops (prefix-accept rule).  
3. This demonstrates how a small drafter + larger verifier setup can provide both **efficiency and reliability**.  

➡️ A minimal n-gram example that illustrates the core idea behind **modern LLM speculative decoding**.




## ▶️ Example Demo

Corpus:  
```text
"the wolf ran into the forest"




