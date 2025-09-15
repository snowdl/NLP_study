# Decoding 🧩

This directory contains experiments and study notes on **text generation (decoding) methods**.  
It is organized from **basics → speculative decoding variants → Medusa**.

---

## 📂 Folder Structure

- **00_decoding_basics/**  
  Fundamental decoding strategies such as Greedy, Top-k, Top-p, and Beam Search.

- **01_Basics/**  
  Minimal / introductory speculative decoding examples (EN & KR versions, simple demos).

- **02_Baseline_vs_Prefix/**  
  Comparisons between baseline decoding and prefix-accept speculative decoding.

- **02_Medusa_Core/**  
  Pure Medusa demos and execution notebooks (Greedy vs Medusa, ultra-min, tiny versions, docs).

- **03_Speculative_Medusa/**  
  Experiments comparing Medusa with Greedy decoding **within the speculative decoding framework**.

- **04_Ngram/**  
  N-gram–based prefix-accept prototypes.

- **05_Soft_Guarded/**  
  Soft and guarded speculative decoding experiments.

- **99_Tutorials_Docs/**  
  Tutorials and walkthroughs (step-by-step guides, English/Korean versions, consolidated docs).

---

## 📌 Notes

- **Speculative Decoding** → A smaller “drafter” model proposes multiple tokens, while a larger “verifier” model checks and accepts as much of the draft as possible. This improves generation speed without much quality loss.  

- **Medusa** → Removes the need for a separate drafter model by adding lightweight “Medusa heads” inside the LLM itself, achieving speculative-style acceleration.  

- **Speculative vs Medusa** → In this repo, you can find both pure Medusa experiments (`02_Medusa_Core`) and speculative decoding experiments that incorporate Medusa (`03_Speculative_Medusa`).
