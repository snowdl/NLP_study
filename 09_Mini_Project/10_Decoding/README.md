# Decoding 🧩

This directory contains experiments and study notes on **text generation (decoding) methods**.  
It is organized from basics → Speculative Decoding → Medusa.

---

## 📂 Folder Structure

- **00_decoding_basics/**  
  Fundamental decoding strategies such as Greedy, Top-k, Top-p, and Beam Search.

- **01_speculative_decoding/**  

  - [01_minimal_speculative.md](01_speculative_decoding/01_minimal_speculative.md)  
  - [speculative_decoding_ngram_prefix_accept.md](01_speculative_decoding/speculative_decoding_ngram_prefix_accept.md)  
  - [speculative_decoding_step_by_step.md](01_speculative_decoding/speculative_decoding_step_by_step.md)

- **02_medusa/**  

  - [medusa_lite_clean.ipynb](02_medusa/medusa_lite_clean.ipynb)

---

## 📌 Notes
- **Speculative Decoding** → a smaller model drafts multiple tokens, while a larger model verifies them, improving generation speed.  
- **Medusa** → eliminates the need for a separate small model by attaching auxiliary heads inside the LLM to achieve similar acceleration.
