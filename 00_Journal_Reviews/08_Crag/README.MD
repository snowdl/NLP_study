# 08_Crag — CRAG & Multi-Agent Collaboration Notes

This folder collects notes, notebooks, and summaries related to **CRAG** (Comprehensive Retrieval-Augmented Generation) and **collaborative/multi-agent** methods that build on or interact with RAG-style systems.

> TL;DR  
> - **`CRAG.pdf`**: benchmark/paper summary.  
> - **`MultiAgent_Collaboration/`**: hands-on analysis and notebooks for collaborative agents (incl. CRAG-related experiments).  

---

## 📁 Structure
08_Crag/
├── CRAG.pdf
├── README.md ← (this file)
└── MultiAgent_Collaboration/
├── Collaborative Agents Analysis.ipynb
├── CollaborativeAgents_Analysis_eng.ipynb
├── CollaborativeAgents_Analysis_eng.v1.md # if present
└── Toolformer_CRAG.ipynb


## 📝 What’s Inside

### 1) CRAG (Benchmark & Concepts)
- **Goal:** Evaluate end-to-end RAG pipelines across retrieval quality, grounding, reasoning, and generation robustness.  
- **You’ll find:**  
  - Failure modes checklists (query reformulation, retrieval noise, hallucination under partial grounding).  
  - Notes on evaluation axes and how to design small-scale replications in practice.

### 2) Multi-Agent Collaboration (Hands-on)
- **Notebooks:**  
  - `Collaborative Agents Analysis.ipynb` — Collaboration patterns (planner/solver/critic), messaging protocols, and ablation ideas.  
  - `CollaborativeAgents_Analysis_eng.ipynb` — English write-up version of the above with code cells.  
  - `CollaborativeAgents_Analysis_eng.v1.md` — Markdown narrative (if present).  
  - `Toolformer_CRAG.ipynb` — Tool-use prompts & API-style actions tested under CRAG-like tasks.
- **Focus:**  
  - Role design (Planner / Researcher / Critic / Executor)  
  - Turn-taking & termination criteria  
  - Evidence-grounding + citation habits in multi-agent chats  
  - Lightweight orchestration patterns you can port into product contexts

---

**CRAG.pdf** — Summary/notes of the CRAG benchmark & key takeaways for RAG systems.  
- **MultiAgent_Collaboration/** — Working notes and experiments analyzing **collaborative agents** (e.g., role assignment, tool use, critique/self-reflection) and their interaction with RAG.
REF: https://arxiv.org/pdf/2401.15884
