# ReAct: Reasoning + Acting on the Iris Task

## Overview
This folder contains notes and experiments related to the **ReAct paper** (*ReAct: Synergizing Reasoning and Acting in Language Models*, arXiv:2210.03629).  
The goal is to replicate and extend the ReAct paradigm by applying it to a small-scale task: **answering questions about the Iris dataset**.  
We compare **Baseline (non-ReAct)** approaches with **ReAct-style reasoning-action loops**.

---

## Contents
- **[ReAct_Reasoning_Acting_in_LLM.pdf](./ReAct_Reasoning_Acting_in_LLM.pdf)**  
  Personal PDF summary of the ReAct paper
- **[ReAct_VS_Non_ReAct_on_iris_v1.md](./ReAct_VS_Non_ReAct_on_iris_v1.md)**  
  Experimental notes and implementation details comparing ReAct vs non-ReAct on the Iris dataset

---

## Experiment Design (Iris Task)
- **Input**: natural language query (e.g., *“What is the difference in average petal length by species in the Iris dataset?”*) and the DataFrame (`df`).
- **Goal**:  
  Produce answers by combining:
  - **Baseline**: directly compute or fetch knowledge in one step
  - **ReAct**: explicitly log *Thought → Action → Observation → Answer* loops for stepwise reasoning

---

## Architecture

### Tool Layer
- `tool_mean_by_species(df, target_col)`: Compute group-wise means in pandas  
- `tool_wikipedia_summary(term)`: Retrieve summary text from Wikipedia  
- `_infer_wiki_term(query)`: Infer candidate term(s) to query on Wikipedia  

### Handler Layer
- **Baseline**
  - `baseline_internal(query, df)`: Direct calculation from the dataframe
  - `baseline_external(query)`: Wikipedia summary without reasoning loops
- **ReAct**
  - `react_internal_avg_by_species(query, df)`:  
    Stepwise Thought–Action–Observation–Answer loop for statistical queries  
  - `react_external_explain_species(query)`:  
    Thought → infer keyword → Action → Wikipedia summary → Answer

### Evaluation Layer
- `eval_internal`, `eval_external`, `summarize` (+ small helpers)  
- Produces:
  - `react_vs_baseline_detail.csv` (detailed logs)  
  - `react_vs_baseline_summary.csv` (summary metrics)

---

## Execution Order (Always Follow These Steps)

