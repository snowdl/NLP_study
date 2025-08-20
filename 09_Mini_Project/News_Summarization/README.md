# News Summarization (Mini Project)

Fetch recent articles from **Google News RSS**, extract full text, and summarize with an **LLM (Mistral via Ollama)**.  
This folder contains two notebook versions:

- **RSS_mistral_ollama_news_basic.ipynb** — ultra-simple, beginner-friendly pipeline  
- **RSS_mistral_ollama_news_advanced.ipynb** — improved success rate (redirect fix, UA headers, final URL resolution, length filter, tidy logs)

---

## What it does

1. **RSS (Step 1)**  
   - Query Google News RSS for a keyword.  
   - Collect basic metadata: `title`, `link`, `published`, `source`.  
   - Deduplicate by `title` and `link`.

2. **Extract (Step 2)**  
   - Resolve Google News redirect to the **final article URL**.  
   - Download HTML (optionally with a simple `User-Agent`) and extract **main body text** via `trafilatura`.

3. **Pipeline (Step 3)**  
   - Combine metadata + full text into a single `DataFrame` with columns:  
     `title`, `link`, `content`.  
   - Filter out short/empty articles using `min_chars`.

4. **Summarize (Step 4)**  
   - Use **Ollama (Mistral)** to produce **3 concise bullet points** per article.

---

## Files

- `RSS_mistral_ollama_news_basic.ipynb`  
  - Minimal code: RSS → extract → (optional) summarize  
  - Easiest to read and tweak

- `RSS_mistral_ollama_news_advanced.ipynb`  
  - Adds helpers: final URL resolver, polite delays, UA headers  
  - More robust extraction and cleaner links


