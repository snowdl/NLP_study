```python
# ============================================
# A0. Set base path
# ============================================
from pathlib import Path

BASE = Path.home() / "NLP_study/09_Mini_Project/13_RAPTOR"
CHUNKS_PATH = BASE / "outputs/chunks.jsonl"
#print("📂 CHUNKS_PATH:", CHUNKS_PATH)
```

    📂 CHUNKS_PATH: /Users/jessicahong/gitclone/NLP_study/09_Mini_Project/13_RAPTOR/outputs/chunks.jsonl



```python
# ============================================
# A1. Check if file exists
# ============================================
import os
print("✅ exists?", os.path.exists(CHUNKS_PATH))
```

    ✅ exists? True



```python
import json

def read_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

chunks_raw = read_jsonl(CHUNKS_PATH)
print("✅ #chunks:", len(chunks_raw))
```

    ✅ #chunks: 227



```python
# ============================================
# A2. Read JSONL file
# - Each line is one JSON object (one chunk)
# ============================================
import json

def read_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

chunks_raw = read_jsonl(CHUNKS_PATH)
print("✅ #chunks:", len(chunks_raw))

```

    ✅ #chunks: 227



```python
# ============================================
# A3. Tokenizer
# - Convert string to lowercase
# - Extract only alphanumeric tokens (a–z, 0–9)
# - Return as a Python set (unique tokens)
# ============================================
import re

def tokenize(s: str):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

```


```python
# --- Test with one sample chunk ---
# Assuming you already loaded chunks_raw from chunks.jsonl:
sample = chunks_raw[0]                     # take first record
sample_text = get_chunk_text(sample)       # extract text field
tokens = tokenize(sample_text)             # tokenize it

print(" Sample text preview:", sample_text[:120], "...")
print(" Tokens(sample):", list(sorted(tokens))[:15])
```

     Sample text preview: M r. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very m ...
     Tokens(sample): ['a', 'about', 'all', 'also', 'although', 'amount', 'and', 'another', 'any', 'anyone', 'anything', 'anywhere', 'arrived', 'as', 'at']



```python
# ============================================
# A4. Overlap score
# - Count how many tokens are shared
#   between query and text
# ============================================
def overlap_score(query_tokens, text_tokens):
    return len(query_tokens & text_tokens)   # set intersection
```


```python
# --- Quick test with sample chunk ---
qt = tokenize("Privet Drive strange events?")    # query tokens
tt = tokenize(get_chunk_text(sample))            # text tokens from sample chunk
score = overlap_score(qt, tt)

print("Query tokens:", qt)
print("Text tokens (sample, first 15):", list(sorted(tt))[:15])
print("Overlap score:", score)
```

    Query tokens: {'events', 'strange', 'privet', 'drive'}
    Text tokens (sample, first 15): ['a', 'about', 'all', 'also', 'although', 'amount', 'and', 'another', 'any', 'anyone', 'anything', 'anywhere', 'arrived', 'as', 'at']
    Overlap score: 3



```python
# ============================================
# A5. Keyword Overlap Search Function
# ============================================
```


```python
"""
    Search chunks using a simple keyword overlap score.

    Steps:
    1. Tokenize the query.
    2. For each chunk:
       - Extract ID and text.
       - Tokenize the text.
       - Compute overlap score with query tokens.
    3. Keep only chunks with score > 0.
    4. Sort by score (descending).
    5. Return top-k results.

    Args:
        query (str): User query text.
        chunks_raw (list): Raw list of chunk dicts.
        topk (int): How many results to return.

    Returns:
        List of tuples (score, chunk_id, chunk_text).
"""
```




    '\n    Search chunks using a simple keyword overlap score.\n\n    Steps:\n    1. Tokenize the query.\n    2. For each chunk:\n       - Extract ID and text.\n       - Tokenize the text.\n       - Compute overlap score with query tokens.\n    3. Keep only chunks with score > 0.\n    4. Sort by score (descending).\n    5. Return top-k results.\n\n    Args:\n        query (str): User query text.\n        chunks_raw (list): Raw list of chunk dicts.\n        topk (int): How many results to return.\n\n    Returns:\n        List of tuples (score, chunk_id, chunk_text).\n'




```python
def search_chunks_keyword_overlap(query: str, chunks_raw, topk: int = 5):
 
    q_tokens = tokenize(query)          # tokenize query once
    scored = []

    for rec in chunks_raw:
        cid = get_chunk_id(rec)         # extract chunk id
        text = get_chunk_text(rec)      # extract chunk text
        if not cid or not text:         # skip invalid chunks
            continue

        s = overlap_score(q_tokens, tokenize(text))
        if s > 0:                       # keep only if overlap exists
            scored.append((s, cid, text))

    # sort results by score, highest first
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]
```


```python
query = "What strange events happened on Privet Drive?"
results = search_chunks_keyword_overlap(query, chunks_raw, topk=3)

for rank, (score, cid, text) in enumerate(results, 1):
    preview = text[:120] + ("..." if len(text) > 120 else "")
    print(f"{rank}. {cid} | score={score}")
    print("   ", preview)
```

    1. C0001 | score=5
        M r. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very m...
    2. C0002 | score=4
        Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing...
    3. C0006 | score=4
        Dursley wondered whether he dared tell her he’d heard the name “Potter.” He decided he didn’t dare. Instead he said, as ...



```python
# ============================================
# A6.Simple Answer Generator
# ============================================
```


```python
"""
    Generate a simple answer by concatenating the top retrieved chunks.

    Args:
        results (list): Search results in the form (score, chunk_id, chunk_text).
        max_chars (int): Maximum number of characters in the final answer.

    Steps:
        1. Start with an empty buffer.
        2. Iterate through results in order.
        3. Take as much text as possible without exceeding max_chars.
        4. Concatenate all collected snippets.
        5. Return final string (or fallback message if empty).
"""
```




    '\n    Generate a simple answer by concatenating the top retrieved chunks.\n\n    Args:\n        results (list): Search results in the form (score, chunk_id, chunk_text).\n        max_chars (int): Maximum number of characters in the final answer.\n\n    Steps:\n        1. Start with an empty buffer.\n        2. Iterate through results in order.\n        3. Take as much text as possible without exceeding max_chars.\n        4. Concatenate all collected snippets.\n        5. Return final string (or fallback message if empty).\n'




```python
def simple_answer(results, max_chars=600):
    buf, used = [], 0
    for score, cid, txt in results:
        if used >= max_chars:
            break
        take = max_chars - used
        snippet = txt[:take]
        buf.append(snippet)
        used += len(snippet)
    return " ".join(buf) if buf else "No evidence found."

print("\n Answer:")
print(simple_answer(hits))

```

    
     Answer:
    M r. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spy



```python

```
