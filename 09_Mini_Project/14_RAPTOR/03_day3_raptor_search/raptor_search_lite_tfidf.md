```python
# ============================================
# B0. Set base path
# ============================================
from pathlib import Path
BASE = Path.home() / "NLP_study/09_Mini_Project/13_RAPTOR"
CHUNKS_PATH = BASE / "outputs/chunks.jsonl"
#print(" CHUNKS_PATH:", CHUNKS_PATH)
```


```python
# ============================================
# B1. Check if file exists
# ============================================
import os
print(" exists?", os.path.exists(CHUNKS_PATH))
```

     exists? True



```python
# ============================================
# B2. Read JSONL file
# Each line is a JSON object (one chunk)
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
print("#chunks:", len(chunks_raw))
```

    #chunks: 227



```python
# ============================================
# B3. Helper functions to extract ID and text
# Handles different possible key names
# ============================================
def get_chunk_id(rec):
    for k in ("id", "chunk_id", "cid"):
        if k in rec:
            return rec[k]
    return None

def get_chunk_text(rec):
    for k in ("text", "content", "body"):
        if k in rec:
            return rec[k]
    return ""
```


```python
# Sample raw chunks with different key names
sample_chunks = [
    {"chunk_id": "C0001", "text": "Mr. and Mrs. Dursley, of number four, Privet Drive..."},
    {"cid": "C0002", "content": "Dudley was now having a tantrum and throwing his cereal..."},
    {"id": "C0003", "body": "There was a tabby cat standing on the corner of Privet Drive..."},
    {"name": "C0004", "desc": "This one has no valid keys"}  # should fail gracefully
]

# Run tests
for i, rec in enumerate(sample_chunks, 1):
    cid = get_chunk_id(rec)
    txt = get_chunk_text(rec)
    print(f"{i}. Raw: {rec}")
    print(f"   → get_chunk_id: {cid}")
    print(f"   → get_chunk_text: {txt[:60]}{'...' if len(txt) > 60 else ''}")
    print("-" * 60)

```

    1. Raw: {'chunk_id': 'C0001', 'text': 'Mr. and Mrs. Dursley, of number four, Privet Drive...'}
       → get_chunk_id: C0001
       → get_chunk_text: Mr. and Mrs. Dursley, of number four, Privet Drive...
    ------------------------------------------------------------
    2. Raw: {'cid': 'C0002', 'content': 'Dudley was now having a tantrum and throwing his cereal...'}
       → get_chunk_id: C0002
       → get_chunk_text: Dudley was now having a tantrum and throwing his cereal...
    ------------------------------------------------------------
    3. Raw: {'id': 'C0003', 'body': 'There was a tabby cat standing on the corner of Privet Drive...'}
       → get_chunk_id: C0003
       → get_chunk_text: There was a tabby cat standing on the corner of Privet Drive...
    ------------------------------------------------------------
    4. Raw: {'name': 'C0004', 'desc': 'This one has no valid keys'}
       → get_chunk_id: None
       → get_chunk_text: 
    ------------------------------------------------------------



```python
# ============================================
# B4. Tokenizer
# - Lowercase
# - Extract alphanumeric tokens
# ============================================
import re

def tokenize(s: str):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

# Optional: Unicode version (for Korean, etc.)
# def tokenize(s: str):
#     return set(re.findall(r"[0-9\w]+", s.lower(), flags=re.UNICODE))
```


```python
# ============================================
# B5. Scoring function
# - overlap: number of shared tokens
# - normalization: divide by log of text length
#   (to avoid bias towards long texts)
# ============================================
import math

def simple_score(query: str, text: str) -> float:
    q = tokenize(query)
    t = tokenize(text)
    if not t:
        return 0.0
    overlap = len(q & t)
    return overlap / (1.0 + math.log(1 + len(t)))
```


```python
# ============================================
# B6. Search function
# - Compute score for each chunk
# - Return top-k results sorted by score
# ============================================
def search_chunks_simple(query: str, chunks_raw, topk=5, min_score=0.0):
    scored = []
    for rec in chunks_raw:
        cid = get_chunk_id(rec)
        text = get_chunk_text(rec)
        if not cid or not text:
            continue
        s = simple_score(query, text)
        if s > min_score:
            scored.append((s, cid, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]

```


```python
# ============================================
# B7. Run search and display results
# ============================================
query = "What strange events happened on Privet Drive?"
hits2 = search_chunks_simple(query, chunks_raw, topk=5)

for rank, (score, cid, text) in enumerate(hits2, 1):
    preview = text[:160] + ("..." if len(text) > 160 else "")
    print(f"{rank}. {cid} | score={score:.4f}")
    print(preview)
    print("-" * 60)
```

    1. C0001 | score=0.7958
    M r. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d exp...
    ------------------------------------------------------------
    2. C0221 | score=0.6495
    I was unfortunate enough in my youth to come across a vomit flavored one, and since then I’m afraid I’ve rather lost my liking for them — but I think I’ll be sa...
    ------------------------------------------------------------
    3. C0030 | score=0.6453
    After a minute of confused fighting, in which everyone got hit a lot by the Smelting stick, Uncle Vernon straightened up, gasping for breath, with Harry’s lette...
    ------------------------------------------------------------
    4. C0026 | score=0.6442
    Uncle Vernon opened his newspaper as usual and Dudley banged his Smelting stick, which he carried everywhere, on the table. They heard the click of the mail slo...
    ------------------------------------------------------------
    5. C0023 | score=0.6414
    He managed to say, “Go — cupboard — stay — no meals,” before he collapsed into a chair, and Aunt Petunia had to run and get him a large brandy. Harry lay in his...
    ------------------------------------------------------------



```python
# ============================================
# B8. Simple answer generator
# - Concatenate top chunk texts
# - Limit output to max_chars
# ============================================
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

print("\n💬 Answer:")
print(simple_answer(hits2))
```

    
    💬 Answer:
    M r. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spy



```python
STOPWORDS = {"the","a","an","and","or","to","of","in","on","for","at","by","with","is","are","was","were"}

def tokenize_nostop(s: str):
    toks = tokenize(s)
    return toks - STOPWORDS

def simple_score_nostop(query: str, text: str) -> float:
    q = tokenize_nostop(query)
    t = tokenize_nostop(text)
    if not t:
        return 0.0
    overlap = len(q & t)
    return overlap / (1.0 + math.log(1 + len(t)))

# 쓰려면 simple_score를 simple_score_nostop으로 바꾸면 됨

```


```python
QUERY_EXPAND = {
    "strange": {"odd","unusual","mysterious"},
    "events": {"incidents","happenings","occurrences"},
}

def expand_tokens(tokens):
    out = set(tokens)
    for t in list(tokens):
        out |= QUERY_EXPAND.get(t, set())
    return out

def simple_score_expanded(query: str, text: str) -> float:
    q = expand_tokens(tokenize(query))
    t = tokenize(text)
    if not t:
        return 0.0
    overlap = len(q & t)
    return overlap / (1.0 + math.log(1 + len(t)))

```


```python

```
