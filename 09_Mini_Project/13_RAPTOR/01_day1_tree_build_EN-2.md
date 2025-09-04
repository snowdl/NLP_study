```python
"""
 BASE = 13_RAPTOR/
 ├── DATA   → 13_RAPTOR/data
 │    └── harry_potter.txt   (원본 문서)
 │
 ├── SRC    → 13_RAPTOR/src
 │    ├── chunking.py
 │    ├── summarize_chunks.py
 │    └── build_tree.py      (코드 모듈들)
 │
 └── OUT    → 13_RAPTOR/outputs
      ├── chunks.jsonl            (Step 1 결과: 문서 → 청크)
      ├── chunk_summaries.jsonl   (Step 2 결과: 청크 → 요약)
      ├── tree_nodes.jsonl        (Step 3 결과: 트리 구조 노드)
      └── tree_root.json          (Step 3 결과: 최종 루트 요약)

"""
```




    '\n BASE = 13_RAPTOR/\n ├── DATA   → 13_RAPTOR/data\n │    └── harry_potter.txt   (원본 문서)\n │\n ├── SRC    → 13_RAPTOR/src\n │    ├── chunking.py\n │    ├── summarize_chunks.py\n │    └── build_tree.py      (코드 모듈들)\n │\n └── OUT    → 13_RAPTOR/outputs\n      ├── chunks.jsonl            (Step 1 결과: 문서 → 청크)\n      ├── chunk_summaries.jsonl   (Step 2 결과: 청크 → 요약)\n      ├── tree_nodes.jsonl        (Step 3 결과: 트리 구조 노드)\n      └── tree_root.json          (Step 3 결과: 최종 루트 요약)\n\n'




```python
!pip install -q sentencepiece tokenizers transformers
```


```python
#Step 0. Prep
```


```python
import json              # Provides functions for working with JSON data (load, dump, etc.)
from pathlib import Path # Object-oriented approach to handle file system paths
from tqdm.auto import tqdm  # Displays progress bars in loops (auto chooses best interface for Jupyter/terminal)
```


```python
BASE = Path.cwd()                 # Current working directory (e.g., /.../09_Mini_Project/13_RAPTOR)
DATA = BASE / "data"              # Path to the "data" subfolder inside BASE
OUT  = BASE / "outputs"           # Path to the "outputs" subfolder inside BASE
SRC  = BASE / "src"               # Path to the "src" (source code) subfolder inside BASE

# Create the "outputs" directory if it doesn’t exist yet
OUT.mkdir(parents=True, exist_ok=True)
```


```python
print("BASE:", BASE)
print("DATA:", DATA)
print("OUT :", OUT)
```

    BASE: /Users/jessicahong/gitclone/NLP_study/09_Mini_Project/13_RAPTOR
    DATA: /Users/jessicahong/gitclone/NLP_study/09_Mini_Project/13_RAPTOR/data
    OUT : /Users/jessicahong/gitclone/NLP_study/09_Mini_Project/13_RAPTOR/outputs



```python
#Step 1. Chunking
"""
Sentence Splitting Function
Simple sentence segmentation function.
Splits text into sentences based on punctuation marks (., ?, !).
    
Args:
text (str): Input text string to be split.
Returns:
list: A list of sentences after splitting.
"""
```




    '\nSentence Splitting Function\nSimple sentence segmentation function.\nSplits text into sentences based on punctuation marks (., ?, !).\n    \nArgs:\ntext (str): Input text string to be split.\nReturns:\nlist: A list of sentences after splitting.\n'




```python
import re
# === Sentence Splitting Function ===
def split_sentences(text: str):
    # Split text whenever a period, question mark, or exclamation mark is followed by whitespace
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Remove any empty strings that may appear after splitting
    return [s for s in sents if s]
```


```python
# === Example check ===
if __name__ == "__main__":
    sample_text = "Hello world! How are you doing today? I'm working on RAPTOR."
    print("Input text:", sample_text)
    print("Split sentences:", split_sentences(sample_text))
```

    Input text: Hello world! How are you doing today? I'm working on RAPTOR.
    Split sentences: ['Hello world!', 'How are you doing today?', "I'm working on RAPTOR."]



```python
"""
Create text chunks by concatenating sentences until the max character limit is reached.
When the current chunk exceeds `max_chars`, a new chunk is started.
    
Args:
sents (list of str): List of sentences to combine into chunks.
max_chars (int, optional): Maximum number of characters allowed per chunk. Defaults to 2000.
    
Returns:
list: A list of text chunks, each containing one or more sentences.
"""
```




    '\nCreate text chunks by concatenating sentences until the max character limit is reached.\nWhen the current chunk exceeds `max_chars`, a new chunk is started.\n    \nArgs:\nsents (list of str): List of sentences to combine into chunks.\nmax_chars (int, optional): Maximum number of characters allowed per chunk. Defaults to 2000.\n    \nReturns:\nlist: A list of text chunks, each containing one or more sentences.\n'




```python
# === Chunk Creation Function ===
def chunk_by_sentences(sents, max_chars=2000):
    chunks, cur, cur_len = [], [], 0
    
    # Iterate over sentences
    for s in sents:
        # If adding the sentence would exceed the limit, finalize current chunk
        if cur_len + len(s) > max_chars and cur:
            chunks.append(" ".join(cur))  # Save the current chunk
            cur, cur_len = [], 0          # Reset for the next chunk
        
        # Add the sentence to the current chunk
        cur.append(s)
        cur_len += len(s) + 1  # +1 accounts for the space between sentences
    
    # Append any remaining sentences as the last chunk
    if cur:
        chunks.append(" ".join(cur))
    
    return chunks
```


```python
# === Example check ===
if __name__ == "__main__":
    sample_text = (
        "Hello world! How are you doing today? "
        "I'm working on RAPTOR. It helps with document chunking. "
        "Sometimes the text can be very long, so we need to split it into chunks. "
        "Each chunk must stay under a certain character limit."
    )
    
    # Step 1: Sentence splitting
    sentences = split_sentences(sample_text)
    print("🔹 Sentences:")
    for i, s in enumerate(sentences, 1):
        print(f"{i}: {s}")
    
    # Step 2: Chunking
    chunks = chunk_by_sentences(sentences, max_chars=50)
    print("\n🔹 Chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"{i}: {c}")
```

    🔹 Sentences:
    1: Hello world!
    2: How are you doing today?
    3: I'm working on RAPTOR.
    4: It helps with document chunking.
    5: Sometimes the text can be very long, so we need to split it into chunks.
    6: Each chunk must stay under a certain character limit.
    
    🔹 Chunks:
    1: Hello world! How are you doing today?
    2: I'm working on RAPTOR.
    3: It helps with document chunking.
    4: Sometimes the text can be very long, so we need to split it into chunks.
    5: Each chunk must stay under a certain character limit.



```python
#2) Step 1: 문서 로드 → 청크 저장
```


```python
# === Specify document path (relative to current working directory) ===
# The document is located inside ../../11_data/ relative to the current script location
DOC_NAME = "01 Harry Potter and the Sorcerers Stone.txt"   # Target document name
doc_path = Path("../../11_data") / DOC_NAME                # Full relative path to the document
```


```python
# Text File  →  Raw String  →  List of Sentences  →  List of Chunks
text   = doc_path.read_text(encoding="utf-8")
sents  = split_sentences(text)
chunks = chunk_by_sentences(sents, max_chars=2000)  
```


```python
# === Sentence Splitting & Chunk Creation ===
# Save chunks to chunks.jsonl

chunk_path = OUT / "chunks.jsonl"   # Output file path

# Open the output file in write mode
with chunk_path.open("w", encoding="utf-8") as f:
    for i, ch in enumerate(chunks, 1):
        # Write each chunk as a JSON object in JSONL format (one line per chunk)
        f.write(json.dumps({
            "chunk_id": f"C{i:04d}",   # Unique chunk ID, zero-padded (e.g., C0001, C0002, ...)
            "text": ch,                # The actual chunk text
            "tokens": len(ch.split())  # Token count (approx. word count using split on whitespace)
        }, ensure_ascii=False) + "\n")

# ✅ Each line in chunks.jsonl now represents one chunk

```


```python
# ✅ Print confirmation after saving chunks
print("✅ chunks.jsonl saved at:", chunk_path)   # Confirm the output file path
print("Total number of sentences:", len(sents)) # Show how many sentences were split
print("Total number of chunks:", len(chunks))   # Show how many chunks were created

# Preview the first chunk (first 300 characters)
print("First chunk preview:\n", chunks[0][:300], "...")

```

    ✅ chunks.jsonl saved at: /Users/jessicahong/gitclone/NLP_study/09_Mini_Project/13_RAPTOR/outputs/chunks.jsonl
    Total number of sentences: 5003
    Total number of chunks: 227
    First chunk preview:
     M r. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense. Mr. Dursley was the director of a fi ...



```python
# Force Transformers to ignore TensorFlow/Flax and stick to PyTorch.
```


```python
# --- Run this at the very top of a fresh cell, before importing transformers ---
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"     # tell transformers to ignore TensorFlow
os.environ["TRANSFORMERS_NO_FLAX"] = "1"   # and Flax/JAX

import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
#Prepare PEGASUS Model
MODEL_NAME = "google/pegasus-xsum"
# Pick device
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
print(f"✅ PEGASUS loaded on {device}")
```

    Some weights of PegasusForConditionalGeneration were not initialized from the model checkpoint at google/pegasus-xsum and are newly initialized: ['model.decoder.embed_positions.weight', 'model.encoder.embed_positions.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


    ✅ PEGASUS loaded on mps



```python
"""
Summarize a single text chunk using PEGASUS.
    
    Args:
        text (str): The input text chunk.
        max_in (int): Maximum number of input tokens (truncate if longer).
        max_out (int): Maximum number of output tokens in the summary.
        num_beams (int): Beam search width (higher = better quality but slower).
    
    Returns:
        str: The generated summary string.
"""
```




    '\nSummarize a single text chunk using PEGASUS.\n    \n    Args:\n        text (str): The input text chunk.\n        max_in (int): Maximum number of input tokens (truncate if longer).\n        max_out (int): Maximum number of output tokens in the summary.\n        num_beams (int): Beam search width (higher = better quality but slower).\n    \n    Returns:\n        str: The generated summary string.\n'




```python
#Step 2 — Chunk Summarization
```


```python
# === Step 2 - Chunk Summarization ===
def summarize_pegasus(text, max_in=512, max_out=64, num_beams=4):
    """
    Summarize a single text chunk using PEGASUS.
    """
    inputs = tokenizer(                 # <- was: tok(...)
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_in
    ).to(device)                        # <- was: .to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=max_out,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)  # <- was: tok.decode(...)

print(f"✅ PEGASUS ready (device={device})")


```

    ✅ PEGASUS ready (device=mps)



```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import cached_file
import torch

model_name = "google/pegasus-xsum"

# === Check if model is already cached locally ===
def have_local_model(model_name: str) -> bool:
    """
    Returns True if the given model is already cached locally,
    otherwise False (will need to download).
    """
    try:
        # Try to locate a config file for the model in the local cache
        _ = cached_file(model_name, "config.json")
        return True
    except Exception:
        return False


if have_local_model(model_name):
    print("✅ PEGASUS model is already cached locally.")
    # You can load directly with local_files_only=True if you want:
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
else:
    print("⬇️ Downloading PEGASUS… (internet required)")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✅ Download complete.")

# === Device selection (MPS > CUDA > CPU) ===
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)

```

    ✅ PEGASUS model is already cached locally.


    Some weights of PegasusForConditionalGeneration were not initialized from the model checkpoint at google/pegasus-xsum and are newly initialized: ['model.decoder.embed_positions.weight', 'model.encoder.embed_positions.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


    Using device: mps



```python
# === Step 3 - Summarization Functions (PEGASUS) ===
```


```python
# === Step 3 - Summarization (PEGASUS) ===
def summarize_pegasus(text, tokenizer, model, device="cpu",
                      max_in=512, max_out=64, num_beams=4):
    """Summarize one text chunk with PEGASUS."""
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=max_in).to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=max_out,
                             num_beams=num_beams,
                             no_repeat_ngram_size=3,
                             early_stopping=True)
    return tokenizer.decode(ids[0], skip_special_tokens=True)

```


```python
def summarize_chunks(chunks, tokenizer, model, device="cpu",
                     max_in=512, max_out=64, num_beams=4):
    """Summarize a list of chunks → list of {chunk_id, summary} dicts."""
    return [
        {"chunk_id": f"C{i+1:04d}",
         "summary": summarize_pegasus(ch, tokenizer, model, device,
                                      max_in, max_out, num_beams)}
        for i, ch in enumerate(chunks)
    ]
```


```python
# === Smoke test ===
```


```python
from tqdm.auto import tqdm
import pandas as pd
import json
```


```python
# --- 4-1) Load only the first 5 chunks for smoke testing ---
chunks_path = OUT / "chunks.jsonl"              # Input file with all chunks
summ_smoke = OUT / "chunk_summaries_smoke.jsonl"  # Output file for smoke test results

sample = []
with open(chunks_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        sample.append(json.loads(line))   # Parse each line into a Python dict
        if i >= 5:                        # Stop after 5 lines
            break
```


```python
# --- 4-2) Run PEGASUS summarization on the sample chunks ---
with open(summ_smoke, "w", encoding="utf-8") as fout:
    for obj in tqdm(sample, desc="Smoke summarizing (5)"):
        cid, text = obj["chunk_id"], obj["text"]

        # Summarize using PEGASUS (slightly longer summaries: max_out=96)
        summ = summarize_pegasus(
            text, tokenizer, model,
            device=device,
            max_in=512, max_out=96, num_beams=4
        )

        # Build result object with summary + key points
        item = {
            "chunk_id": cid,                                     # Original chunk ID
            "summary": summ,                                     # PEGASUS summary text
            "key_points": [s.strip() for s in summ.split(". ")   # Naive split into sentences
                           if s.strip()][:4]                     # Keep up to 4 key points
        }

        # Save one JSON object per line
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Smoke summarization saved to:", summ_smoke)

```


    Smoke summarizing (5):   0%|          | 0/5 [00:00<?, ?it/s]


    ✅ Smoke summarization saved to: /Users/jessicahong/gitclone/NLP_study/09_Mini_Project/13_RAPTOR/outputs/chunk_summaries_smoke.jsonl



```python
# === 4-3) Preview smoke test summaries in a DataFrame ===
# Show full text in DataFrame cells (no truncation)
pd.set_option("display.max_colwidth", None)

# Load JSONL file (one JSON object per line)
df_smoke = pd.read_json(summ_smoke, lines=True)

# Display the DataFrame (works nicely in Jupyter)
display(df_smoke)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chunk_id</th>
      <th>summary</th>
      <th>key_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C0001</td>
      <td>This is the story of the Dursleys and the Potters.</td>
      <td>[This is the story of the Dursleys and the Potters.]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C0002</td>
      <td>The Dursleys left the house for the day, with Mr. Dursley couldn’t bear people who dressed in funny clothes — the getups you saw on young people!</td>
      <td>[The Dursleys left the house for the day, with Mr, Dursley couldn’t bear people who dressed in funny clothes — the getups you saw on young people!]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C0003</td>
      <td>On the morning of the first day of school, Mr.</td>
      <td>[On the morning of the first day of school, Mr.]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C0004</td>
      <td>The first thing Mr.</td>
      <td>[The first thing Mr.]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C0005</td>
      <td>Dudley and Petunia Dursley had a strange day.</td>
      <td>[Dudley and Petunia Dursley had a strange day.]</td>
    </tr>
  </tbody>
</table>
</div>



```python

```


```python
# === Meta summarization builders ===
def _build_meta_with_xsum():
    """
    Build a meta summarization function using PEGASUS-XSum.
    This wraps summarize_pegasus with fixed parameters.
    """
    def meta_func(text):
        return summarize_pegasus(
            text, tokenizer, model,
            device=device,
            max_in=512,   # input token limit
            max_out=96,   # output length (longer summaries)
            num_beams=4
        )
    return meta_func


def _build_meta_with_multinews():
    """
    Build a meta summarization function using PEGASUS-MultiNews.
    You can swap the model_name here if you want MultiNews instead of XSum.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "google/pegasus-multi_news"
    tok_mn = AutoTokenizer.from_pretrained(model_name)
    mdl_mn = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def meta_func(text):
        return summarize_pegasus(
            text, tok_mn, mdl_mn,
            device=device,
            max_in=1024,   # MultiNews handles longer input
            max_out=128,   # longer summaries
            num_beams=5
        )
    return meta_func

```


```python
# --- Prepare leaves for tree building ---
# Convert sample (list of dicts) into list of (chunk_id, text) tuples
leaves = [(obj["chunk_id"], obj["text"]) for obj in sample]   # only the first 5 chunks
```


```python
# 3. Choose meta_summarize function
TRY_MULTINEWS = False
if TRY_MULTINEWS:
    try:
        meta_summarize = _build_meta_with_multinews()
        print("✅ meta: using pegasus-multi_news")
    except Exception as e:
        print(f"ℹ️ multi_news load failed → fallback to XSum: {e}")
        meta_summarize = _build_meta_with_xsum()
        print("✅ meta: using pegasus-xsum (prompt-enhanced)")
else:
    meta_summarize = _build_meta_with_xsum()
    print("✅ meta: using pegasus-xsum (prompt-enhanced)")

# 4. Fanout setting (2 for smoke test, 6 for full run)
fanout = 2 if len(leaves) <= 10 else 6
print(f"fanout = {fanout}, leaves = {len(leaves)}")

```

    ✅ meta: using pegasus-xsum (prompt-enhanced)
    fanout = 2, leaves = 5



```python
#three build
```


```python
# Convert list of dicts into list of (id, text) tuples
leaves = [(obj["chunk_id"], obj["text"]) for obj in sample]   # smoke test
```


```python
# --- 5) Build the hierarchical summary tree ---
level, nodes, current = 0, [], leaves  # current: list of (node_id, text_or_summary)

while len(current) > 1:
    level += 1
    grouped = [current[i:i + fanout] for i in range(0, len(current), fanout)]
    next_level = []
    for gi, group in enumerate(grouped, 1):
        children = [cid for cid, _ in group]
        texts    = [t   for _,   t in group]
        joined = "\n\n".join(texts)           # ← 간단/명확
        summ   = meta_summarize(joined)       # ← meta는 문자열 받음

        node_id = f"L{level}_N{gi:04d}"
        nodes.append({"node_id": node_id, "level": level, "children": children, "summary": summ})
        next_level.append((node_id, summ))
    current = next_level

root_id, root_summary = current[0]
print("✅ Tree built. Root:", root_id)
print("🧾 Root summary preview:\n", root_summary[:500], "...")
```

    ✅ Tree built. Root: L3_N0001
    🧾 Root summary preview:
     All images are copyrighted. ...



```python

```


```python
## 6. Save results + Preview root summary

from pathlib import Path

# Define output file paths
nodes_path = OUT / "tree_nodes.jsonl"   # full hierarchy (all nodes)
root_path  = OUT / "tree_root.json"     # only the root summary

## 6. Save results + Preview root summary

# Unpack root node (the last remaining node after tree building)
root_id, root_text = current[0]

# Save all nodes (entire tree) as JSONL: one JSON object per line
nodes_path.write_text(
    "\n".join(json.dumps(n, ensure_ascii=False) for n in nodes),
    encoding="utf-8"
)

# Save only the root summary as a JSON file (pretty-printed)
root_path.write_text(
    json.dumps({"root_id": root_id, "summary": root_text}, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# Print confirmation and preview the root summary
print("✅ tree_nodes.jsonl:", nodes_path)
print("✅ tree_root.json :", root_path)
print("\n📌 Root Summary:\n", root_path.read_text(encoding="utf-8"))
```

    ✅ tree_nodes.jsonl: /Users/jessicahong/gitclone/NLP_study/09_Mini_Project/13_RAPTOR/outputs/tree_nodes.jsonl
    ✅ tree_root.json : /Users/jessicahong/gitclone/NLP_study/09_Mini_Project/13_RAPTOR/outputs/tree_root.json
    
    📌 Root Summary:
     {
      "root_id": "L3_N0001",
      "summary": "All images are copyrighted."
    }



```python
def _build_meta_with_xsum():
    """
    Build a meta summarization function using PEGASUS-XSum.
    This function returns another function (_fn) that takes a list of texts
    (e.g., child summaries) and produces one higher-level summary.
    """
    def _fn(texts, max_in=512, max_out=220, num_beams=8):
        # Create a prompt from child summaries (bullet point style)
        prompt = (
            "Summarize the following bullet points into a cohesive 4–6 sentence paragraph. "
            "Write declarative sentences only (no questions, no instructions). "
            "Include main characters, setting, key events/conflict, and why it matters.\n\n"
            + "\n".join(f"- {t}" for t in texts)
        )

        # Run PEGASUS summarization on the prompt
        out = summarize_pegasus(prompt, max_in=max_in, max_out=max_out, num_beams=num_beams)

        # Quality check: retry if output looks bad (too short, question form, etc.)
        bad = (
            out.strip().endswith("?")
            or out.strip().lower().startswith(("how ", "do you ", "what "))
            or len(out.split()) < 35
        )
        if bad:
            # Retry once with longer output and wider beam search
            out = summarize_pegasus(prompt, max_in=max_in, max_out=max_out+40, num_beams=num_beams+2)

        return out

    return _fn
```


```python
# --- Prepare meta_summarize (try MultiNews, fallback to XSum) ---
try:
    # Try to build meta summarizer with PEGASUS-MultiNews
    meta_summarize = _build_meta_with_multinews()
    print("✅ meta: using pegasus-multi_news")
except Exception as e:
    # If MultiNews fails, fall back to PEGASUS-XSum
    print(f"ℹ️ multi_news load failed, falling back to XSum: {e}")
    meta_summarize = _build_meta_with_xsum()
    print("✅ meta: using pegasus-xsum (prompt-enhanced)")

```

    Some weights of PegasusForConditionalGeneration were not initialized from the model checkpoint at google/pegasus-multi_news and are newly initialized: ['model.decoder.embed_positions.weight', 'model.encoder.embed_positions.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


    ✅ meta: using pegasus-multi_news



```python

```
