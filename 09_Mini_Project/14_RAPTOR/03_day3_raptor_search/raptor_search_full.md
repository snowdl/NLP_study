```python
# Step 0 — Setup Base Path
from pathlib import Path
import os, json

# Print current working directory
#print("Current working directory:", os.getcwd())

# Set BASE path (up to the folder containing 'outputs')
#BASE = Path.home() / "/NLP_study/09_Mini_Project/13_RAPTOR"
#print("BASE path:", BASE)
```


```python
# Step 1 — JSONL Reader
"""
def read_jsonl(path):
    Read a JSONL (JSON Lines) file line by line.
    Args:path (Path or str): File path to .jsonl file
     Returns:
        list: A list of parsed JSON objects (dicts)
"""
```




    '\ndef read_jsonl(path):\n    Read a JSONL (JSON Lines) file line by line.\n    Args:path (Path or str): File path to .jsonl file\n     Returns:\n        list: A list of parsed JSON objects (dicts)\n'




```python
def read_jsonl(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            out.append(json.loads(line))
    return out
```


```python
# Step 2 — Load Data Files

# Load raw chunks and nodes from outputs folder
chunks_raw = read_jsonl(BASE / "outputs/chunks.jsonl")
nodes_raw  = read_jsonl(BASE / "outputs/tree_nodes.jsonl")

# Print basic stats
print("✅ Number of chunks:", len(chunks_raw))
print("✅ Number of nodes:", len(nodes_raw))
```

    ✅ Number of chunks: 227
    ✅ Number of nodes: 6



```python
# Step 3 — Inspect Key Structure (Sample Records)
```


```python
# Step 3 — Inspect Sample Records

# Print one example record from chunks and nodes
print("🔑 Chunk sample:", chunks_raw[0])
print("🔑 Node sample:", nodes_raw[0])
```

    🔑 Chunk sample: {'chunk_id': 'C0001', 'text': 'M r. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn’t think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley’s sister, but they hadn’t met for several years; in fact, Mrs. Dursley pretended she didn’t have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn’t want Dudley mixing with a child like that. When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair. None of them noticed a large, tawny owl flutter past the window. At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs.', 'tokens': 353}
    🔑 Node sample: {'node_id': 'L1_N0001', 'level': 1, 'children': ['C0001', 'C0002'], 'summary': 'This is the story of the Dursleys and the Potters.'}



```python
#Step 4. Schema Standardization Functions
```


```python
#Step 4.1 — Common Utility: Key Finder
```


```python
def pick_first_key(d, candidates, default=None):
    """
    Find the first matching key from a list of candidate keys
    that exists in the given dictionary.

    Args:
        d (dict): Dictionary to search.
        candidates (list): Possible key names.
        default: Value to return if no candidate is found.

    Returns:
        str or default: The first matching key, or the default value.
    """
    for k in candidates:
        if k in d:
            return k
    return default
```


```python
# Test dictionary
sample_dict = {
    "node_id": "L1_N0001",
    "summary": "This is a test summary",
    "children": ["C0001", "C0002"]
}

# Candidate lists
id_candidates = ["id", "node_id", "nid"]
summary_candidates = ["summary", "text", "desc"]
children_candidates = ["children", "kids", "child_ids"]

print("id key found:      ", pick_first_key(sample_dict, id_candidates))
print("summary key found: ", pick_first_key(sample_dict, summary_candidates))
print("children key found:", pick_first_key(sample_dict, children_candidates))

# Edge case: no matching key
print(" fallback (default):", pick_first_key(sample_dict, ["nonexistent", "ghost"], default="NONE"))

```

    id key found:       node_id
    summary key found:  summary
    children key found: children
     fallback (default): NONE



```python
#2) Normalize a Single Node
"""
    Normalize a single node dictionary into a standard format.
    Input example (raw node may vary):
    {
        "node_id": "L1_N0001",
        "summary": "This is a test summary",
        "children": ["C0001", "C0002"]
    }

    Output (standardized):
    {
        "id": "L1_N0001",
        "summary": "This is a test summary",
        "children": ["C0001", "C0002"]
    }
"""
```




    '\n    Normalize a single node dictionary into a standard format.\n    Input example (raw node may vary):\n    {\n        "node_id": "L1_N0001",\n        "summary": "This is a test summary",\n        "children": ["C0001", "C0002"]\n    }\n\n    Output (standardized):\n    {\n        "id": "L1_N0001",\n        "summary": "This is a test summary",\n        "children": ["C0001", "C0002"]\n    }\n'




```python
def normalize_node(n):

    # Identify possible key names
    id_key = pick_first_key(n, ['id', 'node_id', 'nid', 'name'])
    sum_key = pick_first_key(n, ['summary', 'text', 'desc', 'title'])
    ch_key  = pick_first_key(n, ['children', 'child_ids', 'kids', 'links'])

    # Extract values
    node_id = n.get(id_key, None)
    summary = n.get(sum_key, "")
    children = n.get(ch_key, [])

    # Fix edge cases
    if isinstance(children, str):
        children = [children]     # make single child into list
    if children is None:
        children = []             # ensure list

    # Return standardized format
    if node_id:
        return {"id": node_id, "summary": summary, "children": children}
    else:
        return None

```


```python
# Example raw node (simulate one record)
sample_node = {
    "node_id": "L1_N0001",
    "summary": "This is the story of the Dursleys and the Potters.",
    "children": ["C0001", "C0002"]
}

# Run normalization
normalized = normalize_node(sample_node)

print(" Raw node:")
print(sample_node)
print(" Normalized node:")
print(normalized)
```

     Raw node:
    {'node_id': 'L1_N0001', 'summary': 'This is the story of the Dursleys and the Potters.', 'children': ['C0001', 'C0002']}
     Normalized node:
    {'id': 'L1_N0001', 'summary': 'This is the story of the Dursleys and the Potters.', 'children': ['C0001', 'C0002']}



```python
#3) Standardize Multiple Nodes
```


```python
"""
    Normalize a list of raw node dictionaries into a clean list.

    Each output node will always have:
      - 'id' (string)
      - 'summary' (string)
      - 'children' (list of IDs)

    Example Input:
    [
        {"node_id": "L1_N0001", "summary": "Dursleys intro", "children": ["C0001"]},
        {"nid": "L1_N0002", "text": "Potters appear", "child_ids": ["C0002", "C0003"]}
    ]

    Example Output:
    [
        {"id": "L1_N0001", "summary": "Dursleys intro", "children": ["C0001"]},
        {"id": "L1_N0002", "summary": "Potters appear", "children": ["C0002", "C0003"]}
    ]
"""
```




    '\n    Normalize a list of raw node dictionaries into a clean list.\n\n    Each output node will always have:\n      - \'id\' (string)\n      - \'summary\' (string)\n      - \'children\' (list of IDs)\n\n    Example Input:\n    [\n        {"node_id": "L1_N0001", "summary": "Dursleys intro", "children": ["C0001"]},\n        {"nid": "L1_N0002", "text": "Potters appear", "child_ids": ["C0002", "C0003"]}\n    ]\n\n    Example Output:\n    [\n        {"id": "L1_N0001", "summary": "Dursleys intro", "children": ["C0001"]},\n        {"id": "L1_N0002", "summary": "Potters appear", "children": ["C0002", "C0003"]}\n    ]\n'




```python
def standardize_nodes(nodes):
    out = []
    for n in nodes:
        std = normalize_node(n)   # use single-node normalization
        if std:                   # keep only valid nodes with id
            out.append(std)
    return out

```


```python
# Example list of raw nodes
sample_nodes = [
    {"node_id": "L1_N0001", "summary": "Dursleys intro", "children": ["C0001"]},
    {"nid": "L1_N0002", "text": "Potters appear", "child_ids": ["C0002", "C0003"]},
    {"name": "L1_N0003", "desc": "Mysterious cat shows up", "kids": ["C0004"]}
]

# Apply standardization
normalized_nodes = standardize_nodes(sample_nodes)

print("Normalized nodes:")
for n in normalized_nodes:
    print(n)

```

    Normalized nodes:
    {'id': 'L1_N0001', 'summary': 'Dursleys intro', 'children': ['C0001']}
    {'id': 'L1_N0002', 'summary': 'Potters appear', 'children': ['C0002', 'C0003']}
    {'id': 'L1_N0003', 'summary': 'Mysterious cat shows up', 'children': ['C0004']}



```python
#4)Normalize a Single Chunk
```


```python
def normalize_chunk(c):
    # Identify possible key names
    id_key = pick_first_key(c, ['id', 'chunk_id', 'cid'])
    txt_key = pick_first_key(c, ['text', 'content', 'body'])

    # Extract values
    cid = c.get(id_key, None)
    txt = c.get(txt_key, "")

    if cid:
        return cid, txt
    else:
        return None, None

```


```python
# Example raw chunk
sample_chunk = {
    "chunk_id": "C0001",
    "text": "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say..."
}

# Run normalization
cid, txt = normalize_chunk(sample_chunk)

print("Raw chunk:", sample_chunk)
print("Normalized output:")
print("ID:", cid)
print("Text:", txt[:80] + "...")

```

    Raw chunk: {'chunk_id': 'C0001', 'text': 'Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say...'}
    Normalized output:
    ID: C0001
    Text: Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say......



```python
#5) Standardize Multiple Chunks
```


```python
def standardize_chunks(chunks):
    out = {}
    for c in chunks:
        cid, txt = normalize_chunk(c)  # reuse the single-chunk normalizer
        if cid:
            out[cid] = txt
    return out
```


```python
# Sample raw chunks (mixed schemas)
sample_chunks = [
    {"chunk_id": "C0001", "text": "Mr. and Mrs. Dursley, of number four, Privet Drive..."},
    {"cid": "C0002", "content": "Dudley was now having a tantrum and throwing his cereal..."},
    {"id": "C0003", "body": "There was a tabby cat standing on the corner of Privet Drive..."}
]

chunk_map = standardize_chunks(sample_chunks)

print("✅ Standardized chunk map keys:", list(chunk_map.keys()))
print("📝 C0001 preview:", chunk_map["C0001"][:70] + "...")
print("📝 C0002 preview:", chunk_map["C0002"][:70] + "...")
print("📝 C0003 preview:", chunk_map["C0003"][:70] + "...")
```

    ✅ Standardized chunk map keys: ['C0001', 'C0002', 'C0003']
    📝 C0001 preview: Mr. and Mrs. Dursley, of number four, Privet Drive......
    📝 C0002 preview: Dudley was now having a tantrum and throwing his cereal......
    📝 C0003 preview: There was a tabby cat standing on the corner of Privet Drive......



```python
#Step 5. Run Standardization
```


```python
# Apply normalization to raw data

nodes = standardize_nodes(nodes_raw)
chunk_map = standardize_chunks(chunks_raw)

print("✅ Number of standardized nodes:", len(nodes))
print("✅ Number of standardized chunks:", len(chunk_map))

print("\n📝 Example standardized node:")
print(nodes[0])

print("\n📝 Example standardized chunk:")
first_chunk_id = list(chunk_map.keys())[0]
print(first_chunk_id, "→", chunk_map[first_chunk_id][:80] + "...")

```

    ✅ Number of standardized nodes: 6
    ✅ Number of standardized chunks: 227
    
    📝 Example standardized node:
    {'id': 'L1_N0001', 'summary': 'This is the story of the Dursleys and the Potters.', 'children': ['C0001', 'C0002']}
    
    📝 Example standardized chunk:
    C0001 → M r. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they...



```python
print(f"Collected {len(results)} results")

# save results as 'res' so it looks same as function version
res = results

for r in res:
    print("📌 Node:", r["node_id"], "| score:", round(r["score"], 4))
    print("📝 Summary:", (r["node_summary"] or "")[:140] + ("..." if len(r["node_summary"]) > 140 else ""))
    print("🔗 Chunks:", r["linked_chunk_ids"])
    for i, t in enumerate(r["chunk_texts"], 1):
        print(f"   [{i}] {t[:160]}{'...' if len(t) > 160 else ''}")
    print("-" * 60)
```

    Collected 2 results
    📌 Node: L1_N0003 | score: 0.1081
    📝 Summary: Dudley and Petunia Dursley had a strange day.
    🔗 Chunks: ['C0005']
    ------------------------------------------------------------
    📌 Node: L3_N0001 | score: 0.0
    📝 Summary: All images are copyrighted.
    🔗 Chunks: []
    ------------------------------------------------------------



```python
#Step 6. RAPTOR Search Function

"""
    Retrieve relevant RAPTOR nodes and their linked chunks using a simple TF-IDF search
    over node summaries.

    Args:
        query: user query string
        nodes: standardized node list (each has keys: id, summary, children)
        chunk_map: dict {chunk_id: text}
        topk_nodes: number of top nodes to return
        max_chunks_per_node: max number of chunk texts to attach per node
        chunk_id_prefix: filter for children IDs that represent chunks (e.g., "C")

    Returns:
        List of dicts:
        [
          {
            "node_id": str,
            "node_summary": str,
            "score": float,
            "linked_chunk_ids": [str, ...],
            "chunk_texts": [str, ...]
          },
          ...
        ]
"""
```




    '\n    Retrieve relevant RAPTOR nodes and their linked chunks using a simple TF-IDF search\n    over node summaries.\n\n    Args:\n        query: user query string\n        nodes: standardized node list (each has keys: id, summary, children)\n        chunk_map: dict {chunk_id: text}\n        topk_nodes: number of top nodes to return\n        max_chunks_per_node: max number of chunk texts to attach per node\n        chunk_id_prefix: filter for children IDs that represent chunks (e.g., "C")\n\n    Returns:\n        List of dicts:\n        [\n          {\n            "node_id": str,\n            "node_summary": str,\n            "score": float,\n            "linked_chunk_ids": [str, ...],\n            "chunk_texts": [str, ...]\n          },\n          ...\n        ]\n'




```python

```


```python

```


```python

```


```python
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def raptor_search(
    query: str,
    nodes: List[Dict[str, Any]],
    chunk_map: Dict[str, str],
    topk_nodes: int = 2,
    max_chunks_per_node: int = 2,
    chunk_id_prefix: str = "C",
) -> List[Dict[str, Any]]:
    # Always return a list
    if not nodes or not isinstance(chunk_map, dict) or not chunk_map:
        return []

    # (1) preprocess
    triples = [(n["id"], n.get("summary", "") or "", n.get("children", []) or []) for n in nodes]
    node_ids       = [nid for nid, _, _ in triples]
    node_summaries = [s for _, s, _ in triples]
    node_children  = [ch for _, _, ch in triples]

    # (2)(3) tf-idf + vectorize
    vec  = TfidfVectorizer().fit(node_summaries + [query])
    qvec = vec.transform([query])
    nmat = vec.transform(node_summaries)

    # (4) rank
    sims  = cosine_similarity(qvec, nmat)[0]
    order = sims.argsort()[::-1]

    # (5) collect
    results: List[Dict[str, Any]] = []
    for idx in order[:topk_nodes]:
        nid, nsum, children = node_ids[idx], node_summaries[idx], node_children[idx]
        child_chunk_ids = [c for c in children if isinstance(c, str) and c.startswith(chunk_id_prefix)]
        child_chunk_ids = child_chunk_ids[:max_chunks_per_node]
        texts = [chunk_map[cid] for cid in child_chunk_ids if cid in chunk_map]
        results.append({
            "node_id": nid,
            "node_summary": nsum,
            "score": float(sims[idx]),
            "linked_chunk_ids": child_chunk_ids,
            "chunk_texts": texts
        })
    return results  # ← 반드시 리스트 반환

```


```python
# Step 7 — Run RAPTOR Search & Show Results
query = "What strange events happened on Privet Drive?"
res = raptor_search(query, nodes, chunk_map, topk_nodes=2, max_chunks_per_node=2)

print("res type:", type(res), "len:", len(res))
for r in res:
    print("Node:", r["node_id"], "| score:", round(r["score"], 4))
    print("Summary:", (r["node_summary"] or "")[:140] + ("..." if len(r["node_summary"]) > 140 else ""))
    print("Chunks:", r["linked_chunk_ids"])
    for i, t in enumerate(r["chunk_texts"], 1):
        print(f"   [{i}] {t[:160]}{'...' if len(t) > 160 else ''}")
    print("-"*60)

```

    res type: <class 'list'> len: 2
    Node: L1_N0003 | score: 0.1081
    Summary: Dudley and Petunia Dursley had a strange day.
    Chunks: ['C0005']
       [1] When Dudley had been put to bed, he went into the living room in time to catch the last report on the evening news:
    
    “And finally, bird-watchers everywhere have...
    ------------------------------------------------------------
    Node: L3_N0001 | score: 0.0
    Summary: All images are copyrighted.
    Chunks: []
    ------------------------------------------------------------



```python
print("nodes count:", 0 if nodes is None else len(nodes))
print("chunk_map type/len:", type(chunk_map), 0 if not isinstance(chunk_map, dict) else len(chunk_map))
print("raptor_search is:", raptor_search)
print("raptor_search doc:", getattr(raptor_search, "__doc__", None))
```

    nodes count: 6
    chunk_map type/len: <class 'dict'> 227
    raptor_search is: <function raptor_search at 0x105990040>
    raptor_search doc: None



```python

```


```python

```


```python

```


```python

```


```python

```
