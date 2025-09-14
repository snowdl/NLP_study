```python
import os, json
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Disable parallelism warning from Hugging Face tokenizers (optional)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```


```python
#Step 2. 데이터 불러오기
```


```python
# === 1. Define base paths ===
BASE = Path.cwd().parents[0] / "13_RAPTOR" if (Path.cwd().name != "13_RAPTOR") else Path.cwd()
OUT  = BASE / "outputs"
```


```python
# === 2. Define file paths ===
chunks_path = OUT / "chunks.jsonl"
summ_smoke  = OUT / "chunk_summaries_smoke.jsonl"
summ_all    = OUT / "chunk_summaries.jsonl"
nodes_path  = OUT / "tree_nodes.jsonl"
```


```python
# Use smoke summaries if available, otherwise full summaries
summ_path = summ_smoke if summ_smoke.exists() else summ_all
```


```python
# === 3. Load JSONL data ===
# Original text chunks
chunk_text = {json.loads(l)["chunk_id"]: json.loads(l)["text"] 
              for l in open(chunks_path, encoding="utf-8")}
# Leaf-level summaries
leaf_summary = {json.loads(l)["chunk_id"]: json.loads(l)["summary"] 
                for l in open(summ_path, encoding="utf-8")}
# Node-level summaries
nodes = [json.loads(l) for l in open(nodes_path, encoding="utf-8")]

```


```python
# === 4. Organize node info ===
node_info = {nd["node_id"]: (nd["level"], nd["children"], nd["summary"]) 
             for nd in nodes}
```


```python
# === 5. Build search corpus (nodes + leaves) ===
corpus_ids, corpus_txt = [], []
for nid, (_, _, summ) in node_info.items():
    corpus_ids.append(nid); corpus_txt.append(summ)
for cid, summ in leaf_summary.items():
    corpus_ids.append(cid); corpus_txt.append(summ)

print("✅ Load complete:", len(corpus_ids), "summaries")
```

    ✅ Load complete: 11 summaries



```python
#Step 3. 간단 임베딩 인덱스
```


```python
# === 1. Load libraries & data ===
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError as e:
    raise RuntimeError("Required libraries are missing. Please install first.") from e
```


```python
# === 2. Select embedding backend (SBERT → fallback TF-IDF) ===
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_mat = model.encode(
        corpus_txt,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    backend = "sbert"
    print("✅ Using SBERT embeddings")
except Exception:
    # === 3. Fallback to TF-IDF ===
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)
    emb_mat = vect.fit_transform(corpus_txt)
    backend = "tfidf"
    print("✅ Using TF-IDF (fallback)")

```


    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    ✅ Using SBERT embeddings



```python
# === 3. Encode query ===
def encode_query(query: str):
    if backend == "sbert":
        return model.encode([query], normalize_embeddings=True)[0]
    else:
        return vect.transform([query])
```


```python
# === 4. Search top-k ===
def topk_in_corpus(query: str, k: int = 5):
    qv = encode_query(query)
    if backend == "sbert":
        sims = emb_mat @ qv  # cosine similarity via dot product
    else:
        sims = cosine_similarity(emb_mat, qv).ravel()
    idx = np.argsort(-sims)[:k]
    return [(corpus_ids[i], float(sims[i])) for i in idx]
```


```python
# === 5. Display results immediately ===
query = "large language models"
results = topk_in_corpus(query, k=5)

print("🔎 Query:", query)
for rid, score in results:
    snippet = chunk_text.get(rid, "")[:80]  # show first 80 characters
    print(f"  {rid} | {score:.4f} | {snippet}...")
```

    🔎 Query: large language models
      L1_N0002 | 0.0857 | ...
      L2_N0001 | 0.0857 | ...
      L2_N0002 | 0.0693 | ...
      L3_N0001 | 0.0693 | ...
      C0004 | 0.0534 | Dursley; she always got so upset at any mention of her sister. He didn’t blame h...



```python
#Step 4. Retrieval & 답변
```


```python
# === Text & retrieval utilities (highly modularized) ===
import re
from typing import List, Tuple, Dict, Iterable, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```


```python
# --- 0) Constants & precompiled regex ---
STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","at","for","with",
    "is","are","was","were","do","does","did","what","who","where",
    "when","how","why"
}
SENT_SPLIT_RE = re.compile(r'(?<=[.!?]")\s+|(?<=[.!?])\s+')
MR_FIX_RE     = re.compile(r"\bM\s+r\.")
```


```python
# --- 1) Basic text cleaning / splitting / keywords ---
def clean_text(s: str) -> str:
    """Light cleanup: fix 'M r.' → 'Mr.', collapse spaces."""
    s = MR_FIX_RE.sub("Mr.", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def split_sentences(text: str) -> List[str]:
    """Split text into sentences by quotes/terminal punctuation."""
    return [t.strip() for t in SENT_SPLIT_RE.split(text) if t.strip()]

def extract_keywords(q: str, stop: Iterable[str] = STOPWORDS) -> List[str]:
    """Lowercase, tokenize, remove stopwords/short tokens; return unique sorted."""
    toks = re.findall(r"[A-Za-z']+", q.lower())
    return sorted({t for t in toks if t not in stop and len(t) >= 3})

```


```python
# --- 2) ID helpers & de-duplication ---
def is_chunk_id(x) -> bool:
    """Return True if id looks like a chunk id (e.g., 'C0001')."""
    return isinstance(x, str) and x.startswith("C")

def unique_stable(seq: Iterable[str]) -> List[str]:
    """De-duplicate while preserving order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

```


```python
# --- 3) Graph traversal (nodes → leaf chunks) ---
def descend_to_chunks(
    start_ids: List[str],
    node_info: Dict[str, Tuple[int, List[str], str]],
    leaf_summary: Dict[str, str],
    max_hops: int = 3
) -> List[str]:
    """
    Breadth-first descend from nodes to leaf chunk ids, up to max_hops.
    Accepts that some leaves may appear only in leaf_summary.
    """
    out, frontier = [], list(start_ids)
    for _ in range(max_hops):
        nxt = []
        for _id in frontier:
            if is_chunk_id(_id):
                out.append(_id)
            elif _id in node_info:
                _, children, _ = node_info[_id]
                nxt.extend(children)
            elif _id in leaf_summary:  # Leaf known only by summary
                out.append(_id)
        frontier = nxt
        if not frontier:
            break
    return unique_stable(out)
```


```python
# --- 4) TF-IDF building & ranking helpers ---
def build_tfidf(
    texts: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 50_000,
    min_df: int = 1
) -> Tuple[TfidfVectorizer, "scipy.sparse.spmatrix"]:
    """Fit a TF-IDF vectorizer on texts and return (vectorizer, matrix)."""
    vect = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=min_df)
    M = vect.fit_transform(texts)
    return vect, M

def rank_with_tfidf(
    texts: List[str],
    query: str,
    k: int,
    vect: Optional[TfidfVectorizer] = None,
    M: Optional["scipy.sparse.spmatrix"] = None
) -> List[Tuple[int, float]]:
    """
    Rank texts by cosine similarity between TF-IDF(texts) and TF-IDF(query).
    If vect/M are not provided, fit on the fly.
    Returns list of (index, score) sorted desc by score, top-k.
    """
    if vect is None or M is None:
        vect, M = build_tfidf(texts)
    qv = vect.transform([query])
    sims = cosine_similarity(M, qv).ravel()
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]
```


```python
# --- 5) Candidate chunk collection ---
def collect_candidate_chunks(
    candidate_ids: List[str],
    chunk_text: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """
    From candidate chunk ids, collect aligned (ids, cleaned_texts).
    Filters out chunks not present in chunk_text.
    """
    cids, ctexts = [], []
    for cid in candidate_ids:
        if cid in chunk_text:
            cids.append(cid)
            ctexts.append(clean_text(chunk_text[cid]))
    return cids, ctexts
```


```python
# === Part 1/6: Imports, constants, and regex ===
import re
from typing import List, Tuple, Dict, Iterable, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","at","for","with",
    "is","are","was","were","do","does","did","what","who","where",
    "when","how","why"
}
SENT_SPLIT_RE = re.compile(r'(?<=[.!?]")\s+|(?<=[.!?])\s+')
MR_FIX_RE     = re.compile(r"\bM\s+r\.")

```


```python
# === Part 2/6: Text utilities and ID helpers ===
def clean_text(s: str) -> str:
    """Light cleanup: fix 'M r.' → 'Mr.', collapse spaces."""
    s = MR_FIX_RE.sub("Mr.", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def split_sentences(text: str) -> List[str]:
    """Split text into sentences by quotes/terminal punctuation."""
    return [t.strip() for t in SENT_SPLIT_RE.split(text) if t.strip()]

def extract_keywords(q: str, stop: Iterable[str] = STOPWORDS) -> List[str]:
    """Lowercase tokenize, remove stopwords/short tokens; return unique sorted keywords."""
    toks = re.findall(r"[A-Za-z']+", q.lower())
    return sorted({t for t in toks if t not in stop and len(t) >= 3})

def is_chunk_id(x) -> bool:
    """Return True if id looks like a chunk id (e.g., 'C0001')."""
    return isinstance(x, str) and x.startswith("C")

def unique_stable(seq: Iterable[str]) -> List[str]:
    """De-duplicate while preserving original order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

```


```python
# === Part 3/6: Graph traversal (nodes → leaf chunks) ===
def descend_to_chunks(
    start_ids: List[str],
    node_info: Dict[str, Tuple[int, List[str], str]],
    leaf_summary: Dict[str, str],
    max_hops: int = 3
) -> List[str]:
    """
    Breadth-first descend from nodes to leaf chunk ids, up to max_hops.
    Accepts that some leaves may appear only in leaf_summary.
    """
    out, frontier = [], list(start_ids)
    for _ in range(max_hops):
        nxt = []
        for _id in frontier:
            if is_chunk_id(_id):
                out.append(_id)
            elif _id in node_info:
                _, children, _ = node_info[_id]
                nxt.extend(children)
            elif _id in leaf_summary:  # Leaf known only by summary
                out.append(_id)
        frontier = nxt
        if not frontier:
            break
    return unique_stable(out)

```


```python
# === Part 4/6: TF-IDF helpers (fit and rank) ===
def build_tfidf(
    texts: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 50_000,
    min_df: int = 1
) -> Tuple[TfidfVectorizer, "scipy.sparse.spmatrix"]:
    """Fit a TF-IDF vectorizer on texts and return (vectorizer, matrix)."""
    vect = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=min_df)
    M = vect.fit_transform(texts)
    return vect, M

def rank_with_tfidf(
    texts: List[str],
    query: str,
    k: int,
    vect: Optional[TfidfVectorizer] = None,
    M: Optional["scipy.sparse.spmatrix"] = None
) -> List[Tuple[int, float]]:
    """
    Rank texts by cosine similarity between TF-IDF(texts) and TF-IDF(query).
    If vect/M are not provided, fit on the fly.
    Returns list of (index, score) sorted desc by score, top-k.
    """
    if vect is None or M is None:
        vect, M = build_tfidf(texts)
    qv = vect.transform([query])
    sims = cosine_similarity(M, qv).ravel()
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]

```


```python
# === Part 5/6: Retrieval orchestration (finely split) ===
def get_mixed_hits(query: str, topk_nodes: int, topk_fn) -> List[Tuple[str, float]]:
    """Call embedding/top-k backend to fetch mixed node/leaf candidates."""
    return topk_fn(query, k=topk_nodes)

def get_candidate_chunk_ids(
    hit_ids: List[str],
    node_info: Dict[str, Tuple[int, List[str], str]],
    leaf_summary: Dict[str, str],
    max_hops: int = 3
) -> List[str]:
    """Traverse graph to collect leaf chunk candidates from mixed ids."""
    return descend_to_chunks(hit_ids, node_info=node_info, leaf_summary=leaf_summary, max_hops=max_hops)

def ensure_candidates_with_fallback(candidate_ids: List[str], hit_ids: List[str]) -> List[str]:
    """If traversal produced nothing, fall back to chunk-looking ids from hits."""
    return candidate_ids if candidate_ids else [hid for hid in hit_ids if is_chunk_id(hid)]

def rerank_candidates(
    query: str,
    candidate_chunk_ids: List[str],
    chunk_text: Dict[str, str],
    topk_chunks: int
) -> List[Tuple[str, float]]:
    """Compute TF-IDF similarities over candidate chunk texts and return top-k."""
    # Collect chunk texts
    cids, ctexts = [], []
    for cid in candidate_chunk_ids:
        if cid in chunk_text:
            cids.append(cid); ctexts.append(clean_text(chunk_text[cid]))
    if not ctexts:
        return []
    # Rank and map indices back to ids
    ranks = rank_with_tfidf(ctexts, query, k=topk_chunks)
    return [(cids[i], score) for (i, score) in ranks]

def raptor_retrieve(
    query: str,
    node_info: Dict[str, Tuple[int, List[str], str]],
    leaf_summary: Dict[str, str],
    chunk_text: Dict[str, str],
    topk_nodes: int = 6,
    topk_chunks: int = 5,
    topk_fn=None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    1) get_mixed_hits
    2) get_candidate_chunk_ids
    3) ensure_candidates_with_fallback
    4) rerank_candidates
    """
    if topk_fn is None:
        raise ValueError("topk_fn is required (e.g., topk_in_corpus)")

    hits = get_mixed_hits(query, topk_nodes, topk_fn)  # [(id, score), ...]
    hit_ids = [hid for (hid, _) in hits]

    candidates = get_candidate_chunk_ids(hit_ids, node_info=node_info, leaf_summary=leaf_summary, max_hops=3)
    candidates = ensure_candidates_with_fallback(candidates, hit_ids)

    top_chunks = rerank_candidates(query, candidates, chunk_text, topk_chunks)
    return {"nodes": hits, "chunks": top_chunks}

```


```python
# === Part 6/6: Answer composition + quick test helpers ===
def pick_key_sentences(query: str, text: str, sent_per_chunk: int = 2) -> str:
    """
    From a chunk:
      1) split into sentences,
      2) filter by keyword presence (if any),
      3) TF-IDF re-rank filtered sentences vs. query,
      4) return top-N concatenated.
    """
    if not text:
        return ""
    sents = split_sentences(text)
    if not sents:
        return ""
    kws = extract_keywords(query)
    filtered = [s for s in sents if any(k in s.lower() for k in kws)] or sents
    ranks = rank_with_tfidf(filtered, query, k=sent_per_chunk)
    return " ".join(filtered[i] for (i, _) in ranks)

def compose_answer_from_chunks(
    query: str,
    top_chunks: List[Tuple[str, float]],
    chunk_text: Dict[str, str],
    max_chars: int = 400
) -> str:
    """Build an answer by taking top sentences from each top chunk; truncate by max_chars."""
    snippets = []
    for cid, _ in top_chunks:
        picked = pick_key_sentences(query, clean_text(chunk_text.get(cid, "")))
        if picked:
            snippets.append(f"[{cid}] {picked}")
        if len(" ".join(snippets)) > max_chars:
            break
    return " ".join(snippets).strip()

def answer_query(
    query: str,
    node_info: Dict[str, Tuple[int, List[str], str]],
    leaf_summary: Dict[str, str],
    chunk_text: Dict[str, str],
    topk_nodes: int = 6,
    topk_chunks: int = 5,
    sent_per_chunk: int = 2,
    max_chars: int = 400,
    topk_fn=None
) -> Dict[str, object]:
    """End-to-end: retrieval + light extractive summarization."""
    res = raptor_retrieve(
        query=query,
        node_info=node_info,
        leaf_summary=leaf_summary,
        chunk_text=chunk_text,
        topk_nodes=topk_nodes,
        topk_chunks=topk_chunks,
        topk_fn=topk_fn
    )
    answer = compose_answer_from_chunks(query, res.get("chunks", []), chunk_text, max_chars)
    if not answer:
        # Fallback to leaf summaries if nothing was extracted
        fb = []
        for hid, _ in (topk_fn(query, k=6) if topk_fn else []):
            if is_chunk_id(hid) and hid in leaf_summary:
                fb.append(f"[{hid}-summary] {leaf_summary[hid]}")
                if len(fb) >= 2:
                    break
        answer = " ".join(fb).strip() if fb else "(no matching evidence)"
    return {"retrieval": res, "answer": answer}

def print_top_items(title: str, items: List[Tuple[str, float]], limit: int = 5, preview_map: Dict[str, str] = None, preview_len: int = 80):
    """Pretty-print top (id, score[, preview]) items for quick inspection."""
    print(f"\n{title}")
    print("-" * len(title))
    for i, (iid, score) in enumerate(items[:limit], 1):
        if preview_map is not None:
            snippet = clean_text(preview_map.get(iid, ""))[:preview_len]
            print(f"{i:>2}. {iid} | {score:.4f} | {snippet}...")
        else:
            print(f"{i:>2}. {iid} | {score:.4f}")

def demo_query(query: str, topk_nodes: int = 6, topk_chunks: int = 5, sent_per_chunk: int = 2, max_chars: int = 400):
    """
    One-liner demo: run retrieval + show nodes/chunks + print final answer.
    Requires `topk_in_corpus`, `node_info`, `leaf_summary`, `chunk_text` in scope.
    """
    res = raptor_retrieve(
        query=query,
        node_info=node_info,
        leaf_summary=leaf_summary,
        chunk_text=chunk_text,
        topk_nodes=topk_nodes,
        topk_chunks=topk_chunks,
        topk_fn=topk_in_corpus
    )
    print_top_items("Top nodes/leaves", res["nodes"], preview_map=leaf_summary)
    print_top_items("Top chunks", res["chunks"], preview_map=chunk_text)

    out = answer_query(
        query=query,
        node_info=node_info,
        leaf_summary=leaf_summary,
        chunk_text=chunk_text,
        topk_nodes=topk_nodes,
        topk_chunks=topk_chunks,
        sent_per_chunk=sent_per_chunk,
        max_chars=max_chars,
        topk_fn=topk_in_corpus
    )
    print("\n🧩 Answer")
    print("---------")
    print(out["answer"])
    return out

```


```python
# --- Quick smoke test ---
try:
    _ = (topk_in_corpus, node_info, leaf_summary, chunk_text)
except NameError as e:
    raise RuntimeError(
        "Missing required globals: topk_in_corpus, node_info, leaf_summary, chunk_text.\n"
        "Load them first using your existing RAPTOR setup."
    ) from e

test_query = "large language models evaluation and tree index"
print(f"🔎 Query: {test_query}")
_ = demo_query(test_query, topk_nodes=6, topk_chunks=5, sent_per_chunk=2, max_chars=400)

```

    🔎 Query: large language models evaluation and tree index
    
    Top nodes/leaves
    ----------------
     1. L1_N0002 | 0.0499 | ...
     2. L2_N0001 | 0.0499 | ...
     3. C0004 | -0.0058 | The first thing Mr....
     4. L2_N0002 | -0.0075 | ...
     5. L3_N0001 | -0.0075 | ...
    
    Top chunks
    ----------
     1. C0001 | 0.1231 | Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they ...
     2. C0002 | 0.0969 | Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudl...
     3. C0003 | 0.0740 | Dursley arrived in the Grunnings parking lot, his mind back on drills. Mr. Dursl...
     4. C0004 | 0.0729 | Dursley; she always got so upset at any mention of her sister. He didn’t blame h...
     5. C0005 | 0.0698 | When Dudley had been put to bed, he went into the living room in time to catch t...
    
    🧩 Answer
    ---------
    [C0001] None of them noticed a large, tawny owl flutter past the window. He was a big, beefy man with hardly any neck, although he did have a very large mustache. [C0002] As he drove toward town he thought of nothing except a large order of drills he was hoping to get that day. It was on the corner of the street that he noticed the first sign of something peculiar — a cat reading a map. [C0003] It was on his way back past them, clutching a large doughnut in a bag, that he caught a few words of what they were saying. He didn’t see the owls swooping past in broad daylight, though people down in the street did; they pointed and gazed open-mouthed as owl after owl sped overhead.



```python
#Step 5. 테스트!
```


```python
def pretty_answer(query: str, topk_nodes=6, topk_chunks=5, sent_per_chunk=2, max_chars=400):
    # Safety checks for required globals
    try:
        _ = (node_info, leaf_summary, chunk_text, topk_in_corpus)
    except NameError as e:
        raise RuntimeError(
            "Missing required globals. Make sure these exist in the session:\n"
            " - node_info\n - leaf_summary\n - chunk_text\n - topk_in_corpus"
        ) from e

    out = answer_query(
        query=query,
        node_info=node_info,
        leaf_summary=leaf_summary,
        chunk_text=chunk_text,
        topk_nodes=topk_nodes,
        topk_chunks=topk_chunks,
        sent_per_chunk=sent_per_chunk,
        max_chars=max_chars,
        topk_fn=topk_in_corpus,   # <-- pass the retrieval backend
    )

    print("🔎 Q:", query)
    print("💬 A:", out["answer"])
    return out

```


```python
pretty_answer("Who is Harry Potter's best friend?")
pretty_answer("What strange events happened on Privet Drive?")
```

    🔎 Q: Who is Harry Potter's best friend?
    💬 A: [C0003] He was sure there were lots of people called Potter who had a son called Harry. Come to think of it, he wasn’t even sure his nephew was called Harry. [C0001] Potter was Mrs. None of them noticed a large, tawny owl flutter past the window. [C0004] Rejoice, for You-Know-Who has gone at last! Dursley; she always got so upset at any mention of her sister. [C0002] Dursley couldn’t bear people who dressed in funny clothes — the getups you saw on young people! Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls.
    🔎 Q: What strange events happened on Privet Drive?
    💬 A: [C0002] There was a tabby cat standing on the corner of Privet Drive, but there wasn’t a map in sight. It was now reading the sign that said Privet Drive — no, looking at the sign; cats couldn’t read maps or signs. [C0001] Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country.





    {'retrieval': {'nodes': [('L1_N0003', 0.2585400640964508),
       ('C0005', 0.2585400640964508),
       ('L1_N0001', 0.2353595793247223),
       ('C0001', 0.2353595793247223),
       ('C0002', 0.1981765478849411),
       ('C0003', 0.1439816802740097)],
      'chunks': [('C0002', 0.11857809291952338),
       ('C0001', 0.08515568181128577),
       ('C0003', 0.03252912274508405),
       ('C0005', 0.010153593782822781)]},
     'answer': '[C0002] There was a tabby cat standing on the corner of Privet Drive, but there wasn’t a map in sight. It was now reading the sign that said Privet Drive — no, looking at the sign; cats couldn’t read maps or signs. [C0001] Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country.'}




```python

```
