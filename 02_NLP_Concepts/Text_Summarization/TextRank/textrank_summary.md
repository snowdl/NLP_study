```python
# -------------------------------------------------
# Minimal TextRank pipeline (beginner-friendly)
# 0) sentence_split
# 1) build_sim_matrix
# 2) rank_sentences
# 3) summarize
# -------------------------------------------------

```


```python
import re
import math
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```


```python
from nltk.tokenize import sent_tokenize
text = "Hello. Today I am experimenting with TextRank. Does it work well?"
print(sent_tokenize(text))
```

    ['Hello.', 'Today I am experimenting with TextRank.', 'Does it work well?']



```python
demo_text = (
    "TextRank is a graph-based ranking algorithm for NLP. "
    "It builds a graph of sentences using pairwise similarity. "
    "Then it applies PageRank to score sentences by importance. "
    "This method is often used for extractive summarization. "
    "Even beginners can implement a minimal version quickly."
)

```


```python
print("📌 Original text:")
print(demo_text)
```

    📌 Original text:
    TextRank is a graph-based ranking algorithm for NLP. It builds a graph of sentences using pairwise similarity. Then it applies PageRank to score sentences by importance. This method is often used for extractive summarization. Even beginners can implement a minimal version quickly.



```python
# -----------------------------
# 0) Sentence splitter
#    - Prefer NLTK's sent_tokenize (more accurate)
#    - Fallback to a simple regex if NLTK/punkt is unavailable
# -----------------------------
```


```python
# -----------------------------
def sentence_split(text):
    """
    Split a long text into sentences.

    Steps:
      1) Clean whitespace (collapse multiple spaces/newlines, strip leading/trailing spaces)
      2) Try NLTK's sent_tokenize (more accurate sentence boundary detection)
         - If the 'punkt' model is missing, download it automatically
      3) If NLTK is unavailable or fails, fall back to a simple regex-based splitter
      4) Strip each sentence and remove empty strings
    """
    # Step 1: Basic cleaning
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    try:
        # Step 2: Attempt NLTK-based sentence tokenization
        import nltk
        from nltk.tokenize import sent_tokenize

        # Ensure 'punkt' tokenizer data exists
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        sents = sent_tokenize(text)

    except Exception:
        # Step 3: Fallback to regex-based splitting
        # Regex explanation:
        #   (?<=[.!?]) : position immediately AFTER a period, question mark, or exclamation mark
        #   \s+        : one or more spaces
        sents = re.split(r"(?<=[.!?])\s+", text)

    # Step 4: Final cleanup → strip whitespace and discard empties
    return [s.strip() for s in sents if s and s.strip()]
```


```python
# Step 0) Sentence split
sents = sentence_split(demo_text)
print("\n✅ Step 0 - Sentences:")
for i, s in enumerate(sents, 1):
    print(f"{i}. {s}")

```

    
    ✅ Step 0 - Sentences:
    1. TextRank is a graph-based ranking algorithm for NLP.
    2. It builds a graph of sentences using pairwise similarity.
    3. Then it applies PageRank to score sentences by importance.
    4. This method is often used for extractive summarization.
    5. Even beginners can implement a minimal version quickly.



```python
# -----------------------------
# 1) Similarity matrix (TF-IDF + cosine)
#    * This is where we tune "vector settings"
# -----------------------------
```


```python
def build_sim_matrix(sent_list):
    """
    Build a sentence-by-sentence cosine similarity matrix using TF-IDF.
    Key vector settings:
      - stop_words=None  : good for Korean or mixed text
                           (for pure English, you can set 'english')
      - ngram_range=(1,2): unigrams + bigrams (helps with short sentences)
      - lowercase=True   : normalize casing (mostly for English)
      - max_df=0.95      : ignore terms that appear in >95% of sentences
      - min_df=1         : include terms that appear in >=1 sentence
    """
    if not sent_list:
        return None

    vectorizer = TfidfVectorizer(
        stop_words=None,     # Korean: None, English-only: 'english'
        ngram_range=(1, 2),  # use unigrams + bigrams
        lowercase=True,
        max_df=0.95,
        min_df=1
    )
    X = vectorizer.fit_transform(sent_list)
    sim = cosine_similarity(X)
    return sim
```


```python
# Step 1) Similarity matrix
sim_matrix = build_sim_matrix(sents)
print("\n✅ Step 1 - Similarity matrix (rounded):")
print(sim_matrix.round(2))
```

    
    ✅ Step 1 - Similarity matrix (rounded):
    [[1.   0.05 0.   0.09 0.  ]
     [0.05 1.   0.09 0.   0.  ]
     [0.   0.09 1.   0.   0.  ]
     [0.09 0.   0.   1.   0.  ]
     [0.   0.   0.   0.   1.  ]]



```python
# -----------------------------
# 2) TextRank scores (PageRank on similarity graph)
#    - Build graph from sim matrix
#    - Remove self-loops
#    - Optionally prune very small edges (bottom 25%) to reduce noise
# ----------------------------
```


```python
def rank_sentences(sim):
    if sim is None or sim.size == 0:
        return []

    G = nx.from_numpy_array(sim)
    G.remove_edges_from(nx.selfloop_edges(G))

    positives = sim[sim > 0]
    if positives.size > 0:
        thr = np.percentile(positives, 25)  # bottom 25% threshold
        for i, j in list(G.edges()):
            if sim[i, j] <= thr:
                G.remove_edge(i, j)

    scores = nx.pagerank(G, alpha=0.85)
    return [scores.get(i, 0.0) for i in range(sim.shape[0])]
```


```python
# Step 2) TextRank scores
scores = rank_sentences(sim_matrix)
print("\n✅ Step 2 - TextRank scores:")
for i, score in enumerate(scores, 1):
    print(f"Sentence {i}: {score:.4f}")
```

    
    ✅ Step 2 - TextRank scores:
    Sentence 1: 0.4082
    Sentence 2: 0.0612
    Sentence 3: 0.0612
    Sentence 4: 0.4082
    Sentence 5: 0.0612



```python
# -----------------------------
# 3) Summarize
#    - ratio: fraction of sentences to keep (e.g., 0.2 = 20%)
#    - min_sent: minimum number of sentences
#    - max_sent: optional cap (e.g., fix output to 3 sentences)
# -----------------------------
def summarize(text, ratio=0.2, min_sent=3, max_sent=None):
    sents = sentence_split(text)
    if not sents:
        return []

    sim = build_sim_matrix(sents)
    scores = rank_sentences(sim)

    n = len(sents)
    k = max(min_sent, math.ceil(n * ratio))
    if max_sent is not None:
        k = min(k, max_sent)

    # Take top-k by score, then restore original order for readability
    idx_by_score = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    idx_by_score.sort()
    return [sents[i] for i in idx_by_score]
```


```python
# Step 3) Summarization
summary = summarize(demo_text, ratio=0.4, min_sent=2)
print("\n✅ Step 3 - Summary result:")
for i, s in enumerate(summary, 1):
    print(f"{i}. {s}")
```

    
    ✅ Step 3 - Summary result:
    1. TextRank is a graph-based ranking algorithm for NLP.
    2. This method is often used for extractive summarization.



```python
# -----------------------------
# Final test run (English text)
# -----------------------------
text = """TextRank is an algorithm for extractive summarization.
It uses the PageRank idea to rank sentences by importance.
We compute pairwise sentence similarity to build a graph, then select top-scoring sentences.
This approach is widely used in simple summarization pipelines.
Even beginners can implement a minimal version quickly."""

summary = summarize(text, ratio=0.4, min_sent=2)

print("📌 Original text:")
print(text)
print("\n📌 Summary result:")
for i, s in enumerate(summary, 1):
    print(f"{i}. {s}")
```

    📌 Original text:
    TextRank is an algorithm for extractive summarization.
    It uses the PageRank idea to rank sentences by importance.
    We compute pairwise sentence similarity to build a graph, then select top-scoring sentences.
    This approach is widely used in simple summarization pipelines.
    Even beginners can implement a minimal version quickly.
    
    📌 Summary result:
    1. TextRank is an algorithm for extractive summarization.
    2. It uses the PageRank idea to rank sentences by importance.



```python
# -----------------------------
# 4) Small self-test (you can remove this block)
# -----------------------------
if __name__ == "__main__":
    demo = (
        "TextRank is a graph-based ranking algorithm for NLP. "
        "It builds a graph of sentences using pairwise similarity. "
        "Then it applies PageRank to score sentences by importance. "
        "This method is often used for extractive summarization. "
        "Even beginners can implement a minimal version quickly."
    )

    print("📌 Input:")
    print(demo)
    print("\n📌 Summary:")
    for i, s in enumerate(summarize(demo, ratio=0.4, min_sent=2), 1):
        print(f"{i}. {s}")
```

    📌 Input:
    TextRank is a graph-based ranking algorithm for NLP. It builds a graph of sentences using pairwise similarity. Then it applies PageRank to score sentences by importance. This method is often used for extractive summarization. Even beginners can implement a minimal version quickly.
    
    📌 Summary:
    1. TextRank is a graph-based ranking algorithm for NLP.
    2. This method is often used for extractive summarization.



```python

```
