```python
#uni–bi–tri-gram + backoff + prefix-accept
```


```python
"""
Mini Speculative Decoding Demo (Beginner-Friendly)
--------------------------------------------------
This notebook shows a tiny, self-contained prototype of "speculative decoding":
- A small "drafter" proposes a short draft of k tokens (unigram-based).
- A larger "verifier" checks the draft and accepts only a verified prefix.
- On the first mismatch, we replace with the verified token and stop (prefix-accept).

We provide two verifiers:
  (A) Bigram-only verifier:      uses P(next | prev1)
  (B) Backoff verifier (tri→bi→uni): uses P(next | prev2, prev1), else bigram, else unigram

Everything below is plain Python with simple frequency counts.
"""
```




    '\nMini Speculative Decoding Demo (Beginner-Friendly)\n--------------------------------------------------\nThis notebook shows a tiny, self-contained prototype of "speculative decoding":\n- A small "drafter" proposes a short draft of k tokens (unigram-based).\n- A larger "verifier" checks the draft and accepts only a verified prefix.\n- On the first mismatch, we replace with the verified token and stop (prefix-accept).\n\nWe provide two verifiers:\n  (A) Bigram-only verifier:      uses P(next | prev1)\n  (B) Backoff verifier (tri→bi→uni): uses P(next | prev2, prev1), else bigram, else unigram\n\nEverything below is plain Python with simple frequency counts.\n'




```python
# 1) Imports & reproducibility
import random
from collections import Counter, defaultdict

random.seed(42)  # Make random sampling reproducible
```


```python
# 2) Corpus & tokenization -----------------------------------------------------
#    You can edit the corpus to try different examples.
corpus = "the wolf ran into the forest"
tokens = corpus.lower().split()  # lowercase + whitespace tokenization
```


```python
# 3) Unigram frequency (for drafter & fallback) --------------------------------
#    counts   : global token frequencies
#    vocab    : list of unique tokens
#    weights  : frequency-based weights aligned with vocab order
counts  = Counter(tokens)
vocab   = list(counts.keys())
weights = [counts[w] for w in vocab]

def draft_tokens_unigram(k=3, alpha=1.0):
    """
    Unigram drafter (small model).
    - Samples k tokens with replacement, using frequency-based weights.
    - alpha > 1.0: sharpen distribution (favor frequent words more)
    - alpha < 1.0: flatten distribution (more diversity)
    """
    wts = [wt ** alpha for wt in weights]  # power transform for diversity control
    return random.choices(vocab, weights=wts, k=k)

# (Optional) Display helper: remove only consecutive duplicates for cleaner output
def dedup_consecutive(words):
    """
    Remove consecutive duplicates ONLY: ['a','a','b','b'] -> ['a','b']
    This does NOT enforce uniqueness globally.
    Use this only for printing if you dislike repeated neighbors.
    """
    out = []
    for w in words:
        if not out or out[-1] != w:
            out.append(w)
    return out
```


```python
# 4) N-gram tables (build once) ------------------------------------------------
#    bigram_next : dict(prev1 -> Counter(next))
#    trigram_next: dict((prev2, prev1) -> Counter(next))
bigram_next  = defaultdict(Counter)
trigram_next = defaultdict(Counter)

# Fill bigram table
for a, b in zip(tokens, tokens[1:]):
    bigram_next[a][b] += 1

# Fill trigram table
for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
    trigram_next[(a, b)][c] += 1
```


```python
# 5) Verifiers -----------------------------------------------------------------
def verify_next_bigram(prev1):
    """
    Bigram verifier:
    Return the most frequent next token given prev1 (argmax over bigram counts).
    If there is no entry for prev1, fall back to the global unigram-most-frequent token.
    """
    dist = bigram_next.get(prev1)
    if dist:
        return dist.most_common(1)[0][0]
    # Unigram fallback: global argmax
    return max(counts, key=counts.get)

def verify_next_backoff(prev2, prev1):
    """
    Backoff verifier (trigram -> bigram -> unigram):
    1) If we have trigram stats for (prev2, prev1), use the most frequent next.
    2) Else, if we have bigram stats for prev1, use the most frequent next.
    3) Else, fall back to the global unigram-most-frequent token.
    """
    dist3 = trigram_next.get((prev2, prev1))
    if dist3:
        return dist3.most_common(1)[0][0]
    dist2 = bigram_next.get(prev1)
    if dist2:
        return dist2.most_common(1)[0][0]
    return max(counts, key=counts.get)

```


```python
# 6) Speculative decoding (prefix-accept) --------------------------------------
def spec_prefix_accept_bigram(prompt_tokens, k=5, alpha=1.0, trace=True):
    """
    Speculative step using the BIGRAM verifier.
    - Draft k tokens with the unigram drafter.
    - Walk left→right; if a draft token matches the verifier, accept it and extend context.
    - On the first mismatch, replace with the verified token and STOP (prefix-accept).
    """
    draft = draft_tokens_unigram(k=k, alpha=alpha)
    accepted = []
    prev1 = prompt_tokens[-1]  # last token of the current context

    steps = []
    for t in draft:
        v = verify_next_bigram(prev1)
        ok = (t == v)
        steps.append((prev1, t, v, ok))
        if ok:
            accepted.append(t)
            prev1 = t  # extend context
        else:
            accepted.append(v)
            break

    final = prompt_tokens + accepted

    if trace:
        for i, (p1, t, v, ok) in enumerate(steps, 1):
            print(f"[{i}] prev1='{p1}'  draft='{t}'  verify='{v}'  ->  {'ACCEPT' if ok else 'REPLACE+STOP'}")
        print("Draft   :", draft)
        print("Accepted:", accepted)
        print("Final   :", " ".join(final))
    return draft, accepted, final

```


```python
def spec_prefix_accept_backoff(prompt_tokens, k=5, alpha=1.0, trace=True):
    """
    Speculative step using the TRIGRAM→BIGRAM→UNIGRAM backoff verifier.
    - Same prefix-accept logic, but the verifier considers two-token context first.
    """
    draft = draft_tokens_unigram(k=k, alpha=alpha)
    accepted = []

    # Initialize (prev2, prev1) from the prompt.
    # If prompt is very short, duplicate prev1 as a minimal fallback.
    if len(prompt_tokens) >= 2:
        prev2, prev1 = prompt_tokens[-2], prompt_tokens[-1]
    else:
        prev2, prev1 = prompt_tokens[-1], prompt_tokens[-1]

    steps = []
    for t in draft:
        v = verify_next_backoff(prev2, prev1)
        ok = (t == v)
        steps.append((prev2, prev1, t, v, ok))
        if ok:
            accepted.append(t)
            prev2, prev1 = prev1, t  # shift context window
        else:
            accepted.append(v)
            break

    final = prompt_tokens + accepted

    if trace:
        for i, (p2, p1, t, v, ok) in enumerate(steps, 1):
            print(f"[{i}] prev2='{p2}' prev1='{p1}'  draft='{t}'  verify='{v}'  ->  {'ACCEPT' if ok else 'REPLACE+STOP'}")
        print("Draft   :", draft)
        print("Accepted:", accepted)
        print("Final   :", " ".join(final))
    return draft, accepted, final
```


```python
# 7) Demo runs -----------------------------------------------------------------
prompt = ["the", "wolf", "ran"]

print("=== Bigram verifier demo ===")
_ = spec_prefix_accept_bigram(prompt, k=5, alpha=1.0, trace=True)

print("\n=== Backoff verifier (tri→bi→uni) demo ===")
_ = spec_prefix_accept_backoff(prompt, k=5, alpha=1.0, trace=True)


```

    === Bigram verifier demo ===
    [1] prev1='ran'  draft='ran'  verify='into'  ->  REPLACE+STOP
    Draft   : ['ran', 'the', 'the', 'the', 'into']
    Accepted: ['into']
    Final   : the wolf ran into
    
    === Backoff verifier (tri→bi→uni) demo ===
    [1] prev2='wolf' prev1='ran'  draft='into'  verify='into'  ->  ACCEPT
    [2] prev2='ran' prev1='into'  draft='forest'  verify='the'  ->  REPLACE+STOP
    Draft   : ['into', 'forest', 'the', 'wolf', 'the']
    Accepted: ['into', 'the']
    Final   : the wolf ran into the



```python

```
