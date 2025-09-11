```python
"""
Tokenization → Build & Print n-grams
=> “Data preparation” stage, where you directly check the internal structure.
Backoff distribution + Sampling/Argmax
=>Experimentally confirm the “basic behavior” of the model.
Baseline
=> A simple model that just samples one token at a time and appends it.
Speculative (draft → verify)
=> “Small model proposes, large model verifies” → experience the prefix-accept rule.
100-trial average
=> Compute statistics to see how long the prefix is typically accepted with this method.
"""
```




    '\nTokenization → Build & Print n-grams\n=> “Data preparation” stage, where you directly check the internal structure.\nBackoff distribution + Sampling/Argmax\n=>Experimentally confirm the “basic behavior” of the model.\nBaseline\n=> A simple model that just samples one token at a time and appends it.\nSpeculative (draft → verify)\n=> “Small model proposes, large model verifies” → experience the prefix-accept rule.\n100-trial average\n=> Compute statistics to see how long the prefix is typically accepted with this method.\n'




```python
#Step 0 — Imports & Seed
```


```python
import random
from collections import Counter, defaultdict

random.seed(42)  # same results every run
```


```python
#Step 1 — Define a tiny corpus
```


```python
# Step 1: corpus
corpus = "the wolf ran into the forest"
```


```python
# Step 2: tokenize
tokens = corpus.lower().split()
print(tokens)
```

    ['the', 'wolf', 'ran', 'into', 'the', 'forest']



```python
#Build unigram table
```


```python
# Step 3: Unigram counts
# Count how many times each word appears in the tokens.
# Print as "word : count"
uni = Counter(tokens)
print("=== Unigrams ===")
for w, c in uni.items():
    print(f"{w!r}: {c}")
```

    === Unigrams ===
    'the': 2
    'wolf': 1
    'ran': 1
    'into': 1
    'forest': 1



```python
#Init bigram & trigram tables
```


```python
# Step 4: init bigram & trigram
#defaultdict(Counter) → a dictionary that automatically creates an empty Counter for any new key.
bi  = defaultdict(Counter)            # prev -> Counter(next)
tri = defaultdict(Counter)            # (prev2, prev1) -> Counter(next)
```


```python
#Fill bigram counts
#Check every consecutive word pair, count occurrences, then print the bigram counts.”
```


```python
# Step 5: fill bigram
# Go through each pair of consecutive words (a, b).
# Count how many times word 'a' is followed by word 'b'.
for a, b in zip(tokens, tokens[1:]):
    bi[a][b] += 1

# Print the bigram table: (previous_word -> next_word): count
print("=== Bigrams ===")
for prev, counter in bi.items():
    for nxt, c in counter.items():
        print(f"({prev!r} -> {nxt!r}): {c}")
```

    === Bigrams ===
    ('the' -> 'wolf'): 1
    ('the' -> 'forest'): 1
    ('wolf' -> 'ran'): 1
    ('ran' -> 'into'): 1
    ('into' -> 'the'): 1



```python
#Fill trigram counts
#“Check every consecutive triplet, count how often a two-word context (w1, w2) is followed by another word
# then print the trigram counts.
```


```python
# Step 6: fill trigram
# Go through each triplet of consecutive words (a, b, c).
# Count how many times the pair (a, b) is followed by word c.
for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
    tri[(a, b)][c] += 1

# Print the trigram table: ((word1, word2) -> next_word): count
print("=== Trigrams ===")
for (w1, w2), counter in tri.items():
    for nxt, c in counter.items():
        print(f"(({w1!r}, {w2!r}) -> {nxt!r}): {c}")
```

    === Trigrams ===
    (('the', 'wolf') -> 'ran'): 1
    (('wolf', 'ran') -> 'into'): 1
    (('ran', 'into') -> 'the'): 1
    (('into', 'the') -> 'forest'): 1



```python
#“Backoff = use trigram if possible, else bigram, else unigram.”
```


```python
#Backoff: get next-token distribution# Step 7: backoff distribution
# Rule:
# 1) First, try trigram counts using (prev2, prev1).
# 2) If not available, fall back to bigram counts using (prev1).
# 3) If still not available, fall back to unigram counts.
def next_counts_backoff(prev2, prev1):
    d3 = tri.get((prev2, prev1))
    if d3:
        return d3
    d2 = bi.get(prev1)
    if d2:
        return d2
    return uni
```


```python

```


```python
# Step 7: backoff distribution
# Try trigram((prev2, prev1)), else bigram(prev1), else unigram
def next_counts_backoff(prev2, prev1):
    d3 = tri.get((prev2, prev1))
    if d3:
        return d3
    d2 = bi.get(prev1)
    if d2:
        return d2
    return uni
```


```python
# Test contexts for backoff
contexts = [
    ("wolf", "ran"),   # exact trigram match → 'into'
    ("ran", "into"),   # trigram match → 'the'
    ("the", "wolf"),   # trigram match → 'ran'
    ("hello", "wolf"), # trigram missing → fallback to bigram
    ("hello", "zzz")   # both missing → fallback to unigram
]

for prev2, prev1 in contexts:
    dist = next_counts_backoff(prev2, prev1)
    print(f"\nContext: ({prev2!r}, {prev1!r})")
    print("Distribution:", dict(dist))

```

    
    Context: ('wolf', 'ran')
    Distribution: {'into': 2}
    
    Context: ('ran', 'into')
    Distribution: {'the': 2}
    
    Context: ('the', 'wolf')
    Distribution: {'ran': 2}
    
    Context: ('hello', 'wolf')
    Distribution: {'ran': 2}
    
    Context: ('hello', 'zzz')
    Distribution: {'the': 2, 'wolf': 1, 'ran': 1, 'into': 1, 'forest': 1}



```python

```


```python
#Temperature sampling
"""
“Turn counts into probabilities, adjust sharpness with temperature T, then randomly pick one token.”
Higher T (>1.0): flatter distribution → more random / diverse
Lower T (<1.0): sharper distribution → more deterministic / greedy
"""
```




    '\n“Turn counts into probabilities, adjust sharpness with temperature T, then randomly pick one token.”\nHigher T (>1.0): flatter distribution → more random / diverse\nLower T (<1.0): sharper distribution → more deterministic / greedy\n'




```python
# Step 8: temperature sampling (probabilistic)
def sample_from_counts(dist, T=1.0):
    # Convert frequency counts into sampling weights
    # Apply temperature scaling: weight = count ** (1/T)
    items  = list(dist.items())
    toks   = [t for t, _ in items]              # candidate tokens
    cnts   = [c for _, c in items]              # their counts
    weights = [(c if c > 0 else 1e-9) ** (1.0 / T) for c in cnts]
    # Randomly choose one token according to weights
    return random.choices(toks, weights=weights, k=1)[0]

```


```python
# Example check for Step 8: temperature sampling
prev2, prev1 = "hello", "zzz"   # context "... wolf ran"
dist = next_counts_backoff(prev2, prev1)

print("Context:", (prev2, prev1))
print("Distribution (counts):", dict(dist))

# Try different temperatures
print("\nSampled next (T=1.0):", sample_from_counts(dist, T=1.0))
print("Sampled next (T=0.5):", sample_from_counts(dist, T=0.5))  # greedier
print("Sampled next (T=2.0):", sample_from_counts(dist, T=2.0))  # more random
```

    Context: ('hello', 'zzz')
    Distribution (counts): {'the': 2, 'wolf': 1, 'ran': 1, 'into': 1, 'forest': 1}
    
    Sampled next (T=1.0): the
    Sampled next (T=0.5): ran
    Sampled next (T=2.0): into



```python
#Argmax (deterministic pick)
#“Look at all tokens in the distribution and return the one with the largest count.”
```


```python
# Step 9: argmax (most frequent)
# Pick the token with the highest count (the mode of the distribution).
def pick_most_frequent(dist):
    return max(dist.items(), key=lambda kv: kv[1])[0]
```


```python
#Ensure we have 2-token context
```


```python
# Step 10: safeguard (if prompt has 1 token, duplicate it)
# Ensure at least 2 tokens are available for trigram/bigram context.
# If the prompt is only 1 word, duplicate it to create a pair.
def ensure_two_token_context(seq):
    if len(seq) < 2:
        return [seq[-1], seq[-1]]
    return seq
```


```python
# Test for Step 10: ensure_two_token_context

examples = [
    ["wolf", "ran"],   # already 2 tokens
    ["wolf"],          # only 1 token
    ["the", "wolf", "ran"]  # more than 2 tokens
]

for seq in examples:
    fixed = ensure_two_token_context(seq)
    print(f"Input: {seq} -> Output: {fixed}")
```

    Input: ['wolf', 'ran'] -> Output: ['wolf', 'ran']
    Input: ['wolf'] -> Output: ['wolf', 'wolf']
    Input: ['the', 'wolf', 'ran'] -> Output: ['the', 'wolf', 'ran']



```python
#Baseline generator (one-token-at-a-time)
#“The baseline simply extends the prompt one token at a time, using backoff + sampling.”
```


```python
# Step 11: baseline generation (tri->bi->uni + sampling)
# Generate text step by step using the backoff model.
# Process:
# 1) Ensure we have at least 2 tokens for context.
# 2) Get the next-token distribution with backoff (tri -> bi -> uni).
# 3) Sample one token from the distribution (with temperature T).
# 4) Append the token and slide the context window.
```


```python
# Step 11: baseline generation (tri->bi->uni + sampling)
def generate_baseline(prompt_tokens, steps=5, T=0.7):
    out = list(prompt_tokens)
    out = ensure_two_token_context(out)
    prev2, prev1 = out[-2], out[-1]
    for _ in range(steps):
        dist = next_counts_backoff(prev2, prev1)
        nxt  = sample_from_counts(dist, T)
        out.append(nxt)
        prev2, prev1 = prev1, nxt
    return out
```


```python
# Test for Step 11: baseline generation

prompt = ["the", "wolf", "ran"]

print("Prompt:", prompt)
print("\nBaseline generation (T=0.7, 5 steps):")
result = generate_baseline(prompt, steps=5, T=0.7)
print("Output:", " ".join(result))

print("\nBaseline generation (T=1.5, 5 steps, more random):")
result = generate_baseline(prompt, steps=5, T=1.5)
print("Output:", " ".join(result))
```

    Prompt: ['the', 'wolf', 'ran']
    
    Baseline generation (T=0.7, 5 steps):
    Output: the wolf ran into the forest the wolf
    
    Baseline generation (T=1.5, 5 steps, more random):
    Output: the wolf ran into the forest into the



```python
#12 — Speculative helper: build a draft (small model)
# Step 12: drafter (small model = bigram if available, else unigram)
# Build a draft sequence of k tokens.
# - The "small model" uses bigram counts if available, otherwise unigram.
# - At each step:
#   1) Sample the next token from the small-model distribution.
#   2) Log the context (prev2, prev1) and the draft token.
#   3) Append the token to the draft.
#   4) Advance the context window.
# Returns:
#   draft: the list of proposed tokens
#   trace: log of (prev2, prev1, sampled_token) for each step
```


```python
# Step 12: drafter (small model = bigram if available, else unigram)
def build_draft(context, k=5, T_draft=0.9):
    prev2, prev1 = context[-2], context[-1]
    draft, trace = [], []
    for _ in range(k):
        dist_small = bi.get(prev1, uni)     # small model
        t = sample_from_counts(dist_small, T_draft)
        trace.append((prev2, prev1, t))     # log before advancing
        draft.append(t)
        prev2, prev1 = prev1, t             # drafter advances its own context
    return draft, trace
```


```python
# Test for Step 12: drafter (build_draft)

prompt = ["the", "wolf", "ran"]
context = ensure_two_token_context(prompt)

# Generate a draft of 5 tokens
draft, trace = build_draft(context, k=5, T_draft=0.9)

print("Prompt:", prompt)
print("\nDraft tokens:", draft)
print("\nTrace (prev2, prev1 -> sampled_token):")
for i, (p2, p1, t) in enumerate(trace, 1):
    print(f"[{i}] ({p2!r}, {p1!r}) -> {t!r}")

```

    Prompt: ['the', 'wolf', 'ran']
    
    Draft tokens: ['into', 'the', 'forest', 'the', 'forest']
    
    Trace (prev2, prev1 -> sampled_token):
    [1] ('wolf', 'ran') -> 'into'
    [2] ('ran', 'into') -> 'the'
    [3] ('into', 'the') -> 'forest'
    [4] ('the', 'forest') -> 'the'
    [5] ('forest', 'the') -> 'forest'



```python
#Step 13 — Speculative helper: verifier next (large model)
```


```python
# Step 13: verifier next (large model = backoff + argmax)
# Use the large model:
#   - Get the backoff distribution for (prev2, prev1).
#   - Pick the most frequent (argmax) token deterministically.
def verifier_next(prev2, prev1):
    return pick_most_frequent(next_counts_backoff(prev2, prev1))
```


```python
# Step 14: verify draft with prefix-accept
# Compare the draft against the verifier's predictions step by step:
# 1) For each draft token t:
#    - Verifier predicts v (deterministic argmax from backoff model).
#    - If t == v → accept token and extend context.
#    - If t != v → replace with v and STOP (prefix-accept rule).
# 2) Log each step as (prev2, prev1, draft_token, verify_token, ok_flag).
# Returns:
#    accepted: list of accepted (or replaced) tokens
#    log     : detailed verification trace
```


```python
# Step 14: verify draft with prefix-accept
def prefix_accept_verify(context, draft):
    accepted, log = [], []
    prev2, prev1 = context[-2], context[-1]
    for t in draft:
        v  = verifier_next(prev2, prev1)   # deterministic prediction
        ok = (t == v)
        log.append((prev2, prev1, t, v, ok))
        accepted.append(t if ok else v)
        if ok:
            prev2, prev1 = prev1, t        # extend context
        else:
            break                           # replace & STOP
    return accepted, log
```


```python
#Step 15 — Speculative step (orchestrator)
```


```python
# Step 15: speculative (draft -> verify) — simple version
def speculative_step(prompt_tokens, k=5, T_draft=0.9):
    """
    Speculative decoding (simplified):
    - Drafter: small model (bigram→unigram, with temperature sampling).
    - Verifier: large model (trigram→bigram→unigram, argmax).
    - Rule: accept matching prefix, replace on first mismatch and stop.
    Returns: (draft, accepted, final_sequence)
    """
    ctx = ensure_two_token_context(list(prompt_tokens))
    draft, _ = build_draft(ctx, k=k, T_draft=T_draft)
    accepted, _ = prefix_accept_verify(ctx, draft)
    return draft, accepted, prompt_tokens + accepted
```


```python
# Test for Step 15: speculative_step (simple version)

prompt = ["the", "wolf", "ran"]

draft, accepted, final = speculative_step(prompt, k=5, T_draft=0.9)

print("Prompt   :", prompt)
print("Draft    :", draft)
print("Accepted :", accepted)
print("Final    :", " ".join(final))

```

    Prompt   : ['the', 'wolf', 'ran']
    Draft    : ['into', 'the', 'wolf', 'ran', 'into']
    Accepted : ['into', 'the', 'forest']
    Final    : the wolf ran into the forest



```python
#Step 16 — Quick demo: Baseline vs Speculative
```


```python
# Step 16: demo
prompt = ["the", "wolf", "ran"]

print("---- Baseline (T=0.7), next 5 tokens ----")
print("Baseline:", " ".join(generate_baseline(prompt, steps=5, T=0.7)))

print("\n---- Speculative (k=5, T_draft=0.9) ----")
_ = speculative_step(prompt, k=5, T_draft=0.9, trace=True)
```

    ---- Baseline (T=0.7), next 5 tokens ----
    Baseline: the wolf ran into the forest the forest
    
    ---- Speculative (k=5, T_draft=0.9) ----
    
    ---- Speculative (draft -> verify) ----
    === Draft (small model) ===
    [D1] ('wolf','ran') -> draft='into'
    [D2] ('ran','into') -> draft='the'
    [D3] ('into','the') -> draft='wolf'
    [D4] ('the','wolf') -> draft='ran'
    [D5] ('wolf','ran') -> draft='into'
    
    === Verify (prefix-accept) ===
    [V1] ('wolf','ran') draft='into' verify='into' -> ACCEPT
    [V2] ('ran','into') draft='the' verify='the' -> ACCEPT
    [V3] ('into','the') draft='wolf' verify='forest' -> REPLACE+STOP
    
    Draft   : ['into', 'the', 'wolf', 'ran', 'into']
    Accepted: ['into', 'the', 'forest']
    Final   : the wolf ran into the forest



```python
#Step 17 — 100-trial average accepted prefix length
```


```python
# Step 17: measure average accepted length over 100 trials
trials, total = 100, 0
for _ in range(trials):
    _, acc, _ = speculative_step(["the","wolf","ran"], k=5, T_draft=0.9, trace=False)
    total += len(acc)
print(f"\nAverage accepted length over {trials} trials: {total/trials:.2f} / 5")

```

    
    Average accepted length over 100 trials: 3.72 / 5



```python

```
