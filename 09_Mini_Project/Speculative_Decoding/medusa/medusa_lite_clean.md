```python
# ===== Step 1) Imports =====
import time, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Print library versions to confirm environment is set up
print("✅ imports ok — torch", torch.__version__)

# ===== Step 2) Pick device =====
# Prefer the fastest available backend:
# - On Apple Silicon, use Metal Performance Shaders ('mps')
# - Else if a CUDA-capable GPU is present, use 'cuda'
# - Otherwise fall back to CPU
DEVICE = ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else ("cuda" if torch.cuda.is_available() else "cpu"))

print("✅ device:", DEVICE)
```

    ✅ imports ok — torch 2.8.0
    ✅ device: mps



```python
# ===== Step 3) Load models & tokenizers =====

# Model IDs (Hugging Face Hub names)
DRAFTER_ID  = "distilgpt2"     # small, fast "drafter" model
VERIFIER_ID = "gpt2-medium"    # larger, stronger "verifier" model

# Load tokenizers (text ↔ token IDs)
drafter_tok  = AutoTokenizer.from_pretrained(DRAFTER_ID)
verifier_tok = AutoTokenizer.from_pretrained(VERIFIER_ID)

# GPT-2 family often has no EOS/PAD defined by default.
# Define them so decoding and padding behave consistently.
if verifier_tok.eos_token_id is None:
    verifier_tok.eos_token = ""      # set EOS to the special  token
if verifier_tok.pad_token_id is None:
    verifier_tok.pad_token = verifier_tok.eos_token  # use EOS as PAD to avoid mismatch
EOS_ID = verifier_tok.eos_token_id

# Load models and move them to the selected device.
# .eval() disables dropout etc. for deterministic inference.
drafter  = AutoModelForCausalLM.from_pretrained(DRAFTER_ID).to(DEVICE).eval()
verifier = AutoModelForCausalLM.from_pretrained(VERIFIER_ID).to(DEVICE).eval()

print("✅ models ready:", DRAFTER_ID, "/", VERIFIER_ID)
```

    ✅ models ready: distilgpt2 / gpt2-medium



```python
#3A) Encode / Decode / Normalize
```


```python
# ===== Text I/O utilities =====

# Turn a prompt string into model-ready token IDs.
# We use the *verifier* tokenizer for consistency throughout the pipeline.
def encode_prompt(prompt: str):
    return verifier_tok(prompt, return_tensors="pt").to(DEVICE)["input_ids"]

# Decode token IDs back to text.
# Disable HuggingFace's auto "clean_up_tokenization_spaces" so we can normalize ourselves.
def decode_ids(ids) -> str:
    return verifier_tok.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

# Minimal whitespace normalization:
# - Replace NBSP (U+00A0) with a normal space
# - Collapse multiple spaces/newlines/tabs into a single space
# - Trim leading/trailing spaces
def normalize_text(s: str) -> str:
    return " ".join(s.replace("\u00A0", " ").split())
```


```python
# ===== Quick tests =====
# (Assumes DEVICE, verifier_tok are already defined and models are loaded.)

# 1) Encode → Decode → Normalize round trip
prompt = "In a distant future, " + "\u00A0" + "  the crew   finds   a signal!  "
ids = encode_prompt(prompt)
print("input_ids shape:", ids.shape)

decoded_raw = decode_ids(ids)
decoded_norm = normalize_text(decoded_raw)

print("RAW decoded:", repr(decoded_raw))     # show raw string with possible NBSPs/spaces
print("NORM decoded:", repr(decoded_norm))   # cleaned version

# 2) Extra sanity: encode a simple prompt and confirm tokens grow after appending a word
ids2 = encode_prompt("Hello")
print("len before append:", ids2.shape[1])
# Simulate appending a token (space + 'world' piece) using tokenizer
more = verifier_tok(" world", return_tensors="pt").to(DEVICE)["input_ids"]
ids2_appended = torch.cat([ids2, more[:, 0:1]], dim=1)  # append just one token for demo
print("len after append:", ids2_appended.shape[1])

```

    input_ids shape: torch.Size([1, 19])
    RAW decoded: 'In a distant future, \xa0  the crew   finds   a signal!  '
    NORM decoded: 'In a distant future, the crew finds a signal!'
    len before append: 1
    len after append: 2



```python
# ===== 3B) Drafter: clean sampling utilities (filter invisible tokens, add top-p) =====

```


```python
# Visible token filter:
# - must contain at least one printable, non-whitespace char
# - reject the Unicode replacement char '�' (U+FFFD)
def _is_visible_token(tid: int, tok) -> bool:
    s = tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not any(ch.isprintable() and not ch.isspace() for ch in s):
        return False
    if "\uFFFD" in s:
        return False
    return True

@torch.inference_mode()
def drafter_sample_first_tokens_clean(ids, k: int, temperature: float = 0.9, top_p: float = 0.95):
    """
    Sample up to K distinct first tokens from the drafter:
      - apply temperature
      - nucleus (top-p) filtering
      - filter out pure whitespace / non-visible byte tokens
    Returns a list[int] (length ≤ K).
    """
    logits = get_last_logits(drafter, ids)         # [1, V]
    probs  = logits_to_probs(logits, temperature)  # [V]

    # nucleus (top-p) pool
    sorted_p, sorted_ix = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)
    keep = cumsum <= top_p
    keep[0] = True                                  # always keep top-1
    pool_ix = sorted_ix[keep].tolist()              # candidate ids

    # filter to visible tokens
    visible_ix = [int(t) for t in pool_ix if _is_visible_token(int(t), drafter_tok)]
    if not visible_ix:
        # fallback: top-1 (even if invisible)
        return [int(sorted_ix[0].item())]

    # renormalize over visible pool
    pool_p = probs[visible_ix]
    pool_p = pool_p / pool_p.sum()

    # sample without replacement
    num = min(k, len(visible_ix))
    picks = torch.multinomial(pool_p, num_samples=num, replacement=False).tolist()
    return [visible_ix[i] for i in picks]


```


```python
# ===== Quick test for section 3B (clean sampler) =====
prompt = "In a distant future, "
ids = encode_prompt(prompt)

logits = get_last_logits(drafter, ids)
probs  = logits_to_probs(logits, temperature=0.9)
print("next-token logits shape:", logits.shape)
print("probs sum (≈1.0):", float(probs.sum().item()))

K = 4
candidates = drafter_sample_first_tokens_clean(ids, k=K, temperature=0.9, top_p=0.95)
print(f"\nK={K} filtered token IDs:", candidates)

decoded = [drafter_tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
           for t in candidates]
bpe     = [drafter_tok.convert_ids_to_tokens([t])[0] for t in candidates]

print("decoded (repr):", [repr(s) for s in decoded])
print("BPE pieces   :", [repr(s) for s in bpe])
print("is whitespace:", [s.isspace() for s in decoded])

```

    next-token logits shape: torch.Size([1, 50257])
    probs sum (≈1.0): 0.9999999403953552
    
    K=4 filtered token IDs: [10185, 40493, 9805, 742]
    decoded (repr): ["'!!!'", "'『'", "'????'", "'xt'"]
    BPE pieces   : ["'!!!'", "'ãĢİ'", "'????'", "'xt'"]
    is whitespace: [False, False, False, False]



```python
# ===== 3C) Drafter: rollout helpers =====
```


```python
# Append a single token ID to the current sequence.
# - ids: shape [1, T] (batch size 1)
# - tok: next token id (int)
# Returns a NEW tensor of shape [1, T+1] on the same device as `ids`.
def append_token(ids: torch.Tensor, tok: int) -> torch.Tensor:
    return torch.cat([ids, torch.tensor([[tok]], device=ids.device)], dim=1)

# Get the drafter's greedy next-token choice for the given sequence.
# Uses argmax over the last-position logits.
# - ids: shape [1, T]
# Returns: next token id (int)
@torch.inference_mode()
def drafter_greedy_next(ids: torch.Tensor) -> int:
    logits = get_last_logits(drafter, ids)  # shape: [1, vocab_size]
    return int(torch.argmax(logits, dim=-1)[0])

# Roll out a branch of length `span`, starting from `first_tok`, using greedy steps.
# The returned list includes `first_tok` and (span-1) subsequent greedy tokens.
# - ids:       shape [1, T] (context so far)
# - first_tok: starting token id for this branch
# - span:      total tokens to produce for the branch (>=1)
# Returns: list[int] of length `span`
@torch.inference_mode()
def drafter_rollout_basic(ids: torch.Tensor, first_tok: int, span: int) -> list[int]:
    seq: list[int] = [first_tok]
    cur = append_token(ids, first_tok)
    for _ in range(span - 1):
        nxt = drafter_greedy_next(cur)
        seq.append(nxt)
        cur = append_token(cur, nxt)
    return seq

```


```python
 #Drafter: multi-branch proposal(K × span)
```


```python
@torch.inference_mode()
def drafter_propose_basic(ids, k: int, span: int, temperature: float = 0.8) -> list[list[int]]:
    firsts = drafter_sample_first_tokens_basic(ids, k, temperature)
    return [drafter_rollout_basic(ids, t, span) for t in firsts]
```


```python
# ===== 3E) Verifier: greedy next-token (split into tiny helpers) =====
```


```python
# Get the verifier's logits at the last (next-token) position.
# - ids: shape [1, T]
# Returns: tensor of shape [1, vocab_size]
@torch.inference_mode()
def verifier_last_logits(ids: torch.Tensor) -> torch.Tensor:
    return get_last_logits(verifier, ids)

# Pick the verifier's greedy next token (argmax over logits).
# - ids: shape [1, T]
# Returns: next token id (int)
@torch.inference_mode()
def verifier_greedy_next(ids: torch.Tensor) -> int:
    logits = verifier_last_logits(ids)  # [1, V]
    return int(torch.argmax(logits, dim=-1)[0])
```


```python
# ===== Quick test for section 3E =====
# Assumes: DEVICE, verifier, verifier_tok, encode_prompt(), append_token() are defined.

prompt = "In a distant future, "
ids = encode_prompt(prompt)
print("before length:", ids.shape[1])

# 1-step greedy with the verifier
tid = verifier_greedy_next(ids)
print("next token id:", tid)
print("decoded piece (repr):", repr(verifier_tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)))
print("BPE piece (repr):    ", repr(verifier_tok.convert_ids_to_tokens([tid])[0]))

# Append it and show the new length + a short decode preview
ids2 = append_token(ids, tid)
print("after length:", ids2.shape[1])

preview = verifier_tok.decode(ids2[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("preview:", preview[:120].replace("\n", " "))

```

    before length: 6
    next token id: 1849
    decoded piece (repr): '\xa0'
    BPE piece (repr):     'Âł'
    after length: 7
    preview: In a distant future,  



```python
# ===== 3F) Prefix-accept (split into: accept-one / commit-mismatch / loop) =====
```


```python
# Append a branch token that matches the verifier's prediction.
# - cur_ids: current sequence tensor [1, T]
# - tok:     token id to append
# Returns: new tensor [1, T+1]
def accept_one(cur_ids: torch.Tensor, tok: int) -> torch.Tensor:
    return append_token(cur_ids, tok)

# On first mismatch, append the verifier's predicted token instead of the branch token.
# - pred_tok: verifier-chosen token id
def commit_mismatch(cur_ids: torch.Tensor, pred_tok: int) -> torch.Tensor:
    return append_token(cur_ids, pred_tok)

# Run prefix-accept against a candidate branch:
# For each token in branch_tokens:
#   - if verifier's next == branch token → accept it
#   - else append verifier token and stop
# Returns: (new_ids, accepted_tokens_list, mismatched_flag)
@torch.inference_mode()
def accept_until_mismatch_basic(context_ids: torch.Tensor, branch_tokens: list[int]):
    cur = context_ids.clone()
    accepted: list[int] = []
    mismatched = False
    for t in branch_tokens:
        # NOTE: if your helper is named `verifier_next_token`, use that.
        # Here we use the 3E helper `verifier_greedy_next`.
        pred = verifier_greedy_next(cur)
        if pred == t:
            accepted.append(t)
            cur = accept_one(cur, t)
        else:
            cur = commit_mismatch(cur, pred)
            mismatched = True
            break
    return cur, accepted, mismatched
```


```python
# ===== Quick test for section 3F =====
# Assumes: encode_prompt, drafter_sample_first_tokens_* , drafter_rollout_basic are defined.

prompt = "In a distant future, "
ids = encode_prompt(prompt)

# Pick a first token (try clean sampler if available; else basic)
sampler = globals().get("drafter_sample_first_tokens_clean", globals().get("drafter_sample_first_tokens_basic"))
first = sampler(ids, k=1, temperature=0.9)[0]

# Draft a short candidate branch (e.g., span=3)
branch = drafter_rollout_basic(ids, first_tok=first, span=3)
print("candidate branch token IDs:", branch)

# Run prefix-accept
new_ids, accepted, mism = accept_until_mismatch_basic(ids, branch)
print("accepted len:", len(accepted), "| mismatched?", mism)
print("preview:", decode_ids(new_ids)[:160].replace("\n", " "))

```

    candidate branch token IDs: [11839, 11482, 6527]
    accepted len: 0 | mismatched? True
    preview: In a distant future,  



```python
# ===== 3G) Branch scoring & selection (simplest version) =====
```


```python
# Score a branch by how many tokens were prefix-accepted.
# Apply a small penalty (-1) if a mismatch occurred.
def score_branch_simple(accepted: list[int], mismatched: bool) -> int:
    return len(accepted) - (1 if mismatched else 0)

# Evaluate a single candidate branch:
# - Runs prefix-accept against `branch_tokens`
# - Returns the updated ids and the simple score
@torch.inference_mode()
def evaluate_branch(ids: torch.Tensor, branch_tokens: list[int]):
    new_ids, accepted, mism = accept_until_mismatch_basic(ids, branch_tokens)
    return new_ids, score_branch_simple(accepted, mism)
```


```python
# ===== Quick test for section 3G =====
# Assumes: encode_prompt(), drafter_rollout_basic(), accept_until_mismatch_basic(),
#          and a sampler (drafter_sample_first_tokens_clean/basic) are defined.

prompt = "In a distant future, "
ids = encode_prompt(prompt)

# Pick a sampler (prefer the clean sampler if available)
sampler = globals().get("drafter_sample_first_tokens_clean",
          globals().get("drafter_sample_first_tokens_basic"))

# Draft a small candidate branch (K=1 → take first; span=3 as a demo)
first_tok = sampler(ids, k=1, temperature=0.9)[0]
branch    = drafter_rollout_basic(ids, first_tok=first_tok, span=3)
print("candidate branch:", branch)

# Evaluate it
cand_ids, score = evaluate_branch(ids, branch)
print("score:", score)
print("preview:", decode_ids(cand_ids)[:160].replace("\n", " "))

```

    candidate branch: [933, 12754, 318]
    score: -1
    preview: In a distant future,  



```python
#3H) Medusa 스텝 & 오케스트레이터 (얇게 구성)
```


```python
# ===== 3H) Medusa step & orchestrator (thin version) =====

# One Medusa step:
#  - Ask the drafter to propose K branches (each of length `span`)
#  - Run prefix-accept on each branch
#  - Keep the candidate that scores best (simple length-first scoring)
@torch.inference_mode()
def medusa_step_basic(ids: torch.Tensor, k_branches: int, span: int, temperature: float = 0.8) -> torch.Tensor:
    branches = drafter_propose_basic(ids, k_branches, span, temperature)
    best_score, best_ids = -10**9, None
    for br in branches:
        cand_ids, s = evaluate_branch(ids, br)  # (updated ids, simple score)
        if s > best_score:
            best_score, best_ids = s, cand_ids
    return best_ids

# Orchestrator:
#  - Repeat Medusa steps until we add ~max_new_tokens
#  - Decode and lightly normalize whitespace before returning
@torch.inference_mode()
def medusa_generate_basic(prompt: str,
                          max_new_tokens: int = 40,
                          k_branches: int = 4,
                          span: int = 3,
                          temperature: float = 0.8) -> str:
    ids = encode_prompt(prompt)
    start_len = ids.shape[1]
    steps = math.ceil(max_new_tokens / span)
    for _ in range(steps):
        ids = medusa_step_basic(ids, k_branches, span, temperature)
        if ids.shape[1] - start_len >= max_new_tokens:
            break
    return normalize_text(decode_ids(ids))


```


```python
# ===== Quick test for section 3H =====
# Assumes:
#  - encode_prompt(), decode_ids(), normalize_text()
#  - drafter_propose_basic(), evaluate_branch()
#  - and all earlier sections (3A–3G) are already defined.

prompt = "In a distant future, "
print("=== Medusa-lite (basic) ===")
print(medusa_generate_basic(prompt, max_new_tokens=40, k_branches=3, span=3, temperature=0.8)[:200])

```

    === Medusa-lite (basic) ===
    In a distant future, the world is ruled by a dictator who is obsessed with the idea of controlling the world's resources. He wants to control the

