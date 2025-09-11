```python
#Medusa-lite flow : drafter → verifier → multi-branch prefix-accept
```


```python
from dataclasses import dataclass
import torch, random
```


```python
# Step 2) Device selection

def pick_device():
    # Check if Apple Silicon (MPS) is available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"     # Use MPS on Mac
    # Otherwise, check if CUDA GPU is available
    if torch.cuda.is_available():
        return "cuda"    # Use CUDA if available
    # Fallback to CPU if no GPU/MPS is found
    return "cpu"         

DEVICE = pick_device()
print("✅ DEVICE =", DEVICE)

```

    ✅ DEVICE = mps



```python
assert DEVICE in {"cpu", "cuda", "mps"}
print("OK")
```

    OK



```python
#Seed 고정
```


```python
from dataclasses import dataclass
import torch, random

# Set seeds for reproducibility 
# (not critical if sampling is not used, but still good practice)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

```


```python
#Config
```


```python
from dataclasses import dataclass

@dataclass
class Cfg:
    # IDs of the models used
    DRAFTER_ID: str = "distilgpt2"      # Small, fast draft model
    VERIFIER_ID: str = "gpt2-medium"    # Larger, more accurate verifier model

    # Generation length
    MAX_NEW_TOKENS: int = 30            # Maximum number of tokens to generate

    # Sampling parameters
    TEMPERATURE: float = 0.8            # Controls randomness (lower → more deterministic)
    TOP_P: float = 0.9                  # Nucleus sampling (probability mass cutoff)

    # Repetition control
    REPETITION_PENALTY: float = 1.3     # Penalize repeating tokens
    NO_REPEAT_NGRAM: int = 5            # Prevent repeating n-grams of size 5

    # Speculative decoding settings
    TOPK_BRANCH: int = 4                # How many draft tokens to branch for verification
    DRAFT_SPAN: int = 3                 # Number of tokens the drafter proposes at once

    # Runtime settings
    DEVICE: str = DEVICE                # Device to run on (mps / cuda / cpu)
    DEBUG: bool = False                 # Debug mode toggle

cfg = Cfg()
```


```python
# === Load Draft and Verifier Models (Tokenizer → Model) ===
```


```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizers for drafter and verifier models
drafter_tok  = AutoTokenizer.from_pretrained(cfg.DRAFTER_ID)
verifier_tok = AutoTokenizer.from_pretrained(cfg.VERIFIER_ID)

# 🔧 Fix for GPT-2 family: often eos_token / pad_token are missing
if verifier_tok.eos_token_id is None:
    verifier_tok.eos_token = ""   # Add EOS token if missing
if verifier_tok.pad_token_id is None:
    verifier_tok.pad_token = verifier_tok.eos_token  # Use EOS as padding if missing

# Save EOS token ID for reference
EOS_ID = verifier_tok.eos_token_id

# Load models and move them to the chosen device
drafter  = AutoModelForCausalLM.from_pretrained(cfg.DRAFTER_ID).to(cfg.DEVICE).eval()
verifier = AutoModelForCausalLM.from_pretrained(cfg.VERIFIER_ID).to(cfg.DEVICE).eval()

# Ensure caching is enabled (default is True, but set explicitly)
drafter.config.use_cache  = True
verifier.config.use_cache = True

print("✅ models ready:", cfg.DRAFTER_ID, "/", cfg.VERIFIER_ID)
```

    ✅ models ready: distilgpt2 / gpt2-medium



```python
# === Step 4) Prompt & Context Preparation ===
```


```python
# Define the initial prompt text
prompt = "In a distant future, a small crew of explorers discovers "

# Encode the prompt with the drafter tokenizer
# and move the tensor to the selected DEVICE (mps / cuda / cpu)
ctx = drafter_tok(prompt, return_tensors="pt").to(cfg.DEVICE)

# Extract only the input_ids (token IDs for the prompt)
input_ids = ctx["input_ids"]

# Debug print: confirm context preparation and tensor shape
print("context ok?", ctx is not None, "| shape:", input_ids.shape)
```

    context ok? True | shape: torch.Size([1, 12])



```python
# === Function: Draft k candidate tokens ===
```


```python
@torch.inference_mode()  # Disable gradient calculation for efficiency
def drafter_sample_first_tokens_basic(model, ids, k: int, temperature: float = 0.8):
    # Forward pass → get logits for the last token position
    logits = model(ids).logits[:, -1, :]
    
    # Apply temperature scaling + softmax to convert logits into probabilities
    probs  = torch.softmax(logits / max(temperature, 1e-6), dim=-1)[0]
    
    # Ensure k does not exceed vocabulary size
    k = min(k, probs.numel())
    
    # Sample k distinct token IDs (multinomial sampling without replacement)
    picks = torch.multinomial(probs, num_samples=k, replacement=False)
    
    # Return as a Python list of integers
    return [int(i) for i in picks]
```


```python
# === Debug: Inspect a single token ID ===
tid = 1849

# Decode the token ID back into a string (repr shows invisible characters)
print("token str (repr):", repr(drafter_tok.decode([tid])))

# Show the raw GPT-2 subword token (BPE piece)
print("gpt2 piece:", drafter_tok.convert_ids_to_tokens([tid])[0])

# Check if the decoded token is only whitespace
print("is space?", drafter_tok.decode([tid]).isspace())
```

    token str (repr): '\xa0'
    gpt2 piece: Âł
    is space? True



```python
#멀티-브랜치 Draft 함수
```


```python
import torch
from typing import List, Optional

# =========================
# Small, focused utilities
# =========================

@torch.inference_mode()
def last_token_logits(model, ids: torch.Tensor) -> torch.Tensor:
    """
    Run a forward pass and return logits for the last time step.
    ids: [B, T] LongTensor on the same device as the model.
    returns: [V] 1D logits for the last position (batch assumed 1).
    """
    out = model(ids)
    # shape: [B, T, V] → take last step, squeeze batch
    return out.logits[:, -1, :][0]
```


```python
def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling and softmax.
    Clamps temperature to a minimum to avoid division by zero.
    returns: probability vector over vocabulary [V].
    """
    t = max(float(temperature), 1e-6)
    return torch.softmax(logits / t, dim=-1)
```


```python
def safe_top_p_filter(probs: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    """
    Return the indices of tokens inside the nucleus (top-p) set.
    - Sort by prob desc, take smallest prefix whose cumulative prob ≤ top_p.
    - Always keep at least the top-1 token.
    If top_p is None, returns all indices (torch.arange(V)).
    """
    V = probs.numel()
    if top_p is None:
        return torch.arange(V, device=probs.device)

    # Bound top_p into (0, 1]; treat <=0 as keep only top-1, >1 as keep all.
    if top_p <= 0:
        sorted_p, sorted_ix = torch.sort(probs, descending=True)
        return sorted_ix[:1]
    if top_p >= 1:
        return torch.arange(V, device=probs.device)

    sorted_p, sorted_ix = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)
    keep_mask = cumsum <= top_p
    # Ensure at least one token remains
    keep_mask[0] = True
    return sorted_ix[keep_mask]

```


```python
@torch.inference_mode()
def sample_without_replacement(probs: torch.Tensor, k: int) -> List[int]:
    """
    Multinomial sampling WITHOUT replacement from probs.
    Caps k by available items and returns Python ints.
    """
    k = max(0, min(int(k), probs.numel()))
    if k == 0:
        return []
    picks = torch.multinomial(probs, num_samples=k, replacement=False)
    return [int(i) for i in picks]
```


```python
def append_token(ids: torch.Tensor, tok_id: int) -> torch.Tensor:
    """
    Append a single token id to a [1, T] LongTensor on same device/dtype.
    """
    tok = torch.tensor([[tok_id]], dtype=ids.dtype, device=ids.device)
    return torch.cat([ids, tok], dim=1)

```


```python
# ======================================
# Drafting: first-token + greedy rollout
# ======================================

@torch.inference_mode()
def drafter_sample_first_tokens_basic(model, ids: torch.Tensor, k: int,
                                      temperature: float = 0.8) -> List[int]:
    """
    Simple temperature sampling over the full vocab (no top-p).
    """
    logits = last_token_logits(model, ids)
    probs = softmax_with_temperature(logits, temperature)
    return sample_without_replacement(probs, k)

```


```python
@torch.inference_mode()
def drafter_sample_first_tokens(model, ids: torch.Tensor, k: int,
                                temperature: float = 0.8,
                                top_p: Optional[float] = 0.9) -> List[int]:
    """
    Nucleus (top-p) sampling for the FIRST next-token proposals.
    """
    logits = last_token_logits(model, ids)
    probs = softmax_with_temperature(logits, temperature)
    pool_ix = safe_top_p_filter(probs, top_p)
    pool_probs = probs[pool_ix]
    pool_probs = pool_probs / pool_probs.sum()  # renormalize
    k = min(k, pool_ix.numel())
    if k == 0:
        return []
    picks_local = torch.multinomial(pool_probs, num_samples=k, replacement=False)
    return [int(pool_ix[i]) for i in picks_local]
```


```python
@torch.inference_mode()
def drafter_rollout_greedy(model, ids: torch.Tensor,
                           first_tok: int, span: int) -> List[int]:
    """
    Greedy rollout for 'span' tokens starting with 'first_tok'.
    The first token is fixed; subsequent tokens use argmax.
    """
    span = max(1, int(span))
    cur = append_token(ids, first_tok)
    seq = [first_tok]

    for _ in range(span - 1):
        logits = last_token_logits(model, cur)
        nxt = int(torch.argmax(logits).item())
        seq.append(nxt)
        cur = append_token(cur, nxt)

    return seq
```


```python
@torch.inference_mode()
def drafter_propose(ids: torch.Tensor, k: int, span: int,
                    temperature: float = 0.8, top_p: Optional[float] = 0.9) -> List[List[int]]:
    """
    Propose K branches:
      1) sample K first tokens (nucleus + temperature)
      2) greedy rollout for remaining (span-1) steps per branch
    Uses the global 'drafter' model and current cfg.* settings.
    """
    firsts = drafter_sample_first_tokens(drafter, ids, k, temperature, top_p)
    return [drafter_rollout_greedy(drafter, ids, f, span) for f in firsts]

```


```python
# === Verifier: predict one token (greedy) ===
```


```python
@torch.inference_mode()
def verifier_next_token(ids) -> int:
    """
    Use the verifier model to predict the next token ID greedily.
    - Runs a forward pass on the current ids
    - Takes the last-step logits
    - Returns the argmax token ID as int
    """
    logits = verifier(ids).logits[:, -1, :]
    return int(torch.argmax(logits, dim=-1)[0])
```


```python
# === Pretty-print token information ===
def pretty_token(tokenizer, tid: int):
    """
    Convert a token ID into multiple human-readable formats for debugging.
    
    Returns a dict with:
      - "id": the token ID (int)
      - "decode_repr": decoded string (repr to reveal hidden chars, e.g., '\xa0')
      - "token_repr": raw BPE token string (repr form)
      - "token_fixed": attempt to fix mojibake via latin1 → utf-8 roundtrip
      - "codepoints": list of Unicode codepoints in hex
      - "bytes": list of raw UTF-8 bytes
    """
    # Decode the token ID into text (keep special tokens and spaces)
    s_decode = tokenizer.decode([tid],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)

    # Get the raw BPE subword token string
    s_token = tokenizer.convert_ids_to_tokens([tid])[0]

    # Try to re-encode/decode to fix potential mojibake (encoding artifacts)
    try:
        s_fixed = s_token.encode("latin1").decode("utf-8")
    except Exception:
        s_fixed = s_token

    return {
        "id": tid,
        "decode_repr": repr(s_decode),   # decoded string, repr shows hidden chars
        "token_repr": repr(s_token),     # raw token string as stored by tokenizer
        "token_fixed": repr(s_fixed),    # mojibake-fixed token string
        "codepoints": [hex(ord(c)) for c in s_decode],  # Unicode codepoints
        "bytes": list(s_decode.encode("utf-8")),        # raw UTF-8 bytes
    }
```


```python
@torch.inference_mode()
def next_human_token(ids, tokenizer, tries=10):
    """
    Predict tokens with the verifier until a *human-visible* token appears.
    
    A token is considered "human-visible" if:
      - It contains at least one printable character
      - That character is not just whitespace
    
    Args:
        ids: current input sequence (tensor [1, T])
        tokenizer: the tokenizer used for decoding
        tries: maximum number of attempts before giving up
    
    Returns:
        (tid, s) → the token ID and its decoded string
    """
    cur = ids.clone()
    for _ in range(tries):
        # Predict next token (greedy)
        tid = verifier_next_token(cur)

        # Decode into a string without skipping special tokens
        s = tokenizer.decode(
            [tid],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

        # If the decoded string contains a visible (printable, non-space) character, return
        if any(ch.isprintable() and not ch.isspace() for ch in s):
            return tid, s

        # Otherwise, append token and continue searching
        cur = torch.cat([cur, torch.tensor([[tid]], device=ids.device)], dim=1)

    # If no human-visible token found after max tries, return the last one
    return tid, s

```


```python
# === Example usage ===
vid = verifier_next_token(input_ids)
info = pretty_token(verifier_tok, vid)

print("Predicted token info:", info)

t2, s2 = next_human_token(input_ids, verifier_tok)
print("Next human-visible token:", t2, repr(s2))
```

    Predicted token info: {'id': 488, 'decode_repr': "'ich'", 'token_repr': "'ich'", 'token_fixed': "'ich'", 'codepoints': ['0x69', '0x63', '0x68'], 'bytes': [105, 99, 104]}
    Next human-visible token: 488 'ich'



```python
#Prefix-Accept (mismatch까지)
```


```python
from typing import List, Tuple
import torch
@torch.inference_mode()
def accept_until_mismatch(context_ids, branch_tokens:List[int]) -> Tuple[torch.Tensor, List[int], bool]:
    ids = context_ids.clone()
    accepted = []
    mismatched = False
    for tid in branch_tokens:
        pred = verifier_next_token(ids)
        if pred == tid:
            ids = torch.cat([ids, torch.tensor([[tid]], device=ids.device)], dim=1)
            accepted.append(tid)
        else:
            ids = torch.cat([ids, torch.tensor([[pred]], device=ids.device)], dim=1)
            mismatched = True
            break
    return ids, accepted, mismatched
```


```python
# Generate proposed branches from drafter
branches = drafter_propose(
    input_ids,
    k=cfg.TOPK_BRANCH,
    span=cfg.DRAFT_SPAN,
    temperature=cfg.TEMPERATURE,
    top_p=cfg.TOP_P
)

# Test prefix-accept on the first branch
new_ids, accepted, mism = accept_until_mismatch(input_ids, branches[0])

print("accepted len:", len(accepted), "| mismatched?", mism)
print("new length:", new_ids.shape[1],
      "| tokens added:", new_ids.shape[1] - input_ids.shape[1])
```

    accepted len: 0 | mismatched? True
    new length: 13 | tokens added: 1



```python
# === Branch Scoring Utility ===
import math
import torch
```


```python
import math
import torch
```


```python
#Branch 점수 함수
```


```python
def score_branch(accepted, mismatched):
    """
    Score a drafter branch based on prefix-accept results.

    Args:
        accepted: list of tokens accepted before mismatch
        mismatched: bool, True if a mismatch occurred

    Returns:
        int → simple score = (#accepted tokens) - (1 if mismatch happened else 0)

    Intuition:
        - Longer accepted prefix → higher score
        - If mismatch occurred → apply small penalty (-1)
    """
    return len(accepted) - (1 if mismatched else 0)
```


```python
# Quick checks
print(score_branch([1,2,3], False))  # 3 (3 accepted, no penalty)
print(score_branch([1,2], True))     # 1 (2 accepted, -1 penalty)
```

    3
    1



```python
# === Encode Prompt into Token IDs ===
```


```python
@torch.inference_mode()
def encode_prompt(prompt: str):
    """
    Tokenize a text prompt using the drafter tokenizer and
    move it to the configured device (mps/cuda/cpu).

    Args:
        prompt: input string

    Returns:
        input_ids: tensor of shape [1, T] with token IDs
    """
    ctx = drafter_tok(prompt, return_tensors="pt").to(cfg.DEVICE)
    return ctx["input_ids"]

```


```python
# Quick check
ids = encode_prompt("In a distant future, ")
print("ids.shape:", ids.shape)   # e.g. torch.Size([1, 5])
```

    ids.shape: torch.Size([1, 6])



```python
#한 스텝 수행(multi-branch→검증→최고 점수 채택)
```


```python
# === One Medusa Step ===
@torch.inference_mode()
def medusa_step(ids, topk_branch: int, draft_span: int, temperature: float):
    """
    Perform one Medusa decoding step:
      1) Drafter proposes multiple candidate branches
      2) Each branch is verified with prefix-accept
      3) Branches are scored (longer accepted prefix is better; mismatch penalized)
      4) Best-scoring branch is chosen and returned

    Args:
        ids: tensor [1, T] → current context sequence
        topk_branch: number of branches to propose
        draft_span: number of tokens per branch
        temperature: sampling temperature (≥0.9 enforced for diversity)

    Returns:
        best_ids: updated context tensor after accepting one branch
    """
    # Drafter proposes branches with stronger sampling (top-p = 0.95)
    branches = drafter_propose(
        ids,
        topk_branch,
        draft_span,
        temperature=max(0.9, float(temperature)),  # force ≥ 0.9 for diversity
        top_p=0.95
    )

    # Select the branch with the highest score
    best_score = -10**9
    best_ids = None
    for br in branches:
        new_ids, accepted, mism = accept_until_mismatch(ids, br)
        s = score_branch(accepted, mism)
        if s > best_score:
            best_score, best_ids = s, new_ids

    return best_ids
```


```python
ids2 = medusa_step(ids, cfg.TOPK_BRANCH, cfg.DRAFT_SPAN, cfg.TEMPERATURE)
print("before:", ids.shape[1], "→ after:", ids2.shape[1])
```

    before: 6 → after: 8



```python
#Orchestrator
```


```python
# === Full Orchestrator: Medusa Generate ===
@torch.inference_mode()
def medusa_generate(prompt: str,
                    max_new_tokens: int = None,
                    topk_branch: int = None,
                    draft_span: int = None,
                    temperature: float = None) -> str:
    """
    Generate text using Medusa decoding (drafter + verifier + prefix-accept).

    Args:
        prompt: starting string
        max_new_tokens: max number of tokens to add
        topk_branch: number of branches to propose per step
        draft_span: number of tokens per branch
        temperature: sampling temperature for drafter

    Returns:
        Decoded string (str) including the prompt and generated text
    """
    # Fill with defaults from cfg if not specified
    if max_new_tokens is None: max_new_tokens = cfg.MAX_NEW_TOKENS
    if topk_branch   is None: topk_branch   = cfg.TOPK_BRANCH
    if draft_span    is None: draft_span    = cfg.DRAFT_SPAN
    if temperature   is None: temperature   = cfg.TEMPERATURE

    # Encode the prompt into token IDs
    ids = encode_prompt(prompt)
    start_len = ids.shape[1]

    # Number of Medusa steps (ceil to cover full length)
    steps = math.ceil(max_new_tokens / draft_span)

    # Iteratively expand with Medusa steps
    for _ in range(steps):
        ids = medusa_step(ids, topk_branch, draft_span, temperature)
        if ids.shape[1] - start_len >= max_new_tokens:
            break

    # Decode back into human-readable text
    return drafter_tok.decode(ids[0], skip_special_tokens=True)

```


```python
# ✔️ Example run
out = medusa_generate("In a distant future, ", 40)
print(out)

```

    In a distant future,  the world is ruled by a dictator who is obsessed with the idea of controlling the world's resources.  He wants



```python
# Step 13) Greedy Baseline
```


```python
# === Baseline: Greedy Decoding with Verifier ===
@torch.inference_mode()
def greedy_generate(prompt: str, max_new_tokens: int = None) -> str:
    """
    Generate text using plain greedy decoding with the verifier model.
    Acts as a baseline for comparison against Medusa decoding.

    Args:
        prompt: starting string
        max_new_tokens: number of tokens to generate

    Returns:
        Decoded string (prompt + generated text)
    """
    if max_new_tokens is None:
        max_new_tokens = cfg.MAX_NEW_TOKENS

    # Encode the prompt with verifier tokenizer
    ctx = verifier_tok(prompt, return_tensors="pt").to(cfg.DEVICE)
    ids = ctx["input_ids"]

    # Iteratively add one token at a time (greedy argmax)
    for _ in range(max_new_tokens):
        logits = verifier(ids).logits[:, -1, :]
        nxt = int(torch.argmax(logits, dim=-1)[0])  # pick most likely token
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)

    # Decode back to human-readable text
    return verifier_tok.decode(ids[0], skip_special_tokens=True)
```


```python
txt = greedy_generate("In a distant future, ", 40)
print(txt)
```

    In a distant future,  the world is ruled by a dictator who is obsessed with the idea of controlling the world's resources.  He wants to control the world's resources so that he can rule the world. 



```python
# === 1) A/B Speed & Text Comparison ===
```


```python
import time

def time_it(fn, *args, **kwargs):
    """
    Run a function and measure its execution time.
    Returns:
        (output, elapsed_seconds)
    """
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


# Run both generators on the same prompt
g_txt, g_t = time_it(greedy_generate, "In a distant future, ", 80)
m_txt, m_t = time_it(medusa_generate, "In a distant future, ", 80)

# Compare runtime
print("⏱ greedy:", round(g_t, 3), "s")
print("⏱ medusa:", round(m_t, 3), "s")

# Show a preview of outputs (first 400 chars)
print("\n--- greedy ---\n", g_txt[:400])
print("\n--- medusa ---\n", m_txt[:400])
```

    ⏱ greedy: 7.495 s
    ⏱ medusa: 9.798 s
    
    --- greedy ---
     In a distant future,  the world is ruled by a dictator who is obsessed with the idea of controlling the world's resources.  He wants to control the world's resources so that he can rule the world.  He wants to control the world's resources so that he can rule the world.  He wants to control the world's resources so that he can rule the world.  He wants to
    
    --- medusa ---
     In a distant future,  the world is ruled by a dictator who is obsessed with the idea of controlling the world's resources.  He wants to control the world's resources so that he can rule the world.  He wants to control the world's resources so that he can rule the world.  He



```python

```


```python

```
