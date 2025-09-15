```python

```


```python
# -*- coding: utf-8 -*-
"""
Speculative Decoding — Soft + Guarded Acceptance (modularized)
- drafter  : gpt2
- verifier : gpt2-medium
- acceptance: (p_thresh AND in top-k AND within margin) up to max_accept
"""
```




    '\nSpeculative Decoding — Soft + Guarded Acceptance (modularized)\n- drafter  : gpt2\n- verifier : gpt2-medium\n- acceptance: (p_thresh AND in top-k AND within margin) up to max_accept\n'




```python
import math
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
```


```python
# =========================
# 0) Config & Device
# =========================
@dataclass
class Cfg:
    DRAFTER_ID: str = "gpt2"            # Hugging Face ID for the drafter (small/fast model)
    VERIFIER_ID: str = "gpt2-medium"    # Hugging Face ID for the verifier (larger/more accurate model)
    DRAFT_SPAN: int = 5                 # Number of tokens proposed by the drafter per step

    P_THRESH: float = 0.05              # Probability threshold: accept drafter token if its prob ≥ this under the verifier
    TOPK_GATE: int = 5                  # Top-k gate: drafter token must be within the verifier’s top-k candidates
    MARGIN_GATE: float = 0.20           # Margin gate (absolute prob diff): |p_top1 − p_drafter| ≤ margin

    MAX_ACCEPT: int = 5                 # Max number of consecutive tokens to accept in one step
    DO_SAMPLE_DRAFT: bool = False       # Drafter sampling (False = greedy; True = use temperature/top-p sampling)

    SEED: int = 42                      # Random seed for reproducibility
    DEVICE: str = ""                    # Device override; if empty, auto-pick "mps" → "cuda" → "cpu"
```


```python
def pick_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```


```python
# =========================
# 1A) Models dataclass
# =========================
@dataclass
class Models:
    tok: AutoTokenizer                # Converts input text → token IDs, and token IDs → text (decode)
    drafter: AutoModelForCausalLM     # Small model: generates draft tokens (speculative proposals)
    verifier: AutoModelForCausalLM    # Large model: verifies the drafter’s proposals
    device: str                       # Device where models are loaded (mps / cuda / cpu)
```


```python
# =========================
# 1B) Load models & tokenizer
# =========================
def load_models(cfg: Cfg) -> Models:
    device = cfg.DEVICE or pick_device()    # Auto-select device if not specified
    print(f"✅ device: {device}")
    set_seed(cfg.SEED)                      # Fix random seed for reproducibility

    # Load tokenizer (use verifier's tokenizer for consistency)
    tok = AutoTokenizer.from_pretrained(cfg.VERIFIER_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token       # Ensure padding token is defined

    # Load drafter and verifier
    drafter  = AutoModelForCausalLM.from_pretrained(cfg.DRAFTER_ID).to(device)
    verifier = AutoModelForCausalLM.from_pretrained(cfg.VERIFIER_ID).to(device)

    drafter.eval()
    verifier.eval()

    return Models(tok=tok, drafter=drafter, verifier=verifier, device=device)
```


```python
cfg = Cfg()                 # Use default config
models = load_models(cfg)   # Load models

print("=== Models loaded ===")
print("Tokenizer vocab size :", models.tok.vocab_size)
print("Drafter model ID     :", cfg.DRAFTER_ID)
print("Verifier model ID    :", cfg.VERIFIER_ID)
print("Device               :", models.device)
```

    ✅ device: mps
    === Models loaded ===
    Tokenizer vocab size : 50257
    Drafter model ID     : gpt2
    Verifier model ID    : gpt2-medium
    Device               : mps



```python
# =========================
# 2) Tokenization helpers
# =========================

"""
m.tok(...)
Use Hugging Face tokenizer to convert a string into a sequence of token IDs
Example: "Hello" → [15496]

return_tensors="pt"
Return output as a PyTorch tensor (ready to feed into the model)

.input_ids
Extract the actual ID tensor from the tokenizer output (dict)

.to(m.device)
Move the tensor to the correct device (mps / cuda / cpu), so it matches the model
"""
def to_ids(m: Models, text: str) -> torch.Tensor:
    return m.tok(text, return_tensors="pt").input_ids.to(m.device)


def decode(m: Models, ids: torch.Tensor) -> str:
    return m.tok.decode(ids[0], skip_special_tokens=True)
```


```python
# Example sentence
sample_text = "She walked into the"

# 1) Text → Token IDs (tensor)
ids = to_ids(models, sample_text)
print("Token IDs:", ids)

# 2) Token IDs → Text (decode back)
decoded_text = decode(models, ids)
print("Decoded text:", decoded_text)
```

    Token IDs: tensor([[3347, 6807,  656,  262]], device='mps:0')
    Decoded text: She walked into the



```python
# =========================
# 3) Drafter / Verifier ops
# =========================
@torch.no_grad()  
# Tell PyTorch this function is for inference only (no gradients stored) 
# → saves memory and speeds up computation
def drafter_propose(m: Models, cur_ids: torch.Tensor, k: int, do_sample: bool) -> torch.Tensor:
    """
    Drafter proposes k tokens (either greedy or sampled).
    Return: tensor of shape [k]
    """
    out = m.drafter.generate(
        cur_ids,
        max_new_tokens=k,                 # number of tokens to generate
        do_sample=do_sample,              # if False → greedy; if True → use sampling
        temperature=1.0,                  # sampling temperature
        top_p=1.0,                        # nucleus sampling parameter (1.0 = no truncation)
        pad_token_id=m.tok.eos_token_id   # ensure EOS token is used for padding
    )
    # out[0] → take the first sequence from the batch
    # cur_ids.shape[1]: → slice off the original prompt length, keep only the newly generated k tokens
    return out[0, cur_ids.shape[1]:]
```


```python
prompt = "She walked into the"
cur_ids = to_ids(models, prompt)

# drafter proposes 5 tokens (greedy mode, no sampling)
draft_ids = drafter_propose(models, cur_ids, k=5, do_sample=False)

print("Draft token IDs:", draft_ids.tolist())   # numeric IDs of proposed tokens
print("Draft text:", models.tok.decode(draft_ids, skip_special_tokens=True))  # readable text
```

    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.


    Draft token IDs: [2119, 290, 3114, 379, 262]
    Draft text:  room and looked at the



```python
@torch.no_grad()

# cur_ids = to_ids(models, "She walked into the") => tensor([[1212, 3567, 1234, 262]])
def verifier_logits(m: Models, cur_ids: torch.Tensor) -> torch.Tensor:
    """Verifier's logits for the last token"""
    return m.verifier(cur_ids).logits[:, -1, :]
    # shape = [1, vocab_size]
```


```python
import torch

prompt = "She walked into the"
cur_ids = to_ids(models, prompt)

# 1) Get raw logits from the verifier
logits = verifier_logits(models, cur_ids)

# 2) Convert logits into probabilities
probs = torch.softmax(logits, dim=-1)

# 3) Take the top-5 most likely tokens
top_vals, top_idx = torch.topk(probs, k=5, dim=-1)

# 4) Print results as a table
print("=== Verifier Predictions (Top-5) ===")
for rank, (tid, p) in enumerate(zip(top_idx[0].tolist(), top_vals[0].tolist()), start=1):
    tok_str = models.tok.decode([tid])
    print(f"{rank}. {tok_str!r:>10}  (id={tid:5d})  prob={p:.4f}")
```

    === Verifier Predictions (Top-5) ===
    1.    ' room'  (id= 2119)  prob=0.1380
    2. ' kitchen'  (id= 9592)  prob=0.0665
    3.  ' office'  (id= 2607)  prob=0.0413
    4. ' bathroom'  (id=12436)  prob=0.0182
    5.   ' house'  (id= 2156)  prob=0.0158



```python
@torch.no_grad() 
# Return the most likely next token (greedy decoding)
# Verifier is always deterministic → greedy by default.
def verifier_greedy_next(m: Models, cur_ids: torch.Tensor) -> int:
    return int(torch.argmax(verifier_logits(m, cur_ids), dim=-1)[0])
```


```python
prompt = "She walked into the"
cur_ids = to_ids(models, prompt)

# 1) Get the greedy next token ID from the verifier
next_id = verifier_greedy_next(models, cur_ids)

# 2) Convert token ID → string
next_str = models.tok.decode([next_id])

# 3) Print results
print("=== Verifier Greedy Next ===")
print(f"Prompt     : {prompt!r}")
print(f"Token ID   : {next_id}")
print(f"Token Text : {next_str!r}")
```

    === Verifier Greedy Next ===
    Prompt     : 'She walked into the'
    Token ID   : 2119
    Token Text : ' room'



```python
# =========================
# 4) Acceptance rules
# =========================
def debug_accept_soft_guarded(probs, t_id, 
                              p_thresh, topk_gate, margin_gate, tok):
    p_t = float(probs[0, t_id])
    k = min(topk_gate, probs.shape[-1])
    top_vals, top_idx = torch.topk(probs, k=k, dim=-1)

    in_topk = bool((top_idx[0] == t_id).any())
    p_top1 = float(top_vals[0, 0])
    within_margin = (p_top1 - p_t) <= margin_gate

    print("=== Debug accept_soft_guarded ===")
    print(f"Drafter token : {tok.decode([t_id])!r} (id={t_id})")
    print(f"Drafter prob  : {p_t:.4f}")
    print(f"Top1 prob     : {p_top1:.4f}")
    print(f"In top-{k}?   : {in_topk}")
    print(f"Within margin?: {within_margin} (margin={margin_gate})")
    print(f"Above thresh? : {p_t >= p_thresh} (thresh={p_thresh})")
    print("=> ACCEPT?    :", (p_t >= p_thresh) and in_topk and within_margin)
```


```python
prompt = "She walked into the"
cur_ids = to_ids(models, prompt)

# Verifier distribution
logits = verifier_logits(models, cur_ids)
probs = torch.softmax(logits, dim=-1)

# Assume drafter proposed: ' room'
t_id = models.tok.encode(" room")[0]

# Run the debug function
debug_accept_soft_guarded(
    probs, t_id,
    p_thresh=0.05,
    topk_gate=5,
    margin_gate=0.20,
    tok=models.tok
)
```

    === Debug accept_soft_guarded ===
    Drafter token : ' room' (id=2119)
    Drafter prob  : 0.1380
    Top1 prob     : 0.1380
    In top-5?   : True
    Within margin?: True (margin=0.2)
    Above thresh? : True (thresh=0.05)
    => ACCEPT?    : True



```python
# =========================
# 5) Speculative steps (refactored)
# _init_cur_ids : Convert input string → token tensor
# _get_draft_ids : Drafter proposals (k tokens)
# _verifier_probs : Verifier’s next-token probability distribution
# _should_accept : Acceptance decision (threshold / Top-k / margin)
# _append_token : Common utility to append one token
# _accept_token : Append token if accepted
# _reject_and_take_verifier : On rejection, append verifier’s greedy token & stop
# _process_one_draft_token : Handle a single draft token (decide accept/reject)
# speculative_step_soft_guarded : Orchestration (loop over draft tokens)
# =========================
```


```python
@torch.no_grad()
def _init_cur_ids(m: Models, text: str) -> torch.Tensor:
    """Convert prompt string into a token ID tensor"""
    return to_ids(m, text)
```


```python
prompt = "She walked into the"

cur_ids = _init_cur_ids(models, prompt)

print("=== _init_cur_ids Debug ===")
print(f"Prompt     : {prompt!r}")
print(f"Tensor     : {cur_ids}")
print(f"Shape      : {tuple(cur_ids.shape)}")
print(f"As list    : {cur_ids.tolist()}")
```

    === _init_cur_ids Debug ===
    Prompt     : 'She walked into the'
    Tensor     : tensor([[3347, 6807,  656,  262]], device='mps:0')
    Shape      : (1, 4)
    As list    : [[3347, 6807, 656, 262]]



```python
@torch.no_grad()
def _get_draft_ids(m: Models, cur: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    """Drafter proposes cfg.DRAFT_SPAN tokens (greedy or sampling)"""
    return drafter_propose(m, cur, cfg.DRAFT_SPAN, cfg.DO_SAMPLE_DRAFT)
```


```python
prompt = "She walked into the"

# 1) Tokenize the prompt
cur_ids = _init_cur_ids(models, prompt)

# 2) Drafter proposes k tokens
draft_ids = _get_draft_ids(models, cur_ids, cfg)

print("=== _get_draft_ids Debug ===")
print(f"Prompt        : {prompt!r}")
print(f"Draft IDs     : {draft_ids}")
print(f"Shape         : {tuple(draft_ids.shape)}")
print(f"As list       : {draft_ids.tolist()}")
print(f"Draft decoded : {models.tok.decode(draft_ids, skip_special_tokens=True)!r}")
```

    === _get_draft_ids Debug ===
    Prompt        : 'She walked into the'
    Draft IDs     : tensor([2119,  290, 3114,  379,  262], device='mps:0')
    Shape         : (5,)
    As list       : [2119, 290, 3114, 379, 262]
    Draft decoded : ' room and looked at the'



```python
@torch.no_grad()
def _verifier_probs(m: Models, cur: torch.Tensor) -> torch.Tensor:
    """Verifier's probability distribution for the next token (softmax)"""
    logits = verifier_logits(m, cur)            # shape [1, vocab_size]
    probs = torch.softmax(logits, dim=-1)       # normalize to probabilities
    return probs
```


```python
prompt = "She walked into the"

# 1) Convert prompt to token IDs
cur_ids = _init_cur_ids(models, prompt)

# 2) Get probability distribution from the verifier
probs = _verifier_probs(models, cur_ids)

print("=== _verifier_probs Debug ===")
print(f"Prompt       : {prompt!r}")
print(f"Shape        : {tuple(probs.shape)}")      # (1, vocab_size)
print(f"Sum probs    : {float(probs.sum()):.4f}") # should be ≈ 1.0 since it's softmax

# 3) Show top-5 predictions
top_vals, top_idx = torch.topk(probs, k=5, dim=-1)
for rank, (tid, p) in enumerate(zip(top_idx[0].tolist(), top_vals[0].tolist()), start=1):
    tok_str = models.tok.decode([tid])
    print(f"{rank}. {tok_str!r:>10} (id={tid})  prob={p:.4f}")
```

    === _verifier_probs Debug ===
    Prompt       : 'She walked into the'
    Shape        : (1, 50257)
    Sum probs    : 1.0000
    1.    ' room' (id=2119)  prob=0.1380
    2. ' kitchen' (id=9592)  prob=0.0665
    3.  ' office' (id=2607)  prob=0.0413
    4. ' bathroom' (id=12436)  prob=0.0182
    5.   ' house' (id=2156)  prob=0.0158



```python
# Final decision function: should we accept the drafter's token or not?
@torch.no_grad()
def _should_accept(m: Models, cur: torch.Tensor, t_id: int, cfg: Cfg) -> bool:
    """
    Calls accept_soft_guarded(...)

    Returns True (accept) only if all three conditions hold:
      - Drafter token probability ≥ threshold (p_thresh)
      - Drafter token is within top-k candidates (topk_gate)
      - Probability gap with top-1 ≤ margin (margin_gate)
    """
    # Compute verifier’s next-token probability distribution
    # (softmax output, shape = [1, vocab_size])
    probs = _verifier_probs(m, cur)

    return accept_soft_guarded(
        probs, t_id,
        cfg.P_THRESH, cfg.TOPK_GATE, cfg.MARGIN_GATE
    )
```


```python
prompt = "She walked into the"
cur_ids = _init_cur_ids(models, prompt)

# Assume the drafter proposed " room"
t_id = models.tok.encode(" room")[0]

ok = _should_accept(models, cur_ids, t_id, cfg)

print("=== _should_accept Debug ===")
print(f"Prompt        : {prompt!r}")
print(f"Drafter token : {models.tok.decode([t_id])!r} (id={t_id})")
print(f"Accept?       : {ok}")
```

    === _should_accept Debug ===
    Prompt        : 'She walked into the'
    Drafter token : ' room' (id=2119)
    Accept?       : True



```python
def _append_token(cur: torch.Tensor, token_id: int) -> torch.Tensor:
    """Append a single token to the end of the current sequence"""
    return torch.cat([cur, torch.tensor([[token_id]], device=cur.device)], dim=1)
```


```python
prompt = "She walked into the"
cur_ids = _init_cur_ids(models, prompt)

print("Before IDs:", cur_ids.tolist())
print("Before Decoded:", models.tok.decode(cur_ids[0]))

# Append the drafter/verifier proposed token " room" (id=2119)
new_ids = _append_token(cur_ids, 2119)

print("\nAfter IDs:", new_ids.tolist())
print("After Decoded:", models.tok.decode(new_ids[0]))
```

    Before IDs: [[3347, 6807, 656, 262]]
    Before Decoded: She walked into the
    
    After IDs: [[3347, 6807, 656, 262, 2119]]
    After Decoded: She walked into the room



```python
@torch.no_grad()
def _accept_token(cur: torch.Tensor, t_id: int) -> torch.Tensor:
    """Accept the drafter token by appending it to the sequence"""
    return _append_token(cur, t_id)
```


```python
@torch.no_grad()
# When the drafter’s proposal is rejected, instead append the verifier’s most likely token (greedy next) and stop
def _reject_and_take_verifier(m: Models, cur: torch.Tensor) -> torch.Tensor:
    """
    Reject: append one greedy token from the verifier and then stop
    (Matches the "stop at first mismatch" rule used in the paper/baseline)
    """
    logits = verifier_logits(m, cur)          # Scores for all possible next tokens given the current sequence
    g_id = int(torch.argmax(logits, dim=-1)[0])  # Select the token ID with the highest probability (argmax)
    return _append_token(cur, g_id)
```


```python
prompt = "She walked into the"
cur_ids = _init_cur_ids(models, prompt)

print("Before:", models.tok.decode(cur_ids[0]))

# Assume the drafter token was rejected → append verifier’s greedy token instead
new_ids = _reject_and_take_verifier(models, cur_ids)

print("After :", models.tok.decode(new_ids[0]))
```

    Before: She walked into the
    After : She walked into the room



```python
@torch.no_grad()
def _process_one_draft_token(m: Models, cur: torch.Tensor, t_id: int, cfg: Cfg, accepted: int) -> tuple[torch.Tensor, int, bool]:
    """
    Process a single drafter token:
      - If acceptance conditions are met AND accepted < MAX_ACCEPT → accept token, continue (continue=True)
      - Otherwise → reject and append verifier’s greedy token, then stop (continue=False)
    Returns: (updated cur sequence, increment in accepted count (0/1), continue flag)
    """
    # Check conditions:
    # If the number of accepted tokens so far is less than MAX_ACCEPT
    # AND the drafter token t_id satisfies _should_accept (threshold, top-k, margin)
    if (accepted < cfg.MAX_ACCEPT) and _should_accept(m, cur, t_id, cfg):
        cur = _accept_token(cur, t_id)
        return cur, 1, True
    # Rejection case
    else:
        cur = _reject_and_take_verifier(m, cur)
        return cur, 0, False
```


```python
prompt = "She walked into the"
cur_ids = _init_cur_ids(models, prompt)

# Assume the drafter proposed " room" (id=2119)
t_id = models.tok.encode(" room")[0]

new_cur, inc, cont = _process_one_draft_token(models, cur_ids, t_id, cfg, accepted=0)

print("=== _process_one_draft_token Debug ===")
print("Before prompt :", models.tok.decode(cur_ids[0]))
print("Draft token   :", models.tok.decode([t_id]))
print("After update  :", models.tok.decode(new_cur[0]))
print("Accepted increment:", inc)
print("Continue?     :", cont)
```

    === _process_one_draft_token Debug ===
    Before prompt : She walked into the
    Draft token   :  room
    After update  : She walked into the room
    Accepted increment: 1
    Continue?     : True



```python
@torch.no_grad()
def speculative_step_soft_guarded(
    m: Models, text: str, cfg: Cfg
) -> dict:
    cur = _init_cur_ids(m, text)
    draft_ids = _get_draft_ids(m, cur, cfg)

    accepted = 0
    for t in draft_ids.tolist():
        cur, inc, cont = _process_one_draft_token(m, cur, t, cfg, accepted)
        accepted += inc
        if not cont:
            break

    return {
        # draft_ids is already a tensor → no need for torch.tensor(...)
        "draft_text": m.tok.decode(draft_ids, skip_special_tokens=True),
        "accepted_len": accepted,
        "final_text": decode(m, cur),
    }
```


```python
# Example usage (print return values)
prompt = "She walked into the"
res = speculative_step_soft_guarded(models, prompt, cfg)

print("=== Speculative Step Result ===")
print("Prompt       :", prompt)
print("Draft text   :", res["draft_text"])
print("Accepted len :", res["accepted_len"])
print("Final text   :", res["final_text"])
```

    === Speculative Step Result ===
    Prompt       : She walked into the
    Draft text   :  room and looked at the
    Accepted len : 3
    Final text   : She walked into the room and looked around



```python
# =========================
# 6) Greedy baseline - compare with soft+guarded
# =========================
@torch.no_grad()
def speculative_step_greedy_exact(m: Models, text: str, k: int = 5) -> dict:
    # Convert text into token IDs
    cur = to_ids(m, text)

    # Drafter proposes k tokens (always greedy here, no sampling)
    draft_ids = drafter_propose(m, cur, k, do_sample=False)

    accepted = 0
    for t in draft_ids.tolist():
        # Verifier’s greedy prediction
        g = verifier_greedy_next(m, cur)

        if g == t:
            # If drafter’s token matches verifier’s greedy prediction → accept
            cur = torch.cat([cur, torch.tensor([[g]], device=cur.device)], dim=1)
            accepted += 1
        else:
            # If mismatch → append verifier’s greedy token and stop
            cur = torch.cat([cur, torch.tensor([[g]], device=cur.device)], dim=1)
            break

    return {
        "draft_text": m.tok.decode(draft_ids, skip_special_tokens=True),
        "accepted_len": accepted,
        "final_text": decode(m, cur),
    }
```


```python
prompt = "She walked into the"

# Run soft+guarded speculative decoding
res_soft = speculative_step_soft_guarded(models, prompt, cfg)

# Run greedy baseline (using the same draft span)
res_greedy = speculative_step_greedy_exact(models, prompt, k=cfg.DRAFT_SPAN)

print("=== Compare Results ===")
print(f"Prompt       : {prompt}")

print("--- Soft+Guarded ---")
print(f"Draft text   : {res_soft['draft_text']}")
print(f"Accepted len : {res_soft['accepted_len']}")
print(f"Final text   : {res_soft['final_text']}")

print("--- Greedy Baseline ---")
print(f"Draft text   : {res_greedy['draft_text']}")
print(f"Accepted len : {res_greedy['accepted_len']}")
print(f"Final text   : {res_greedy['final_text']}")
```

    === Compare Results ===
    Prompt       : She walked into the
    --- Soft+Guarded ---
    Draft text   :  room and looked at the
    Accepted len : 3
    Final text   : She walked into the room and looked around
    --- Greedy Baseline ---
    Draft text   :  room and looked at the
    Accepted len : 3
    Final text   : She walked into the room and looked around



```python
# =========================
# 7) Benchmark helpers
# =========================
def run_prompts(m: Models, cfg: Cfg, prompts: List[str]) -> List[Dict[str, Any]]:
    results = []
    for p in prompts:
        out = speculative_step_soft_guarded(m, p, cfg)
        results.append({"prompt": p, **out})
    return results


def print_results(title: str, rows: List[Dict[str, Any]]):
    print(f"\n=== {title} ===")
    for r in rows:
        print("\n--- Prompt ---")
        print(r["prompt"])
        print("--- Draft ---")
        print(r["draft_text"])
        print(f"Accepted length: {r['accepted_len']}")
        print("--- Final ---")
        print(r["final_text"])
```


```python
# =========================
# 8) Main (example)
# =========================
if __name__ == "__main__":
    cfg = Cfg()
    models = load_models(cfg)

    prompts = [
        "Once upon a time,",
        "The capital of France is",
        "She walked into the",
        "It is important to",
        "This is a simple",
        "In the beginning,"
    ]

    # Soft + Guarded
    soft_rows = run_prompts(models, cfg, prompts)
    print_results("Soft + Guarded Acceptance", soft_rows)

    # Greedy exact (optional 비교)
    greedy_rows = []
    for p in prompts:
        out = speculative_step_greedy_exact(models, p, k=cfg.DRAFT_SPAN)
        greedy_rows.append({"prompt": p, **out})
    print_results("Greedy Exact (baseline)", greedy_rows)
```

    ✅ device: mps
    
    === Soft + Guarded Acceptance ===
    
    --- Prompt ---
    Once upon a time,
    --- Draft ---
     the world was a place
    Accepted length: 1
    --- Final ---
    Once upon a time, the world
    
    --- Prompt ---
    The capital of France is
    --- Draft ---
     the capital of the French
    Accepted length: 5
    --- Final ---
    The capital of France is the capital of the French
    
    --- Prompt ---
    She walked into the
    --- Draft ---
     room and looked at the
    Accepted length: 3
    --- Final ---
    She walked into the room and looked around
    
    --- Prompt ---
    It is important to
    --- Draft ---
     note that the current state
    Accepted length: 3
    --- Final ---
    It is important to note that the data
    
    --- Prompt ---
    This is a simple
    --- Draft ---
     example of how to use
    Accepted length: 0
    --- Final ---
    This is a simple,
    
    --- Prompt ---
    In the beginning,
    --- Draft ---
     the only way to get
    Accepted length: 1
    --- Final ---
    In the beginning, the world
    
    === Greedy Exact (baseline) ===
    
    --- Prompt ---
    Once upon a time,
    --- Draft ---
     the world was a place
    Accepted length: 0
    --- Final ---
    Once upon a time, there
    
    --- Prompt ---
    The capital of France is
    --- Draft ---
     the capital of the French
    Accepted length: 0
    --- Final ---
    The capital of France is Paris
    
    --- Prompt ---
    She walked into the
    --- Draft ---
     room and looked at the
    Accepted length: 3
    --- Final ---
    She walked into the room and looked around
    
    --- Prompt ---
    It is important to
    --- Draft ---
     note that the current state
    Accepted length: 3
    --- Final ---
    It is important to note that the data
    
    --- Prompt ---
    This is a simple
    --- Draft ---
     example of how to use
    Accepted length: 0
    --- Final ---
    This is a simple,
    
    --- Prompt ---
    In the beginning,
    --- Draft ---
     the only way to get
    Accepted length: 1
    --- Final ---
    In the beginning, the world



```python

```
