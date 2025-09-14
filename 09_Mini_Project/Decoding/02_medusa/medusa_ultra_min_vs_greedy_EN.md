```python
"""
Medusa Ultra-Min (Greedy vs Medusa) — module header with thorough English comments.

This file is organized into small, toggleable steps so you can learn and run the
pipeline incrementally. Turn individual steps ON/OFF with the RUN_STEP_* flags
below. The typical execution order is STEP0 → STEP1 → STEP2 → … → STEP7.

Workflow overview (translated from your Korean outline):

STEP0 — Device selection
    - Detect a compute device in this priority: Apple Silicon `mps` → `cuda` → `cpu`.
    - Optionally fix random seeds for reproducibility.

STEP1 — Tokenizer / Model preparation
    - Load `tok` (AutoTokenizer) and `model` (AutoModelForCausalLM).
    - Ensure `eos_token` / `pad_token` are set; cache `EOS_ID`.

STEP2 — Utility functions
    - `encode`, `decode`, `append_token`, `last_logits`, `greedy_next`.

STEP3 — Sampling functions  ➜ (this is the part you said you’re focusing on now)
    - `softmax_temp`, `top_p_indices`, `sample_next` (with optional EOS-ban and repetition-ban).

STEP4 — Drafter (`propose_branch`)
    - Use sampling to propose a short branch of tokens (span length).

STEP5 — Prefix-accept (`prefix_accept_once`)
    - Compare drafter tokens vs greedy tokens; accept matching prefix.
    - On the first mismatch, take the greedy token and stop.

STEP6 — Loop execution (`medusa_tiny` / `run_greedy`)
    - Repeat propose + prefix-accept until `max_new_tokens` is reached.

STEP7 — Demo (print Greedy vs Medusa outputs)
"""
```




    '\nMedusa Ultra-Min (Greedy vs Medusa) — module header with thorough English comments.\n\nThis file is organized into small, toggleable steps so you can learn and run the\npipeline incrementally. Turn individual steps ON/OFF with the RUN_STEP_* flags\nbelow. The typical execution order is STEP0 → STEP1 → STEP2 → … → STEP7.\n\nWorkflow overview (translated from your Korean outline):\n\nSTEP0 — Device selection\n    - Detect a compute device in this priority: Apple Silicon `mps` → `cuda` → `cpu`.\n    - Optionally fix random seeds for reproducibility.\n\nSTEP1 — Tokenizer / Model preparation\n    - Load `tok` (AutoTokenizer) and `model` (AutoModelForCausalLM).\n    - Ensure `eos_token` / `pad_token` are set; cache `EOS_ID`.\n\nSTEP2 — Utility functions\n    - `encode`, `decode`, `append_token`, `last_logits`, `greedy_next`.\n\nSTEP3 — Sampling functions  ➜ (this is the part you said you’re focusing on now)\n    - `softmax_temp`, `top_p_indices`, `sample_next` (with optional EOS-ban and repetition-ban).\n\nSTEP4 — Drafter (`propose_branch`)\n    - Use sampling to propose a short branch of tokens (span length).\n\nSTEP5 — Prefix-accept (`prefix_accept_once`)\n    - Compare drafter tokens vs greedy tokens; accept matching prefix.\n    - On the first mismatch, take the greedy token and stop.\n\nSTEP6 — Loop execution (`medusa_tiny` / `run_greedy`)\n    - Repeat propose + prefix-accept until `max_new_tokens` is reached.\n\nSTEP7 — Demo (print Greedy vs Medusa outputs)\n'




```python
from __future__ import annotations


import random
from dataclasses import dataclass
from typing import List, Tuple


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================
# RUN SWITCHES (turn individual steps ON/OFF)
# =============================
RUN_STEP_0 = True  # Device selection (and optional seeding)
RUN_STEP_1 = True  # Tokenizer/model load
RUN_STEP_2 = True  # Utilities
RUN_STEP_3 = True  # Sampling
RUN_STEP_4 = True  # Drafter
RUN_STEP_5 = True  # Prefix-accept
RUN_STEP_6 = True  # Loops
RUN_STEP_7 = True  # Demo
```


```python
# =============================
# 0) Config
# =============================
@dataclass
class Cfg:
    MODEL_ID: str = "distilgpt2"
    TEMPERATURE: float = 0.9
    TOP_P: float = 0.95
    SPAN: int = 3
    MAX_NEW_TOKENS: int = 30
    BAN_EOS_FIRST_N: int = 0
    REP_BAN_N: int = 0
    SEED: int | None = 7
    DEBUG: bool = False
cfg = Cfg()
```


```python
if RUN_STEP_0:
    DEVICE = (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    def set_seed(seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(cfg.SEED)
    print(f"[STEP0] DEVICE = {DEVICE}")

```

    [STEP0] DEVICE = mps



```python
# =============================
# STEP 1 —  Tokenizer/model load (eos/pad 보정)
# =============================
if RUN_STEP_1:
    def load_tokenizer(model_id: str):
        tok = AutoTokenizer.from_pretrained(model_id)
         # If the tokenizer does not already define an EOS token,
        # assign an empty string "" as a placeholder.
        # (This ensures the model has a valid end-of-sequence marker.)
        if tok.eos_token_id is None:
            tok.eos_token = ""
          # If the tokenizer does not already define a PAD token,
        # reuse the EOS token as padding.
        # (Many GPT-style models do not have a PAD token by default,
        # but they can safely use EOS as padding since extra EOS tokens
        # at the end of input do not change the model’s behavior.)          
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return tok

    def load_model(model_id: str, device: str):
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        model.config.use_cache = True
        return model

    tok = load_tokenizer(cfg.MODEL_ID)
    model = load_model(cfg.MODEL_ID, DEVICE)
    EOS_ID = tok.eos_token_id

    print(f"[STEP1] MODEL = {cfg.MODEL_ID}, EOS_ID={EOS_ID}")
```

    [STEP1] MODEL = distilgpt2, EOS_ID=50256



```python
# =============================
# STEP 2 — Utility functions
# (encode / decode / append / last_logits / greedy_next)
# =============================
if RUN_STEP_2:
    def encode(text: str) -> torch.Tensor:
        """
        Convert a string into token IDs.
        - Input: raw text string
        - Output: tensor of shape [1, T] moved onto DEVICE (mps/cuda/cpu)
        """
        return tok(text, return_tensors="pt").to(DEVICE)["input_ids"]

    def decode(ids: torch.Tensor) -> str:
        """
        Convert token IDs back into a string.
        - skip_special_tokens=True removes EOS, PAD, etc. from the output
        """
        return tok.decode(ids[0], skip_special_tokens=True)

    def append_token(ids: torch.Tensor, token_id: int) -> torch.Tensor:  #Helper to add one token to the end of the sequence.
        """
        Append a single token ID to the current sequence.
        - ids: tensor [1, T]
        - token_id: int, the new token to add
        - return: tensor [1, T+1]
        """
        t = torch.tensor([[token_id]], device=ids.device)
        return torch.cat([ids, t], dim=1)

    @torch.inference_mode()
    def last_logits(ids: torch.Tensor) -> torch.Tensor:  #Get the model’s score distribution for the next token.
        """
        Get the logits (unnormalized scores) of the last position.
        - Forward pass the model on 'ids'
        - Take logits from the last time step → shape [V]
        where V = vocabulary size
        """
        return model(ids).logits[0, -1, :]

    @torch.inference_mode()
    def greedy_next(ids: torch.Tensor) -> int: #Pick the highest-scoring token (greedy decoding).
        """
        Select the next token using greedy decoding.
        - Take argmax over the last logits
        - Return the token ID (int)
        """
        return int(torch.argmax(last_logits(ids)).item())

    print("[STEP2] utils ready: encode/decode/append/last_logits/greedy_next")

```

    [STEP2] utils ready: encode/decode/append/last_logits/greedy_next



```python
# =============================
# STEP 3 — Sampling Functions
# (softmax_temp / top_p_indices / sample_next)
# =============================
```


```python
# =============================
# STEP 3a — softmax_temp
# =============================
if RUN_STEP_3:
    @torch.inference_mode()
    def softmax_temp(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Apply temperature-scaled softmax to logits.
        - logits: tensor [V], raw scores for each vocabulary token
        - temperature < 1.0 → sharper distribution (more confident)
        - temperature > 1.0 → flatter distribution (more random)
        - returns: probability vector [V]
        """
        t = max(float(temperature), 1e-6)   # avoid division by zero
        return torch.softmax(logits / t, dim=-1)

    print("[STEP3a] ready: softmax_temp")
```

    [STEP3a] ready: softmax_temp



```python
# =============================
# STEP 3b — top_p_indices/Nucleus Sampling (Top-p Sampling)
# =============================
if RUN_STEP_3:
    @torch.inference_mode()
    def top_p_indices(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Perform nucleus (top-p) filtering.
        - Sort tokens by probability (descending)
        - Keep the smallest set of tokens whose cumulative prob ≤ top_p
        - Always keep at least the top-1 token
        - returns: indices of selected tokens
        """
        V = probs.numel()
        if top_p is None or top_p >= 1:
            # keep all tokens
            return torch.arange(V, device=probs.device)
        sp, sx = torch.sort(probs, descending=True)   # sorted probs & indices
        csum = torch.cumsum(sp, dim=0)                # cumulative sum
        keep = csum <= top_p
        keep[0] = True                                # ensure top-1 is included
        return sx[keep]

    print("[STEP3b] ready: top_p_indices")
```

    [STEP3b] ready: top_p_indices



```python
# =============================
# STEP 3c — sample_next
#Apply temperature scaling + nucleus (top-p) filtering + repetition ban + EOS ban option to sample the next token.
# =============================
```


```python
if RUN_STEP_3:
    @torch.inference_mode()
    def sample_next(ids: torch.Tensor, temperature: float, top_p: float,
                    ban_eos: bool = False, rep_ban_n: int = 0) -> int:

        # 1. Get logits for the last position and apply temperature-scaled softmax
        logits = last_logits(ids)
        probs = softmax_temp(logits, temperature)

        # 2. Repetition ban: set probability of the last N tokens to 0
        if rep_ban_n > 0:
            tail = ids[0, -rep_ban_n:].tolist()
            probs[tail] = 0
            s = probs.sum()
            # Re-normalize if valid; if all zero, fall back to original distribution
            probs = probs if s <= 0 else probs / s
            if s <= 0:
                probs = softmax_temp(logits, temperature)  # fallback

        # 3. Apply nucleus (top-p) filtering
        pool_ix = top_p_indices(probs, top_p)
        pool = probs[pool_ix]
        pool = pool / pool.sum()

        # 4. Randomly sample one token from the filtered distribution
        pick_local = int(torch.multinomial(pool, 1)[0].item())
        picked = int(pool_ix[pick_local].item())

        # 5. EOS ban (optional): if EOS is chosen, resample once without EOS
        if ban_eos and EOS_ID is not None and picked == EOS_ID:
            mask = pool_ix != EOS_ID
            if mask.any():
                pool_ix2 = pool_ix[mask]
                pool2 = pool[mask]
                pool2 = pool2 / pool2.sum()
                pick_local = int(torch.multinomial(pool2, 1)[0].item())
                picked = int(pool_ix2[pick_local].item())

        # Return the final sampled token ID
        return picked

    print("[STEP3c] ready: sample_next")
```

    [STEP3c] ready: sample_next



```python
# =============================
# STEP 4 — Drafter (propose_one / propose_branch)
# =============================
if RUN_STEP_4:
    @torch.inference_mode()
    def propose_one(cur_ids: torch.Tensor, temperature: float, top_p: float,
                    accepted_so_far: int) -> int:
        """
        Draft a single token proposal.
        - Decide whether to ban EOS (ban_eos=True) based on how many tokens
          have already been accepted (before BAN_EOS_FIRST_N threshold).
        - Call sample_next() once with the given settings.
        - Return: one proposed token ID.
        """
        ban_eos = accepted_so_far < cfg.BAN_EOS_FIRST_N
        return sample_next(cur_ids, temperature, top_p,
                           ban_eos=ban_eos, rep_ban_n=cfg.REP_BAN_N)

    @torch.inference_mode()
    def propose_branch(ids: torch.Tensor, span: int, temperature: float, top_p: float,
                       accepted_so_far: int = 0) -> List[int]:
        """
        Draft a short branch of tokens (length = span).
        - Repeatedly call propose_one() 'span' times.
        - Append each proposed token to the running sequence so that the next
          proposal is conditioned on the previous draft tokens.
        - Collect all proposed token IDs in a list.
        - If DEBUG is enabled, print each drafted token and its decoded form.
        - Return: list of proposed token IDs (length = span).
        """
        cur = ids.clone()
        out: List[int] = []
        for i in range(span):
            t = propose_one(cur, temperature, top_p, accepted_so_far)
            out.append(t)
            cur = append_token(cur, t)
            if cfg.DEBUG:
                print(f"  [draft {i}] pick={t} ({tok.decode([t])!r})")
        return out

    print("[STEP4] drafter ready: propose_one/propose_branch")
```

    [STEP4] drafter ready: propose_one/propose_branch



```python
# =============================
# STEP 5 — Prefix-Accept (accept_one / prefix_accept_once)
# =============================
if RUN_STEP_5:
    @torch.inference_mode()
    def accept_one(cur: torch.Tensor, token: int) -> Tuple[torch.Tensor, bool]:
        """
        Compare one drafted token vs the greedy token.
        - Compute the greedy token 'g' for the current sequence.
        - If the drafted token == greedy token → accept it and continue.
        - If different → append the greedy token instead, then stop.
        - Return: (new sequence, continue_flag)
        """
        g = greedy_next(cur)
        if cfg.DEBUG:
            print(f"    compare draft={token} ({tok.decode([token])!r}) "
                  f"vs greedy={g} ({tok.decode([g])!r})")
        if g == token:
            return append_token(cur, token), True   # match → keep going
        else:
            return append_token(cur, g), False      # mismatch → stop

    @torch.inference_mode()
    def prefix_accept_once(ids: torch.Tensor, branch: List[int]) -> Tuple[torch.Tensor, int]:
        """
        Compare an entire drafted branch (list of tokens) against greedy decoding.
        - Iterate token by token:
            - Accept while tokens match greedy predictions.
            - On the first mismatch: insert greedy token and stop.
        - Count how many tokens were accepted from the branch.
        - Return: (new sequence including accepted tokens, number_accepted)
        """
        cur = ids.clone()
        accepted = 0
        for t in branch:
            cur, ok = accept_one(cur, t)
            if ok:
                accepted += 1
            else:
                break
        return cur, accepted

    print("[STEP5] prefix-accept ready: accept_one/prefix_accept_once")

```

    [STEP5] prefix-accept ready: accept_one/prefix_accept_once



```python
# =============================
# STEP 6 — Loop execution (ultra-split)
# =============================
if RUN_STEP_6:
    @torch.inference_mode()
    def medusa_loop_step(ids: torch.Tensor, span: int, temperature: float,
                         top_p: float, accepted_total: int, loop_idx: int):
        """
        One Medusa loop step:
        1) Draft a short branch with the drafter (propose_branch).
        2) Apply prefix-accept once to compare with greedy and extend the sequence.
        Returns:
            - ids: the updated sequence after accepting tokens (and possibly 1 greedy token on mismatch)
            - acc: how many draft tokens were accepted in this step
        """
        if cfg.DEBUG:
            print(f"[loop {loop_idx}] cur_len={ids.shape[1]}")
        branch = propose_branch(ids, span, temperature, top_p, accepted_total)
        ids, acc = prefix_accept_once(ids, branch)
        return ids, acc

    @torch.inference_mode()
    def medusa_tiny(prompt: str, max_new_tokens: int, span: int,
                    temperature: float, top_p: float) -> str:
        """
        End-to-end Medusa-tiny decoding:
        - Start from the encoded prompt.
        - Repeat medusa_loop_step until we generate max_new_tokens.
        - Keep track of how many draft tokens were accepted total (optional metric).
        - Decode the final sequence to text.
        """
        ids = encode(prompt)
        start = ids.shape[1]
        accepted_total = 0
        loop_idx = 0
        while ids.shape[1] - start < max_new_tokens:
            ids, acc = medusa_loop_step(ids, span, temperature, top_p,
                                        accepted_total, loop_idx)
            accepted_total += acc
            loop_idx += 1
        return decode(ids)

    @torch.inference_mode()
    def run_greedy(prompt: str, max_new_tokens: int) -> str:
        """
        Plain greedy decoding baseline:
        - Iteratively pick argmax (greedy_next) and append it to the sequence.
        - Stop if EOS appears or we reach max_new_tokens.
        """
        ids = encode(prompt)
        start = ids.shape[1]
        while ids.shape[1] - start < max_new_tokens:
            nxt = greedy_next(ids)
            if EOS_ID is not None and nxt == EOS_ID:
                break
            ids = append_token(ids, nxt)
        return decode(ids)

    print("[STEP6] loops ready: medusa_loop_step/medusa_tiny/run_greedy")
```

    [STEP6] loops ready: medusa_loop_step/medusa_tiny/run_greedy



```python
# =============================
# STEP 7 — 데모 (Greedy vs Medusa 결과)
# =============================
if RUN_STEP_7:
    prompt = "In a distant future, "
    print("\n=== Greedy ===")
    print(run_greedy(prompt, cfg.MAX_NEW_TOKENS), "\n")

    print("=== Medusa-tiny ===")
    print(medusa_tiny(prompt, cfg.MAX_NEW_TOKENS,
                      cfg.SPAN, cfg.TEMPERATURE, cfg.TOP_P))

    # 디버그 보고 싶으면 아래 두 줄로 토글 후 재실행
    # cfg.DEBUG = True
    # print(medusa_tiny(prompt, cfg.MAX_NEW_TOKENS,
    #                   cfg.SPAN, cfg.TEMPERATURE, cfg.TOP_P))

```

    
    === Greedy ===
    In a distant future,   the world is a place where the world is a place where the world is a place where the world is a place where the world is a place 
    
    === Medusa-tiny ===
    In a distant future,   the world is a place where the world is a place where the world is a place where the world is a place where the world is a place where the



```python

```


```python

```
