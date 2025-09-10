```python
#import
```


```python
import torch
import torch.nn.functional as F
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM
```


```python
#device selection
# Pick device (Apple Silicon → mps, else cuda if available, else cpu)
```


```python
device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else ("cuda" if torch.cuda.is_available() else "cpu"))
print("✅ device:", device)
```

    ✅ device: mps



```python
#Tokenizer Draft Vs Target
```


```python
draft_id  = "distilgpt2"  # draft model (smaller/faster)
target_id = "gpt2"        # target model (larger/higher quality)

# Use the target model's tokenizer so both models share the same tokenization/vocab
tok = AutoTokenizer.from_pretrained(target_id)
tok.pad_token = tok.eos_token  # set pad token to eos to silence padding warnings

# Load models and move to the selected device; switch to eval() for inference
draft  = AutoModelForCausalLM.from_pretrained(draft_id).to(device).eval()
target = AutoModelForCausalLM.from_pretrained(target_id).to(device).eval()
```


```python
# Optional: reproducibility
_ = torch.manual_seed(42) 
```


```python
#Draft : Token proposal
```


```python
@torch.no_grad()
def draft_next_token(input_ids: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
    # 1) Forward pass: logits shape is [batch=1, seq_len=T, vocab_size=V]
    out = draft(input_ids=input_ids)

    # 2) Take logits at the last position (distribution for the next token): [1, V]
    logits = out.logits[:, -1, :]

    # 3) Greedy top-1 over the vocabulary: resulting shape [1, 1]
    next_id = torch.argmax(logits, dim=-1, keepdim=True)

    # 4) Append the chosen token to the sequence along the time dimension: [1, T+1]
    new_ids = torch.cat([input_ids, next_id], dim=1)

    # Return the extended sequence, the scalar token id, and the last-step logits
    return new_ids, int(next_id.item()), logits
```


```python
# Minimal test: print what the draft just proposed

prompt = "In a quiet village by the sea,"
enc = tok(prompt, return_tensors="pt")
ids0 = enc.input_ids.to(device)              # [1, T]

new_ids, token_id, logits = draft_next_token(ids0)

print("prompt:", prompt)
print("proposed token id:", token_id)
print("proposed token   :", repr(tok.decode([token_id])))
print("length:", ids0.shape[1], "->", new_ids.shape[1])     # T -> T+1
print("appended OK?", new_ids[0, -1].item() == token_id)    # sanity check
```

    prompt: In a quiet village by the sea,
    proposed token id: 262
    proposed token   : ' the'
    length: 8 -> 9
    appended OK? True



```python
"""
def draft_propose_k: Propose K next tokens sequentially with the draft model (greedy).
 Notes:
        - This is compatible with `draft_next_token` that returns (new_ids, token_id, logits).
        - The `temperature` argument is kept for API compatibility but not used here
          because `draft_next_token` is greedy in this setup.
Args:
     input_ids: Current context token IDs. Shape: [1, T]
        k : Number of tokens to propose in a row.
Returns:
        A Python list of length K with the proposed token IDs (ints).
    """
```




    '\ndef draft_propose_k: Propose K next tokens sequentially with the draft model (greedy).\n Notes:\n        - This is compatible with `draft_next_token` that returns (new_ids, token_id, logits).\n        - The `temperature` argument is kept for API compatibility but not used here\n          because `draft_next_token` is greedy in this setup.\nArgs:\n     input_ids: Current context token IDs. Shape: [1, T]\n        k : Number of tokens to propose in a row.\nReturns:\n        A Python list of length K with the proposed token IDs (ints).\n    '




```python
# Draft → propose K tokens (compatible with draft_next_token returning (new_ids, int, logits))
@torch.no_grad()
def draft_propose_k(input_ids: torch.Tensor, k: int = 4, temperature: float = 0.8) -> List[int]:
    ids = input_ids.clone()        # work on a copy; do not mutate the caller's tensor
    proposals: List[int] = []
    for _ in range(k):
        ids, nid, _ = draft_next_token(ids)  # (new_ids, proposed_token_id, logits)
        proposals.append(nid)
    return proposals  # e.g., [1234, 42, 50256, ...]
```


```python
prompt = "In a quiet village by the sea,"
enc = tok(prompt, return_tensors="pt")
ids0 = enc.input_ids.to(device)

props = draft_propose_k(ids0, k=5)  # propose 5 tokens
print("proposed ids :", props)
print("proposed toks:", [repr(tok.decode([t])) for t in props])
print("original len :", ids0.shape[1])  # ids0 is unchanged
```

    proposed ids : [262, 7404, 318, 257, 1402]
    proposed toks: ["' the'", "' village'", "' is'", "' a'", "' small'"]
    original len : 8



```python
# Target → top-1 token (Jupyter cell)
#Returns  int: The ID of the most likely next token according to the target model.
@torch.no_grad()
def target_top1(input_ids: torch.Tensor) -> int:
    out = target(input_ids=input_ids)      # logits shape: [1, T, V]
    logits = out.logits[:, -1, :]          # take last-step logits: [1, V]
    return int(torch.argmax(logits, dim=-1).item())  # greedy argmax → scalar token id
```


```python
prompt = "In a quiet village by the sea,"
enc = tok(prompt, return_tensors="pt")
ids = enc.input_ids.to(device)

tid = target_top1(ids)
print("top-1 id:", tid)
print("top-1 tok:", repr(tok.decode([tid])))
```

    top-1 id: 262
    top-1 tok: ' the'



```python
"""
def target_sample_one : Sample ONE next token from the target model (stochastic decoding).
Args:
input_ids : Current context token IDs. Shape: [1, T]
temperature: Softens/sharpens the distribution (>1 = more random, <1 = more greedy)
Returns=> int: Sampled next-token ID.
"""
```




    '\ndef target_sample_one : Sample ONE next token from the target model (stochastic decoding).\nArgs:\ninput_ids : Current context token IDs. Shape: [1, T]\ntemperature: Softens/sharpens the distribution (>1 = more random, <1 = more greedy)\nReturns=> int: Sampled next-token ID.\n'




```python
@torch.no_grad()
def target_sample_one(input_ids: torch.Tensor, temperature: float = 0.7) -> int:
    # Forward pass → logits over the vocabulary at each step: [1, T, V]
    out = target(input_ids=input_ids)

    # Take only the last-step logits (distribution for the next token): [1, V]
    logits = out.logits[:, -1, :]

    # Temperature scaling, then convert logits → probabilities
    probs = F.softmax(logits / max(1e-6, temperature), dim=-1)

    # Multinomial sampling: draw exactly one token id from the categorical distribution
    next_id = torch.multinomial(probs, num_samples=1)  # shape: [1, 1]

    # Return as a Python int (batch=1 assumed)
    return int(next_id.item())
```


```python
prompt = "In a quiet village by the sea,"
enc = tok(prompt, return_tensors="pt")
ids = enc.input_ids.to(device)

tid = target_sample_one(ids, temperature=0.8)
print("sampled id:", tid)
print("sampled tok:", repr(tok.decode([tid])))
```

    sampled id: 262
    sampled tok: ' the'



```python
#Verify the draft's proposed tokens in order using the target model.
```


```python
#Process:
#If target top-1 == proposed token → accept (append) and continue.
#Otherwise → sample ONE token from the target, append it, and STOP this cycle.
#Args:
#-input_ids: Current context IDs. Shape: [1, T]
#- proposed  : List of K proposed token IDs from the draft (ints)
#-temperature: Used only for the rejection path sampling
#Returns:new_ids : Updated context after this cycle. Shape: [1, T + accepted] or [1, T + accepted + 1] if rejected
#accepted: Number of proposals accepted in this cycle (0..K)
```


```python
@torch.no_grad()
def verify_one_cycle(
    input_ids: torch.Tensor,
    proposed: List[int],
    temperature: float = 0.7
) -> Tuple[torch.Tensor, int]:
    ids = input_ids.clone()
    accepted = 0
    for t in proposed:
        top1 = target_top1(ids)
        if top1 == t:  # accept
            ids = torch.cat([ids, torch.tensor([[t]], device=ids.device)], dim=1)
            accepted += 1
        else:          # reject → target samples one token, then stop
            samp = target_sample_one(ids, temperature=temperature)
            ids = torch.cat([ids, torch.tensor([[samp]], device=ids.device)], dim=1)
            break
    return ids, accepted

```


```python
prompt = "In a quiet village by the sea,"
enc = tok(prompt, return_tensors="pt")
ids0 = enc.input_ids.to(device)

props = draft_propose_k(ids0, k=3)
ids1, acc = verify_one_cycle(ids0, props, temperature=0.7)

print("proposed:", props)
print("accepted:", acc)
print("len:", ids0.shape[1], "->", ids1.shape[1])
print("partial:", tok.decode(ids1[0, :], skip_special_tokens=True))

```

    proposed: [262, 7404, 318]
    accepted: 2
    len: 8 -> 11
    partial: In a quiet village by the sea, the village did



```python
"""
    Minimal speculative decoding loop.

    Steps per cycle:
      1) Draft proposes K tokens (greedy in this setup).
      2) Target verifies them in order:
         - if target top-1 == proposed → accept (append) and continue
         - else → target samples ONE token, append it, stop the cycle
      3) Repeat cycles until `max_new_tokens` are generated.

    Args:
        prompt         : Seed text.
        max_new_tokens : Generation budget (number of new tokens to add).
        k              : How many tokens the draft proposes per cycle.
        draft_temp     : Present for API symmetry; not used because draft is greedy here.
        target_temp    : Temperature for the target's sampling on reject.

    Returns:
        generated_text : Decoded string.
        total_accepted : Total number of draft proposals accepted by the target.
"""
```




    "\n    Minimal speculative decoding loop.\n\n    Steps per cycle:\n      1) Draft proposes K tokens (greedy in this setup).\n      2) Target verifies them in order:\n         - if target top-1 == proposed → accept (append) and continue\n         - else → target samples ONE token, append it, stop the cycle\n      3) Repeat cycles until `max_new_tokens` are generated.\n\n    Args:\n        prompt         : Seed text.\n        max_new_tokens : Generation budget (number of new tokens to add).\n        k              : How many tokens the draft proposes per cycle.\n        draft_temp     : Present for API symmetry; not used because draft is greedy here.\n        target_temp    : Temperature for the target's sampling on reject.\n\n    Returns:\n        generated_text : Decoded string.\n        total_accepted : Total number of draft proposals accepted by the target.\n"




```python
@torch.no_grad()
def speculative_generate_minimal(
    prompt: str,
    max_new_tokens: int = 60,
    k: int = 4,
    draft_temp: float = 0.8,   # kept for API compatibility; ignored by greedy draft_propose_k
    target_temp: float = 0.7,  # used inside verify_one_cycle (sampling on reject)
) -> Tuple[str, int]:

    # Encode prompt and move to device
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    base_len = input_ids.shape[1]
    total_accepted = 0

    # Keep generating until we reach the budget
    while (input_ids.shape[1] - base_len) < max_new_tokens:
        # 1) Draft proposes K tokens (greedy; `draft_temp` ignored here)
        proposed = draft_propose_k(input_ids, k=k, temperature=draft_temp)

        # 2) Target verifies proposals and merges accepted ones
        input_ids, acc = verify_one_cycle(input_ids, proposed, temperature=target_temp)
        total_accepted += acc

        # 3) If nothing was accepted this cycle, advance one step with target top-1
        if acc == 0 and (input_ids.shape[1] - base_len) < max_new_tokens:
            nid = target_top1(input_ids)
            input_ids = torch.cat([input_ids, torch.tensor([[nid]], device=device)], dim=1)

    # Decode the final sequence
    text = tok.decode(input_ids[0], skip_special_tokens=True)
    return text, total_accepted
```


```python
prompt = "In a quiet village by the sea,"
text, accepted = speculative_generate_minimal(
    prompt, max_new_tokens=60, k=3, draft_temp=0.7, target_temp=0.7
)
print("📝 Prompt:", prompt)
print("✅ Accepted tokens (by target):", accepted)
print("\n=== Output ===\n", text)

```

    📝 Prompt: In a quiet village by the sea,
    ✅ Accepted tokens (by target): 26
    
    === Output ===
     In a quiet village by the sea, the village of Kshira, Flora, with its beautiful trees, is a town of ordinary people. It is a place of quiet, and lived in a time when the people had no power to control the weather. It is a place of beauty, and is a place of fair play.
    



```python

```
