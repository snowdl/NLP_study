```python
import random, torch
DEVICE = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else ("cuda" if torch.cuda.is_available() else "cpu"))
print("DEVICE =", DEVICE)

def set_seed(seed=7):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(7)
```

    DEVICE = mps



```python
#step1 : Tokenizer
```


```python
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_ID = "distilgpt2"

# Load tokenizer for the given model
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# If the tokenizer does not define an EOS (end-of-sequence) token,
# assign an empty string as a placeholder.
if tok.eos_token_id is None:
    tok.eos_token = ""

# If the tokenizer does not define a PAD (padding) token,
# reuse the EOS token as PAD. (Common practice for GPT-like models.)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# Load the causal language model and move it to the chosen device (CPU, CUDA, or MPS).
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()

# Enable caching of past key/values to speed up generation.
model.config.use_cache = True

# Save the EOS token ID for later checks during generation.
EOS_ID = tok.eos_token_id
```


```python
# STEP2 — Utilities: encode / decode / greedy_next

def encode(text: str):
    """
    Convert text into token IDs tensor.
    - Returns: shape [1, T] on the chosen DEVICE.
    """
    return tok(text, return_tensors="pt").to(DEVICE)["input_ids"]

def decode(ids):
    """
    Convert token IDs back into text.
    - Skip special tokens like <pad> or <eos>.
    """
    return tok.decode(ids[0], skip_special_tokens=True)

@torch.inference_mode()
def greedy_next(ids):
    """
    Select the next token using greedy decoding.
    - Run the model on the current sequence.
    - Take logits (scores) from the last position.
    - Return the token ID with the highest score (argmax).
    """
    logits = model(ids).logits[0, -1, :]      # last position logits
    return int(torch.argmax(logits).item())   # ID of best token

print("[STEP2] utils ready")

```

    [STEP2] utils ready



```python
prompt = "Artificial intelligence is changing the way humans ,"
ids = encode(prompt)
start = ids.shape[1]

for _ in range(30):
    nxt = greedy_next(ids)
    if EOS_ID is not None and nxt == EOS_ID:
        break
    ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)

print("=== Greedy ===")
print(decode(ids))
```

    === Greedy ===
    Artificial intelligence is changing the way humans , and it’s changing the way we interact with other people.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



```python
#Sampling(only temperature)
```


```python
@torch.inference_mode()
def softmax_temp(logits, temperature=1.0):
    """
    Apply temperature scaling to logits and convert to probabilities.
    - logits: raw scores for each vocabulary token (tensor [V]).
    - temperature:
        < 1.0 → sharper distribution (more deterministic).
        > 1.0 → flatter distribution (more random).
    - A very small floor (1e-6) is applied to avoid division by zero.
    - Returns: probability vector (tensor [V]) that sums to 1.0.
    """
    t = max(float(temperature), 1e-6)
    return torch.softmax(logits / t, dim=-1)

@torch.inference_mode()
def sample_next_temp_only(ids, temperature=0.9):
    """
    Sample the next token using temperature-scaled softmax only
    (no top-p or repetition penalties here).

    Steps:
      1. Run the model on the current sequence `ids`.
      2. Take the logits from the last position (next-token scores).
      3. Convert logits → probabilities using `softmax_temp`.
      4. Draw one token index at random according to these probabilities
         using `torch.multinomial`.
      5. Return the chosen token ID (int).
    """
    logits = model(ids).logits[0, -1, :]        # last-step logits
    probs  = softmax_temp(logits, temperature)  # apply temperature
    pick   = torch.multinomial(probs, 1)[0].item()  # sample 1 token ID
    return int(pick)
```


```python
temperature = 0.9
t = max(float(temperature), 1e-6)
print("Temperature actually used:", t)
```

    Temperature actually used: 0.9



```python
#test
```


```python
ids2 = encode(prompt)
for _ in range(30):
    nxt = sample_next_temp_only(ids2, temperature=0.9)
    if EOS_ID is not None and nxt == EOS_ID: break
    ids2 = torch.cat([ids2, torch.tensor([[nxt]], device=ids2.device)], dim=1)

print("=== Sample (temp only) ===")
print(decode(ids2))
```

    === Sample (temp only) ===
    Artificial intelligence is changing the way humans , and understanding human biology is changing the way humans are prepared for its challenges, a presentation published this week in the journal Current Biology Proceedings of the National Academy



```python
#STEP 5 — nucleus(top-p)
```


```python
@torch.inference_mode()
def top_p_indices(probs, top_p=0.95):
    """
    Nucleus (top-p) filtering:
    - Select smallest set of tokens whose cumulative prob ≤ top_p.
    - Always keep the top-1 token.
    """
    if top_p is None or top_p >= 1:
        return torch.arange(probs.numel(), device=probs.device)

    sp, sx = torch.sort(probs, descending=True)  # sorted probs & indices
    csum = torch.cumsum(sp, dim=0)               # cumulative sum
    keep = csum <= top_p
    keep[0] = True
    return sx[keep]                              # candidate indices
```


```python
@torch.inference_mode()
def sample_next(ids, temperature=0.9, top_p=0.95):
    """
    Sample the next token:
    - Apply temperature scaling.
    - Apply nucleus (top-p) filtering.
    - Draw 1 token at random from the filtered set.
    """
    logits = model(ids).logits[0, -1, :]      # last-step logits
    probs  = softmax_temp(logits, temperature)

    pool_ix = top_p_indices(probs, top_p)     # step 1: filter candidates
    pool = probs[pool_ix]                     # step 2: restrict probs
    pool = pool / pool.sum()                  # step 3: normalize
    pick_local = torch.multinomial(pool, 1)[0].item()  # step 4: sample one
    return int(pool_ix[pick_local].item())    # step 5: map back to vocab ID
```


```python
# 프롬프트 문장
prompt = "Artificial intelligence is transforming the world because "
ids = encode(prompt)

# 1. 모델이 준 마지막 위치 로짓
logits = model(ids).logits[0, -1, :]
print("Logits shape:", logits.shape)

# 2. softmax + temperature 적용
probs = softmax_temp(logits, temperature=0.9)
print("Sum of probs (should be ~1.0):", probs.sum().item())

# 3. top-p 필터링으로 후보 뽑기
candidates = top_p_indices(probs, top_p=0.95)
print("Number of candidate tokens:", len(candidates))

# 4. 실제로 하나 샘플링
picked_id = sample_next(ids, temperature=0.9, top_p=0.95)
print("Picked token ID:", picked_id)
print("Picked token str:", tok.decode([picked_id]))
```

    Logits shape: torch.Size([50257])
    Sum of probs (should be ~1.0): 1.0
    Number of candidate tokens: 66
    Picked token ID: 933
    Picked token str: vern



```python
#Drafter
```


```python
 @torch.inference_mode()
def propose_branch(ids, span=3, temperature=0.9, top_p=0.95):
    cur = ids.clone()
    out = []
    for _ in range(span):
        t = sample_next(cur, temperature, top_p)
        out.append(t)
        cur = torch.cat([cur, torch.tensor([[t]], device=cur.device)], dim=1)
    return out  # list[int]
```


```python
prompt = "Artificial intelligence is changing the way humans "
ids = encode(prompt)

branch = propose_branch(ids, span=5, temperature=0.9, top_p=0.95)

print("Proposed token IDs:", branch)                # 숫자 리스트
print("Decoded tokens:", [tok.decode([t]) for t in branch])  # 각각 글자로
print("Joined as text:", tok.decode(branch))        # 한 번에 이어붙인 결과
```

    Proposed token IDs: [1133, 511, 9017, 11, 355]
    Decoded tokens: ['ute', ' their', ' minds', ',', ' as']
    Joined as text: ute their minds, as



```python
#STEP 7 — prefix-accept
```


```python
@torch.inference_mode()
def prefix_accept_once(ids, branch):
    """
    Compare a drafted branch with greedy decoding, one token at a time.
    
    Process:
      1. Start with the current sequence `ids`.
      2. For each token `t` in the proposed branch:
         - Compute greedy_next(cur): the model’s best next token.
         - If greedy == proposed token:
             → Accept it, append `t` to the sequence, and continue.
         - If greedy != proposed token:
             → Reject the branch at this point, 
               append the greedy token instead, and stop checking further.
      3. Return:
         - The new sequence with accepted tokens (and possibly one greedy token).
         - The count of how many proposed tokens were accepted before the mismatch.
    
    Args:
        ids (torch.Tensor): Current token IDs [1, T].
        branch (list[int]): Drafted token IDs.
    
    Returns:
        (torch.Tensor, int): (updated sequence, number of accepted tokens)
    """
    cur = ids.clone()
    accepted = 0
    for t in branch:
        # 1) Greedy prediction for the next token
        g = greedy_next(cur)
        if g == t:
            # 2) If they match → accept the proposed token
            cur = torch.cat([cur, torch.tensor([[t]], device=cur.device)], dim=1)
            accepted += 1
        else:
            # 3) If mismatch → append greedy token and stop
            cur = torch.cat([cur, torch.tensor([[g]], device=cur.device)], dim=1)
            break
    return cur, accepted
```


```python
#STEP 8 — Medusa-tiny
```


```python
@torch.inference_mode()
def medusa_tiny(prompt, max_new_tokens=30, span=3, temperature=0.9, top_p=0.95):
    ids = encode(prompt)
    start = ids.shape[1]
    steps = 0
    max_steps = max_new_tokens * 3  
    while ids.shape[1] - start < max_new_tokens and steps < max_steps:
        branch = propose_branch(ids, span=span, temperature=temperature, top_p=top_p)
        ids, _ = prefix_accept_once(ids, branch)
        steps += 1
    return decode(ids)
```


```python
print("=== Medusa-tiny ===")
print(medusa_tiny("In a distant future, ", max_new_tokens=30, span=3, temperature=0.9, top_p=0.95))
```

    === Medusa-tiny ===
    In a distant future,   the world is a place where the world is a place where the world is a place where the world is a place where the world is a place



```python

```
