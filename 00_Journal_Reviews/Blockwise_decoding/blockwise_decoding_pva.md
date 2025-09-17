```python
#Greedy (argmax)
#Always selects the token with the highest probability
#→ Deterministic, always the same output
#Sampling
#Selects a token randomly according to the probability distribution
#Example: softmax = [0.7, 0.2, 0.1] →70% chance for the first token,
#20% chance for the second,
#10% chance for the third
#→ Nondeterministic, output may differ on each run
```


```python
import torch, random, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(42); random.seed(42); np.random.seed(42)
```


```python
device = "cpu"
print("device:", device)
```

    device: cpu



```python
# Load tokenizer and model
```


```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
```


```python
#prompt ready
```


```python
prompt = "In a distant future,"
print("prompt:", prompt)
```

    prompt: In a distant future,



```python
#Check PAD/EOS settings
```


```python
# distilgpt2 does not have a default pad_token, so we set it to eos.
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("pad_token:", tok.pad_token)
print("eos_token:", tok.eos_token)
```

    pad_token: <|endoftext|>
    eos_token: <|endoftext|>



```python
#Tokenize input context
```


```python
ctx = tok(prompt, return_tensors="pt").input_ids.to(device)
print("ctx shape:", ctx.shape)
print("ctx tokens:", ctx[0].tolist())
print("ctx decoded:", tok.decode(ctx[0]))
```

    ctx shape: torch.Size([1, 5])
    ctx tokens: [818, 257, 12899, 2003, 11]
    ctx decoded: In a distant future,



```python
# draft_block (for debugging: drafter also generates only 1 block using greedy)
```


```python
@torch.no_grad()
def draft_block_greedy(drafter, input_ids, block_size=3, pad_token_id=None):
    # Running on CPU → just align device
    input_ids = input_ids.to("cpu")
    # When pad==eos, it's safer to explicitly provide attention_mask
    attn = (input_ids != tok.pad_token_id)   # attention mask :creates an attention mask that sets padding token positions to 0 and keeps only the actual input tokens as 1

    out = drafter.generate(
        input_ids,
        attention_mask=attn,
        max_new_tokens=block_size,
        do_sample=False,          # disable sampling (sanity check)
        pad_token_id=pad_token_id
    )
    # Return only the newly generated block (after the prompt)
    return out[:, input_ids.shape[1]:]   # shape: [1, block_size]
```


```python
#drafter 
```


```python
# Load drafter model
from transformers import AutoModelForCausalLM

# drafter: small model (distilgpt2)
drafter = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cpu")

print("drafter ready")
```

    drafter ready



```python
# First block proposal test
```


```python
# ctx는 이전 셀에서 만든 토큰 시퀀스
block = draft_block_greedy(drafter, ctx, block_size=3, pad_token_id=tok.eos_token_id)

print("draft ids:", block[0].tolist())# Load verifier
print("draft decoded:", tok.decode(block[0], skip_special_tokens=True))
```

    draft ids: [262, 995, 318]
    draft decoded:  the world is



```python
#verifier= the same model as drafter
```


```python
# Load verifier (debug mode: same model as drafter)
from transformers import AutoModelForCausalLM

verifier = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cpu")
print("verifier ready ")
```

    verifier ready 



```python
# Generate greedy sequence with verifier
# Verifier: a model that checks whether the block proposed by the drafter is valid.
# Greedy sequence generation: at each step, select the token with the highest probability (argmax) and append it.
```


```python
greedy_seq = []
gctx = ctx.clone() # clone → keep ctx intact, work on a copy

with torch.no_grad():  # no gradient calc (inference mode)
    for _ in range(block.shape[1]):  # repeat = length of draft block
        out = verifier(input_ids=gctx) # run verifier forward pass
        g = out.logits[:, -1, :].argmax(dim=-1, keepdim=True) # greedy step (argmax)
        greedy_seq.append(int(g.item()))  # save token id/ verifier baseline sequence
        gctx = torch.cat([gctx, g], dim=1) # extend context with new token

print("greedy ids:", greedy_seq)
print("greedy decoded:", tok.decode(greedy_seq, skip_special_tokens=True))
```

    greedy ids: [262, 995, 318]
    greedy decoded:  the world is



```python
# Compare draft vs greedy token-by-token from the start and accept accordingly
```


```python
accepted_ids = []
cur = ctx.clone() # cur = current sequence
 
for i in range(block.shape[1]):
    d_id = int(block[0, i].item())
    g_id = greedy_seq[i]
    print(f"[{i}] draft={tok.decode([d_id])!r} vs greedy={tok.decode([g_id])!r} ->", end=" ")

    
    if d_id == g_id:
        #if they match → accept draft token,
        accepted_ids.append(d_id)
        cur = torch.cat([cur, block[:, i:i+1]], dim=1)
        print("ACCEPT")
    else:
        # if not → replace with greedy token and stop
        g = torch.tensor([[g_id]], device=cur.device)
        cur = torch.cat([cur, g], dim=1)
        print("MISMATCH -> take greedy and STOP")
        break

print("accepted count:", len(accepted_ids), "/", block.shape[1])

```

    [0] draft=' the' vs greedy=' the' -> ACCEPT
    [1] draft=' world' vs greedy=' world' -> ACCEPT
    [2] draft=' is' vs greedy=' is' -> ACCEPT
    accepted count: 3 / 3



```python
#Check text after one step
```


```python
print("new text:\n", tok.decode(cur[0], skip_special_tokens=True))
```

    new text:
     In a distant future, the world is



```python
# Define one-step function (pva_step_once)
```


```python
@torch.no_grad()
def pva_step_once(ctx, block_size=3):
    """
    Drafter proposes block_size tokens → 
    Verifier generates greedy predictions of the same length →
    Accept only the matching prefix from the start, 
    at the first mismatch append the greedy token and stop.
    Returns: new_ctx, accepted_count
    """
    # 1) Draft (debug mode: greedy, no sampling)
    block = draft_block_greedy(drafter, ctx, block_size=block_size, pad_token_id=tok.eos_token_id)

    # 2) Verifier greedy sequence
    greedy_seq = []
    gctx = ctx.clone()
    for _ in range(block.shape[1]):
        out = verifier(input_ids=gctx)
        g = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        greedy_seq.append(int(g.item()))
        gctx = torch.cat([gctx, g], dim=1)

    # 3) Compare and accept
    accepted = 0
    cur = ctx.clone()
    for i in range(block.shape[1]):
        d_id = int(block[0, i].item())
        g_id = greedy_seq[i]
        if d_id == g_id:
            cur = torch.cat([cur, block[:, i:i+1]], dim=1)
            accepted += 1
        else:
            g = torch.tensor([[g_id]], device=cur.device)
            cur = torch.cat([cur, g], dim=1)
            break
    return cur, accepted
```


```python
# Run one speculative step with block size 3
new_ctx, accepted = pva_step_once(ctx, block_size=3)

# For visualization: compare each token in the draft block with greedy
block = draft_block_greedy(drafter, ctx, block_size=3, pad_token_id=tok.eos_token_id)

greedy_seq = []
gctx = ctx.clone()
for _ in range(block.shape[1]):
    out = verifier(input_ids=gctx)
    g = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    greedy_seq.append(int(g.item()))
    gctx = torch.cat([gctx, g], dim=1)

# Step-by-step comparison
print("\n--- Draft vs Greedy comparison ---")
for i in range(block.shape[1]):
    d_id = int(block[0, i].item())
    g_id = greedy_seq[i]
    print(f"[{i}] draft={tok.decode([d_id])!r} vs greedy={tok.decode([g_id])!r} -> ", end="")
    if d_id == g_id:
        print("ACCEPT")
    else:
        print("MISMATCH -> STOP")
        break

print(f"\nAccepted {accepted} / {block.shape[1]} tokens")
print("New ctx decoded:", tok.decode(new_ctx[0], skip_special_tokens=True))
```

    
    --- Draft vs Greedy comparison ---
    [0] draft=' the' vs greedy=' the' -> ACCEPT
    [1] draft=' world' vs greedy=' world' -> ACCEPT
    [2] draft=' is' vs greedy=' is' -> ACCEPT
    
    Accepted 3 / 3 tokens
    New ctx decoded: In a distant future, the world is



```python
# Run one more speculative step
```


```python
# Run one more speculative step
cur2, acc2 = pva_step_once(cur, block_size=3)

# Number of draft tokens accepted in this step
print("accepted this step:", acc2)

# Decode the updated context after this step
print(tok.decode(cur2[0], skip_special_tokens=True))
```

    accepted this step: 3
    In a distant future, the world is in a state



```python
# Short loop (run only 5 steps)
```


```python
ctx_loop = ctx.clone()
for step in range(5):
    # Run one speculative decoding step (drafter + verifier)
    ctx_loop, acc = pva_step_once(ctx_loop, block_size=3)

    # Print how many draft tokens were accepted and the current decoded text
    print(f"[step {step+1}] accepted={acc}, text='{tok.decode(ctx_loop[0], skip_special_tokens=True)}'")
```

    [step 1] accepted=3, text='In a distant future, the world is'
    [step 2] accepted=3, text='In a distant future, the world is in a state'
    [step 3] accepted=3, text='In a distant future, the world is in a state of flux.'
    [step 4] accepted=3, text='In a distant future, the world is in a state of flux. The world is'
    [step 5] accepted=3, text='In a distant future, the world is in a state of flux. The world is in a state'



```python
#Sampling version - drafter 
```


```python
@torch.no_grad()
def draft_block_sampled(
    drafter, input_ids, block_size=3,
    top_k=20, top_p=0.9, temperature=0.7,
    pad_token_id=None, repetition_penalty=1.05, no_repeat_ngram_size=3
):
    # Move input to CPU
    input_ids = input_ids.to("cpu")
    # Build attention mask (ignore pad tokens)
    attn = (input_ids != tok.pad_token_id)

    # Drafter generates a block of tokens using sampling
    out = drafter.generate(
        input_ids,
        attention_mask=attn,
        max_new_tokens=block_size,
        do_sample=True,                # enable sampling instead of greedy
        top_k=top_k,                   # restrict to top-k candidates
        top_p=top_p,                   # nucleus sampling (top cumulative probability p)
        temperature=temperature,       # controls randomness (lower = more greedy)
        pad_token_id=pad_token_id,     # pad token handling
        repetition_penalty=repetition_penalty,  # penalize repeating tokens
        no_repeat_ngram_size=no_repeat_ngram_size,  # block repeated n-grams
    )
    # Return only the newly generated tokens
    return out[:, input_ids.shape[1]:]
```


```python
# Example prompt
prompt = "The future of AI is"
ctx = tok(prompt, return_tensors="pt").input_ids

# Greedy draft block (deterministic)
block_greedy = draft_block_greedy(
    drafter, ctx, block_size=5, pad_token_id=tok.eos_token_id
)

# Sampling draft block (nondeterministic)
block_sampled = draft_block_sampled(
    drafter, ctx, block_size=5,
    top_k=20, top_p=0.9, temperature=0.7,
    pad_token_id=tok.eos_token_id
)

# Decode both results
print("Greedy draft :", tok.decode(block_greedy[0], skip_special_tokens=True))
print("Sampled draft:", tok.decode(block_sampled[0], skip_special_tokens=True))
```

    Greedy draft :  not yet clear.
    
    Sampled draft:  still a mystery, but



```python
# Realistic mode: one speculative decoding step (pva_step_real)
```


```python
@torch.no_grad()
#pva_step_real: drafter uses guided drafting + sampling → closer to actual speculative decoding behavior
def pva_step_real(ctx, block_size=3): 
    block = draft_block_guided(
        drafter, verifier, ctx, block_size=block_size,
        guide_topk=20,       
        sample_topk=40, top_p=0.95, temperature=0.7,
        pad_token_id=tok.eos_token_id,
        repetition_penalty=1.05, no_repeat_ngram_size=3
    )

 # Verifier greedy decoding (same as before)
    greedy_seq = []
    gctx = ctx.clone()
    for _ in range(block.shape[1]):
        attn = (gctx != tok.pad_token_id)
        out = verifier(input_ids=gctx, attention_mask=attn)
        g = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        greedy_seq.append(int(g.item()))
        gctx = torch.cat([gctx, g], dim=1)

    
    # Compare draft vs greedy and accept prefix
    accepted = 0
    cur = ctx.clone()
    for i in range(block.shape[1]):
        d_id = int(block[0, i])
        g_id = greedy_seq[i]
        if d_id == g_id:
            cur = torch.cat([cur, block[:, i:i+1]], dim=1)
            accepted += 1
        else:
            g = torch.tensor([[g_id]], device=cur.device)
            cur = torch.cat([cur, g], dim=1)
            break
    return cur, accepted
```


```python
#guided drafting (wrapper around sampling)
```


```python
@torch.no_grad()
def draft_block_guided(
    drafter, verifier, input_ids, block_size=3,
    guide_topk=20, sample_topk=40, top_p=0.95, temperature=0.7,
    pad_token_id=None, repetition_penalty=1.05, no_repeat_ngram_size=3
):
    # Move to CPU & build attention mask
    input_ids = input_ids.to("cpu")
    attn = (input_ids != tok.pad_token_id) if pad_token_id is not None else None

    # Just sample with the drafter (no fancy guiding)
    out = drafter.generate(
        input_ids,
        attention_mask=attn,
        max_new_tokens=block_size,
        do_sample=True,
        top_k=sample_topk,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=pad_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    # Return only the newly generated tokens (the block)
    return out[:, input_ids.shape[1]:]

```


```python
# Example prompt
prompt = "The future of AI is"
ctx = tok(prompt, return_tensors="pt").input_ids

# Run one speculative step (debug mode: greedy drafter)
ctx_once, acc_once = pva_step_once(ctx, block_size=5)

# Run one speculative step (realistic mode: guided + sampling drafter)
ctx_real, acc_real = pva_step_real(ctx, block_size=5)

# Decode outputs
print("=== pva_step_once (greedy drafter, debug mode) ===")
print("Accepted tokens:", acc_once)
print("Decoded text   :", tok.decode(ctx_once[0], skip_special_tokens=True))

print("\n=== pva_step_real (guided + sampling drafter, realistic mode) ===")
print("Accepted tokens:", acc_real)
print("Decoded text   :", tok.decode(ctx_real[0], skip_special_tokens=True))
```

    === pva_step_once (greedy drafter, debug mode) ===
    Accepted tokens: 5
    Decoded text   : The future of AI is not yet clear.
    
    
    === pva_step_real (guided + sampling drafter, realistic mode) ===
    Accepted tokens: 0
    Decoded text   : The future of AI is not



```python
 # Test a single step in realistic mode
ctx_real = ctx.clone()
ctx_real, acc_real = pva_step_real(ctx_real, block_size=3)

# Print how many draft tokens were accepted in this step
print("accepted (real step):", acc_real)

# Decode and print the updated context as text
print(tok.decode(ctx_real[0], skip_special_tokens=True))
```

    accepted (real step): 1
    The future of AI is not yet



```python
# Short loop (realistic mode, run 8 steps)
```


```python
# Short loop: run 8 steps in realistic mode
def run_short_real(ctx0, steps=8, block_size=3):
    ctx = ctx0.clone()
    proposed = accepted = 0
    for i in range(steps):
        ctx, acc = pva_step_real(ctx, block_size=block_size)
        proposed += block_size
        accepted += acc
        print(f"[{i+1}] accepted={acc}, acc_rate_so_far={round(100*accepted/proposed,1)}%")
    print("text:")
    print(tok.decode(ctx[0], skip_special_tokens=True))
    return ctx

_ = run_short_real(ctx, steps=8, block_size=2)

```

    [1] accepted=0, acc_rate_so_far=0.0%
    [2] accepted=0, acc_rate_so_far=0.0%
    [3] accepted=2, acc_rate_so_far=33.3%
    [4] accepted=0, acc_rate_so_far=25.0%
    [5] accepted=2, acc_rate_so_far=40.0%
    [6] accepted=0, acc_rate_so_far=33.3%
    [7] accepted=0, acc_rate_so_far=28.6%
    [8] accepted=0, acc_rate_so_far=25.0%
    text:
    The future of AI is not yet clear.
    
    
    
    
    
    



```python

```
