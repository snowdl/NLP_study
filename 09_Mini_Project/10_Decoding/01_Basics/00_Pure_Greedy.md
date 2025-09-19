```python
"""
Level 0 — Pure Greedy (argmax) Decoding
- No drafting/verifying separation
- No sampling, beams, or tricks
- Generate K tokens by always picking the top-1 (argmax) at each step
"""
```




    '\nLevel 0 — Pure Greedy (argmax) Decoding\n- No drafting/verifying separation\n- No sampling, beams, or tricks\n- Generate K tokens by always picking the top-1 (argmax) at each step\n'




```python
# ================================================================
# 1) Imports
# ================================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
```


```python
# ================================================================
# 2) Tokenizer n model
# ================================================================
tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
```


```python
# ================================================================
# 3) PAD/EOS (GPT-2 family has no PAD by default → set PAD to EOS)
# ================================================================
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
```


```python
# ================================================================
# 4) Prompt → token IDs
# ================================================================
prompt = "In a distant future,"
ctx = tok(prompt, return_tensors="pt").input_ids
```


```python
print(ctx)
```

    tensor([[  818,   257, 12899,  2003,    11]])



```python
# ================================================================
# 5)Generate K tokens with greedy decoding (single generate call)
# ================================================================
K = 20
out = model.generate(
    input_ids=ctx,
    max_new_tokens=K,
    do_sample=False,                 # ← greedy (argmax)
    pad_token_id=tok.pad_token_id,   # safe when PAD == EOS
)
```

    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.



```python
# ---------------------------------------------------------------------
# 5) Decode / print the result
# ---------------------------------------------------------------------
print(tok.decode(out[0], skip_special_tokens=True))
```

    In a distant future, the world is in a state of flux. The world is in a state of flux. The world



```python

```
