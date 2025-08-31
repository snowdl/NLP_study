```python
# -----------------------------------------------------------
# 0. Version & Device check
# -----------------------------------------------------------
import torch, transformers, peft

print("transformers:", transformers.__version__)  # should be 4.43.3
print("peft:", peft.__version__)                  # should be 0.13.2

if torch.backends.mps.is_available():
    device = "mps"   # Apple Silicon
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"   # fallback
print("device:", device)
```

    transformers: 4.43.3
    peft: 0.13.2
    device: mps



```python
# -----------------------------------------------------------
# 1. Simple script to install exact versions that work well
#    with Prefix-Tuning.
# -----------------------------------------------------------
import sys, subprocess

py = sys.executable
print("Using python:", py)

def run(cmd):
    """Run a shell command and print it (super simple)."""
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)

# Step 1: uninstall old versions (safe cleanup)
run([py, "-m", "pip", "uninstall", "-y",
     "peft", "transformers", "accelerate"])

# Step 2: install a known-good combo
run([py, "-m", "pip", "install",
     "transformers==4.43.3",
     "peft==0.13.2",
     "accelerate==0.33.0",
     "datasets>=2.19",
     "evaluate>=0.4",
     "scikit-learn",
     "matplotlib",
     "torch"])  # torch will auto-select build (CPU/CUDA/MPS)

print("\n[Done] ✅ Please restart your Python kernel/terminal if in Jupyter.")
```

    Using python: /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/bin/python
    
    $ /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/bin/python -m pip uninstall -y peft transformers accelerate
    Found existing installation: peft 0.13.2
    Uninstalling peft-0.13.2:
      Successfully uninstalled peft-0.13.2
    Found existing installation: transformers 4.43.3
    Uninstalling transformers-4.43.3:
      Successfully uninstalled transformers-4.43.3
    Found existing installation: accelerate 0.33.0
    Uninstalling accelerate-0.33.0:
      Successfully uninstalled accelerate-0.33.0
    
    $ /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/bin/python -m pip install transformers==4.43.3 peft==0.13.2 accelerate==0.33.0 datasets>=2.19 evaluate>=0.4 scikit-learn matplotlib torch
    Collecting transformers==4.43.3
      Using cached transformers-4.43.3-py3-none-any.whl.metadata (43 kB)
    Collecting peft==0.13.2
      Using cached peft-0.13.2-py3-none-any.whl.metadata (13 kB)
    Collecting accelerate==0.33.0
      Using cached accelerate-0.33.0-py3-none-any.whl.metadata (18 kB)
    Requirement already satisfied: datasets>=2.19 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (4.0.0)
    Requirement already satisfied: evaluate>=0.4 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (0.4.5)
    Requirement already satisfied: scikit-learn in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (1.7.1)
    Requirement already satisfied: matplotlib in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (3.10.6)
    Requirement already satisfied: torch in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (2.8.0)
    Requirement already satisfied: filelock in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (3.19.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (0.34.4)
    Requirement already satisfied: numpy>=1.17 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (1.23.5)
    Requirement already satisfied: packaging>=20.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (25.0)
    Requirement already satisfied: pyyaml>=5.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (6.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (2022.10.31)
    Requirement already satisfied: requests in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (2.32.5)
    Requirement already satisfied: safetensors>=0.4.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (0.4.5)
    Requirement already satisfied: tokenizers<0.20,>=0.19 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (0.19.1)
    Requirement already satisfied: tqdm>=4.27 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from transformers==4.43.3) (4.67.1)
    Requirement already satisfied: psutil in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from peft==0.13.2) (7.0.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers==4.43.3) (2025.3.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers==4.43.3) (4.14.1)
    Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers==4.43.3) (1.1.8)
    Requirement already satisfied: pyarrow>=15.0.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from datasets>=2.19) (21.0.0)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from datasets>=2.19) (0.3.8)
    Requirement already satisfied: pandas in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from datasets>=2.19) (2.3.1)
    Requirement already satisfied: xxhash in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from datasets>=2.19) (3.5.0)
    Requirement already satisfied: multiprocess<0.70.17 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from datasets>=2.19) (0.70.16)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from fsspec[http]<=2025.3.0,>=2023.1.0->datasets>=2.19) (3.9.5)
    Requirement already satisfied: scipy>=1.8.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from scikit-learn) (1.10.1)
    Requirement already satisfied: joblib>=1.2.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from scikit-learn) (1.5.1)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: contourpy>=1.0.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from matplotlib) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from matplotlib) (4.59.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from matplotlib) (1.4.9)
    Requirement already satisfied: pillow>=8 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from matplotlib) (11.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from matplotlib) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: sympy>=1.13.3 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from torch) (1.14.0)
    Requirement already satisfied: networkx in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from torch) (2.6.3)
    Requirement already satisfied: jinja2 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from torch) (3.1.6)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets>=2.19) (1.4.0)
    Requirement already satisfied: attrs>=17.3.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets>=2.19) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets>=2.19) (1.7.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets>=2.19) (5.2.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets>=2.19) (1.20.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets>=2.19) (4.0.3)
    Requirement already satisfied: idna>=2.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from yarl<2.0,>=1.0->aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets>=2.19) (3.10)
    Requirement already satisfied: propcache>=0.2.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from yarl<2.0,>=1.0->aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets>=2.19) (0.3.2)
    Requirement already satisfied: six>=1.5 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
    Requirement already satisfied: charset_normalizer<4,>=2 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from requests->transformers==4.43.3) (3.4.3)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from requests->transformers==4.43.3) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from requests->transformers==4.43.3) (2025.8.3)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from sympy>=1.13.3->torch) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from jinja2->torch) (3.0.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from pandas->datasets>=2.19) (2022.7.1)
    Requirement already satisfied: tzdata>=2022.7 in /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages (from pandas->datasets>=2.19) (2025.2)
    Using cached transformers-4.43.3-py3-none-any.whl (9.4 MB)
    Using cached peft-0.13.2-py3-none-any.whl (320 kB)
    Using cached accelerate-0.33.0-py3-none-any.whl (315 kB)
    Installing collected packages: accelerate, transformers, peft
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3/3[0m [peft][32m1/3[0m [transformers]
    [1A[2KSuccessfully installed accelerate-0.33.0 peft-0.13.2 transformers-4.43.3
    
    [Done] ✅ Please restart your Python kernel/terminal if in Jupyter.



```python
import sys, importlib, transformers, peft, torch
print("python exe:", sys.executable)
print("transformers:", transformers.__version__, "| file:", importlib.import_module("transformers").__file__)
print("peft:", peft.__version__, "| file:", importlib.import_module("peft").__file__)
print("device:", "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
```

    python exe: /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/bin/python
    transformers: 4.43.3 | file: /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/transformers/__init__.py
    peft: 0.13.2 | file: /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/peft/__init__.py
    device: mps



```python
# ===== Imports =====
import torch, transformers, peft, random, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ===== Seed & Device =====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


```


```python
#Dataset
```


```python
# ===== Dataset & Model =====
MODEL = "t5-small"  
```


```python
# 1) Dataset
ds = load_dataset("PolyAI/banking77")
label_names = ds["train"].features["label"].names
print("labels:", len(label_names), "| train/test:", len(ds["train"]), "/", len(ds["test"]))

```

    Using the latest cached version of the dataset since PolyAI/banking77 couldn't be found on the Hugging Face Hub
    Found the latest cached dataset configuration 'default' at /Users/jessicahong/.cache/huggingface/datasets/PolyAI___banking77/default/1.1.0/17ffc2ed47c2ed928bee64127ff1dbc97204cb974c2f980becae7c864007aed9 (last modified on Sat Aug 30 22:48:44 2025).


    labels: 77 | train/test: 10003 / 3080



```python
# 2) Tokenizer / Model
tok  = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL)  # fp32 권장 (mps)
base.to(device)
```




    T5ForConditionalGeneration(
      (shared): Embedding(32128, 512)
      (encoder): T5Stack(
        (embed_tokens): Embedding(32128, 512)
        (block): ModuleList(
          (0): T5Block(
            (layer): ModuleList(
              (0): T5LayerSelfAttention(
                (SelfAttention): T5Attention(
                  (q): Linear(in_features=512, out_features=512, bias=False)
                  (k): Linear(in_features=512, out_features=512, bias=False)
                  (v): Linear(in_features=512, out_features=512, bias=False)
                  (o): Linear(in_features=512, out_features=512, bias=False)
                  (relative_attention_bias): Embedding(32, 8)
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (1): T5LayerFF(
                (DenseReluDense): T5DenseActDense(
                  (wi): Linear(in_features=512, out_features=2048, bias=False)
                  (wo): Linear(in_features=2048, out_features=512, bias=False)
                  (dropout): Dropout(p=0.1, inplace=False)
                  (act): ReLU()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (1-5): 5 x T5Block(
            (layer): ModuleList(
              (0): T5LayerSelfAttention(
                (SelfAttention): T5Attention(
                  (q): Linear(in_features=512, out_features=512, bias=False)
                  (k): Linear(in_features=512, out_features=512, bias=False)
                  (v): Linear(in_features=512, out_features=512, bias=False)
                  (o): Linear(in_features=512, out_features=512, bias=False)
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (1): T5LayerFF(
                (DenseReluDense): T5DenseActDense(
                  (wi): Linear(in_features=512, out_features=2048, bias=False)
                  (wo): Linear(in_features=2048, out_features=512, bias=False)
                  (dropout): Dropout(p=0.1, inplace=False)
                  (act): ReLU()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (final_layer_norm): T5LayerNorm()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (decoder): T5Stack(
        (embed_tokens): Embedding(32128, 512)
        (block): ModuleList(
          (0): T5Block(
            (layer): ModuleList(
              (0): T5LayerSelfAttention(
                (SelfAttention): T5Attention(
                  (q): Linear(in_features=512, out_features=512, bias=False)
                  (k): Linear(in_features=512, out_features=512, bias=False)
                  (v): Linear(in_features=512, out_features=512, bias=False)
                  (o): Linear(in_features=512, out_features=512, bias=False)
                  (relative_attention_bias): Embedding(32, 8)
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (1): T5LayerCrossAttention(
                (EncDecAttention): T5Attention(
                  (q): Linear(in_features=512, out_features=512, bias=False)
                  (k): Linear(in_features=512, out_features=512, bias=False)
                  (v): Linear(in_features=512, out_features=512, bias=False)
                  (o): Linear(in_features=512, out_features=512, bias=False)
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (2): T5LayerFF(
                (DenseReluDense): T5DenseActDense(
                  (wi): Linear(in_features=512, out_features=2048, bias=False)
                  (wo): Linear(in_features=2048, out_features=512, bias=False)
                  (dropout): Dropout(p=0.1, inplace=False)
                  (act): ReLU()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (1-5): 5 x T5Block(
            (layer): ModuleList(
              (0): T5LayerSelfAttention(
                (SelfAttention): T5Attention(
                  (q): Linear(in_features=512, out_features=512, bias=False)
                  (k): Linear(in_features=512, out_features=512, bias=False)
                  (v): Linear(in_features=512, out_features=512, bias=False)
                  (o): Linear(in_features=512, out_features=512, bias=False)
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (1): T5LayerCrossAttention(
                (EncDecAttention): T5Attention(
                  (q): Linear(in_features=512, out_features=512, bias=False)
                  (k): Linear(in_features=512, out_features=512, bias=False)
                  (v): Linear(in_features=512, out_features=512, bias=False)
                  (o): Linear(in_features=512, out_features=512, bias=False)
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (2): T5LayerFF(
                (DenseReluDense): T5DenseActDense(
                  (wi): Linear(in_features=512, out_features=2048, bias=False)
                  (wo): Linear(in_features=2048, out_features=512, bias=False)
                  (dropout): Dropout(p=0.1, inplace=False)
                  (act): ReLU()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (final_layer_norm): T5LayerNorm()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (lm_head): Linear(in_features=512, out_features=32128, bias=False)
    )




```python
# ===== Cell 3 — Prefix-Tuning wrapping =====
from peft import PrefixTuningConfig, get_peft_model, TaskType

peft_cfg = PrefixTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # T5 models are Seq2Seq
    num_virtual_tokens=16,            # recommended to start with 8–16
)

model = get_peft_model(base, peft_cfg)
```


```python
# Check which parameters are trainable
model.print_trainable_parameters()

# Move the model to device (MPS / CUDA / CPU)
model.to(device)
```

    trainable params: 98,304 || all params: 60,604,928 || trainable%: 0.1622





    PeftModelForSeq2SeqLM(
      (base_model): T5ForConditionalGeneration(
        (shared): Embedding(32128, 512)
        (encoder): T5Stack(
          (embed_tokens): Embedding(32128, 512)
          (block): ModuleList(
            (0): T5Block(
              (layer): ModuleList(
                (0): T5LayerSelfAttention(
                  (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    (relative_attention_bias): Embedding(32, 8)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                  (DenseReluDense): T5DenseActDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    (act): ReLU()
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
            (1-5): 5 x T5Block(
              (layer): ModuleList(
                (0): T5LayerSelfAttention(
                  (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                  (DenseReluDense): T5DenseActDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    (act): ReLU()
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
          )
          (final_layer_norm): T5LayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (decoder): T5Stack(
          (embed_tokens): Embedding(32128, 512)
          (block): ModuleList(
            (0): T5Block(
              (layer): ModuleList(
                (0): T5LayerSelfAttention(
                  (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    (relative_attention_bias): Embedding(32, 8)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                  (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                  (DenseReluDense): T5DenseActDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    (act): ReLU()
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
            (1-5): 5 x T5Block(
              (layer): ModuleList(
                (0): T5LayerSelfAttention(
                  (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                  (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                  (DenseReluDense): T5DenseActDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    (act): ReLU()
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
          )
          (final_layer_norm): T5LayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (lm_head): Linear(in_features=512, out_features=32128, bias=False)
      )
      (prompt_encoder): ModuleDict(
        (default): PrefixEncoder(
          (embedding): Embedding(16, 6144)
        )
      )
      (word_embeddings): Embedding(32128, 512)
    )




```python
# ===== Cell 4 — Preprocess & tokenize for T5 (Banking77) =====
# This function turns raw Banking77 examples into T5-friendly inputs/targets.
```


```python
# Notes:
#   * T5 is a text-to-text model => we feed a short instruction ("classify intent:")
#     plus the sentence, and ask it to generate the label name as text.
#   * We use `text_target=...` which is the correct way to tokenize targets for
#     seq2seq models in recent Transformers versions.
#   * We set only `truncation=True` here (no padding yet); padding will be handled
#     by a DataCollator at DataLoader time (dynamic per-batch padding).
```


```python
from torch.utils.data import DataLoader

def preprocess(batch):
    # 1) Build the *source* strings with a simple instruction prefix
    inputs  = [f"classify intent: {t}" for t in batch["text"]]

    # 2) Convert numeric labels -> label names (strings), which T5 can generate
    targets = [label_names[i] for i in batch["label"]]

    # 3) Tokenize sources (encoder side)
    enc_in = tok(inputs, truncation=True)

    # 4) Tokenize targets (decoder side)
    #    Using `text_target` ensures special handling of labels for seq2seq.
    lab = tok(text_target=targets, truncation=True)

    # 5) Attach tokenized targets as "labels" (what the model should generate)
    enc_in["labels"] = lab["input_ids"]
    return enc_in
```


```python
# Apply preprocessing to the whole dataset (batched for speed).
# We drop the original "text" and "label" columns because we've converted them.
ds_tok = ds.map(preprocess, batched=True, remove_columns=["text", "label"])

# Make the dataset return PyTorch tensors for each example.
# (Each example can have variable-length tensors; batching will pad later.)
ds_tok.set_format(type="torch")

```


    Map:   0%|          | 0/10003 [00:00<?, ? examples/s]



    Map:   0%|          | 0/3080 [00:00<?, ? examples/s]



```python
#Quick check
print("Tokenized columns:", ds_tok["train"].features)
print("Train/Test sizes:", len(ds_tok["train"]), "/", len(ds_tok["test"]))
```

    Tokenized columns: {'input_ids': List(Value('int32')), 'attention_mask': List(Value('int8')), 'labels': List(Value('int64'))}
    Train/Test sizes: 10003 / 3080



```python
# ===== Cell 5 — DataLoader with padding and label masking =====
```


```python
#   1) Pads "input_ids" and "attention_mask" dynamically per batch.
#   2) Pads "labels" (decoder targets) dynamically as well.
#   3) Replaces all PAD tokens in labels with -100.
#        → -100 is the default ignore index for loss computation in PyTorch,
#          so the model will NOT be penalized for predicting PAD tokens.
```


```python
def collate_fn(features):
    # Separate encoder inputs (input_ids, attention_mask)
    ins = [
        {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
        for f in features
    ]
    # Separate decoder labels
    labs = [{"input_ids": f["labels"]} for f in features]

    # Pad encoder inputs → returns dict with input_ids + attention_mask
    batch = tok.pad(ins, return_tensors="pt")

    # Pad decoder labels
    lab = tok.pad(labs, return_tensors="pt")["input_ids"]

    # Replace PAD token ids with -100 (ignored in loss calculation)
    lab[lab == tok.pad_token_id] = -100
    batch["labels"] = lab

    return batch
```


```python
# ===== Quick sanity check for collate_fn =====
sample_batch = [ds_tok["train"][i] for i in range(3)]  # take 3 examples
collated = collate_fn(sample_batch)

print("Keys in batch:", collated.keys())
print("input_ids shape:", collated["input_ids"].shape)
print("attention_mask shape:", collated["attention_mask"].shape)
print("labels shape:", collated["labels"].shape)

# Check that PAD tokens in labels are replaced by -100
print("labels (first row):", collated["labels"][0][:20])

```

    You're using a T5TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.


    Keys in batch: dict_keys(['input_ids', 'attention_mask', 'labels'])
    input_ids shape: torch.Size([3, 23])
    attention_mask shape: torch.Size([3, 23])
    labels shape: torch.Size([3, 5])
    labels (first row): tensor([  895,   834,   291, 25295,     1])



```python
# -----------------------------------------------------------------------------
# * batch_size = 16 (small and safe for most setups)
# * train_dl uses shuffle=True for stochasticity
# * test_dl uses shuffle=False for deterministic evaluation
# -----------------------------------------------------------------------------
train_dl = DataLoader(
    ds_tok["train"], batch_size=16, shuffle=True, collate_fn=collate_fn
)
test_dl = DataLoader(
    ds_tok["test"], batch_size=16, shuffle=False, collate_fn=collate_fn
)

# Quick check: how many batches in each split?
print("train batches:", len(train_dl), "| test batches:", len(test_dl))

```

    train batches: 626 | test batches: 193



```python
# ===== Cell 6 — Sanity check (single forward pass with loss) =====
# We already have:
#   - Prefix-wrapped model: `model`
#   - DataLoader: `train_dl`
#   - Selected device in `device` and model moved to it
```


```python
model.train()  # enable training mode (dropout etc.)
batch = next(iter(train_dl))  # take a single batch

# Move tensors to the selected device (MPS / CUDA / CPU)
batch = {k: v.to(device) for k, v in batch.items()}
```


```python
#NOTE:
# - On MPS, autocast is limited; on CPU it’s different; to avoid dtype surprises,
#   we simply DISABLE autocast here for the sanity check.
# - If you're on CUDA and want speed later, enable autocast only for CUDA.
with torch.autocast(device_type=("cpu" if device == "cpu" else device), enabled=False):
    out = model(**batch)  # forward pass computes loss because batch has `labels`

print("sanity loss:", float(out.loss))  # if you see a finite number, you're good!
print("logits shape:", tuple(out.logits.shape))  # optional: (B, T, vocab_size) for T5
```

    sanity loss: 6.557995796203613
    logits shape: (16, 21, 32128)


    /var/folders/6y/xtl4b0cx1cs9zrr9n5y814_h0000gn/T/ipykernel_80632/3054933449.py:8: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
    Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
      print("sanity loss:", float(out.loss))  # if you see a finite number, you're good!



```python
# ===== Cell 7 — Training preparation (optimizer & hyperparameters) =====
import torch
from math import inf
```


```python
# Collect only the parameters that require gradients
# ---------------------------------------------------------------------------
# In Prefix-Tuning, the base model is frozen and only the prefix parameters
# are trainable. So we filter with `requires_grad=True`.
# ---------------------------------------------------------------------------
trainable_params = [p for p in model.parameters() if p.requires_grad]
print("trainable params:", sum(p.numel() for p in trainable_params))

```

    trainable params: 98304



```python
# Optimizer
```


```python
# ---------------------------------------------------------------------------
# AdamW is a common choice for transformer fine-tuning.
# Learning rate here is set to 5e-4 (safe starting point).
# ---------------------------------------------------------------------------
optim = torch.optim.AdamW(trainable_params, lr=5e-4)

```


```python
# Training hyperparameters
```


```python
# ---------------------------------------------------------------------------
epochs = 2       # run 2 epochs first (reduce to 1 for a very quick test)
log_every = 100  # log the average loss every 100 steps
grad_clip = 1.0  # clip gradients at 1.0 to prevent exploding gradients
# ---------------------------------------------------------------------------

# Set model to training mode
# (important for dropout, layer norm, etc. to behave correctly)
model.train()
```




    PeftModelForSeq2SeqLM(
      (base_model): T5ForConditionalGeneration(
        (shared): Embedding(32128, 512)
        (encoder): T5Stack(
          (embed_tokens): Embedding(32128, 512)
          (block): ModuleList(
            (0): T5Block(
              (layer): ModuleList(
                (0): T5LayerSelfAttention(
                  (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    (relative_attention_bias): Embedding(32, 8)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                  (DenseReluDense): T5DenseActDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    (act): ReLU()
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
            (1-5): 5 x T5Block(
              (layer): ModuleList(
                (0): T5LayerSelfAttention(
                  (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerFF(
                  (DenseReluDense): T5DenseActDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    (act): ReLU()
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
          )
          (final_layer_norm): T5LayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (decoder): T5Stack(
          (embed_tokens): Embedding(32128, 512)
          (block): ModuleList(
            (0): T5Block(
              (layer): ModuleList(
                (0): T5LayerSelfAttention(
                  (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                    (relative_attention_bias): Embedding(32, 8)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                  (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                  (DenseReluDense): T5DenseActDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    (act): ReLU()
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
            (1-5): 5 x T5Block(
              (layer): ModuleList(
                (0): T5LayerSelfAttention(
                  (SelfAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (1): T5LayerCrossAttention(
                  (EncDecAttention): T5Attention(
                    (q): Linear(in_features=512, out_features=512, bias=False)
                    (k): Linear(in_features=512, out_features=512, bias=False)
                    (v): Linear(in_features=512, out_features=512, bias=False)
                    (o): Linear(in_features=512, out_features=512, bias=False)
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (2): T5LayerFF(
                  (DenseReluDense): T5DenseActDense(
                    (wi): Linear(in_features=512, out_features=2048, bias=False)
                    (wo): Linear(in_features=2048, out_features=512, bias=False)
                    (dropout): Dropout(p=0.1, inplace=False)
                    (act): ReLU()
                  )
                  (layer_norm): T5LayerNorm()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
          )
          (final_layer_norm): T5LayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (lm_head): Linear(in_features=512, out_features=32128, bias=False)
      )
      (prompt_encoder): ModuleDict(
        (default): PrefixEncoder(
          (embedding): Embedding(16, 6144)
        )
      )
      (word_embeddings): Embedding(32128, 512)
    )




```python
# ===== Cell 8 — Training loop =====
```


```python
# Assumes:
#   - `model`, `device` already set
#   - `train_dl` is a DataLoader yielding batches with input_ids/attention_mask/labels
#   - `optim`, `epochs`, `log_every`, `grad_clip` defined in the previous cellimport math
from statistics import mean

global_step = 0
print("[Train] start")
```

    [Train] start



```python
for ep in range(1, epochs + 1):
    running_losses = []  # keep per-step losses to compute averages
    for step, batch in enumerate(train_dl, start=1):
        # 1) Move tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # 2) Forward pass (we keep it simple: no mixed precision to avoid surprises)
        out = model(**batch)
        loss = out.loss

        # 3) Backward pass
        optim.zero_grad()
        loss.backward()

        # 4) (Optional) Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # 5) Optimizer step
        optim.step()

        # 6) Logging
        running_losses.append(loss.item())
        global_step += 1
        if step % log_every == 0:
            avg = mean(running_losses[-log_every:])  # average over recent window
            print(f"[ep {ep}] step {step}/{len(train_dl)}  loss {avg:.4f}")

    # End of epoch summary
    epoch_avg = mean(running_losses) if running_losses else math.nan
    print(f"[ep {ep}] epoch_avg_loss {epoch_avg:.4f}")

print("[Train] done")
```

    [ep 1] step 100/626  loss 4.4890
    [ep 1] step 200/626  loss 4.3414
    [ep 1] step 300/626  loss 4.1939
    [ep 1] step 400/626  loss 4.0593
    [ep 1] step 500/626  loss 3.9101
    [ep 1] step 600/626  loss 3.8191
    [ep 1] epoch_avg_loss 4.1214
    [ep 2] step 100/626  loss 3.6778
    [ep 2] step 200/626  loss 3.6308
    [ep 2] step 300/626  loss 3.5387
    [ep 2] step 400/626  loss 3.4567
    [ep 2] step 500/626  loss 3.4296
    [ep 2] step 600/626  loss 3.3384
    [ep 2] epoch_avg_loss 3.5064
    [Train] done



```python
from sklearn.metrics import accuracy_score, f1_score

model.eval()
gen_preds, gen_labels = [], []
```


```python
with torch.no_grad():
    for batch in test_dl:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        # generate short label strings
        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=6,          # labels are short
            num_beams=1,               # greedy is fine
        )
        # decode predictions
        gen_texts = tok.batch_decode(gen_out, skip_special_tokens=True)
        gen_preds.extend(gen_texts)

        # prepare gold labels (undo -100 -> pad)
        gold = batch["labels"].clone()
        gold[gold == -100] = tok.pad_token_id
        gold_texts = tok.batch_decode(gold, skip_special_tokens=True)
        gen_labels.extend(gold_texts)
```


```python
acc = accuracy_score(gen_labels, gen_preds)
f1  = f1_score(gen_labels, gen_preds, average="macro")
print(f"[Eval:generate] accuracy={acc:.4f}  macro_F1={f1:.4f}")
```

    [Eval:generate] accuracy=0.0006  macro_F1=0.0001



```python
# ===== Cell 9 (fixed) — Evaluation with proper label decoding =====
from sklearn.metrics import accuracy_score, f1_score

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_dl:
        # move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)  # teacher forcing logits

        # token-level argmax (simple baseline)
        preds = torch.argmax(out.logits, dim=-1)              # [B, T_dec]
        labels = batch["labels"].clone()                      # [B, T_dec]
        labels[labels == -100] = tok.pad_token_id             # 🔧 undo ignore index

        # collect as python lists
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# decode to strings (now all IDs are non-negative)
decoded_preds  = tok.batch_decode(all_preds,  skip_special_tokens=True)
decoded_labels = tok.batch_decode(all_labels, skip_special_tokens=True)

# compute metrics (label strings vs predicted strings)
acc = accuracy_score(decoded_labels, decoded_preds)
f1  = f1_score(decoded_labels, decoded_preds, average="macro")

print(f"[Eval] accuracy={acc:.4f}  macro_F1={f1:.4f}")

# show a few qualitative samples
print("\nSample predictions:")
for i in range(5):
    print(f"Text : {ds['test'][i]['text']}")
    print(f"True : {decoded_labels[i]!r}")
    print(f"Pred : {decoded_preds[i]!r}")
    print("---")

```

    [Eval] accuracy=0.0003  macro_F1=0.0000
    
    Sample predictions:
    Text : How do I locate my card?
    True : 'card_arrival'
    Pred : '_card_'
    ---
    Text : I still have not received my new card, I ordered over a week ago.
    True : 'card_arrival'
    Pred : 'new_card_'
    ---
    Text : I ordered a card but it has not arrived. Help please!
    True : 'card_arrival'
    Pred : 'card_rebitled'
    ---
    Text : Is there a way to know when my card will arrive?
    True : 'card_arrival'
    Pred : 'card_cardbitled'
    ---
    Text : My card has not arrived yet.
    True : 'card_arrival'
    Pred : 'card_card_'
    ---


    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/sklearn/metrics/_classification.py:99: UserWarning: The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.
      type_pred = type_of_target(y_pred, input_name="y_pred")
    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/sklearn/metrics/_classification.py:99: UserWarning: The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.
      type_pred = type_of_target(y_pred, input_name="y_pred")
    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/sklearn/utils/multiclass.py:79: UserWarning: The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.
      ys_types = set(type_of_target(x) for x in ys)
    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/sklearn/metrics/_classification.py:99: UserWarning: The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.
      type_pred = type_of_target(y_pred, input_name="y_pred")
    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/sklearn/utils/multiclass.py:79: UserWarning: The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.
      ys_types = set(type_of_target(x) for x in ys)



```python

```
