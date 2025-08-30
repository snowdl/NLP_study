```python
# ============================================================
# LoRA fine-tuning example using roberta-base on BANKING77
# With seed setting, warmup, and weight decay for stability
# ============================================================
```


```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from peft import LoraConfig, get_peft_model
from evaluate import load as load_metric
import numpy as np
```


```python
# ------------------------------
# Configuration
# ------------------------------
```


```python
set_seed(42)  # fix random seed for reproducibility

MODEL = "roberta-base"   # backbone model
NUM_LABELS = 77          # number of classes (BANKING77 dataset)
EPOCHS = 3               # number of epochs
LR = 2e-4                 # learning rate
BTR, BTE = 16, 32        # batch sizes (train/eval)

```


```python
# ------------------------------
# Load dataset & tokenizer
# ------------------------------
ds = load_dataset("PolyAI/banking77")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
```

    Using the latest cached version of the dataset since PolyAI/banking77 couldn't be found on the Hugging Face Hub
    Found the latest cached dataset configuration 'default' at /Users/jessicahong/.cache/huggingface/datasets/PolyAI___banking77/default/1.1.0/17ffc2ed47c2ed928bee64127ff1dbc97204cb974c2f980becae7c864007aed9 (last modified on Sat Aug 30 16:35:53 2025).



```python
# tokenize function
def tok_fn(batch):
    return tok(batch["text"], truncation=True)
```


```python
# remove original text column after tokenization
ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tokenizer=tok)
```


    Map:   0%|          | 0/10003 [00:00<?, ? examples/s]



    Map:   0%|          | 0/3080 [00:00<?, ? examples/s]



```python
# ------------------------------
# Metrics
# ------------------------------
```


```python
acc = load_metric("accuracy")
f1 = load_metric("f1")
```


```python
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": acc.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "macro_f1": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
    }
```


```python
# ------------------------------
# Model + LoRA
# ------------------------------
base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)
```

    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
# LoRA target modules for RoBERTa (query, key, value, dense)
targets = ["query", "key", "value", "dense"]
```


```python
lora_cfg = LoraConfig(
    r=16,              
    lora_alpha=64,     
    lora_dropout=0.05,
    target_modules=["query","key","value","dense"],
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(base, lora_cfg)
```


```python
# ------------------------------
# TrainingArguments
# ------------------------------
args = TrainingArguments(
    output_dir="./out_lora_roberta",
    learning_rate=LR,
    per_device_train_batch_size=BTR,
    per_device_eval_batch_size=BTE,
    num_train_epochs=EPOCHS,
    report_to="none",       # disable WandB/Hub logging
    warmup_ratio=0.06,      # small warmup for stability
    weight_decay=0.01       # add weight decay for generalization
)

```


```python
# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["test"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics
)
```

    /var/folders/6y/xtl4b0cx1cs9zrr9n5y814_h0000gn/T/ipykernel_63601/1085237875.py:4: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
      trainer = Trainer(



```python
# ------------------------------
# Run training + evaluation
# ------------------------------
trainer.train()
print(trainer.evaluate())
```

    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py:684: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
      warnings.warn(warn_msg)




    <div>

      <progress value='1878' max='1878' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1878/1878 05:22, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500</td>
      <td>1.960900</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>0.422900</td>
    </tr>
    <tr>
      <td>1500</td>
      <td>0.261100</td>
    </tr>
  </tbody>
</table><p>


    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py:684: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
      warnings.warn(warn_msg)
    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py:684: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
      warnings.warn(warn_msg)
    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py:684: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
      warnings.warn(warn_msg)




<div>

  <progress value='97' max='97' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [97/97 00:12]
</div>



    {'eval_loss': 0.25878748297691345, 'eval_accuracy': 0.9292207792207792, 'eval_macro_f1': 0.929054045685224, 'eval_runtime': 12.9636, 'eval_samples_per_second': 237.588, 'eval_steps_per_second': 7.482, 'epoch': 3.0}



```python
# ------------------------------
# save LoRA adapter
# ------------------------------
trainer.model.save_pretrained("./out_lora_roberta_adapter")
```


```python
# ========== Quick Inference Test ==========
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
```


```python
# 1) Tokenizer + label names
tok = AutoTokenizer.from_pretrained("roberta-base")
ds = load_dataset("PolyAI/banking77")
label_names = ds["train"].features["label"].names  # id -> name mapping
```

    Using the latest cached version of the dataset since PolyAI/banking77 couldn't be found on the Hugging Face Hub
    Found the latest cached dataset configuration 'default' at /Users/jessicahong/.cache/huggingface/datasets/PolyAI___banking77/default/1.1.0/17ffc2ed47c2ed928bee64127ff1dbc97204cb974c2f980becae7c864007aed9 (last modified on Sat Aug 30 16:46:43 2025).



```python
# 2) Base + adapter load
base = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_names))
model = PeftModel.from_pretrained(base, "./out_lora_roberta_adapter").eval()
```

    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
# 3) Prediction
text = "I lost my card and need help."
inputs = tok(text, return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
pred_id = int(logits.argmax(-1))
print("pred id:", pred_id, "| label:", label_names[pred_id])
```

    pred id: 41 | label: lost_or_stolen_card



```python

```
