```python
# 1) Imports
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from peft import LoraConfig, get_peft_model
from evaluate import load as load_metric
import numpy as np
```


```python
# 2) Config
set_seed(42)  # reproducibility

MODEL = "distilbert-base-uncased"
NUM_LABELS = 77              # BANKING77
EPOCHS = 3                  
LR = 2e-4
BTR, BTE = 16, 32            # train/eval batch sizes
```


```python
# 3) Dataset & tokenizer
ds = load_dataset("PolyAI/banking77")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
```

    Using the latest cached version of the dataset since PolyAI/banking77 couldn't be found on the Hugging Face Hub
    Found the latest cached dataset configuration 'default' at /Users/jessicahong/.cache/huggingface/datasets/PolyAI___banking77/default/1.1.0/17ffc2ed47c2ed928bee64127ff1dbc97204cb974c2f980becae7c864007aed9 (last modified on Sat Aug 30 18:07:46 2025).



```python
def tok_fn(batch):
    # truncation=True ensures consistent sequence length
    return tok(batch["text"], truncation=True)
```


```python
# remove_columns=["text"] avoids "too many dimensions 'str'" errors later
ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tokenizer=tok)
```


    Map:   0%|          | 0/3080 [00:00<?, ? examples/s]



```python
# 4) Metrics
acc = load_metric("accuracy")
f1  = load_metric("f1")
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
# 5) Model + LoRA (DistilBERT module names!)
base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)

targets = ["q_lin", "k_lin", "v_lin", "out_lin"]  # DistilBERT attention projections
lora_cfg = LoraConfig(
    r=16,                 
    lora_alpha=64,      
    lora_dropout=0.05,
    target_modules=targets,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(base, lora_cfg)
```

    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
# 6) Training arguments
args = TrainingArguments(
    output_dir="./out_lora_distilbert",
    learning_rate=LR,
    per_device_train_batch_size=BTR,
    per_device_eval_batch_size=BTE,
    num_train_epochs=EPOCHS,
    report_to="none",      # disable external loggers
    warmup_ratio=0.06,     # tiny warmup helps stability
    weight_decay=0.01      # mild regularization
)
```


```python
# 7) Trainer + train/eval
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["test"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics
)

trainer.train()
print(trainer.evaluate())
# lora_distilbert_min.py  (END)

```

    /var/folders/6y/xtl4b0cx1cs9zrr9n5y814_h0000gn/T/ipykernel_63875/4109794826.py:2: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
      trainer = Trainer(
    /Users/jessicahong/.pyenv/versions/3.10.12/envs/nlp_env/lib/python3.10/site-packages/torch/utils/data/dataloader.py:684: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
      warnings.warn(warn_msg)




    <div>

      <progress value='1878' max='1878' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1878/1878 02:17, Epoch 3/3]
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
      <td>2.303800</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>0.633500</td>
    </tr>
    <tr>
      <td>1500</td>
      <td>0.414900</td>
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
  [97/97 00:06]
</div>



    {'eval_loss': 0.37591156363487244, 'eval_accuracy': 0.8925324675324675, 'eval_macro_f1': 0.8924499805083215, 'eval_runtime': 6.886, 'eval_samples_per_second': 447.288, 'eval_steps_per_second': 14.087, 'epoch': 3.0}



```python

```
