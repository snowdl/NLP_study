```python
import pandas as pd
from sklearn.model_selection import train_test_split
```


```python
#data processing
```


```python
train_df = pd.read_csv("../11_data/nlp-getting-started/train.csv")
```


```python
print(train_df.head())
```

       id keyword location                                               text  \
    0   1     NaN      NaN  Our Deeds are the Reason of this #earthquake M...   
    1   4     NaN      NaN             Forest fire near La Ronge Sask. Canada   
    2   5     NaN      NaN  All residents asked to 'shelter in place' are ...   
    3   6     NaN      NaN  13,000 people receive #wildfires evacuation or...   
    4   7     NaN      NaN  Just got sent this photo from Ruby #Alaska as ...   
    
       target  
    0       1  
    1       1  
    2       1  
    3       1  
    4       1  



```python
#preprocessing
```


```python
texts = train_df['text'].tolist()
labels = train_df['target'].tolist()
```


```python
 train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
```


```python
#Tokenizer + dataset
```


```python
from transformers import RobertaTokenizer
import torch
```


```python
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
```


```python
#tokenizer func
```


```python
 # Takes a list of texts as input
def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
```


```python
#dataset
```


```python
class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        """
        Initialize the dataset with texts and their corresponding labels.

        Args:
            texts (List[str]): List of input text strings.
            labels (List[int]): List of integer labels for classification.
        """
        # Tokenize all texts at once using the tokenize() function
        # This returns a dictionary with 'input_ids', 'attention_mask', etc.
        self.encodings = tokenize(texts)
        
        # Store the labels corresponding to each text
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieve the tokenized inputs and label for a given index.

        Args:
            idx (int): Index of the data item to retrieve.

        Returns:
            dict: A dictionary containing:
                - input_ids (torch.Tensor): Token IDs for the input text.
                - attention_mask (torch.Tensor): Attention mask to avoid attending to padding tokens.
                - labels (torch.Tensor): The label tensor for supervised learning.
        """
        # For each key (input_ids, attention_mask, etc.), get the idx-th tensor
        item = {key: val[idx] for key, val in self.encodings.items()}
        
        # Add the label tensor for this index
        item['labels'] = torch.tensor(self.labels[idx])
        
        return item

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of text samples.
        """
        return len(self.labels)

```


```python
# Create the training dataset by wrapping the training texts and labels using the DisasterDataset class
train_dataset = DisasterDataset(train_texts, train_labels)

# Create the validation dataset by wrapping the validation texts and labels using the DisasterDataset class
val_dataset = DisasterDataset(val_texts, val_labels)
```


```python
print(f"Train dataset size: {len(train_dataset)}")
```

    Train dataset size: 6090



```python
print(f"Validation dataset size: {len(val_dataset)}")
```

    Validation dataset size: 1523



```python
sample = train_dataset[0]
```


```python
print(sample)
```

    {'input_ids': tensor([    0, 43309,  1580,  1827,     8,  5322,  1966,     9,   240,     7,
              304, 30169, 14521,    11, 21123,     4,   849,   725,  4712,  1193,
             4261,  3083,  2898,   831,  3179, 13446,     4,  1205,   640,    90,
                4,   876,    73,   846, 13492,  2553,   565,  3320, 11621,     2,
                1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
                1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
                1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
                1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
                1,     1,     1,     1,     1]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'labels': tensor(1)}



```python
input_ids = sample['input_ids']
```


```python
# Decode the input_ids back into a readable text string, skipping special tokens like [CLS], [SEP], etc.
decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

# Print the decoded text for verification or debugging purposes
print("Decoded text:", decoded_text)

```

    Decoded text: Courageous and honest analysis of need to use Atomic Bomb in 1945. #Hiroshima70 Japanese military refused surrender. https://t.co/VhmtyTptGR



```python
print("Label:", sample['labels'].item())
```

    Label: 1



```python
#roberta
# Reason for choosing RoBERTa
#- Removing the Next Sentence Prediction (NSP) task for better language understanding.
#- Using dynamic masking during pre-training.
#- Achieving state-of-the-art performance on many NLP benchmarks.
```


```python
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
```


```python
# Load the pretrained 'roberta-base' model configured for sequence classification tasks
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
```

    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
import transformers
print(transformers.__version__)
```

    4.55.0



```python
training_args = TrainingArguments(
    output_dir='./results',                 # Directory to save training outputs
    num_train_epochs=1,                     # Number of training epochs (1 epoch here due to dataset size)
    per_device_train_batch_size=16,        # Batch size for training per device (GPU/CPU)
    per_device_eval_batch_size=32,         # Batch size for evaluation per device
    logging_dir='./logs',                   # Directory to save training logs
    logging_steps=50,                       # Log training info every 50 steps
    save_steps=200,                        # Save model checkpoint every 200 steps
    save_total_limit=2,                     # Limit total saved checkpoints to 2
)
```


```python
#Trainer
```


```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

```


```python
trainer.train()
```

    /Users/jessicahong/.pyenv/versions/3.11.11/lib/python3.11/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
      warnings.warn(warn_msg)




    <div>

      <progress value='381' max='381' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [381/381 01:53, Epoch 1/1]
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
      <td>50</td>
      <td>0.593800</td>
    </tr>
    <tr>
      <td>100</td>
      <td>0.520500</td>
    </tr>
    <tr>
      <td>150</td>
      <td>0.489400</td>
    </tr>
    <tr>
      <td>200</td>
      <td>0.457200</td>
    </tr>
    <tr>
      <td>250</td>
      <td>0.437100</td>
    </tr>
    <tr>
      <td>300</td>
      <td>0.444300</td>
    </tr>
    <tr>
      <td>350</td>
      <td>0.433000</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=381, training_loss=0.4762486184988748, metrics={'train_runtime': 119.3863, 'train_samples_per_second': 51.011, 'train_steps_per_second': 3.191, 'total_flos': 266014526967000.0, 'train_loss': 0.4762486184988748, 'epoch': 1.0})




```python
# Evaluate the trained model on the evaluation dataset using the Trainer API
eval_results = trainer.evaluate()

# Print the evaluation metrics such as loss and accuracy
print("Evaluation results:", eval_results)

```

    /Users/jessicahong/.pyenv/versions/3.11.11/lib/python3.11/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
      warnings.warn(warn_msg)




<div>

  <progress value='48' max='48' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [48/48 00:06]
</div>



    Evaluation results: {'eval_loss': 0.39824098348617554, 'eval_runtime': 6.9016, 'eval_samples_per_second': 220.673, 'eval_steps_per_second': 6.955, 'epoch': 1.0}



```python
#save model->roberta-base model training and saving
```


```python
# Fine-tune the model on your specific downstream task (e.g., sentiment analysis)

# Save the fine-tuned model weights and configuration to the specified directory
model.save_pretrained('./checkpoints/roberta-base')

# Save the tokenizer configuration and vocabulary files to the same directory
tokenizer.save_pretrained('./checkpoints/roberta-base')

```




    ('./checkpoints/roberta-base/tokenizer_config.json',
     './checkpoints/roberta-base/special_tokens_map.json',
     './checkpoints/roberta-base/vocab.json',
     './checkpoints/roberta-base/merges.txt',
     './checkpoints/roberta-base/added_tokens.json')




```python
#RoBERTa-large Fine-tuning
```


```python
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
```


```python
# Use MPS (Apple Silicon GPU) if available; otherwise, use CPU
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
```


```python
# Load the fine-tuned model and tokenizer from saved checkpoint
model = RobertaForSequenceClassification.from_pretrained('./checkpoints/roberta-base').to(device)
tokenizer = RobertaTokenizer.from_pretrained('./checkpoints/roberta-base')

```


```python
model.eval()  # Set model to evaluation mode
```




    RobertaForSequenceClassification(
      (roberta): RobertaModel(
        (embeddings): RobertaEmbeddings(
          (word_embeddings): Embedding(50265, 768, padding_idx=1)
          (position_embeddings): Embedding(514, 768, padding_idx=1)
          (token_type_embeddings): Embedding(1, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): RobertaEncoder(
          (layer): ModuleList(
            (0-11): 12 x RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSdpaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (classifier): RobertaClassificationHead(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (out_proj): Linear(in_features=768, out_features=2, bias=True)
      )
    )




```python
def predict(texts):
    # Tokenize input texts and convert to PyTorch tensors
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Move input tensors to the target device (MPS or CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)  # Forward pass through the model
        
        probs = torch.softmax(outputs.logits, dim=1)  # Apply softmax to get class probabilities
        
        preds = torch.argmax(probs, dim=1)  # Get the predicted class with highest probability
    
    return preds.cpu().numpy()  # Move predictions to CPU and convert to NumPy array

```


```python
# Example test sentences
test_texts = ["There's a fire in the city.", "Lovely sunny day today!"]
print(predict(test_texts))
```

    [1 0]



```python
def predict_with_probs(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()

label_names = ['Non-disaster', 'Disaster']
texts = ["There's a fire.", "It's a nice day."]

preds, probs = predict_with_probs(texts)
for text, pred, prob in zip(texts, preds, probs):
    print(f"Text: {text}")
    print(f"Predicted label: {label_names[pred]}, Probability: {prob[pred]:.4f}")
    print()
```

    Text: There's a fire.
    Predicted label: Disaster, Probability: 0.7732
    
    Text: It's a nice day.
    Predicted label: Non-disaster, Probability: 0.7919
    



```python
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
```


```python
#roberta-large model training and saving
model_large = RobertaForSequenceClassification.from_pretrained('roberta-large')
```

    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-large and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
# TrainingArguments 
training_args_large = TrainingArguments(
    output_dir='./results-large',
    num_train_epochs=1,
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=16,
    logging_dir='./logs-large',
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
)
```


```python
trainer_large = Trainer(
    model=model_large,
    args=training_args_large,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
```


```python
trainer_large.train()
```



    <div>

      <progress value='762' max='762' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [762/762 20:32, Epoch 1/1]
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
      <td>50</td>
      <td>0.692900</td>
    </tr>
    <tr>
      <td>100</td>
      <td>0.731800</td>
    </tr>
    <tr>
      <td>150</td>
      <td>0.704500</td>
    </tr>
    <tr>
      <td>200</td>
      <td>0.707500</td>
    </tr>
    <tr>
      <td>250</td>
      <td>0.699100</td>
    </tr>
    <tr>
      <td>300</td>
      <td>0.708000</td>
    </tr>
    <tr>
      <td>350</td>
      <td>0.682200</td>
    </tr>
    <tr>
      <td>400</td>
      <td>0.672500</td>
    </tr>
    <tr>
      <td>450</td>
      <td>0.686200</td>
    </tr>
    <tr>
      <td>500</td>
      <td>0.652500</td>
    </tr>
    <tr>
      <td>550</td>
      <td>0.658100</td>
    </tr>
    <tr>
      <td>600</td>
      <td>0.685400</td>
    </tr>
    <tr>
      <td>650</td>
      <td>0.672000</td>
    </tr>
    <tr>
      <td>700</td>
      <td>0.663500</td>
    </tr>
    <tr>
      <td>750</td>
      <td>0.654900</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=762, training_loss=0.6843721046848247, metrics={'train_runtime': 1238.506, 'train_samples_per_second': 4.917, 'train_steps_per_second': 0.615, 'total_flos': 942215371536600.0, 'train_loss': 0.6843721046848247, 'epoch': 1.0})




```python
#save the model
```


```python
model_large.save_pretrained('./checkpoints/roberta-large')
tokenizer.save_pretrained('./checkpoints/roberta-large')
```




    ('./checkpoints/roberta-large/tokenizer_config.json',
     './checkpoints/roberta-large/special_tokens_map.json',
     './checkpoints/roberta-large/vocab.json',
     './checkpoints/roberta-large/merges.txt',
     './checkpoints/roberta-large/added_tokens.json')




```python
#ensemble
```


```python
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import numpy as np
```


```python

```


```python

```


```python
# Set device
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

```


```python
# Load base model & tokenizer
tokenizer_base = RobertaTokenizer.from_pretrained('./checkpoints/roberta-base')
model_base = RobertaForSequenceClassification.from_pretrained('./checkpoints/roberta-base').to(device)
model_base.eval()
```




    RobertaForSequenceClassification(
      (roberta): RobertaModel(
        (embeddings): RobertaEmbeddings(
          (word_embeddings): Embedding(50265, 768, padding_idx=1)
          (position_embeddings): Embedding(514, 768, padding_idx=1)
          (token_type_embeddings): Embedding(1, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): RobertaEncoder(
          (layer): ModuleList(
            (0-11): 12 x RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSdpaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (classifier): RobertaClassificationHead(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (out_proj): Linear(in_features=768, out_features=2, bias=True)
      )
    )




```python

```


```python
# Load large model & tokenizer
tokenizer_large = RobertaTokenizer.from_pretrained('roberta-large')  # or from checkpoint if saved
model_large = RobertaForSequenceClassification.from_pretrained('./checkpoints/roberta-large').to(device)
model_large.eval()

```


    tokenizer_config.json:   0%|          | 0.00/25.0 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]





    RobertaForSequenceClassification(
      (roberta): RobertaModel(
        (embeddings): RobertaEmbeddings(
          (word_embeddings): Embedding(50265, 1024, padding_idx=1)
          (position_embeddings): Embedding(514, 1024, padding_idx=1)
          (token_type_embeddings): Embedding(1, 1024)
          (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): RobertaEncoder(
          (layer): ModuleList(
            (0-23): 24 x RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSdpaSelfAttention(
                  (query): Linear(in_features=1024, out_features=1024, bias=True)
                  (key): Linear(in_features=1024, out_features=1024, bias=True)
                  (value): Linear(in_features=1024, out_features=1024, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=1024, out_features=1024, bias=True)
                  (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=1024, out_features=4096, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=4096, out_features=1024, bias=True)
                (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (classifier): RobertaClassificationHead(
        (dense): Linear(in_features=1024, out_features=1024, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (out_proj): Linear(in_features=1024, out_features=2, bias=True)
      )
    )




```python

```


```python

```


```python

```


```python
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import onnx
import onnxruntime
import numpy as np
```


```python
def predict_with_probs(texts):
    # Tokenize input texts and convert to PyTorch tensors on the target device
    inputs = tokenizer_base(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Forward pass through the roberta-base model
        outputs = model_base(**inputs)
        
        # Compute softmax probabilities over the output logits
        probs = torch.softmax(outputs.logits, dim=1)
        
        # Get predicted class indices with highest probability
        preds = torch.argmax(probs, dim=1)
    
    # Return predictions and probabilities as NumPy arrays on CPU
    return preds.cpu().numpy(), probs.cpu().numpy()
```


```python
# 1. 디바이스 설정 (MPS 우선, 안되면 CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")
```

    Using device: mps



```python
# Large model predict function
def predict_with_probs_large(texts):
    inputs = tokenizer_large(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_large(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()
```


```python

```


```python
# Ensemble prediction by averaging probabilities
def ensemble_predict(texts):
    preds_base, probs_base = predict_with_probs(texts)
    preds_large, probs_large = predict_with_probs_large(texts)
    avg_probs = (probs_base + probs_large) / 2
    ensemble_preds = np.argmax(avg_probs, axis=1)
    return ensemble_preds, avg_probs

# Test sentences
test_texts = ["There's a fire in the city.", "Lovely sunny day today!"]

# Run ensemble
ensemble_preds, ensemble_probs = ensemble_predict(test_texts)

print("Ensemble Predictions:", ensemble_preds)
print("Ensemble Probabilities:", ensemble_probs)
```

    Ensemble Predictions: [1 0]
    Ensemble Probabilities: [[0.39559686 0.60440314]
     [0.6921458  0.30785418]]



```python
save_dir = "saved_model"  # . 없이

# Hugging Face에서 모델 다운로드 (예: roberta-base)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")

# 저장
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
```

    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
#inference
```


```python

```


```python
import torch
import onnxruntime as ort
import numpy as np
from scipy.special import softmax

texts = ["I love this movie!", "This is terrible."]

```


```python

```


```python
# 1. Tokenize input texts and convert to tensors on CPU
inputs_cpu = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to('cpu')
```


```python
#Export the model to ONNX format (opset_version 14 or higher recommended)
torch.onnx.export(
    model_cpu,  # PyTorch model to export
    (inputs_cpu['input_ids'], inputs_cpu['attention_mask']),  # Model inputs as a tuple
    "roberta_model.onnx",  # Output ONNX file name
    input_names=['input_ids', 'attention_mask'],  # ONNX input tensor names
    output_names=['logits'],  # ONNX output tensor name
    dynamic_axes={  # Allow dynamic batch size and sequence length
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    },
    opset_version=14  # ONNX opset version supporting scaled_dot_product_attention
)

```


```python

```


```python
# Create an ONNX Runtime inference session
ort_session = ort.InferenceSession("roberta_model.onnx")
```


```python
# Convert input tensors to numpy arrays
input_ids = inputs_cpu['input_ids'].numpy()
attention_mask = inputs_cpu['attention_mask'].numpy()
```


```python
#Prepare the input dictionary for ONNX Runtime
ort_inputs = {
    'input_ids': input_ids,
    'attention_mask': attention_mask
}
```


```python
# Run inference with ONNX Runtime
ort_outs = ort_session.run(None, ort_inputs)
```


```python
#Check the output logits
logits = ort_outs[0]  # shape: (batch_size, num_classes)
print("Logits:", logits)
```

    Logits: [[ 1.2654392  -0.99142563]
     [ 1.0565486  -0.86827624]]



```python
#Apply softmax function to convert logits to probabilities
probs = softmax(logits, axis=1)
print("Probabilities:", probs)
```

    Probabilities: [[0.905241   0.09475897]
     [0.8726755  0.12732449]]



```python
#Get predicted classes with the highest probability
preds = np.argmax(probs, axis=1)
print("Predicted labels:", preds)
```

    Predicted labels: [0 0]



```python

```
