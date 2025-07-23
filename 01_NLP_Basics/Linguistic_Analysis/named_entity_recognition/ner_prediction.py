# 1. Install required libraries (run once in Jupyter or terminal)
!pip install numpy==1.24.3
!pip install folium geopy datasets transformers seqeval

# 2. Load the CoNLL-2003 NER dataset
from datasets import load_dataset
dataset = load_dataset("conll2003")
print(dataset)

# 3. Import necessary modules
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import torch

# 4. Load BERT tokenizer and base model to get hidden states (not for NER directly)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model_basic = AutoModel.from_pretrained("bert-base-cased")

# 5. Tokenize a sample sentence and get hidden states from the base model
inputs = tokenizer("My name is Sarah", return_tensors="pt")
outputs_basic = model_basic(**inputs)

# 6. Load token classification model with 9 NER labels (BIO scheme)
model_ner = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)

# 7. Load pre-trained NER model fine-tuned on CoNLL-2003 dataset
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_ner = AutoModelForTokenClassification.from_pretrained(model_name)

# 8. Print the mapping from label IDs to NER tag names
label_list = model_ner.config.id2label
print("NER Label mapping:", label_list)

# 9. Define example sentence and tokenize it
sentence = "Barack Obama visited Microsoft headquarters in Washington."
inputs = tokenizer(sentence, return_tensors="pt")

# 10. Perform model inference without gradient calculation for efficiency
with torch.no_grad():
    outputs = model_ner(**inputs)

# 11. Get predicted label IDs by selecting the highest scoring label per token
predictions = torch.argmax(outputs.logits, dim=2)
print("Predicted label IDs:", predictions)

# 12. Convert predicted label IDs to their corresponding string labels
id2label = model_ner.config.id2label  
predicted_labels = [id2label[idx.item()] for idx in predictions[0]]
print("Predicted labels:", predicted_labels)

# 13. Convert input token IDs back to tokens for readable output
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 14. Print each token with its predicted NER tag
for token, label in zip(tokens, predicted_labels):
    print(f"{token:15} -> {label}")

# 15. Filter out special tokens ([CLS], [SEP]) from tokens and labels
import pandas as pd
filtered = [(tok, lbl) for tok, lbl in zip(tokens, predicted_labels) if tok not in ["[CLS]", "[SEP]"]]
filtered_tokens_labels = filtered
tokens_filtered, predicted_labels_filtered = zip(*filtered)

# 16. Create a DataFrame for tokens and their predicted NER tags
df = pd.DataFrame(filtered_tokens_labels, columns=["Token", "NER_Tag"])

# 17. Count occurrences of each NER tag and print
tag_counts = df["NER_Tag"].value_counts()
print("\nNER Tag Counts:")
print(tag_counts)

# 18. Visualize the distribution of NER tags using a bar plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.barplot(x=tag_counts.index, y=tag_counts.values)
plt.title("NER Tag Distribution")
plt.xlabel("NER Tag")
plt.ylabel("Count")
plt.show()
