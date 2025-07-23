# Named Entity Recognition (NER) Project

This folder contains code examples and notebooks for Named Entity Recognition (NER) tasks using BERT-based models.

## Files

- `ner_prediction.ipynb`: Jupyter Notebook demonstrating NER prediction on example sentences with visualization.
- `ner_prediction.py`: Python script version of the NER prediction code (not included if deleted).

## Overview

- Uses the `dbmdz/bert-large-cased-finetuned-conll03-english` model fine-tuned on the CoNLL-2003 dataset for token classification.
- Performs tokenization, model inference, and outputs entity tags for tokens.
- Filters special tokens ([CLS], [SEP]) for cleaner output.
- Provides visualization of NER tag distributions using pandas, matplotlib, and seaborn.

## How to Use

1. Install required packages:

   ```bash
   pip install numpy folium geopy datasets transformers seqeval

References
Hugging Face Transformers

CoNLL-2003 Dataset


