# 🧠 Linguistic Analysis

This folder contains basic linguistic analysis tasks using Python and NLP libraries such as **NLTK** and **spaCy**.

## 📂 Folder Structure

Each subfolder contains code files and notebooks for a specific task:

Linguistic_Analysis/
├── pos_tagging/
│   ├── pos_tagging.ipynb           # Jupyter notebook for POS tagging
│   └── pos_tagging.py              # Script version of POS tagging
│
├── dependency_parsing/
│   ├── dependency_parsing.ipynb    # Jupyter notebook for dependency parsing
│   └── dependency_parsing.py       # Script version of dependency parsing
│
├── constituency_parsing/
│   └── constituency_parsing.py     # Script for constituency parsing
│
├── named_entity_recognition/
│   └── named_entity_recognition.py # Script for NER



## ✅ Tasks Covered

- **POS Tagging**  
  Tokenization and Part-of-Speech tagging using NLTK and the Brown corpus.

- **Dependency Parsing**  
  Extracting subject-verb-object structures and syntactic dependencies using spaCy.

- **Constituency Parsing**  
  Parsing phrase structures using constituency trees.

- **Named Entity Recognition (NER)**  
  Identifying named entities such as persons, organizations, and locations.

## 📦 Requirements

Make sure the following libraries are installed:

```bash
pip install nltk spacy
python -m nltk.downloader all
python -m spacy download en_core_web_sm

