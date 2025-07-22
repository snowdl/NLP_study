# 🧠 Linguistic Analysis

This folder contains basic linguistic analysis tasks using Python and NLP libraries such as **NLTK** and **spaCy**.

## 📂 Folder Structure

Each subfolder contains code files and notebooks for a specific task:
| Folder                                         | Description                           |
| ---------------------------------------------- | ------------------------------------- |
| `Linguistic_Analysis/pos_tagging`              | POS tagging task folder               |
| `Linguistic_Analysis/dependency_parsing`       | Dependency parsing task folder        |
| `Linguistic_Analysis/constituency_parsing`     | Constituency parsing task folder      |
| `Linguistic_Analysis/named_entity_recognition` | Named Entity Recognition (NER) folder |



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

