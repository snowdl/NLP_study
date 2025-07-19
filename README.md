
Welcome! This repository is a comprehensive personal study notebook documenting my journey through various Natural Language Processing (NLP) and LLM-related online courses and projects.

The materials come from online lectures (Coursera, Udemy, K-MOOC, etc.) and include hands-on coding, math fundamentals, mini projects, and certificate logs.

---

## :pushpin: Repository Purpose

-  Systematically organize concepts and code from NLP/LLM courses  
-  Practice key techniques from scratch using Python  
-  Build a clean, traceable portfolio for NLP/AI graduate program applications  
-  Serve as a long-term learning and revision hub

---
## :package: Tech Stack & Tools

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python)](https://www.python.org)  
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=Jupyter)](https://jupyter.org)  
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy)](https://numpy.org)  
[![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org)  
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib)](https://matplotlib.org)  
[![Seaborn](https://img.shields.io/badge/Seaborn-76B900?style=flat-square)](https://seaborn.pydata.org)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)  
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat-square)](https://spacy.io)  
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21F?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)  
[![Shell](https://img.shields.io/badge/Shell-FFD500?style=flat-square&logo=gnu-bash)](https://www.gnu.org/software/bash)

---
##  Scripts

| File                          | Purpose                                                       |
| ----------------------------- | ------------------------------------------------------------- |
| `02_organize.sh`              | Batch rename folders/files with consistent numbering          |
| `convert_paths.py`            | Convert all absolute paths to relative ones                   |
| `convert_plotly_notebooks.sh` | Enable consistent project-root plotting                       |
| `sync_notebooks.sh`           | Convert Jupyter notebooks to `.py` files for version tracking |
| `setup_path.py`               | Dynamically manage sys.path for notebook execution            |


---
##  Data Location

All datasets and data files are located in the `12_data/` folder.  
All code and notebooks use **relative paths** (e.g., `./12_data/filename`).  
 **Important:** Please run all code from the project **root directory** (`NLP_study/`).
##  Notes

This repository is intended for **study**, **practice**, and **portfolio building**.  
Some materials are adapted from online courses for **educational purposes only**.  

 Feel free to **fork**, **star**, or **contribute**!  
 If you have questions or suggestions, please **open an issue** or **contact me**.


---

## Folder Structure

<details>
<summary>Click to expand 📁</summary>

```markdown
NLP_study/
├── 01_NLP_Basics/                      # Foundational NLP concepts and preprocessing
│   ├── Lemmatization_Stemming/        # Word normalization techniques
│   ├── pattern_matching_analysis/     # Rule-based pattern matching with spaCy
│   ├── spacy_text_classification/     # Text classification using spaCy pipelines
│   └── vector_semantics/              # Word vector arithmetic and similarity
├── 02_NLP_Concepts/                   # Core theoretical concepts in NLP
├── 03_nlp_architectures/             # NLP model architectures and custom implementations
├── 04_llm_related/                    # Projects and experiments with Large Language Models
│   ├── applications/                  # Real-world LLM applications and agents
│   └── embeddings/                    # Embedding generation and vector analysis
├── 05_Data_Visualization/            # Plotting and visualization tools
│   ├── Plotly/                        # Interactive plots with Plotly
│   └── Seaborn/                       # Statistical visualization with Seaborn
├── 06_Pandas_Numpy/                  # Data manipulation and analysis with Pandas & Numpy
├── 07_sklearn/                       # Machine Learning using Scikit-learn
│   ├── KNN/                           # K-Nearest Neighbors classifier
│   ├── RandomForest_Analysis/        # Random Forest implementation and analysis
│   ├── Text_classification/          # Text classification using various models
│   │   ├── notebooks/                # Jupyter Notebooks for experimentation
│   │   └── scripts/                  # Clean Python scripts
│   ├── Topic_Modeling/               # Topic modeling with NLP techniques
│   └── linear_regression/            # Linear regression model and metrics
├── 08_core_math_concepts/           # Essential math for machine learning and NLP
│   └── Linear_Algebra/               # Linear algebra basics
├── 09_Mini_Project/                 # End-to-end ML & NLP mini projects
│   ├── Breast_Cancer_Binary_Classification/  # Classification project with cancer dataset
│   ├── IMDB_Movie_Review_Sentiment_Analysis/ # Sentiment analysis using IMDB data
│   └── Recommendation_Systems/               # Collaborative filtering & content-based recommenders
├── 10_framework/                    # Deep learning frameworks
│   ├── pytorch/                      # PyTorch-based experiments
│   └── tensorflow_keras/            # TensorFlow/Keras projects
├── 11_certificates/                # Completed course certificates
├── 12_data/                        # Datasets used across the projects
└── 13_NLTK/                        # Experiments using the NLTK library
└- 14_Kaggle/
    ├── Titanic_Survival/
    ├── Ensemble_Methods/
    ├── NLP_Competitions/
    └── Code_Notebooks/
---
## :bulb: Usage

```bash
# Clone the repository
git clone https://github.com/snowdl/NLP_study.git
cd NLP_study

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt


