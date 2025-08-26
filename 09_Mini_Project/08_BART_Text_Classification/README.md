# 📌 BART-based Multi-Class Text Classification (AG News)

## 📅 Date: 2025-07-13

## ✅ Overview
This mini-project focuses on fine-tuning a pre-trained transformer (BERT) on a multi-class text classification task using the AG News dataset.

## ✅ Key Components

- **Dataset**: AG News (4 classes: World, Sports, Business, Sci/Tech)
- **Preprocessing**: HuggingFace Tokenizer, select subset for fast experimentation
- **Model**: `bert-base-uncased` using `AutoModelForSequenceClassification`
- **Training**: `Trainer` API with 3 epochs
- **Evaluation**:
  - Accuracy, Precision, Recall, F1 (macro & weighted)
  - Confusion Matrix (Seaborn & Plotly)
  - Class-wise metric bar chart (Plotly)
- **Bonus**: Logistic Regression with TF-IDF + GridSearchCV for comparison

## 🔍 Next Steps
- Ensemble model training (BERT + DistilBERT) with voting strategy
- GPT-based generative tasks (summarization, QnA)
- FastAPI model deployment

## 👩‍💻 File
- `model_evaluation_visualization.pt1.ipynb`: Contains full pipeline from preprocessing to evaluation

---

*Contributions, refactoring, or visual enhancements are welcome!*

