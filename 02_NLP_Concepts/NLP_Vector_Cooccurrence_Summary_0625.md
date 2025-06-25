
# Summary: Co-occurrence, Vector Representation, Matrix Construction

## 1. Co-occurrence
- **Definition**: Two words appear together within a context window (typically size `k`).
- **Example**: In "I love music more than any other genre", "love" and "music" co-occur if `k=2`.

## 2. How to Say 'Co-occurrence' in English
- **Co-occurrence relationship**
- "They co-occur" or "They occur together"

## 3. Vector Representation
- Representing words as numerical vectors for computation.
- Methods:
  - One-hot encoding
  - Count Vector / TF-IDF
  - Word Embeddings (e.g., Word2Vec, GloVe)

## 4. Vector Space
- A space where vectors (e.g., word vectors) live.
- Enables comparison using distance, angle, or similarity.

## 5. W/D vs W/W Matrices

| Type  | Word-by-Document (W/D) | Word-by-Word (W/W) |
|-------|------------------------|---------------------|
| **Structure** | Rows: words / Columns: documents | Rows and columns: words |
| **Values**    | Word frequency in documents | Frequency of two words co-occurring |
| **Use**       | Document classification, retrieval | Word similarity, embeddings (e.g., Word2Vec) |

## 6. Matrix Construction

### W/D Matrix
- Count how many times each word appears in each document.

### W/W Matrix (window size = 1)
- Slide a window over the text and count all word pairs within that window.

## 7. Beyond Word Pairs
- Use n-grams (e.g., 3-gram = "I love NLP")
- Higher-dimensional co-occurrence tensors
- Word combinations within a window

---

## Word-by-Word (W/W) Co-occurrence Matrix (window size = 1)

| Word     | I | love | NLP | deep | learning | loves | me |
|----------|---|------|-----|------|----------|--------|----|
| I        | 0 | 2    | 0   | 0    | 0        | 0      | 0  |
| love     | 2 | 0    | 1   | 1    | 0        | 0      | 0  |
| NLP      | 0 | 1    | 0   | 0    | 0        | 0      | 0  |
| deep     | 0 | 1    | 0   | 0    | 2        | 0      | 0  |
| learning | 0 | 0    | 0   | 2    | 0        | 1      | 0  |
| loves    | 0 | 0    | 0   | 0    | 1        | 0      | 1  |
| me       | 0 | 0    | 0   | 0    | 0        | 1      | 0  |

---

## Word-by-Document (W/D) Matrix

| Word     | Doc1 | Doc2 | Doc3 |
|----------|------|------|------|
| I        | 1    | 1    | 0    |
| love     | 1    | 1    | 0    |
| NLP      | 1    | 0    | 0    |
| deep     | 0    | 1    | 1    |
| learning | 0    | 1    | 1    |
| loves    | 0    | 0    | 1    |
| me       | 0    | 0    | 1    |
