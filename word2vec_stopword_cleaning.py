# Word2Vec_Stopword_Cleaning_and_Training.py

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Sample corpus: Replace this with your actual dataset
corpus = [
    "Natural language processing is an exciting field.",
    "Deep learning models like Word2Vec are powerful.",
    "NLP is used in chatbots, translators, and search engines.",
    "GloVe and FastText are alternatives to Word2Vec.",
    "Word embeddings capture semantic meaning."
]

# 1. Text preprocessing: lowercase all text and remove punctuation
cleaned_corpus = [re.sub(r"[^\w\s]", "", sentence.lower()) for sentence in corpus]

# 2. Tokenization and stopword removal
#    Tokenize each sentence into words, then filter out common stopwords
stop_words = set(stopwords.words('english'))
tokenized_corpus = [
    [word for word in word_tokenize(sentence) if word not in stop_words]
    for sentence in cleaned_corpus
]

# 3. Train Word2Vec model using the cleaned and tokenized corpus
#    Parameters:
#    - vector_size: Dimensionality of the word vectors (100)
#    - window: Context window size (5 words to left and right)
#    - min_count: Minimum word frequency to consider (1)
#    - workers: Number of CPU cores to use (2)
#    - sg: Training algorithm (1 = skip-gram; 0 = CBOW)
#    - epochs: Number of training iterations (50)
model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=2,
    sg=1,
    epochs=50
)

# 4. Extract word vectors for the top 10 most frequent words in the vocabulary
words = list(model.wv.index_to_key)[:10]
vectors = [model.wv[word] for word in words]

# 5. Use PCA to reduce the 100-dimensional vectors to 2 dimensions for visualization
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# 6. Plot the 2D PCA results with word annotations
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(result[i, 0], result[i, 1])
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title("Word2Vec - PCA Visualization")
plt.show()

# 7. Find and print the top 5 words most similar to the word "embeddings"
print("Top 5 words similar to 'embeddings':")
print(model.wv.most_similar("embeddings", topn=5))

