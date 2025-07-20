import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

API_KEY = "dc56f112cff9403c84bbf44e74acb04d"

def fetch_news(query="AI", page_size=100):
    # Construct the URL for the NewsAPI request with given query parameters
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"language=en&"
        f"pageSize={page_size}&"
        f"sortBy=publishedAt&"
        f"apiKey={API_KEY}"
    )
    # Send a GET request to the NewsAPI
    response = requests.get(url)
    # Parse the JSON response
    data = response.json()

    # Extract articles list from the response
    articles = data.get("articles", [])
    docs = []       # List to store combined title and description of each article
    meta_info = []  # List to store source name and published date for each article
    for article in articles:
        # Combine the title and description as the document text
        text = f"{article['title']} {article.get('description') or ''}".strip()
        docs.append(text)
        # Get the source name (e.g., CNN, BBC)
        source = article.get("source", {}).get("name", "Unknown source")
        # Get the publication date
        published = article.get("publishedAt", "Unknown date")
        # Store formatted metadata string
        meta_info.append(f"{source} | {published}")

    # Return the list of documents and their metadata
    return docs, meta_info

def recommend_similar_articles(sim_matrix, article_index, top_n=5):
    # Get similarity scores for the target article against all others
    sim_scores = sim_matrix[article_index]
    # Sort indices by similarity score in descending order
    similar_indices = np.argsort(-sim_scores)
    # Remove the index of the article itself
    similar_indices = similar_indices[similar_indices != article_index]
    # Return the top N most similar article indices
    top_indices = similar_indices[:top_n]
    return top_indices

def main():
    # Fetch news articles and metadata using the query "AI"
    docs, meta_info = fetch_news("AI", 100)
    print(f"Fetched {len(docs)} articles.")

    # Compute TF-IDF vectors for all documents
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_vectors = tfidf.fit_transform(docs)
    # Calculate cosine similarity matrix based on TF-IDF vectors
    tfidf_sim = cosine_similarity(tfidf_vectors)

    # Load pre-trained SBERT model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Compute SBERT embeddings for all documents
    sbert_vectors = sbert_model.encode(docs, show_progress_bar=True)
    # Calculate cosine similarity matrix based on SBERT embeddings
    sbert_sim = cosine_similarity(sbert_vectors)

    target_index = 0  # Select the first article as the target

    # Print the target article text and metadata
    print("\nTarget Article:")
    print(f"- {docs[target_index]}")
    print(f"  ({meta_info[target_index]})")

    # Print recommended articles based on TF-IDF similarity
    print("\nTF-IDF based Recommendations:")
    for idx in recommend_similar_articles(tfidf_sim, target_index):
        print(f"- {docs[idx]}")
        print(f"  ({meta_info[idx]})")

    # Print recommended articles based on SBERT similarity
    print("\nSBERT based Recommendations:")
    for idx in recommend_similar_articles(sbert_sim, target_index):
        print(f"- {docs[idx]}")
        print(f"  ({meta_info[idx]})")

if __name__ == "__main__":
    main()

