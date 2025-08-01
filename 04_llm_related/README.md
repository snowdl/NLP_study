cat > 04_llm_related/README.md << 'EOF'
# 04_llm_related

This folder contains notebooks related to Large Language Models (LLMs).  
It mainly includes notebooks covering fundamental concepts, model generation, and practical applications.

## Key Notebooks Examples
- `movieRecommendation_llm_generator_20250629.ipynb`: A movie recommendation system using GPT‑2 and semantic search
- `vector_similarity_cosine_euclidean.ipynb`: Practice notebook for vector embeddings with cosine similarity and Euclidean distance calculations

### Folder Structure

The `04_llm_related` folder organizes LLM-related notebooks by topic:

- `embeddings/`  
  Notebooks related to vector embeddings and similarity calculations (e.g., cosine similarity, Euclidean distance).  
  Example: `vector_similarity_cosine_euclidean.ipynb`

- `applications/`  
  Notebooks demonstrating practical LLM applications such as recommendation systems and generative models.  
  Example: `movieRecommendation_llm_generator_20250629.ipynb`

---

## Movie Recommendation System Overview

This notebook implements a movie recommendation system using the GPT‑2 model combined with semantic search.

### 🧠 Key Features
- **Model**: GPT‑2 (via Hugging Face Transformers), a lightweight LLM suitable for rapid prototyping
- **Search**: Semantic search to retrieve relevant movie descriptions based on keyword similarity
- **Input**: User preferences such as genre, tone, style, etc.
- **Output**: 3–5 personalized movie recommendations generated by GPT‑2

### 🛠️ Components
- `semantic_search.py` – Embedding generation and similarity-based retrieval
- `prompt_template` – Few-shot examples for structured prompting
- GPT‑2 inference code – Tokenizes input, generates text, and extracts recommendation list

### 💡 Example Use
- Users provide their movie preferences as input  
- The system performs semantic search over movie descriptions to find relevant matches  
- GPT‑2 generates a curated list of movie recommendations tailored to user preferences
