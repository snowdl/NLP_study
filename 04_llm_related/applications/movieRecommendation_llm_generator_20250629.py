#!/usr/bin/env python
# coding: utf-8

import sys
import os

def set_project_root(levels_up=2):
    """
    Add the project root directory (levels_up above current file) to sys.path
    so Python can find modules/packages there.
    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    project_root = current_path
    for _ in range(levels_up):
        project_root = os.path.dirname(project_root)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Project root set to ( {levels_up} levels up): {project_root}")

set_project_root(levels_up=2)  # Adjust levels_up depending on your folder structure

import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # progress bar
from sklearn.metrics.pairwise import cosine_similarity

# Define base data folder relative to project root
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../12_data')

# Load movies and ratings data
movies_path = os.path.join(DATA_DIR, 'movies.json')
ratings_path = os.path.join(DATA_DIR, 'ratings.json')

movies = pd.read_json(movies_path)
print("movies preview:")
print(movies.head())
print("movies shape:", movies.shape)

ratings = pd.read_json(ratings_path)
print("\nratings preview:")
print(ratings.head())
print("ratings shape:", ratings.shape)

# Sample 50 ratings for quick tests
ratings_small = ratings.sample(n=50, random_state=42)
print("\nSampled ratings_small:")
print(ratings_small.head())
print(ratings_small.shape)

# Create user-item matrix from sampled data (fill NaN with 0)
user_item_matrix_small = ratings_small.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    aggfunc='first',
    fill_value=0
)
print("\nUser-Item matrix (small sample):")
print(user_item_matrix_small.head())

# Create full user-item matrix (may have NaNs)
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
print("\nFull User-Item matrix:")
print(user_item_matrix.head())

# Get top 10 most rated movies
top_movies = ratings['movieId'].value_counts().head(10)
print("\nTop 10 movies info:")
print(movies[movies['movieId'].isin(top_movies.index)])

# Compute user similarity matrix using cosine similarity
user_item_filled = user_item_matrix.fillna(0)
user_sim = cosine_similarity(user_item_filled)
print('User similarity matrix shape:', user_sim.shape)

# ----- GPT-2 text generation on prompt-response data -----

PROMPT_FILE = os.path.join(DATA_DIR, 'movielens_prompt_response.json')
MODEL_NAME = "gpt2"

# Load prompt-response pairs
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

results = []
errors = []

for item in tqdm(data, desc="Processing prompts"):
    try:
        prompt = item['prompt']
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "reference_response": item['response'],
            "gpt2_response": generated
        })
    except Exception as e:
        print(f"\n[Error] prompt: {item['prompt']}\n{e}")
        errors.append({"prompt": item['prompt'], "error": str(e)})

# Save outputs
output_file = os.path.join(os.path.dirname(__file__), 'gpt2_generated_responses.json')
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

if errors:
    error_file = os.path.join(os.path.dirname(__file__), 'gpt2_generation_errors.json')
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"\n[Notice] Errors occurred for {len(errors)} prompts, saved to {error_file}.")

# Print a sample result
if results:
    print("\n[Sample Result] First item:")
    print("Prompt:", results[0]['prompt'])
    print("Reference response:", results[0]['reference_response'])
    print("GPT-2 generated response:", results[0]['gpt2_response'])
else:
    print("\n[Notice] No results were generated.")
