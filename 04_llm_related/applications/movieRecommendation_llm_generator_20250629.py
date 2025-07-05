#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# movies.json 파일 읽기
movies = pd.read_json('./12_data/movies.json')
print("movies 데이터 미리보기:")
print(movies.head())
print("movies shape:", movies.shape)

# ratings.json 파일 읽기
ratings = pd.read_json('./12_data/ratings.json')
print("\nratings 데이터 미리보기:")
print(ratings.head())
print("ratings shape:", ratings.shape)


# In[3]:


ratings_small = ratings.sample(n=50, random_state=42)
print("\n샘플링된 ratings_small:")
print(ratings_small.head())
print(ratings_small.shape)


# In[4]:


print(movies.head())
print(ratings.head())


# In[5]:


user_item_matrix = ratings_small.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    aggfunc='first',      # 중복 없으면 mean 대신 first
    fill_value=0          # 결측치 0으로 대체
)
print(user_item_matrix.head())


# In[6]:


#create a user-item rating matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
# pivot table=crete a matrix with userId as rows (index) movieid as columns and rating as values

print(user_item_matrix.head())


# In[7]:


top_movies = ratings['movieId'].value_counts().head(10)
print(movies[movies['movieId'].isin(top_movies.index)])


# In[8]:


#imports the consine_similarity func from sklearn
from sklearn.metrics.pairwise import cosine_similarity


# In[9]:


#user_item_matrix.fillna(0) -> replaces all missing values (NaN) in the user-tim rating matrix with 0
user_item_filled = user_item_matrix.fillna(0)

#computes the consine similarity btw every pair of users based on their rating vectors
user_sim = cosine_similarity(user_item_filled)
print('User similarity matrix shape:', user_sim.shape)


# In[ ]:


#install surprize library 
get_ipython().system('pip install scikit-surprise')
get_ipython().system('conda install -c conda-forge scikit-surprise')


#import required modules
from surprise import Dataset, Reader, SVD


#import the cross-validation func for model evaluation
from surprise.model_selection import cross_validate


# In[ ]:


#install surprize library 
get_ipython().system('pip install scikit-surprise')
get_ipython().system('conda install -c conda-forge scikit-surprise')


#import required modules
from surprise import Dataset, Reader, SVD


#import the cross-validation func for model evaluation
from surprise.model_selection import cross_validate


# In[ ]:





# In[1]:


import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # For progress bar (install with pip install tqdm if needed)

MODEL_NAME = "gpt2"
PROMPT_FILE = "./12_data/movielens_prompt_response.json"

# Load prompt-response data from JSON file
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

results = []
errors = []

# Iterate through all prompts with a progress bar
for item in tqdm(data, desc="Processing prompts"):
    try:
        prompt = item['prompt']
        # Tokenize the prompt and prepare input tensor
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        # Generate model response
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        # Decode the generated response
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Store prompt, reference response, and model response
        results.append({
            "prompt": prompt,
            "reference_response": item['response'],
            "gpt2_response": generated
        })
    except Exception as e:
        print(f"\n[Error] prompt: {item['prompt']}\n{e}")
        errors.append({"prompt": item['prompt'], "error": str(e)})

# Save results to a JSON file
with open("gpt2_generated_responses.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Save errors to a separate file if any occurred
if errors:
    with open("gpt2_generation_errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"\n[Notice] Errors occurred for {len(errors)} prompts, saved to gpt2_generation_errors.json.")

# Print a sample result (optional)
if results:
    print("\n[Sample Result] First item:")
    print("Prompt:", results[0]['prompt'])
    print("Reference response:", results[0]['reference_response'])
    print("GPT-2 generated response:", results[0]['gpt2_response'])
else:
    print("\n[Notice] No results were generated.")



# In[ ]:


#----------------gpt-j-6B---------------------------


# In[ ]:


#!pip install "transformers[torch]>=4.28.1,<5" "torch>=1.13.1,<2" "accelerate>=0.16.0,<1"


# In[ ]:


#model_name = "databricks/dolly-v2-3b"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)


# In[ ]:


# Prepare the prompt for the language model
#prompt = "Recommend a good thriller movie."

# Tokenize the prompt and convert it to PyTorch tensors, with truncation
#inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

# Generate a response from the model with a maximum length of 100 tokens
#outputs = model.generate(**inputs, max_length=100)

# Decode the generated tokens into a readable string and print the result
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))

