#!/usr/bin/env python
# coding: utf-8

# In[13]:


#Find the most relevant document using sentence embeddings → Ask a question using that context → Generate an answer with GPT"


# In[35]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# In[36]:


# Data preparation: list of documents for retrieval
documents = [
    "Transformers are a type of neural network architecture.",
    "BERT stands for Bidirectional Encoder Representations from Transformers.",
    "GPT models are good at generating human-like text.",
    "Retrieval-Augmented Generation combines search and generation.",
]


# In[37]:


# Load the pre-trained sentence embedding model (MiniLM) for converting sentences into dense vector representations
embedder = SentenceTransformer('all-MiniLM-L6-v2')


# In[38]:


# Generate vector embeddings for each document to capture their semantic meaning
doc_embeddings = embedder.encode(documents)


# In[39]:


# Define the user’s query or question to find the most relevant document
query = "What does BERT mean?"


# In[40]:


# Convert the user’s query into a vector embedding for similarity comparison
query_embedding = embedder.encode([query])


# In[41]:


# Compute cosine similarity scores between the query embedding and each document embedding
similarities = cosine_similarity(query_embedding, doc_embeddings)


# In[42]:


# Identify the index of the document most similar to the query
#argmax()  returns the index (position) of the largest value (highest similarity score) in the array.
top_doc_idx = similarities.argmax()
# In other words, compare the query embedding with all document embeddings using cosine similarity
# and return the index of the closest (most similar) document.


# In[43]:


# Initialize the GPT-2 text generation pipeline to generate answers based on the given prompt
# This will generate a text response using GPT-2 based on the context and question provided.
generator = pipeline('text-generation', model='gpt2', max_length=100)


# In[44]:


#RAG:framework that combines information retrieval and text generation.
"""
It first retrieves relevant documents from a large collection based on a query, then uses a language generation model (like GPT) to produce answers grounded in the retrieved information.

"""


# In[45]:


# Select the most relevant document as context based on similarity
context = documents[top_doc_idx]


# In[46]:


# Create a prompt combining the context and the user’s question to guide text generation
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"


# In[47]:


# Generate a text response using GPT-2 based on the constructed prompt
result = generator(prompt, max_length=100, num_return_sequences=1)


# In[48]:


# Print the generated answer text from GPT-2 output
print(result[0]['generated_text'])


# In[ ]:




