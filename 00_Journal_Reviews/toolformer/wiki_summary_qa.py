"""
Wikipedia-based NLP pipeline:
1. Fetches Wikipedia article
2. Summarizes it using DistilBART
3. Embeds sentences using DistilBERT
4. Answers questions based on summary using FLAN-T5
"""

"""
Summarization
→ sshleifer/distilbart-cnn-12-6
A model optimized for condensing long Wikipedia texts into concise summaries.

Embedding
→ distilbert-base-uncased (via SentenceTransformer)
Used to convert text into vector embeddings for semantic similarity calculation or extracting important sentences.

Question Answering (Q&A) Generation
→ google/flan-t5-small
Generates natural answers to questions based on the summarized content or embedding results.
"""
# pip install wikipedia transformers sentence-transformers torch
import wikipedia
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
# 1. Wikipedia page content retrieval function with error handling
def wiki_search(query):
    try:
        # Try to get the exact Wikipedia page content without auto-suggestion
        return wikipedia.page(query, auto_suggest=False).content
    except wikipedia.exceptions.PageError:
        # Handle case when the page does not exist
        return "Page not found."
#Model Setting : "sshleifer/distilbart-cnn-12-6"
"""
# Why we chose this model:
# - "distilbart-cnn-12-6" is a distilled, smaller version of BART optimized for summarization.
# - It balances performance and speed, making it suitable for environments with limited resources.
# - Its lightweight nature allows faster inference on CPU, like on a Mac, while still producing good-quality summaries.
"""
# 2. Model setting for text summarization
summary_model_name = "sshleifer/distilbart-cnn-12-6"

# Load tokenizer for the summarization model
summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)

# Load pre-trained Seq2Seq model for summarization tasks
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)

# 3. Setting up the sentence embedding model
embedding_model = SentenceTransformer('distilbert-base-uncased')

# Explanation:
# - 'distilbert-base-uncased' is a smaller, faster version of BERT.
# - It efficiently converts sentences into fixed-size vector embeddings.
# - These embeddings help measure semantic similarity, useful for tasks like filtering relevant sentences or clustering.
# - Its lightweight architecture makes it suitable for CPU-based environments without sacrificing too much accuracy.

# 4. Setting up the Question Answering (Q&A) generation model
qa_model_name = "google/flan-t5-small"

# Load tokenizer for the Q&A model
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

# Load pre-trained Seq2Seq model for generating answers to questions
qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)

# 5. Document summarization function
def summarize(text):
    # Tokenize the input text with truncation and max length
    inputs = summary_tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
    
    # Generate the summary using the model with specific generation parameters:
    # - max_length: maximum tokens in the summary
    # - min_length: minimum tokens in the summary
    # - length_penalty: encourages shorter or longer summaries (2.0 favors shorter)
    # - num_beams: beam search size for better quality
    # - early_stopping: stop when an end condition is met
    summary_ids = summary_model.generate(
        **inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode the generated token ids to readable text, skipping special tokens
    summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary
# 6. Question Answering (Q&A) function
def generate_answer(question, context):
    # Prepare the prompt by combining the question and context
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    
    # Tokenize the prompt into model input format
    inputs = qa_tokenizer(prompt, return_tensors="pt")
    
    # Generate the answer tokens with a limit on max new tokens
    outputs = qa_model.generate(**inputs, max_new_tokens=50)
    
    # Decode the generated tokens back to readable text, ignoring special tokens
    answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

query = "Artificial intelligence"
wiki_text = wiki_search(query)
summary = summarize(wiki_text)
print("Summary:", summary)

question = "What is artificial intelligence?"
answer = generate_answer(question, summary)
print("Answer:", answer)

query = "Japan"
wiki_text = wiki_search(query)
print(wiki_text[:500])

