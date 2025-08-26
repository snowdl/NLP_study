from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the pretrained tokenizer and model for flan-t5-base
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define the input prompt/question clearly
prompt = "Question: What is the capital city of France? Answer:"

# Tokenize the input prompt to convert it into model-readable format
inputs = tokenizer(prompt, return_tensors="pt")

print("🌀 Generating...")

# Generate output tokens from the model
# - max_new_tokens: limit the length of generated tokens
# - num_beams: use beam search to improve output quality
# - early_stopping: stop generation when an end condition is met
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_beams=5,
    early_stopping=True
)

# Decode the generated tokens back into human-readable text
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the model's response
print("📤 Model output:", result)
