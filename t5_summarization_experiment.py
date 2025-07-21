from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import pandas as pd

def load_data():
    # Load a small portion of CNN/DailyMail dataset (for quick testing)
    train_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
    test_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:5]")
    return train_dataset, test_dataset

def initialize_model_tokenizer(model_name="t5-small"):
    # Load pre-trained tokenizer and model from Hugging Face
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(tokenizer, model, text):
    # Generate a summary for a single input text using the T5 model
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary ids with beam search and length constraints
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    # Decode generated ids to string, skipping special tokens
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def preprocess_function(examples, tokenizer):
    # Prepare inputs and labels for fine-tuning
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Tokenize the target summaries with the new `text_target` argument (transformers v4.27+)
    labels = tokenizer(text_target=examples["highlights"], max_length=150, truncation=True, padding="max_length")
    
    # Set labels for model training (decoder inputs)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def fine_tune_model(tokenizer, model, train_dataset, test_dataset):
    # Tokenize the train and test datasets for model fine-tuning
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Set training arguments - small batch size and 1 epoch for fast testing
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no",    # Disable checkpoint saving for faster training
        disable_tqdm=False
    )
    
    # Initialize Hugging Face Trainer with tokenized datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train.select(range(10)),  # Use a small subset for quick fine-tuning
        eval_dataset=tokenized_test.select(range(5)),
        tokenizer=tokenizer,
    )
    
    # Start fine-tuning
    trainer.train()

def evaluate_summaries(tokenizer, model, test_dataset):
    # Evaluate model summaries using ROUGE metric
    rouge = evaluate.load("rouge")
    
    # Generate summaries for each article in test dataset
    predictions = [summarize_text(tokenizer, model, sample["article"]) for sample in test_dataset]
    references = [sample["highlights"] for sample in test_dataset]
    
    # Compute ROUGE scores between generated summaries and reference highlights
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

def main():
    print("Loading datasets...")
    train_dataset, test_dataset = load_data()

    print("Initializing model and tokenizer...")
    tokenizer, model = initialize_model_tokenizer()

    # Evaluate ROUGE score before fine-tuning to establish baseline quality
    print("\nEvaluating baseline model performance before fine-tuning...")
    baseline_rouge = evaluate_summaries(tokenizer, model, test_dataset)
    print("Baseline ROUGE scores:", baseline_rouge)

    print("\nExample summarization:")
    example_text = test_dataset[0]["article"]
    print("Original article:\n", example_text[:300], "...")
    print("Summary:\n", summarize_text(tokenizer, model, example_text))

    print("\nFine-tuning model on small subset...")
    fine_tune_model(tokenizer, model, train_dataset, test_dataset)

    # Evaluate ROUGE score after fine-tuning to check for improvements
    print("\nEvaluating fine-tuned model performance...")
    fine_tuned_rouge = evaluate_summaries(tokenizer, model, test_dataset)
    print("Fine-tuned ROUGE scores:", fine_tuned_rouge)

    # Display comparison of ROUGE scores before and after fine-tuning
    data = {
        "Metric": ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        "Before Fine-tuning": [f"{baseline_rouge['rouge1']*100:.1f}%", f"{baseline_rouge['rouge2']*100:.1f}%", f"{baseline_rouge['rougeL']*100:.1f}%", f"{baseline_rouge['rougeLsum']*100:.1f}%"],
        "After Fine-tuning": [f"{fine_tuned_rouge['rouge1']*100:.1f}%", f"{fine_tuned_rouge['rouge2']*100:.1f}%", f"{fine_tuned_rouge['rougeL']*100:.1f}%", f"{fine_tuned_rouge['rougeLsum']*100:.1f}%"]
    }
    df = pd.DataFrame(data)
    print("\nPerformance Comparison:\n", df)

if __name__ == "__main__":
    main()
