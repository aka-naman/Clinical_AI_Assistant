import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Optional: Disable symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Optional: Token (if not logged in via huggingface-cli)
hf_token = "Your_HuggingFace_Token"  # Replace or set to None if already logged in

# Model name (you can change it to any QA model)
model_name = "deepset/minilm-uncased-squad2"

# Detect GPU (CUDA) if available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

print("Loading model...")
model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=hf_token)

# Create QA pipeline with device support
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

# Example input
context = """
Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns and make decisions with minimal human intervention.
"""
question = "What is machine learning?"

# Get the answer
result = qa_pipeline(question=question, context=context)
print("\nQuestion:", question)
print("Answer:", result['answer'])
