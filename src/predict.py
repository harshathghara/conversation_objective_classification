import pandas as pd
import numpy as np
import joblib
from transformers import BertTokenizer, BertModel
import torch

# Load model
model = joblib.load("models/bert_model.pkl")

# Load BERT model for feature extraction
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Define objective mapping
objective_mapping = {0: "Customer Support", 1: "Sales Inquiry", 2: "Technical Issue"}

# Function to extract BERT features
def extract_bert_features(text):
    tokens = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()

# Predict function
def predict_objective(text):
    features = extract_bert_features(text).reshape(1, -1)
    prediction = model.predict(features)[0]
    return objective_mapping[prediction]

# Test predictions
sample_text = "I want a refund for my last purchase"
predicted_objective = predict_objective(sample_text)
print(f"Predicted Objective: {predicted_objective}")
