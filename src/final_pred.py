import pandas as pd
import numpy as np
import joblib
from transformers import BertTokenizer, BertModel
import torch
from collections import Counter

# Load trained model
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

# Function to predict objectives for multiple sentences
def predict_overall_objective(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = file.readlines()

    predictions = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # Ignore empty lines
            features = extract_bert_features(sentence).reshape(1, -1)
            prediction = model.predict(features)[0]
            predictions.append(prediction)

    # Get the most common prediction
    if predictions:
        overall_prediction = Counter(predictions).most_common(1)[0][0]
        return objective_mapping[overall_prediction]
    else:
        return "No valid sentences found in the file."

# Example usage
file_path = "data/transcripts.txt"  # Replace with your text file path
overall_objective = predict_overall_objective(file_path)
print(f"Overall Predicted Objective: {overall_objective}")
