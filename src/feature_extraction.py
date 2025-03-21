import pandas as pd
import numpy as np

# Load preprocessed data
df = pd.read_csv("data/preprocessed_data.csv")

# Simulating BERT embeddings (replace this with your actual embedding extraction)
def dummy_bert_embedding(text):
    return np.random.rand(768)  # Example: 768-dimensional random vector

# Extract features
df["bert_features"] = df["text"].apply(dummy_bert_embedding)

# Convert list to properly formatted CSV (comma-separated values)
df["bert_features"] = df["bert_features"].apply(lambda x: ",".join(map(str, x)))

# Save formatted features
df[["bert_features", "label"]].to_csv("data/bert_features.csv", index=False)

print("BERT feature extraction completed. Features saved to data/bert_features.csv")
