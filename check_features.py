import pandas as pd
import json

# Load BERT features
df = pd.read_csv("data/bert_features.csv")

# Convert JSON string back to list
df["bert_features"] = df["bert_features"].apply(lambda x: json.loads(x))

# Check if all feature vectors are 768-dimensional
incorrect_rows = df[df["bert_features"].apply(lambda x: len(x) != 768)]

if incorrect_rows.empty:
    print("✅ All feature vectors have 768 dimensions.")
else:
    print("⚠️ Some feature vectors are incomplete! Check these rows:")
    print(incorrect_rows)
