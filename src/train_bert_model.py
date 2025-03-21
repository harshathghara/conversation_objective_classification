import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ast import literal_eval

# Load feature data
df = pd.read_csv("data/bert_features.csv", converters={"bert_features": literal_eval})

# Convert to numpy array
X = np.vstack(df["bert_features"].values)
y = df["label"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Training completed. Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "models/bert_model.pkl")
print("Model saved to models/bert_model.pkl")
