Conversation Objective Classification 🚀

# 📌 Project Overview

This project classifies conversation transcripts into predefined objectives using BERT-based feature extraction and a logistic regression model.

# 📂 Folder Structure

CONVERSATION_OBJECTIVE_CLASSIFICATION/
│── data/                     # Stores dataset files
│   ├── bert_features.csv      # Extracted BERT features
│   ├── preprocessed_data.csv  # Processed conversation data
│   ├── transcripts.txt        # Raw conversation transcripts
│
│── models/                    # Trained machine learning models
│   ├── bert_model.pkl         # Saved logistic regression model
│
│── src/                       # Core project scripts
│   ├── feature_extraction.py  # Extracts BERT features from text
│   ├── final_pred.py          # Predicts objective for full transcript files
│   ├── predict.py             # Predicts objective for single sentences
│   ├── preprocess.py          # Preprocesses raw transcripts
│   ├── train_bert_model.py    # Trains the logistic regression model
│
│── transcripts_venv/          # Virtual environment (ignored in Git)
│── check_features.py          # Debugging script for checking BERT features
│── .gitignore                 # Files to exclude from Git
│── README.md                  # Project documentation


# 🛠 Setup Instructions

1️⃣ Install Dependencies

python -m venv transcripts_venv
source transcripts_venv/bin/activate  # On macOS/Linux
transcripts_venv\Scripts\activate     # On Windows
pip install -r requirements.txt


2️⃣ Preprocess Data

Run the following command to process raw conversation transcripts:

python src/preprocess.py


3️⃣ Extract Features Using BERT

Generate BERT embeddings for each sentence in the dataset:

python src/feature_extraction.py

4️⃣ Train the Model

Train the logistic regression model using extracted features:

python src/train_bert_model.py


5️⃣ Predict Objective for a Single Sentence

Run predictions on a single input sentence:

python src/predict.py

6️⃣ Predict Objective for Full Transcripts

Run classification on an entire transcript file:

python src/final_pred.py


# 🏆 Model Performance

Accuracy: 1.00 (on the training dataset)

Classification Objectives:
0 → Customer Support
1 → Sales Inquiry
2 → Technical Issue

# 📌 How It Works?

Preprocessing (preprocess.py): Cleans the conversation transcript.
Feature Extraction (feature_extraction.py): Converts text into BERT embeddings.
Model Training (train_bert_model.py): Trains a logistic regression model.
Prediction (predict.py & final_pred.py): Uses the trained model to classify conversations.

# 📝 Future Improvements

🔹 Fine-tune BERT instead of using precomputed embeddings.
🔹 Expand the dataset for better generalization.
🔹 Deploy as a web API for real-time classification.

# 🤝 Contributors

👤 Harsh Kumar
📌 CMR Institute of Technology | AIML Department

