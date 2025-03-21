import pandas as pd
import re

# Load the raw transcripts
with open("data/transcripts.txt", "r", encoding="utf-8") as file:
    conversations = file.readlines()

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Define objective labels (manually updated based on actual conversations)
objectives = {
    "customer_support": 0,
    "sales_inquiry": 1,
    "technical_issue": 2
}


# Assign objectives manually++

preprocessed_data = []
for line in conversations:
    cleaned_text = clean_text(line)
    if "refund" in cleaned_text or "support" in cleaned_text:
        label = 0  # Customer Support
    elif "buy" in cleaned_text or "discount" in cleaned_text:
        label = 1  # Sales Inquiry
    elif "error" in cleaned_text or "bug" in cleaned_text:
        label = 2  # Technical Issue
    else:
        label = -1  # Skip unknown cases

    preprocessed_data.append([cleaned_text, label])
# Convert to DataFrame
df = pd.DataFrame(preprocessed_data, columns=["text", "label"]) 

# Save processed data
df.to_csv("data/preprocessed_data.csv", index=False)

print("✅ Preprocessed data saved to data/preprocessed_data.csv")
