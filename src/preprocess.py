import pandas as pd
import re
import os

# Load dataset
input_path = "data/Romanized_kannada_reviews.csv"

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Keep only required columns
df = df[["Cleaned_Comment", "Final_Label"]].copy()

# Remove missing values
df.dropna(subset=["Cleaned_Comment", "Final_Label"], inplace=True)

# Convert text to string
df["Cleaned_Comment"] = df["Cleaned_Comment"].astype(str)

# Text preprocessing
def clean_text(text):
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["Processed_Text"] = df["Cleaned_Comment"].apply(clean_text)

# Remove duplicate comments
df.drop_duplicates(subset=["Processed_Text"], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Create results directory
os.makedirs("results", exist_ok=True)

# Save processed dataset
output_path = "data/processed_dataset.csv"
df.to_csv(output_path, index=False)

print("\nPreprocessing completed!")
print("Processed dataset shape:", df.shape)

print("\nSentiment distribution:")
print(df["Final_Label"].value_counts())

print("\nSample processed data:")
print(df[["Processed_Text", "Final_Label"]].head())

print("\nSaved to:", output_path)