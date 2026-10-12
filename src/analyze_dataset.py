import pandas as pd
import re

# Load processed dataset
df = pd.read_csv("data/processed_dataset.csv")

print("=" * 60)
print("KANNADA-ENGLISH SENTIMENT DATASET ANALYSIS")
print("=" * 60)

# -------------------------------
# 1. Dataset size
# -------------------------------
print("\n1. DATASET SIZE")
print("-" * 40)
print("Total samples:", len(df))
print("Total columns:", len(df.columns))
print("Columns:", list(df.columns))

# -------------------------------
# 2. Sentiment distribution
# -------------------------------
print("\n2. SENTIMENT DISTRIBUTION")
print("-" * 40)

label_counts = df["Final_Label"].value_counts()

print(label_counts)

print("\nPercentage distribution:")
print((label_counts / len(df) * 100).round(2))

# -------------------------------
# 3. Missing values
# -------------------------------
print("\n3. MISSING VALUES")
print("-" * 40)

print(df.isnull().sum())

# -------------------------------
# 4. Duplicate comments
# -------------------------------
print("\n4. DUPLICATE COMMENTS")
print("-" * 40)

duplicates = df["Cleaned_Comment"].duplicated().sum()

print("Duplicate comments:", duplicates)

# -------------------------------
# 5. Empty comments
# -------------------------------
print("\n5. EMPTY COMMENTS")
print("-" * 40)

empty_comments = (
    df["Cleaned_Comment"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print("Empty comments:", empty_comments)

# -------------------------------
# 6. Average text length
# -------------------------------
print("\n6. TEXT LENGTH")
print("-" * 40)

df["text_length"] = (
    df["Cleaned_Comment"]
    .fillna("")
    .astype(str)
    .apply(lambda x: len(x.split()))
)

print("Average words per comment:",
      round(df["text_length"].mean(), 2))

print("Minimum words:",
      df["text_length"].min())

print("Maximum words:",
      df["text_length"].max())

# -------------------------------
# 7. Sentiment-wise text length
# -------------------------------
print("\n7. TEXT LENGTH BY SENTIMENT")
print("-" * 40)

print(
    df.groupby("Final_Label")["text_length"]
    .mean()
    .round(2)
)

# -------------------------------
# 8. Important sentiment words
# -------------------------------
print("\n8. SAMPLE SENTIMENT WORD ANALYSIS")
print("-" * 40)

positive_words = [
    "chennagide",
    "super",
    "excellent",
    "best",
    "good",
    "love",
    "awesome"
]

negative_words = [
    "worst",
    "bad",
    "sari illa",
    "waste",
    "poor",
    "hate",
    "problem"
]

text = (
    df["Cleaned_Comment"]
    .fillna("")
    .astype(str)
    .str.lower()
)

print("\nPositive word occurrences:")

for word in positive_words:
    count = text.str.contains(
        re.escape(word),
        regex=True
    ).sum()

    print(f"{word}: {count}")

print("\nNegative word occurrences:")

for word in negative_words:
    count = text.str.contains(
        re.escape(word),
        regex=True
    ).sum()

    print(f"{word}: {count}")

# -------------------------------
# 9. Sample records
# -------------------------------
print("\n9. SAMPLE DATA")
print("-" * 40)

print(
    df[["Cleaned_Comment", "Final_Label"]]
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("DATASET ANALYSIS COMPLETED")
print("=" * 60)