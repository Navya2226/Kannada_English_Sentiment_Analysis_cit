import joblib
import re

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/logistic_regression.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def preprocess_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep English/Kannada Romanized characters and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("=" * 55)
print("Kannada-English Code-Mixed Sentiment Analyzer")
print("=" * 55)

while True:

    text = input(
        "\nEnter a Kannada-English sentence "
        "(or type 'exit'): "
    )

    if text.lower() == "exit":
        print("Program ended.")
        break

    processed_text = preprocess_text(text)

    text_vector = vectorizer.transform([processed_text])

    prediction = model.predict(text_vector)[0]

    print("Processed Text:", processed_text)
    print("Predicted Sentiment:", prediction)