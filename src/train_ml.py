import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# --------------------------------------------------
# Settings
# --------------------------------------------------
MAX_VOCAB = 20000
MAX_LEN = 60
EMBED_DIM = 128
LSTM_UNITS = 64
EPOCHS = 15
BATCH_SIZE = 32

tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("results/confusion_matrices", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --------------------------------------------------
# 1. Load data (same as train_ml.py)
# --------------------------------------------------
df = pd.read_csv("data/processed_dataset.csv")
df = df.dropna(subset=["Processed_Text", "Final_Label"]).copy()
df["Processed_Text"] = df["Processed_Text"].astype(str)

X = df["Processed_Text"]
y = df["Final_Label"]

# --------------------------------------------------
# 2. Same train/test split as train_ml.py
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 3. Encode labels and convert text to sequences
# --------------------------------------------------
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)
class_names = list(label_encoder.classes_)
num_classes = len(class_names)
print("Classes:", class_names)

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(
    tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post", truncating="post"
)
X_test_pad = pad_sequences(
    tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding="post", truncating="post"
)

y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)

weights = compute_class_weight("balanced", classes=np.unique(y_train_enc), y=y_train_enc)
class_weight_dict = {i: w for i, w in enumerate(weights)}

# --------------------------------------------------
# 4. Build BiLSTM model
# --------------------------------------------------
vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, mask_zero=True),
    Bidirectional(LSTM(LSTM_UNITS, dropout=0.3)),
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(num_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# --------------------------------------------------
# 5. Train
# --------------------------------------------------
print("\n" + "=" * 50)
print("Training BiLSTM...")
print("=" * 50)

history = model.fit(
    X_train_pad,
    y_train_cat,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    verbose=2,
)

# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------
pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)

accuracy = accuracy_score(y_test_enc, pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_enc, pred, average="weighted", zero_division=0
)

print("\nBiLSTM Results")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test_enc, pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_test_enc, pred)
print("Confusion Matrix:")
print(cm)

# confusion matrix image
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
)
ax.set_title("Confusion Matrix - BiLSTM")
plt.tight_layout()
plt.savefig("results/confusion_matrices/bilstm.png", dpi=150)
plt.close()

# training curves (useful for the report)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["loss"], label="train loss")
axes[0].plot(history.history["val_loss"], label="val loss")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history["accuracy"], label="train acc")
axes[1].plot(history.history["val_accuracy"], label="val acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig("results/bilstm_training_curves.png", dpi=150)
plt.close()

# --------------------------------------------------
# 7. Save model, tokenizer, label encoder
# --------------------------------------------------
model.save("models/bilstm.keras")
with open("models/bilstm_tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())
joblib.dump(label_encoder, "models/bilstm_label_encoder.pkl")

# --------------------------------------------------
# 8. Add BiLSTM row to results/model_comparison.csv
# --------------------------------------------------
row = {
    "Model": "BiLSTM",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
}

csv_path = "results/model_comparison.csv"
if os.path.exists(csv_path):
    comparison = pd.read_csv(csv_path)
    comparison = comparison[comparison["Model"] != "BiLSTM"]
    comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True)
else:
    comparison = pd.DataFrame([row])

comparison.to_csv(csv_path, index=False)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))
print("\nSaved: models/bilstm.keras, results/model_comparison.csv")