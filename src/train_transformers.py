import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Load Dataset
data_path = 'data/Romanized_kannada_reviews.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Check filename in data folder.")

df = pd.read_csv(data_path)

# Ensure correct column names (maps text and sentiment)
text_col = 'text' if 'text' in df.columns else df.columns[0]
label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]

# Map text sentiment labels to integers
unique_labels = df[label_col].unique().tolist()
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}
df['label'] = df[label_col].map(label2id)

print(f"Dataset Loaded: {len(df)} rows")
print(f"Labels mapped: {label2id}")

# Split Dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 2. Tokenize with XLM-RoBERTa
MODEL_NAME = "xlm-roberta-base"
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

# 3. Evaluation Metrics Setup
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 4. Load Model and Set Up Training
print(f"Initializing {MODEL_NAME} model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./results_xlm",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 5. Start Training
print("Starting XLM-RoBERTa Fine-Tuning...")
trainer.train()

# 6. Save Fine-Tuned Model
save_dir = "models/xlm_roberta_sentiment"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model saved successfully to {save_dir}!")