import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load XLM-RoBERTa
model_path = "models/xlm_roberta_sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. Load Dataset
test_df = pd.read_csv('data/Romanized_kannada_reviews.csv')

# Use detected columns
text_col = 'Cleaned_Comment'
label_col = 'Annotator_1'

# Clean labels and convert string labels to integer IDs
label_map = {'Neutral': 0, 'Negative': 1, 'Positive': 2}
test_df['label_id'] = test_df[label_col].astype(str).str.strip().map(label_map)

# Drop any unmapped/NaN labels
test_df = test_df.dropna(subset=['label_id'])
test_df['label_id'] = test_df['label_id'].astype(int)

texts = test_df[text_col].astype(str).tolist()
true_labels = test_df['label_id'].tolist()

# 3. Model Predictions
predictions = []
model.eval()
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(pred_label)

# 4. Calculate Final Performance Metrics
acc = accuracy_score(true_labels, predictions)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

print("\n==========================================")
print("XLM-RoBERTa Final Evaluation Results:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print("==========================================")

# 5. Append Results to model_comparison.csv
try:
    comparison_df = pd.read_csv('results/model_comparison.csv')
    new_row = pd.DataFrame([{'Model': 'XLM-RoBERTa', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1_Score': f1}])
    comparison_df = pd.concat([comparison_df, new_row], ignore_index=True)
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("Successfully updated results/model_comparison.csv!")
except Exception as e:
    print("Could not update model_comparison.csv:", e)