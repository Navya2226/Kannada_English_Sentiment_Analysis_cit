# Kannada–English Sentiment Analysis

Sentiment analysis of Romanized Kannada (Kannada–English code-mixed) reviews. The project compares classical machine learning models, a BiLSTM and a fine-tuned XLM-RoBERTa transformer.

## Dataset

- `data/Romanized_kannada_reviews.csv`: raw reviews
- `data/processed_dataset.csv`: dataset after preprocessing
- About 9,165 reviews, split 80% train / 20% test

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes | 0.7152 | 0.7137 | 0.7152 | 0.6935 |
| Linear SVM | 0.7469 | 0.7446 | 0.7469 | 0.7445 |
| Logistic Regression | 0.7670 | 0.7631 | 0.7670 | 0.7614 |
| BiLSTM | 0.7632 | 0.7753 | 0.7632 | 0.7668 |
| **XLM-RoBERTa (fine-tuned)** | **0.7867** | **0.7848** | **0.7867** | **0.7856** |

XLM-RoBERTa performs best, about 2 points above the strongest classical model. The XLM-RoBERTa scores are from the final evaluation after 3 epochs (`results/xlm/trainer_state.json`).

Charts and confusion matrices for the other models are in the `results/` folder:
- `accuracy_comparison.png`, `f1_score_comparison.png`, `precision_comparison.png`, `recall_comparison.png`
- `bilstm_training_curves.png`
- `confusion_matrices/` for Logistic Regression, Linear SVM, Naive Bayes and BiLSTM
- `model_comparison.csv`

## Project structure

```
├── data/                     # dataset
├── results/                  # charts, confusion matrices, metrics
├── src/
│   ├── preprocess.py         # cleaning and preparing the data
│   ├── analyze_dataset.py    # dataset statistics
│   ├── train_ml.py           # Naive Bayes, Linear SVM, Logistic Regression
│   ├── train_bilstm.py       # BiLSTM model
│   ├── train_transformers.py # XLM-RoBERTa fine-tuning
│   ├── evaluate_models.py    # comparison of all models
│   ├── predict.py            # predict sentiment for new text
│   └── app.py                # web app
├── requirements.txt
└── test.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run these from the project root:

```bash
python src/preprocess.py
python src/analyze_dataset.py
python src/train_ml.py
python src/train_bilstm.py
python src/train_transformers.py
python src/evaluate_models.py
python src/predict.py
python src/app.py
```

## Note

Trained models (`models/`) and training checkpoints (`results_xlm/`) are not in this repository because of their size (hundreds of MB). Run `src/train_transformers.py` to regenerate the fine-tuned XLM-RoBERTa model in `models/xlm_roberta_sentiment`.
