# 🎭 Kannada–English Code-Mixed Sentiment Analysis

A complete Natural Language Processing (NLP) pipeline and interactive web application that classifies sentiment in transliterated Kannada and code-mixed English user reviews. The project explores model performance transitioning from traditional ML baselines to modern deep learning and fine-tuned transformer architectures.

---

## 📌 Project Architecture & Tech Stack

* **Language**: Python 3.13
* **Libraries**: PyTorch, Transformers (Hugging Face), TensorFlow/Keras, Scikit-Learn, Pandas, Streamlit
* **Models Trained**:
  * Naive Bayes (Baseline)
  * Linear SVM (Baseline)
  * Logistic Regression (Baseline)
  * BiLSTM (Deep Learning)
  * XLM-RoBERTa (Fine-Tuned Multilingual Transformer)

---

## 📊 Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Naive Bayes | 0.7152 | 0.7137 | 0.7152 | 0.6935 |
| Linear SVM | 0.7468 | 0.7446 | 0.7468 | 0.7445 |
| Logistic Regression | 0.7670 | 0.7631 | 0.7670 | 0.7614 |
| BiLSTM | 0.7632 | 0.7752 | 0.7632 | 0.7667 |
| **XLM-RoBERTa** | **0.82+** | **0.82+** | **0.82+** | **0.82+** |

*Note: XLM-RoBERTa outperforms baseline models due to its contextual multilingual pre-training on code-switched text structures.*

---

## 📁 Repository Structure

```text
Kannada_English_Sentiment_Analysis/
├── data/                       # Dataset files
├── models/                     # Saved model checkpoints & tokenizers
│   ├── bilstm.keras
│   ├── xlm_roberta_sentiment/
│   └── *.pkl
├── results/                    # Output comparison CSVs and confusion matrix plots
├── src/                        # Core Python scripts
│   ├── app.py                  # Streamlit web application
│   ├── evaluate_models.py      # Model comparison script
│   ├── train_bilstm.py         # BiLSTM training
│   ├── train_ml.py             # ML baseline training
│   └── train_transformers.py   # Transformer fine-tuning script
└── README.md