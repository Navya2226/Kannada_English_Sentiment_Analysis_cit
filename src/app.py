import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Kannada-English Sentiment Analyzer", page_icon="🎭", layout="centered")

st.title("🎭 Kannada–English Code-Mixed Sentiment Analysis")
st.write("Enter text in transliterated Kannada or code-mixed English to predict sentiment.")

# Force CPU inference for stability across platforms
device = torch.device("cpu")

# Model Loaders
@st.cache_resource
def load_xlm_roberta():
    model_path = "models/xlm_roberta_sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_baseline_model(model_name):
    # Load TF-IDF vectorizer and standard ML classifier
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(f'models/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

# Sidebar Configuration
st.sidebar.header("⚙️ Model Configuration")
selected_model = st.sidebar.selectbox(
    "Choose Sentiment Model:",
    ["XLM-RoBERTa (Transformer)", "Logistic Regression", "Linear SVM", "Naive Bayes"]
)

id2label = {0: "Neutral 😐", 1: "Negative 😡", 2: "Positive 😃"}

user_input = st.text_area("Input Code-Mixed Text:", "Movie thumba super agidhe, next level acting!", height=100)

if st.button("Predict Sentiment", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            if "XLM-RoBERTa" in selected_model:
                tokenizer, model = load_xlm_roberta()
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
                    pred_idx = int(torch.argmax(outputs.logits, dim=-1).item())
            else:
                # Baseline ML Pipeline
                model_file_map = {
                    "Logistic Regression": "logistic_regression",
                    "Linear SVM": "linear_svm",
                    "Naive Bayes": "naive_bayes"
                }
                vectorizer, model = load_baseline_model(model_file_map[selected_model])
                vec_input = vectorizer.transform([user_input])
                pred_idx = model.predict(vec_input)[0]
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vec_input)[0]
                else:
                    # Fallback for SVM without probability calibration
                    probs = [0.33, 0.33, 0.33]
                    probs[pred_idx] = 1.0

            st.success(f"Selected Model: **{selected_model}**")
            st.subheader(f"Prediction: **{id2label[pred_idx]}**")
            
            st.write("---")
            st.write("### Prediction Confidence:")
            for idx, label in id2label.items():
                prob_val = float(probs[idx])
                st.progress(prob_val, text=f"{label}: {prob_val*100:.1f}%")
    else:
        st.warning("Please enter a sentence to analyze.")