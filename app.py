# app.py

import streamlit as st
import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Prediction function
def predict_news(news):
    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

# Streamlit App UI
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🧠", layout="centered")

st.markdown(
    """
    <div style="text-align:center">
        <h1 style="color:#1F77B4;">🧠 Fake News Detection App</h1>
        <p style="font-size:18px;">This app uses a machine learning model to detect whether a news article is <b>Fake</b> or <b>Real</b>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### 🔍 Enter a news article below:")
news_input = st.text_area("📝 Paste news text here", height=200)

if st.button("🚀 Predict"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news content to analyze.")
    else:
        prediction = predict_news(news_input)
        if prediction == 1:
            st.success("✅ This news seems to be **REAL**.")
            st.markdown("### 🟢 Trustworthy News")
        else:
            st.error("❌ This news seems to be **FAKE**.")
            st.markdown("### 🔴 Beware of Misinformation!")

st.markdown("---")
st.markdown("💡 *Built using Logistic Regression & TF-IDF Vectorization.*")
