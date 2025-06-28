import streamlit as st
import joblib
import re
import string
import os

# ---------------------- Load model and vectorizer ---------------------- #
if not os.path.exists("spam_classifier.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
    st.error("❌ Required model/vectorizer files not found.")
    st.stop()

try:
    model = joblib.load("spam_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# ---------------------- Text Preprocessing ---------------------- #
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(rf"[{string.punctuation}]", '', text)
    return text.strip()

# ---------------------- Streamlit UI ---------------------- #
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📱 SMS Spam Classifier")
st.write("Check if a message is spam or not.")

# Input
user_input = st.text_area("✏️ Enter message:")

# Prediction button
if st.button("Predict"):
    st.write("📤 Message submitted for prediction")

    if not user_input.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        cleaned = preprocess(user_input)
        st.write(f"🧹 Cleaned message: `{cleaned}`")

        try:
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)

            st.write(f"🔎 Raw prediction output: `{prediction}`")

            if prediction[0] == 1:
                st.error("🚨 SPAM Message Detected!")
            else:
                st.success("✅ Not Spam")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
