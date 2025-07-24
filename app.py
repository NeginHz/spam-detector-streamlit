import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("svm2_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer_path2.pkl")

# App title
st.title("📨 Spam Detector App")
st.write("Enter a message and the model will predict whether it's spam or not.")

# Text input
user_input = st.text_area("✍️ Enter your message here:", "")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        label = "📢 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
        st.success(f"Prediction: {label}")
