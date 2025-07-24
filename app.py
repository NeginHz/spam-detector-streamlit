import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if needed
nltk.download("stopwords")
nltk.download("wordnet")

nltk.data.path.append("./nltk_data")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Load model and vectorizer
model = joblib.load("svm2_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer_path2.pkl")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
custom_stop_words = set(stopwords.words("english")) - set(["won"])

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "URL_TOKEN", text)
    text = re.sub(r"\d+", "NUM_TOKEN", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stop_words]
    return " ".join(words)

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
        cleaned_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        label = "📢 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
        st.success(f"Prediction: {label}")
