import streamlit as st
import joblib
import re
import string
import nltk
import os
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer

# Add custom nltk data path (if bundled with the app)
nltk_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_path)

# Load stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    st.error("❌ NLTK stopwords not found. Make sure 'nltk_data/corpora/stopwords/' exists.")
    st.stop()

try:
    lemmatizer = WordNetLemmatizer()
    _ = lemmatizer.lemmatize("test")  # test WordNet availability
except LookupError:
    st.error("❌ WordNet not found. Make sure 'nltk_data/corpora/wordnet/' and 'omw-1.4/' exist.")
    st.stop()

# Load the trained model
try:
    model = joblib.load("svm2_model.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Load the fitted vectorizer
try:
    vectorizer = joblib.load("tfidf_vectorizer_path2.pkl")
    _ = vectorizer.vocabulary_  # check if fitted
except (NotFittedError, AttributeError):
    st.error("❌ Vectorizer is not fitted. Train and save it after calling `.fit()`.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load vectorizer: {e}")
    st.stop()

# Define custom stopwords (excluding 'won')
custom_stop_words = set(stop_words) - {"won"}

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "URL_TOKEN", text)
    text = re.sub(r"\d+", "NUM_TOKEN", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stop_words]
    return " ".join(words)

# Streamlit app interface
st.title("📨 Spam Detector App")
st.write("Enter a message below. The model will classify it as spam or not.")

# User input field
user_input = st.text_area("✍️ Enter your message here:")

# When 'Predict' button is clicked
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Preprocess the input
        cleaned_input = preprocess_text(user_input)
        try:
            # Transform and predict
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]

            # Display result
            label = "📢 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
            st.success(f"Prediction: {label}")
        except NotFittedError:
            st.error("❌ Vectorizer is not fitted properly.")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
