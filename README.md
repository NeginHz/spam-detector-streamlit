# 📧 Spam Detector - Streamlit App

A lightweight and accurate **spam detection web app** powered by **Scikit-learn**, **Streamlit**, and a well-evaluated **SVM model (SVM-2)**.

> 👨‍🎓 Developed as part of an **NLP course Project** on text classification using machine learning.  
> ✅ The selected model demonstrated the best generalization across multiple datasets after extensive evaluation.

---

## 🧠 About the Project

This app takes a **text message** as input and classifies it as either:

- ✅ `HAM` (Not Spam)  
- 🚫 `SPAM` (Unwanted or Junk Message)

The backend model is a **Support Vector Machine (SVM)** trained on real-world SMS messages.  
It uses **TF-IDF vectorization** to extract meaningful features from text for spam detection.

---

## 🧪 Model Selection Process

This project includes a **3-phase model evaluation pipeline** implemented in Jupyter/Colab notebooks:

📂 `notebooks/`

- 📘 **Notebook 1 (Path 1)**  
  Trains NB-1 and SVM-1 using the **Small Dataset**. Evaluated on validation/test and general unseen data.

- 📘 **Notebook 2 (Path 2)**  
  Trains NB-2 and SVM-2 using a **large Kaggle dataset**.. Shows improved performance across test datasets.

- 📘 **Notebook 3 (Cross Evaluation)**  
  Compares all 4 models across 4 datasets.  
  ✅ **SVM-2** was selected as the final model based on best F1 score and stable performance across validation, test, and unseen data.

---
  ## 📊 Dataset Information

- 🧾 **SMS Spam Collection Dataset** (Kaggle) 
- 📘 **Small Dataset** (from GitHub by justmarkham) 

> 📌 Source:  
> - [Tinu Kumar - Kaggle Dataset](https://www.kaggle.com/datasets/tinu10kumar/sms-spam-dataset)  
> - [justmarkham - GitHub Dataset](https://github.com/justmarkham/DAT8/blob/master/data/sms.tsv)  

---

## 💡 Why Streamlit?

Streamlit is used to quickly turn the final trained model into a usable web app:

- 🖊 Simple input box for message  
- 🔍 Predicts whether message is spam or ham  
- 🧠 Uses saved `SVM-2` model and `TF-IDF` vectorizer (`joblib`)

---

## ☁️ Used Google Colab

All preprocessing, training, and evaluation were performed in **Google Colab** for easy scalability and reproducibility.  
Notebooks were saved and exported from Colab into the project.

---

## 🛠 Tech Stack

- Python 3  
- Scikit-learn  
- Pandas  
- Joblib  
- Streamlit

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/spam-detector-streamlit.git
cd spam-detector-streamlit
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the App
```bash
streamlit run app.py
```
Then open the browser at: http://localhost:8501/
---
### Credits

Model trained by: Negin Hezarjaribi

NLP Projct - IU University

Powered by: Scikit-learn, Streamlit, Google Colaba

