🛡 Toxic Comment Detection System

A Machine Learning and N LP based web application that detects cyberbullying and toxic comments in real time.
The system analyzes user input text and classifies it into six toxicity categories using a trained ML model.

🚀 Project Overview

Online platforms often suffer from abusive or harmful comments. This project aims to detect toxic language automatically using Natural Language Processing and Machine Learning.
The system processes user comments, converts them into numerical features using TF-IDF, and predicts toxicity categories using a Logistic Regression model.
The model is deployed through an interactive Streamlit web application where users can analyze comments instantly.

🎯 Features

✔ Detects 6 types of toxic content

Toxic
Severe Toxic
Obscene
Threat
Insult
Identity Hate

✔ Real-time comment analysis
✔ Clean and interactive Streamlit UI
✔ Shows prediction probabilities and toxicity levels
✔ Displays detailed analysis results
✔ Fully automated NLP pipeline

🧠 Machine Learning Pipeline

The system follows an end-to-end ML workflow:

User Comment
      ↓
Text Preprocessing
(lowercase, remove symbols, stopwords)
      ↓
TF-IDF Feature Extraction
      ↓
Logistic Regression Model
      ↓
Multi-Label Toxicity Prediction
      ↓
Streamlit Dashboard Results
🛠 Tech Stack

Programming Language

Python
Machine Learning
Scikit-learn
TF-IDF Vectorization
Logistic Regression

Libraries

Pandas
NumPy
NLTK
Joblib
Streamlit

📂 Project Structure
toxic-comment-detector
│
├── app.py                     # Streamlit application
├── preprocess.py              # Text preprocessing script
├── train_model.py             # Model training script
├── toxicity_model.pkl         # Trained ML model
├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore

📊 Dataset

This project uses the Jigsaw Toxic Comment Classification Dataset from Kaggle.
The dataset contains 159,000+ labeled comments across multiple toxicity categories.
Dataset link:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

⚙️ Installation

1.Clone the repository
git clone https://github.com/YOUR_USERNAME/toxic-comment-detector.git

2.Navigate to the project folder
cd toxic-comment-detector

3.Install dependencies
pip install -r requirements.txt

4.Download NLTK stopwords (first time only)
import nltk
nltk.download('stopwords')

5.Run the application
streamlit run app.py

🖥 Example Output

User Input

You are such a stupid person

Model Prediction

Toxic → Detected
Insult → Detected
Threat → Not Detected

The system also displays toxicity probabilities and visual indicators.

🔒 Privacy

All analysis runs locally
No user data is stored
Comments are not shared externally

📈 Future Improvements

Possible enhancements for this project:
Deep Learning models (LSTM / BERT)
Real-time moderation API
Multi-language toxicity detection

Browser extension for comment moderation

GitHub:
https://github.com/GitHubManeesh

⭐ If you like this project

Give the repository a star ⭐ on GitHub.
