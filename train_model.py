import pandas as pd
import re
from nltk.corpus import stopwords
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

#Clean Text 
df=pd.read_csv("train.csv")

stop_words=set(stopwords.words("english"))

def clean_text(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z\s]", "", text)
    words=text.split()
    words=[word for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_comment"]=df["comment_text"].apply(clean_text)

#TF_IDF Features
vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(df["cleaned_comment"])

y = df[[
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]]
#Train/Test Split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#Logic Regression Training
model=MultiOutputClassifier(LogisticRegression())
model.fit(X_train, y_train)
print("Model training Completed..")

#Model Evaluation
y_pred=model.predict(X_test)
print(classification_report(y_test, y_pred))

#Save Model & Vectorizer

joblib.dump(model, "toxicity_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorized saved..")
