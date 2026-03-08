import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#preprocessing the dataset
df=pd.read_csv("train.csv")

comments=df["comment_text"]

stop_words=set(stopwords.words("english"))

def clean_text(text):
    text=text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words=text.split()
    words=[word for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_comment"]=comments.apply(clean_text)

print("Cleaning completed..")
print(df[["comment_text" ,"cleaned_comment"]].head)

#TF-IDF Adding
vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(df["cleaned_comment"])
print("TF-IDF shape:", X.shape)

#saving vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Vectoried saved!")
