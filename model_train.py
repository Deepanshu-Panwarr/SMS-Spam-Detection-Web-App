import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

data = data.iloc[:, :2]
data.columns = ["label", "message"]

data["label"] = data["label"].map(
    {"ham": 0, "spam": 1}
)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    data["message"]
)

y = data["label"]

model = MultinomialNB()

model.fit(X, y)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("Model saved successfully")
