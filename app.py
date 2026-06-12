from flask import Flask, render_template, request
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)

stemmer = PorterStemmer()

# Load trained ML model and TF-IDF vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


def preprocess_message(message):
    """
    Cleans and preprocesses SMS text before prediction.
    """

    message = message.lower()
    words = nltk.word_tokenize(message)

    processed_words = []

    for word in words:
        if word.isalnum():
            processed_words.append(word)

    processed_words = [
        stemmer.stem(word)
        for word in processed_words
        if word not in stopwords.words("english")
        and word not in string.punctuation
    ]

    return " ".join(processed_words)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form.get("message")

    if not message:
        return render_template(
            "index.html",
            result="Please enter a message"
        )

    cleaned_message = preprocess_message(message)

    vectorized_message = vectorizer.transform(
        [cleaned_message]
    )

    prediction = model.predict(vectorized_message)[0]

    result = (
        "🚨 Spam Message"
        if prediction == 1
        else "✅ Not Spam Message"
    )

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run()
