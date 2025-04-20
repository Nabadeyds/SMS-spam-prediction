from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# Download stopwords if not already present
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Initialize Flask app
app = Flask(__name__)

# Text preprocessing function
def sentconv(seinp):
    lowersent = seinp.lower()
    clsent = re.sub(r"<.*?>", "", lowersent)
    clsent = re.sub(r"[!@#$%^&*()_+{}[\]:;,.<>/?\\|'\-=\"~]", " ", clsent)
    clsent = re.sub(r"\s+", " ", clsent)
    clsent = clsent.strip().split(" ")
    stopwor = stopwords.words("english")
    cleanstopwds = [i for i in clsent if i not in stopwor]
    p = PorterStemmer()
    splitx = [p.stem(i) for i in cleanstopwds]
    return " ".join(splitx)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    message = request.form['message']
    transformed_msg = sentconv(message)
    vect_input = vectorizer.transform([transformed_msg])
    prediction = model.predict(vect_input)[0]
    result = "Spam" if prediction == 1 else "Ham"
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
