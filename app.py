from flask import Flask, request, render_template, redirect, url_for, session
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session to work

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

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    result = session.pop('result', None)  # Clear after use
    return render_template('index.html', prediction_text=result)

@app.route('/predict', methods=["POST"])
def predict():
    message = request.form['message']
    transformed_msg = sentconv(message)
    vect_input = vectorizer.transform([transformed_msg])
    prediction = model.predict(vect_input)[0]
    result = "Prediction: Spam" if prediction == 1 else "Prediction: Ham"
    session['result'] = result
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
