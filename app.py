from flask import Flask, render_template, request
# import model

app = Flask(__name__)

import string
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
# prepare regex for char filtering
    re_punc = re.compile( ' [%s] ' % re.escape(string.punctuation))
# remove punctuation from each word
    tokens = [re_punc.sub( '' , w) for w in tokens]
# remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
# filter out stop words
    stop_words = set(stopwords.words( 'english' ))
    tokens = [w for w in tokens if not w in stop_words]
# filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
# pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding= 'post' )
    return padded


# classify a review as negative or positive
@app.route('/detect', methods=['POST', 'OPTIONS'])
def predict_sentiment(review, vocab, tokenizer, max_length, model):
# clean review
    line = clean_doc(review, vocab)
# encode and pad review
    padded = encode_docs(tokenizer, max_length, [line])
# predict sentiment
    yhat = model.predict(padded, verbose=0)
# retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'Depression'
    return percent_pos, 'No Depression'


@app.route("/", methods = ["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        text = request.form.get("testbox")
        return render_template("index.html"),
        output = backend.meters_feet(float(text)),
        user_text = text

if __name__ == "__main__":
    app.run()