import nltk
from nltk.corpus import stopwords
from keras.models import load_model
import pickle
import re
import string
from flask import Flask, render_template, request

app = Flask(__name__)


nltk.download('stopwords')


def load_doc(filename):
    '''Load the file and return the text of the given a filename'''
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
# prepare regex for char filtering
    re_punc = re.compile(' [%s] ' % re.escape(string.punctuation))
# remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
# remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
# filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
# filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# classify a review as negative or positive
def predict_sentiment():
    # get the review text from the form data
    review = request.form["review"]
    print("got review: " + review)
    print(str(request.form))

    vocab_filename = 'test.txt'
    vocab = load_doc(vocab_filename)
    vocab = set(vocab.split())

    print("Vocab loaded")

    model = load_model('model_ml.h5')

    print("Model loaded")

    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

    print("Tokenizer loaded")

    # clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to a line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='binary')

    yhat = model.predict(encoded, verbose=0)

    percent_pos = yhat[0, 0]
    # round to whole percentage
    percent_pos = round(percent_pos * 100)
    print("Predicted: %d%%" % percent_pos)
    if percent_pos == 0:
        # update html with id of result
        return 'No Depression'
    return 'Depression'


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        val = predict_sentiment()
        return render_template("index.html", data=val)


if __name__ == "__main__":
    app.run(debug=True)
