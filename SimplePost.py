#A user submits text and this flask returns the tokens from that test

from flask import Flask, render_template, request, jsonify
import nltk
nltk.download('punkt')
from nltk import word_tokenize

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('three_button_form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    stuff = word_tokenize(text)
    return jsonify(stuff)


if __name__ == "__main__":
    app.run(port=8004)
