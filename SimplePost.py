#A user submits text and this flask returns the tokens from that test

from flask import Flask, render_template, request, render_template_string
import nltk
nltk.download('punkt')


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('three_button_form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    coin = request.form.get('selectCoin')
    return render_template_string('If you invested [x dollars] in {{ what }}', what = coin)


if __name__ == "__main__":
    app.run(port=8004)