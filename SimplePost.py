#This flask asks users to select a coin, investment value, and investment date and will return the current worth

from flask import Flask, render_template, request, render_template_string



app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('three_button_form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    coin = request.form.get('selectCoin')
    investment = request.form.get('investValue')
    timeline = request.form.get('investDate')
    return render_template_string('If you invested {{ much }} US Dollars in {{ what }} on {{ when }} that would now be worth [x dollars]', what = coin, much = investment, when = timeline)


if __name__ == "__main__":
    app.run(port=8004)