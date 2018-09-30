#This flask asks users to select a coin, investment value, and investment date and will return the current worth

from flask import Flask, render_template, request, render_template_string
from datetime import datetime
import time
import requests
import pandas as pd

app = Flask(__name__)

#Here I am setting up the template for the data entry page
@app.route('/')
def my_form():
    return render_template('three_button_form.html')

#Here I am defining the operations of the app and what the app will return
@app.route('/', methods=['GET' , 'POST'])
def my_form_post():
    coin = request.form.get('selectCoin')
    investment = request.form.get('investValue')
    timeline = request.form.get('investDate')

    #Here I compare these inputs agianst historical data
    # step 1, need to convert timeline to datetime to unix
    date_str = timeline
    format_str = "%Y-%m-%dT%H:%M:%S%z"
    addtime = 'T00:00:00-0000'
    comb_str = timeline + addtime
    datetime_obj = datetime.strptime(comb_str, format_str)
    unixtime = (time.mktime(datetime_obj.timetuple())) - 14400
    unixtime = int(str(unixtime)[:10])

    # Step 2, call the api to get previous and current values
    d = requests.get(
        "https://min-api.cryptocompare.com/data/histoday?fsym=" + coin + "&tsym=USD&limit=1000000000").json()
    df = pd.DataFrame.from_dict(d['Data'])
    df = df[['time', 'close', 'open']]

    # step 2, calc % change (current-original/original) then investment + (investment * % change) = investmentToday
    orig = df.loc[df['time'] == unixtime]
    today = df.tail(1)
    calc1 = int(today["open"])
    calc2 = int(orig["open"])
    pct = ((calc1 - calc2) / calc2)
    investmentToday = int(investment) + ((int(investment)) * pct)


    #for debugging purposes I will print out the inputs
    print(coin)
    print(investment)
    print(timeline)

    #Here I combine the inputs with the comparision calculations to respond to the user
    return render_template_string('If you invested {{ much }} US Dollars '
                                  'in {{ what }} on {{ when }} that would '
                                  'now be worth {{ moola }}', what = coin,
                                  much = investment, when = timeline, moola = investmentToday)

if __name__ == "__main__":
    app.run(port=8004)