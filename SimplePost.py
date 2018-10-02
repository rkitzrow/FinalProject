#This flask asks users to select a coin, investment value, and investment date and will return the current worth

from flask import Flask, render_template, request, render_template_string
import time
import calendar
import requests
from money import Money
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import datetime
import numpy as np
import seaborn as sns
import base64
import io
import matplotlib
matplotlib.use('Agg')

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
    format_str = "%Y-%m-%d"
    unixtime = calendar.timegm(time.strptime(timeline, "%Y-%m-%d"))

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
    investmentToday = str(round(investmentToday, 2))
    investmentToday = Money(investmentToday, 'USD')
    investmentToday.format('en_US')


    #for debugging purposes I will print out the inputs
    print(coin)
    print(investment)
    print(timeline)

    #Here I create the plot
    # step 1, need to convert timeline to datetime to unix
    date_str = timeline
    format_str = "%Y-%m-%d"
    unixtime = calendar.timegm(time.strptime(timeline, "%Y-%m-%d"))

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
    investmentToday = str(round(investmentToday, 2))
    investmentToday = Money(investmentToday, 'USD')
    investmentToday.format('en_US')
    print(investmentToday)

    # Next I will plot a graph of returns to show the daily returns from the selected date
    # Reduce the df to only dates on and after the selected date
    df_return = df[df['time'] >= unixtime]

    # select the original open price
    df_orig = df_return.iloc[0, 2].copy()

    # calculate daily returns based that open price
    df_return['orig'] = df_orig
    df_return['return'] = (df_return['open'] - df_return['orig']) / df_return['orig']
    df_return['return'] = df_return['return']*100

    # convert unix date back to standard date
    df_return['time'] = pd.to_datetime(df_return['time'], unit='s')


    # select only columns for graph
    return_plot = pd.DataFrame(df_return, columns=["time", "return"])

    # max return value for annotation
    ymax_index = int((np.argmax(return_plot['return'])))
    ymax = return_plot['return'].loc[ymax_index]
    xpos_max = return_plot['time'].loc[ymax_index]
    xmax = return_plot['time'].loc[ymax_index]

    # min return value for annotation
    ymin_index = int((np.argmin(return_plot['return'])))
    ymin = return_plot['return'].loc[ymin_index]
    xpos_min = return_plot['time'].loc[ymin_index]
    xmin = return_plot['time'].loc[ymin_index]

    # plot graph using seaborn
    img = io.BytesIO()
    plt.figure(figsize=(10,3))

    # annotation
    plt.annotate('Max Value on %s' % xmax, xy=(xpos_max, ymax),
                 xycoords='data',
                 xytext=(40, 5), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"),
                 horizontalalignment='left', verticalalignment='top')

    plt.annotate('Min Value on %s' % xmin, xy=(xpos_min, ymin),
                 xycoords='data',
                 xytext=(30, 60), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"),
                 horizontalalignment='left', verticalalignment='bottom')

    sns.lineplot(x='time', y='return', data=return_plot)
    plt.title("Return of Your Investment Over Time")
    plt.xlabel("Time of Investment")
    plt.ylabel("Return of Investment (%)")
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()



    #Here I combine the inputs with the comparision calculations to respond to the user
    return render_template('return_page.html', what=coin, much=investment, when=timeline, moola=investmentToday, graph_url=graph_url)



if __name__ == "__main__":
    app.run(port=8004)