#This flask asks users to select a coin, investment value, and investment date and will return the current worth
#This flask will also return a plot of the coin returns from the initial investment date to today

#I import all the packages and libraries needed for this app..
from flask import Flask, render_template, request, url_for, redirect
import time
import datetime
from datetime import datetime
import calendar
import requests
from money import Money
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import seaborn as sns
import base64
import io
import matplotlib
matplotlib.use('Agg')

# packages imported by Max (Gang Ping) Zhu
from cryptocmd import CmcScraper
from scipy import stats
import statsmodels.api as sm
from itertools import product
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

#I set the app name
app = Flask(__name__)


#Here I create an entry page that allows the user to identify their investment level
@app.route('/')
def my_start():
    # Look in the templates folder for this html page which includes the input fields
    return render_template('start_page.html')

#Landing page for a future investor
@app.route('/FutureInvestor', methods=['POST' , 'GET'])
def my_evaluation():
    # Look in the templates folder for this html page which includes the input fields
    return render_template('futureinvestor_page.html')

@app.route('/predictions', methods=['POST'])
def crypto_predict():
    # Create the variable for the chosen cryptocurrency
    coin = request.form.get('ChooseCoin')

    # initialise scraper without passing time interval
    scraper = CmcScraper(coin)

    # Pandas dataFrame for the same data
    df = scraper.get_dataframe()

    # Format the dates in the dataframe
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year

    # Update the format of the Volume column
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Provide the log returns using the close price in the dataframe
    # Use the value to provide a volatility (rolling standard deviation function).
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # We're using a window of 365 trading days as the cryptocurrency market doesn't open/close.
    df['volatility'] = df['log_ret'].rolling(365).std() * np.sqrt(365)
    df['dv'] = (df['Close'] * df['Volume'] / 1e6)[1:]
    df['lret'] = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    df['daily_illiq'] = np.abs(df['lret']) / df['dv']

    # Creating a variable to display the adjusted results
    df_display = df.head(5)

    # Aligning the date to the index in the dataframe
    df.index = df.Date
    df = df.resample('D').mean()  # Resampling to daily frequency for the cryptocurrency
    df_month = df.resample('M').mean()  # Resampling to monthly frequency for cryptocurrency

    # Creating the Box-Cox Transformations
    df_month['high_box'], pred_input = stats.boxcox(df_month.High)

    # Initial approximation of parameters
    Qs = range(0, 2)
    qs = range(0, 3)
    Ps = range(0, 3)
    ps = range(0, 3)
    D = 1
    d = 1
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    len(parameters_list)

    # Model Selection for BTC
    results = []
    best_aic = float("inf")
    for param in parameters_list:
        try:
            model = sm.tsa.statespace.SARIMAX(df_month.high_box, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
        except ValueError:
            print('wrong parameters:', param)
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    # Best Models
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']

    # Creating a function for the Inverse Box-Cox Transformation
    def invboxcox(y, pred_input):
        if pred_input == 0:
            return (np.exp(y))
        else:
            return (np.exp(np.log(pred_input * y + 1) / pred_input))

    # Creating the Prediction for BTC
    df_month2 = df_month[['High']]
    date_list = [datetime(2018, 6, 30),
                 datetime(2018, 7, 31),
                 datetime(2018, 8, 31),
                 datetime(2018, 9, 30),
                 datetime(2018, 10, 31),
                 datetime(2018, 11, 30),
                 datetime(2018, 12, 31),
                 datetime(2019, 1, 31),
                 datetime(2019, 2, 28),
                 datetime(2019, 3, 31),
                 datetime(2019, 4, 30),
                 datetime(2019, 5, 31),
                 datetime(2019, 6, 30),
                 datetime(2019, 7, 31)]

    future = pd.DataFrame(index=date_list, columns=df_month.columns)
    df_month2 = pd.concat([df_month2, future])
    df_month2['forecast'] = invboxcox(best_model.predict(start=0, end=75), pred_input)
    plt.figure(figsize=(15, 7))
    df_month2.High.plot()
    df_month2.forecast.plot(color='r', ls='--', label='predicted high')
    plt.legend()
    plt.title('Cryptocurrency Prediction, by months')
    plt.ylabel('mean USD')

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # The return below shows the latest results in a table format.
    return render_template('predictions.html', coin=coin, result=figdata_png.decode('utf8'), adjusted_results=df_display.to_html())

#Landing page for a current investor
@app.route('/CurrentInvestor', methods=['POST' , 'GET'])
def my_form():
    # Look in the templates folder for this html page which includes the input fields
    return render_template('three_button_form.html')

#Here I am defining the operations of the app and what the app will return. Is accepts post only right now
@app.route('/results', methods=['POST'])
def my_form_post():
    # Here I create variables that align to form inputs from the three_button_form.html page
    # I can then refer to these variables for return calculations and for use in the API
    coin = request.form.get('selectCoin')
    investment = request.form.get('investValue')
    timeline = request.form.get('investDate')

    #Now I need to change date format
    timeline2 = datetime.strptime(timeline, "%Y-%m-%d").strftime("%m/%d/%Y")

    #Here I need to change investment format

    # I set the investmetn to an integer
    investment2 = int(investment)

    # I then convert the integer to a string with two decimals
    investment2 = str(round(investment2, 2))

    # I turn the investmetn into USD money format in english
    investment2 = Money(investment2, 'USD')
    investment2.format('en_US')

    # Here I compare these inputs against historical data
    # I need to convert timeline to datetime to unix
    unixtime = calendar.timegm(time.strptime(timeline, "%Y-%m-%d"))

    # I need to  call the api to get previous and current values
    # As noted above, I use the dynamic variable of coin for the GET API call
    # I hardcode the currency to USE and return the max possible entries
    d = requests.get(
        "https://min-api.cryptocompare.com/data/histoday?fsym=" + coin + "&tsym=USD&limit=2000").json()

    #I pull the data from the dictinary
    df = pd.DataFrame.from_dict(d['Data'])

    #I don't want all the fields and select only time, close, and open values to make my return calculations
    df = df[['time', 'close', 'open']]

    #I need to calc % change (current-original/original) then investment + (investment * % change) = investmentToday

    # I select the origin date and identify it as unixtime
    orig = df.loc[df['time'] == unixtime]

    # I identify today's date which is the last date in the tale
    today = df.tail(1)

    #I make interim calculations for computing return (close-open)/open
    calc1 = float(today["open"])
    calc2 = float(orig["open"])
    pct = ((calc1 - calc2) / calc2)

    # I calculate the return and set to money format to USD and and language to english
    investmentToday = int(investment) + ((int(investment)) * pct)
    investmentToday = str(round(investmentToday, 2))
    investmentToday = Money(investmentToday, 'USD')
    investmentToday.format('en_US')


    #Here I create the plot
    # I need to convert timeline to datetime to unix
    unixtime = calendar.timegm(time.strptime(timeline, "%Y-%m-%d"))

    # I need to call the api to get previous and current values
    d = requests.get(
        "https://min-api.cryptocompare.com/data/histoday?fsym=" + coin + "&tsym=USD&limit=2000").json()
    df = pd.DataFrame.from_dict(d['Data'])
    df = df[['time', 'close', 'open']]

    # I need to calc % change (current-original/original) then investment + (investment * % change) = investmentToday
    orig = df.loc[df['time'] == unixtime]
    today = df.tail(1)
    calc1 = float(today["open"])
    calc2 = float(orig["open"])
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

    # separate out the parts of the date
    #df_return['time'] = df_return['time'].dt.year

    # select only columns for graph
    return_plot = pd.DataFrame(df_return, columns=["time", "return"])

    # max return value for annotation
    ymax_index = int((np.argmax(return_plot['return'])))
    ymax = return_plot['return'].loc[ymax_index]
    xpos_max = return_plot['time'].loc[ymax_index]
    xmax = return_plot['time'].loc[ymax_index]
    xmax = datetime.strftime(xmax, "%m/%d/%Y")


    # min return value for annotation
    ymin_index = int((np.argmin(return_plot['return'])))
    ymin = return_plot['return'].loc[ymin_index]
    xpos_min = return_plot['time'].loc[ymin_index]
    xmin = return_plot['time'].loc[ymin_index]
    xmin = datetime.strftime(xmin, "%m/%d/%Y")

    # plot graph using seaborn
    # this is required so we can return the image on the return_page.html
    img = io.BytesIO()
    plt.figure(figsize=(10,3))

    # I annotate the graph with an arrow and statement of max and min value
    plt.annotate('Max Value on\n%s' % xmax, xy=(xpos_max, ymax),
                 xycoords='data',
                 xytext=(40, 5), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"),
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=10, fontweight="bold", color='r')

    plt.annotate('Min Value on\n%s' % xmin, xy=(xpos_min, ymin),
                 xycoords='data',
                 xytext=(30, 60), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"),
                 horizontalalignment='left', verticalalignment='bottom',
                 fontsize=10, fontweight="bold", color='r')

    # I return the plot and add my x and y labels and title
    sns.lineplot(x='time', y='return', data=return_plot)
    plt.title("Return of Your Investment Over Time")
    plt.xlabel("Time of Investment")
    plt.ylabel("Return of Investment (%)")
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.clf()
    plt.cla()
    plt.close('all')

    #Here I combine the inputs with the comparision calculations to respond to the user
    return render_template('return_page.html', what=coin, much=investment2, when=timeline2, moola=investmentToday, graph_url=graph_url)

#Help and FAQ page
@app.route('/Help', methods=['GET'])
def my_help():
    # Look in the templates folder
    return render_template('help.html')

#I created 404 and 500 errors (although using the same html message)
@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    # If we get a 500 error, I am returning the same generic error message as the 400 error
    return render_template('404.html'), 500

# Issue:
# When running this locally on windows, an error will prevent a second call.
# This error is specifc to windows and does not appear when hosted on linux
# On aws linux instance the error does not appear and unlimited calls can be made



# Set port, host, and debug status for the flask app
if __name__ == "__main__":
    app.run(debug=False, port=8020)
