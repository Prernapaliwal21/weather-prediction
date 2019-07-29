
from flask import Flask, request,jsonify, render_template
from sklearn.externals import joblib
import pandas as pd
import traceback
import requests
from werkzeug.wsgi import DispatcherMiddleware

from hello import app as app2
import dash
import dash_html_components as html
import dash_core_components as dcc
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import numpy as np
import json
import sys

# import json
# import requests


server = Flask(__name__)
app = dash.Dash()
def start(df):
    N = 40
    # x = df['temp']
    # y = np.random.randn(N)
    # df = pd.DataFrame({'x': x, 'y': y})  # creating a sample dataframe
    print(type(df.items()))

    data = [
        go.Bar(
            x =list(df.keys()),  # assign x as the dataframe column 'x'
            y =list(df.values())
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@server.route("/")
def hello():

   return render_template("index.html")

@server.route("/app")
def dash_chart():
    bar = start()
    return render_template("show.html",plot=bar)




@server.route("/index", methods=["GET","POST"])
def predict():
    lr = joblib.load("rain_today.pkl")  # Load "rain_today.pkl"
    today_columns = joblib.load("today_columns.pkl")  # Load "today_columns.pkl"

    gr = joblib.load("rain_tomorrow.pkl")  # Load "rain_tomorrow.pkl"
    tomorrow_columns = joblib.load("tomorrow_columns.pkl")  # Load "tomorrow_columns.pkl"

    if lr:
        try:
            # json_ = request.form

            city = request.form['city']
            # json_= json_.to_dict(flat=False)
            api_key = "3b53b34a3ccf134df29d33f0e6db2f1b"
            r = requests.get('http://api.openweathermap.org/data/2.5/weather?q=' + city + '&APPID=' + api_key)
            data = (r.json())

            s = []


            s.append(data['main']['temp_min'])
            s.append(data['main']['temp_max'])
            s.append(data['sys']['message'])
            s.append(data['wind']['speed'])
            s.append(data['main']['humidity'])
            s.append(data['main']['pressure'])
            s.append(data['clouds']['all'])
            s.append(data['main']['temp'])

            print(s)
            percent = (data['sys']['message'])*10000
            k = ['temp_min', 'temp_max', 'rain', 'wind', 'humidity', 'pressure', 'clouds', 'temp']

            df = dict(zip(k, s))
            # print(json(df))
            # json_ = pd.DataFrame(df)
            # print(json_)
            query = pd.get_dummies(df)
            print(query)
            query_today = query.reindex(columns=today_columns, fill_value=0)

            print(query_today)
            query_tomorrow = query.reindex(columns=tomorrow_columns, fill_value=0)

            prediction = list(lr.predict(query_today))
            predict = list(lr.predict(query_tomorrow))
            # result = jsonify({'prediction': str(prediction)})
            today = (prediction)
            tomorrow = (predict)

            bar = start(df)

            return render_template("show.html", TODAY= today[0] , TOMORROW = tomorrow[0],temp =data['main']['temp'],
                                   pressure = data['main']['humidity'],percentage=percent,plot =bar)
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')



if __name__ == '__main__':
    # try:
    #     port = int(sys.argv[1]) # This is for a command-line input
    # except:
    #     port = 12345 # If you don't provide any port the port will be set to 12345

    # lr = joblib.load("model.pkl") # Load "model.pkl"
    # print ('Model loaded')
    # model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    # print ('Model columns loaded')
    # port=port,
    server.run( debug=True)
