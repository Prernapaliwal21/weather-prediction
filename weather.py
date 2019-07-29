import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#  read file
df = pd.read_csv("weather.csv")
df.dropna(inplace=True)

#  train model
r = df[['RainToday']]
Tommorow = df[['RainTomorrow']]
weather_current = df.drop(
    ['RainToday', 'RISK_MM', 'RainTomorrow', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
     'Evaporation', 'Sunshine', 'WindSpeed9am', 'Humidity9am', 'Pressure9am', 'Cloud9am', 'Temp9am'], axis=1)
rainT = {'Yes': 1, 'No': 0}

r.RainToday = [rainT[item] for item in r.RainToday]
Tommorow.RainTomorrow = [rainT[item] for item in Tommorow.RainTomorrow]

X,x,Y,y = train_test_split(weather_current,r, test_size = 0.5, random_state = 0)

XX,xx,YY,yy = train_test_split(weather_current,Tommorow, test_size = 0.5, random_state = 0)


logModel = LogisticRegression()
lr = logModel.fit(X,Y)
# rain tommorow
gr = logModel.fit(XX,YY)
lr.predict(x)


from sklearn.externals import joblib
joblib.dump(lr, 'rain_today.pkl')


joblib.dump(gr, 'rain_tomorrow.pkl')


today_columns = list(X.columns)
joblib.dump(today_columns, 'today_columns.pkl')
print("Models columns dumped!")

tomorrow_columns = list(XX.columns)
joblib.dump(tomorrow_columns, 'tomorrow_columns.pkl')
print("Models columns dumped!")