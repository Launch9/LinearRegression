import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import math
import datetime
import pickle
IS_USING_PICKLE = False
print("Hello world!")
#Loading csv files
df = pd.read_csv("Bittrex_BTCUSD_1h.csv")
df2 = pd.read_csv("Bittrex_BTCUSD_1h copy.csv")
df3 = pd.read_csv("Bittrex_ETHUSD_1h.csv")
df4 = pd.read_csv("Bittrex_ETHBTC_1h.csv")
df5 = pd.read_csv("Bittrex_LTCBTC_1h.csv")
df6 = pd.read_csv("Bittrex_NEOBTC_1h.csv")
df7 = pd.read_csv("Bittrex_WAVESBTC_1h.csv")

def formatDate(dataFrame, date_format, attribute):
    dataFrame[attribute] = pd.to_datetime(dataFrame[attribute], format=date_format)
    dataFrame.set_index(attribute, inplace=True)

def formatData(dataFrame):
    dataFrame['HL_PCT'] = (dataFrame['High'] - dataFrame['Low']) / dataFrame['Close'] * 100.0
    dataFrame['PCT_change'] = (dataFrame['Close'] - dataFrame['Open']) / dataFrame['Open'] * 100.0

def concatData(dataFrame1, dataFrame2, label,at1,at2):
    dataFrame1['Close#' + label] = dataFrame2["Close"]
    dataFrame1['HL_PCT#' + label] = dataFrame2['HL_PCT']
    dataFrame1['PCT_change#' + label] = dataFrame2['PCT_change']
    dataFrame1['Volume1#' + label] = dataFrame2[at1]
    dataFrame1['Volume2#' + label] = dataFrame2[at2]
formatDate(df, '%Y-%m-%d %I-%p', "Date")
formatDate(df2, '%Y-%m-%d %I-%p', "Date")
formatDate(df3, '%Y-%m-%d %I-%p', "Date")
formatDate(df4, '%Y-%m-%d %I-%p', "Date")
formatDate(df5, '%Y-%m-%d %I-%p', "Date")
formatDate(df6, '%Y-%m-%d %I-%p', "Date")
formatDate(df7, '%Y-%m-%d %I-%p', "Date")

formatData(df)
formatData(df3)
formatData(df4)
formatData(df5)
formatData(df6)
formatData(df7)
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume USD', "Volume BTC"]]

concatData(df,df3,"ETH-USD", "Volume ETH", "Volume USD")
concatData(df,df4,"ETH-BTC", "Volume ETH", "Volume BTC")
concatData(df,df5,"LTC-BTC", "Volume LTC", "Volume BTC")
concatData(df,df6,"NEO-BTC", "Volume NEO", "Volume BTC")
concatData(df,df7,"WAV-BTC", "Volume WAV", "Volume ESBTC")


forecast_col = 'Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(200))#0.2 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])
if(not IS_USING_PICKLE):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    with open('./linearregression.pickle','wb') as f:
        pickle.dump(clf, f)
    print("Confidence: " + str(confidence))
else:
    pickle_in = open('./linearregression.pickle','rb')
    clf = pickle.load(pickle_in)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[0].name
last_close = df.iloc[0].Close
print("Printing last date!")
print(last_date)
print("Done printing last date")
last_unix = last_date.timestamp()
one_day = 60 * 60#86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 60 * 60
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


#df["Prediction"] += 5





df['Forecast'] = df['Forecast'].rolling(window=30).mean()
forecastDif = last_close - df.iloc[-150].Forecast
print("ForecastDif: " + str(forecastDif))
df.Forecast += forecastDif
print(df.iloc[-496])

#df['Close'].plot()

df2["Close"].plot()
df['Forecast'].plot()
#df["Close_ETH"].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

#plt.plot(df.index,df.Close)

plt.show()
