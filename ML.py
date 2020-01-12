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


def formatDate(dataFrame, date_format, attribute):
    dataFrame[attribute] = pd.to_datetime(dataFrame[attribute], format=date_format)
    dataFrame.set_index(attribute, inplace=True)

def formatData(dataFrame):
    dataFrame['HL_PCT'] = (dataFrame['High'] - dataFrame['Low']) / dataFrame['Close'] * 100.0
    dataFrame['PCT_change'] = (dataFrame['Close'] - dataFrame['Open']) / dataFrame['Open'] * 100.0

def concatData(dataFrame1, dataFrame2, label,at1,at2):
    dataFrame1['Close' + label] = dataFrame2["Close"]
    dataFrame1['HL_PCT' + label] = dataFrame2['HL_PCT']
    dataFrame1['PCT_change' + label] = dataFrame2['PCT_change']
    dataFrame1['Volume1' + label] = dataFrame2[at1]
    dataFrame1['Volume2' + label] = dataFrame2[at2]

def createGraph(forecast_col, df):

    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(20))#0.2 * len(df)))
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
    last_close = df.iloc[0][forecast_col]

    last_unix = last_date.timestamp()
    one_day = 60 * 60#86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 60 * 60
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    df['Forecast'] = df['Forecast'].rolling(window=1).mean()


    df = df.sort_values('Date')
    firstForecast = 0
    forecastShift = 0
    for i in range(forecast_out):
        row = df.iloc[-(forecast_out - i)]
        if(not math.isnan(row.Forecast)):
            forecastShift = i
         
            firstForecast = row.Forecast
           
            break


    print("LC: " + str(last_close))
    print("FF: " + str(firstForecast))
 
    forecastDif = last_close - firstForecast
    #df.iloc[-167].Forecast += 2000
   

    #df['Forecast'] += forecastDif
    
    return {"fd":forecastDif, "fs":forecastShift}

df = pd.read_csv("Bittrex__1h.csv")
df3 = pd.read_csv("Bittrex_ETHUSD_1h.csv")
df4 = pd.read_csv("Bittrex_ETHBTC_1h.csv")
df5 = pd.read_csv("Bittrex_LTCBTC_1h.csv")
df6 = pd.read_csv("Bittrex_NEOBTC_1h.csv")
df7 = pd.read_csv("Bittrex_WAVBTC_1h.csv")

formatDate(df, '%Y-%m-%d %I-%p', "Date")

formatDate(df3, '%Y-%m-%d %I-%p', "Date")
formatDate(df4, '%Y-%m-%d %I-%p', "Date")
formatDate(df5, '%Y-%m-%d %I-%p', "Date")
formatDate(df6, '%Y-%m-%d %I-%p', "Date")
formatDate(df7, '%Y-%m-%d %I-%p', "Date")
end = pd.to_datetime('2019-08-17', format='%Y-%m-%d')
start = pd.to_datetime('2018-07-01 11-AM', format='%Y-%m-%d %I-%p')
df = df[end:start]
df3 = df3[end:start]
df4 = df4[end:start]
df5 = df5[end:start]
df6 = df6[end:start]
df7 = df7[end:start]
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

trades = ["ETH-USD", "", "ETH-BTC", "LTC-BTC", "NEO-BTC", "WAV-BTC"]
results = []
defaultdf = df.copy()
for i in trades:
    
    df2 = pd.read_csv("Bittrex_" + i.replace('-', '') + "_1h.csv")
    formatDate(df2, '%Y-%m-%d %I-%p', "Date")
    print(i)
    fd = createGraph("Close" + i, df)
    df['Forecast'] += fd['fd']
    df['Forecast'] = df['Forecast'].shift(-fd['fs'])
    df[:-fd['fs']]
    df2["Close"].plot()
    df['Forecast'].plot()
    #df["Close_ETH"].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    df = defaultdf.copy()





#df['Close'].plot()



#plt.plot(df.index,df.Close)


