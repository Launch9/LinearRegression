# Begin by importing all necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection, svm
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
import pickle
USING_PICKLE = False
def createLabels(df, directory):
    df["type"] = ""
    df["color"] = 0
    for index, row in df.iterrows():
        if(index == 0):
            continue
        if(index == len(df.index) - 1):
            break
        pct_change = (df.iloc[index]["Close"] - df.iloc[index - 1]["Close"]) / df.iloc[index - 1]["Close"] * 100.0
        print(pct_change)
        if(pct_change > 0.0):
            print("Rise")
            df.at[index, "type"] = 'Rise'
            df.at[index, "color"] = 2.0
        elif(pct_change < 0.0 and pct_change > -0.0):
            print("Stag")
            df.at[index, "type"] = 'Stag'
            df.at[index, "color"] = 1.0
        else:
            print("Fall")
            df.at[index, "type"] = 'Fall'
            df.at[index, "color"] = 0.0
    df.to_csv(directory)

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
#df = pd.read_csv("./test.csv")
df = pd.read_csv("./freshCSV/Coinbase_BTCUSD_1h.csv")
df2 = pd.read_csv("./freshCSV/Coinbase_ETHUSD_1h.csv")
df3 = pd.read_csv("./freshCSV/Coinbase_LTCUSD_1h.csv")

"""df = pd.read_csv("./CSV/Coinbase_BTCUSD_1h.csv")
df2 = pd.read_csv("./CSV/Coinbase_ETHUSD_1h.csv")
df3 = pd.read_csv("./CSV/Coinbase_LTCUSD_1h.csv")"""


createLabels(df, "./CSV/Coinbase_BTCUSD_1h.csv")
print("Finished 1")
createLabels(df2, "./CSV/Coinbase_ETHUSD_1h.csv")
print("Finsiehd 2")
createLabels(df3, "./CSV/Coinbase_LTCUSD_1h.csv")
print("Finsihed 3")



#Volume NEO,Volume BTC,type,color,HL_PCT,PCT_change

formatDate(df, '%Y-%m-%d %I-%p', "Date")
formatData(df)
formatDate(df2, '%Y-%m-%d %I-%p', "Date")
formatData(df2)
formatDate(df3, '%Y-%m-%d %I-%p', "Date")
formatData(df3)





#df.set_index("Date", inplace=True)

#end = pd.to_datetime('2019-07-17 01-AM', format='%Y-%m-%d %I-%p')
#start = pd.to_datetime('2017-07-01 11-AM', format='%Y-%m-%d %I-%p')



colorSer = df["color"]
df = df[['Close','HL_PCT','PCT_change','color']]
concatData(df,df2,"ETH-USD", "Volume ETH", "Volume USD")
concatData(df,df3,"LTC-USD", "Volume LTC", "Volume USD")




# Pandas ".iloc" expects row_indexer, column_indexer  
start = 0
end = 400
#df.fillna(value=-99999, inplace=True)
#forecast_out = int(math.ceil(1))#0.2 * len(df)))
X = df.drop(["color"], axis=1)
X = preprocessing.scale(X)
X_lately = X[start:end]
print(X_lately)
# Now let's tell the dataframe which column we want for the target/labels.  
y = df['color']
Y_lately = y[start:end]
#le = preprocessing.LabelEncoder()

#y = y.apply(le.fit_transform)
# Test size specifies how much of the data you want to set aside for the testing set. 
# Random_state parameter is just a random seed we can use.
# You can use it if you'd like to reproduce these specific results.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.01)


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
def algorize(X_train, y_train, X_test, y_test):
    if(not USING_PICKLE):
        mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=3000)
        mlp.fit(X_train, y_train.values.ravel())
        print("MLP score: " + str(mlp.score(X_test, y_test)))
        with open('./pickle/neural.pickle','wb') as f:
            pickle.dump(mlp, f)
        mlp_pred = mlp.predict(X_lately)
        SVC_model = svm.SVC()
        SVC_model.fit(X_train, y_train)
        SVC_pred = SVC_model.predict(X_lately)
        print("SVC_model score: " + SVC_model.score(X_test, y_test))
        with open('./pickle/SVC.pickle','wb') as f:
            pickle.dump(SVC_model, f)

        KNN_model = KNeighborsClassifier(n_neighbors=20)
        KNN_model.fit(X_train, y_train)
else:
    pickle_in = open('./neural.pickle','rb')
    mlp = pickle.load(pickle_in)


#predictions = mlp.predict(X_test)
#print(predictions)

# KNN model requires you to specify n_neighbors,
# the number of points the classifier will look at to determine what class a new point belongs to
#KNN_model = KNeighborsClassifier(n_neighbors=20)

#KNN_model.fit(X_train, y_train)
#SVC_prediction = SVC_model.predict(X_test)
#KNN_prediction = KNN_model.predict(X_test)

#print(prediction)
# Accuracy score is the simplest way to evaluate
#print(accuracy_score(prediction, Y_lately))
#print(accuracy_score(KNN_prediction, y_test))
# But Confusion Matrix and Classification Report give more details about performance
#print(confusion_matrix(SVC_prediction, y_test))
#print(classification_report(KNN_prediction, y_test))

#print(accuracy_score(Y_lately,prediction))
#print(classification_report(Y_lately,prediction))
#print(X_test)
df["Close"].plot()
plt.scatter(df[start:end].index, df[start:end]["Close"], c=prediction)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()
