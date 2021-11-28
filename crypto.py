import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pandas_ta
from sklearn import preprocessing, model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import datetime
# load data
#df = pd.read_csv('currency_daily_BTC_CNY.csv')
df = pd.read_csv('BTC-USD.csv')


# summary statistics
print(df.describe())

# Reindex data using DateTimeIndex
df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)

# keep only Adj Close
df = df[['Adj Close']]

# inspect data
#print(df)

#print(df.info())

forecast_col = 'Adj Close'

#print(df.info())
# inspect data
#print(df)


# Drop the first n-rows
df = df.iloc[10:]
# View our newly-formed dataset
print(df.head(10))
# 10 percent of data to predict
forecast_out = int(math.ceil(0.05*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)

#features
X = np.array(df.drop(['label'],1))
# label

X = X[:-forecast_out]

X = preprocessing.scale(X)
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

# Split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)

clf = LinearRegression(n_jobs=-1)
#clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

# Printout relevant metrics

print("Accuracy: ", accuracy)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
#sec in day
one_day = 86400
#predict next day from last timestamp
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


style.use('ggplot')
df['Adj Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

print(df)