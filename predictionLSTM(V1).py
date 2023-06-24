import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime , timedelta

scaler = MinMaxScaler(feature_range=(0,1))


def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return dt.datetime(year=year, month=month, day=day)

def getStockHistory(stockTag , startDate):
    stock = yf.download(stockTag , start=startDate)
    return stock

def adjustStockData(stock):
    stock = pd.DataFrame(stock)
    stock.insert(loc=1,column='Date',value=stock.index)
    stock = stock[['Date','Close']]
    return stock

def visualizeStockHistory(stock):
   plt.plot(stock['Date'] , stock['Close'])
   plt.title("Stock Price History")
   plt.show()

def dataframeToMemorizedDataframe(stock , first_date_str , last_date_str , n=3):
  first_date = str_to_datetime(first_date_str)
  last_date  = str_to_datetime(last_date_str)

  target_date = first_date
  
  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = stock.loc[:target_date].tail(n+1)
    
    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = stock.loc[target_date:target_date+dt.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = dt.datetime(day=int(day), month=int(month), year=int(year))
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target'] = Y

  return ret_df


def normalizeData(stock):
  dataset = scaler.fit_transform(stock.filter(['Close']))
  stock['Close'] = dataset
  return stock

def denormalizeData(price):
  price = scaler.inverse_transform(price)
  return price

def memorizedDataFrameToDateXY(memorizedStock):
  dataframeAsNP = memorizedStock.to_numpy()

  dates = dataframeAsNP[:, 0]

  middle_matrix = dataframeAsNP[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
  
  Y = dataframeAsNP[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32) 

def showDataAfterSplitting(dates_train , y_train , dates_val , y_val , dates_test , y_test):
  plt.plot(dates_train, y_train)
  plt.plot(dates_val, y_val)
  plt.plot(dates_test, y_test)
  plt.legend(['Train', 'Validation', 'Test'])
  plt.title('Data after splitting')
  plt.show()


stock = getStockHistory(stockTag="AAPL" , startDate='2013-01-01')
stock = adjustStockData(stock=stock)
#visualizeStockHistory(stock=stock)
stock = normalizeData(stock=stock)

memorizedDF = dataframeToMemorizedDataframe (stock=stock , 
                                            first_date_str='2013-03-25' , 
                                            last_date_str= '2023-05-30' , 
                                            n=3)
print(memorizedDF)

dates, X, y = memorizedDataFrameToDateXY(memorizedStock=memorizedDF)

Q80 = int(len(dates) * .8)
Q90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:Q80], X[:Q80], y[:Q80]

dates_val, X_val, y_val = dates[Q80:Q90], X[Q80:Q90], y[Q80:Q90]
dates_test, X_test, y_test = dates[Q90:], X[Q90:], y[Q90:]


model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)),
                    layers.LSTM(64, return_sequences=False),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

train_predictions = model.predict(X_train)
train_predictions = scaler.inverse_transform(train_predictions)

y_train = pd.DataFrame(y_train)
y_train = scaler.inverse_transform(y_train)

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.legend(['Training Predictions', 'Training Observations']) 
plt.show()

val_predictions = model.predict(X_val)
val_predictions = scaler.inverse_transform(val_predictions)

y_val = pd.DataFrame(y_val)
y_val = scaler.inverse_transform(y_val)

plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.legend(['Validation Predictions', 'Validation Observations'])
plt.show()

test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)

y_test = pd.DataFrame(y_test)
y_test = scaler.inverse_transform(y_test)

results = pd.DataFrame(test_predictions)
results.columns = ['Predicted Price']
results.insert(loc=0,column='Date',value=dates_test)
results.insert(loc=1,column='Actual Price',value=y_test)

print(results)

plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Testing Predictions', 'Testing Observations'])
plt.show()

# plt.plot(dates_train, train_predictions)
# plt.plot(dates_train, y_train)
# plt.plot(dates_val, val_predictions)
# plt.plot(dates_val, y_val)
# plt.plot(dates_test, test_predictions)
# plt.plot(dates_test, y_test)
# plt.legend(['Training Predictions', 
#             'Training Observations',
#             'Validation Predictions', 
#             'Validation Observations',
#             'Testing Predictions', 
#             'Testing Observations'])
# plt.show()

