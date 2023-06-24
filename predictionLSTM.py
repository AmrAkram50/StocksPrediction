import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime , timedelta

scaler = MinMaxScaler(feature_range=(0,1))

current_date = datetime.now().date()
yesterday = current_date - timedelta(days=1)
if yesterday.weekday() >= 5:
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)
formatted_date = yesterday.strftime("%Y-%m-%d")
prediction_dates = []
end_date = current_date + timedelta(days=15)

def getPredictionDates():
  global current_date
  global prediction_dates
  prediction_dates = []
  today = current_date
  while today <= end_date:
    if today.weekday() < 5: 
        formattedDate = today.strftime("%Y-%m-%d")
        prediction_dates.append(formattedDate)
    today += timedelta(days=1)

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return dt.datetime(year=year, month=month, day=day)

def getStockHistory(stockTag , startDate):
    stock = yf.download(stockTag , start=startDate)
    global formatted_date
    print(stock)
    #formatted_date= stock
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


def getPredictedPrices(stockName):
  getPredictionDates()
  stock = getStockHistory(stockTag=stockName , startDate='2019-01-01')
  stock = adjustStockData(stock=stock)
  #visualizeStockHistory(stock=stock)
  stock = normalizeData(stock=stock)

  memorizedDF = dataframeToMemorizedDataframe (stock=stock , 
                                              first_date_str='2019-03-25' , 
                                              last_date_str= formatted_date , 
                                              n=3)
  print(memorizedDF)

  dates, X, y = memorizedDataFrameToDateXY(memorizedStock=memorizedDF)


  model = Sequential([Input((3, 1)),
                      LSTM(units=50, return_sequences=True, input_shape= (X.shape[1], 1)),
                      Dropout(0.2),
                      LSTM(units = 50, return_sequences = True),
                      Dropout(0.2),
                      LSTM(units = 50, return_sequences = True),
                      Dropout(0.2),
                      LSTM(units = 50, return_sequences = True),
                      Dropout(0.2),
                      LSTM(units = 50),
                      Dropout(0.2),
                      Dense(units = 1)])
  
  

  model.compile(loss='mean_squared_error', 
                optimizer='adam',
                )

  model.fit(X, y, epochs=50 , batch_size=32)

  data = {
     'Date':prediction_dates[0],
     'Day-3':X[-1,1],
     'Day-2':X[-1,2],
     'Day-1':y[-1]
  }

  resultDataframe = pd.DataFrame(data)
  toPredict = resultDataframe.iloc[-1,[1,2,3]].to_numpy()
  predictionmatrix = toPredict.reshape((1,toPredict.shape[0],1))
  print(resultDataframe)
  sixMonthsPrediction = model.predict(predictionmatrix.astype(np.float32))
  resultDataframe['Price'] = sixMonthsPrediction

  for i in range(1,len(prediction_dates)):
    newRow = pd.DataFrame({
       'Date':prediction_dates[i],
       'Day-3':resultDataframe.iloc[-1,2],
       'Day-2':resultDataframe.iloc[-1,3],
       'Day-1':resultDataframe.iloc[-1,4]
    },index=[prediction_dates[i]])
    toPredict = newRow.iloc[-1,[1,2,3]].to_numpy()
    predictionmatrix = toPredict.reshape((1,toPredict.shape[0],1))
    sixMonthsPrediction = model.predict(predictionmatrix.astype(np.float32))
    newRow['Price'] = sixMonthsPrediction
    resultDataframe = pd.concat([resultDataframe , newRow])
  
  print(resultDataframe)  
  modelPredictions = resultDataframe['Price']
  modelPredictions = scaler.inverse_transform(modelPredictions.to_frame())
  #plt.plot(prediction_dates,modelPredictions)
  #plt.legend(['15 Days Predictions'])
  #plt.show()

  result = pd.DataFrame({'Date':prediction_dates,'Price':modelPredictions.tolist()})
  result['Price'] = result['Price'].astype(str)
  return result

