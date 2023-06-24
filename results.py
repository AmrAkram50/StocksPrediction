import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


dataApple = pd.read_csv("AAPL.csv")
dataAmazon = pd.read_csv("AMZN.csv")
dataMicrosoft = pd.read_csv("MSFT.csv")
dataTesla = pd.read_csv("TSLA.csv")

def calc(data):
    actual = np.array(data['Actual Price'])

    predicted = np.array(data['Predicted Price'])

    mse = np.sqrt(mean_squared_error(actual, predicted))
    mae = np.mean(np.abs(predicted - actual))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)


    return mse , mae , mape , r2

APmse , APmae , APmape , APr2 = calc(data=dataApple)
AMmse , AMmae , AMmape , AMr2 = calc(data=dataAmazon)
MSmse , MSmae , MSmape , MSr2 = calc(data=dataMicrosoft)
TSmse , TSmae , TSmape , TSr2 = calc(data=dataTesla)

result = {'Apple': [APmse, APmae , APmape , APr2],
          'Amazon': [AMmse, AMmae , AMmape , AMr2],
          'Microsoft':[MSmse, MSmae , MSmape , MSr2],
          'Tesla':[TSmse, TSmae , TSmape , TSr2]}
df = pd.DataFrame(result)
df.index = ['Root Mean Square Error','Mean Absolute Error' , 'Mean Absolute Percentage Error' , 'R2-Score'] # type: ignore
print(df)
df.to_csv('Statistical Results.csv')