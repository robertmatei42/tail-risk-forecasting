import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import warnings

warnings.filterwarnings('ignore')

def clean_download(ticker, start):
    data = yf.download(tickers=ticker, start=start)
    if ticker[0].isalpha()==False: ticker=ticker[1:]
    data=data.droplevel(level=1, axis=1)
    data=data.drop(columns='Volume')
    data=data[['Open', 'High', 'Low', 'Close']]
    data=data.reset_index()

    return data


df=clean_download('AAPL', '2020-01-01')

print(df)