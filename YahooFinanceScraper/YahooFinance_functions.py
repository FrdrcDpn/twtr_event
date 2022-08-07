import pandas_datareader as pdr
import datetime as dt
import yfinance

def Getfinancedata(start, end, ticker):
    data = yfinance.download(tickers=ticker, start=start, end=end, interval="1d")
   # data = pdr.get_data_yahoo(ticker, start, end, interval='1m')
    return data
