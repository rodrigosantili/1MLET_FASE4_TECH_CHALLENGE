import yfinance
import pandas


def fetch_data(ticker, period):
    data = yfinance.download(tickers=ticker, period=period)
    return data
