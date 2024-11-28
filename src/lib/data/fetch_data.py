import yfinance as yf
import pandas as pd


def fetch_yfinance_data(ticker, period):
    """
    Fetches financial asset data using yfinance, with error handling.

    Parameters:
        ticker (str): Financial asset symbol.
        period (str): Data collection period (e.g., '1mo', '1y', '10y').

    Returns:
        pd.DataFrame: DataFrame with historical data containing the 'Close' column.
    """
    try:
        data = yf.download(tickers=ticker, period=period)
        if data.empty:
            raise ValueError(f"No data was downloaded for ticker '{ticker}' in the period '{period}'.")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data: {e}")
