import logging
import yfinance as yf

from datetime import datetime, timedelta

from pandas import DataFrame

from ..routers.interfaces import HistoricalDataService

logging.basicConfig(level=logging.INFO)


class HistoricalDataServiceImpl(HistoricalDataService):
    async def fetch_historical_data(self, ticker: str = "BTC-USD") -> DataFrame:
        """
        Fetches the last 60 days of closing prices for the given ticker.

        :param ticker: The ticker symbol (default: BTC-USD for Bitcoin).
        :return: A list of closing prices for the last 60 days.
        """
        logging.info(f"Fetching historical data for ticker '{ticker}' from Yahoo Finance.")
        try:
            # Define the date range: last 60 days
            end_date = datetime.today()
            start_date = end_date - timedelta(days=60)

            # Fetch data from Yahoo Finance
            return yf.download(ticker, start=start_date, end=end_date)
        except Exception as ex:
            logging.error(f"Error fetching data from Yahoo Finance: {ex}")
            raise
