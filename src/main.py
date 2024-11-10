import os
from dotenv import load_dotenv

from data import fetch_data


load_dotenv()
YFINANCE_TICKER = os.getenv('YFINANCE_TICKER')
YFINANCE_PERIOD = os.getenv('YFINANCE_PERIOD')


def main():
    data = fetch_data(ticker=YFINANCE_TICKER,period=YFINANCE_PERIOD)
    print(data.head())


if __name__ == '__main__':
    main()
