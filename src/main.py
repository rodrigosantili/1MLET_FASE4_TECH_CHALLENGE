import os
from dotenv import load_dotenv

from data import fetch_data, preprocess_data


load_dotenv()
YFINANCE_TICKER = os.getenv('YFINANCE_TICKER')
YFINANCE_PERIOD = os.getenv('YFINANCE_PERIOD')


def main():
    data = fetch_data(ticker=YFINANCE_TICKER,period=YFINANCE_PERIOD)
    print(f"Dataset:\n\n{data.head()}\n\n")

    scaled_data, scaler = preprocess_data(data)
    print(f"""
    {scaled_data} \n\n
    {scaler}
    """)


if __name__ == '__main__':
    main()
