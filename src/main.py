import os
from dotenv import load_dotenv

from data import fetch_data, preprocess_data

from mlflow_setup import init_mlflow, log_params


load_dotenv()


def main():
    init_mlflow()

    params = {
        "YFINANCE_TICKER": os.getenv('YFINANCE_TICKER'),
        "YFINANCE_PERIOD": os.getenv('YFINANCE_PERIOD')
    }
    log_params(params)

    data = fetch_data(ticker=params["YFINANCE_TICKER"],period=params["YFINANCE_PERIOD"])
    scaled_data, scaler = preprocess_data(data)


if __name__ == '__main__':
    main()
