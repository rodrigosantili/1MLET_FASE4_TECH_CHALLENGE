import os
from dotenv import load_dotenv

from data import fetch_data, preprocess_data

from utils.sequence_utils import create_sequences
from utils.tensor_utils import prepare_tensors
from utils.device_utils import get_device
from mlflow_setup import init_mlflow, log_params

load_dotenv()


def main():
    init_mlflow()

    params = {
        "YFINANCE_TICKER": os.getenv('YFINANCE_TICKER'),
        "YFINANCE_PERIOD": os.getenv('YFINANCE_PERIOD'),
        "FRAMEWORK": "pytorch",  # Can be "pytorch" or "keras"
        "SEQ_LENGTH": 60,
    }
    log_params(params)

    device = get_device(params["FRAMEWORK"])
    print(f"device {device}")

    data = fetch_data(ticker=params["YFINANCE_TICKER"],period=params["YFINANCE_PERIOD"])
    scaled_data, scaler = preprocess_data(data)

    sequences, targets = create_sequences(scaled_data, params["SEQ_LENGTH"])
    print(f"sequences {sequences}")
    print(f"targets {targets}")


if __name__ == '__main__':
    main()
