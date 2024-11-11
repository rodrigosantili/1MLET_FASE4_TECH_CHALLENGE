import os
from dotenv import load_dotenv

from data import fetch_data, preprocess_data

import matplotlib.pyplot as plt

import torch
from models.model_pytorch import StockLSTM, train_model
from predict.predict_pytorch import evaluate_model, plot_results, future_predictions

from utils.sequence_utils import create_sequences
from utils.tensor_utils import prepare_tensors_pytorch
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
        "EPOCHS": 100,
        "LEARNING_RATE": 0.001,
        "HIDDEN_LAYER_SIZE": 50,
        "future_days": 10
    }
    log_params(params)

    device = get_device(params["FRAMEWORK"])

    data = fetch_data(ticker=params["YFINANCE_TICKER"],period=params["YFINANCE_PERIOD"])
    scaled_data, scaler = preprocess_data(data)

    # Create sequences and prepare tensors for PyTorch
    if params["FRAMEWORK"] == "pytorch":
        sequences, targets = create_sequences(scaled_data, params["SEQ_LENGTH"])
        train_size = int(0.8 * len(sequences))
        X_train, y_train = prepare_tensors_pytorch(sequences[:train_size], targets[:train_size], device)
        X_test, y_test = prepare_tensors_pytorch(sequences[train_size:], targets[train_size:], device)

        # Initialize and train the PyTorch model
        model = StockLSTM(input_size=1, hidden_layer_size=params["HIDDEN_LAYER_SIZE"], output_size=1).to(device)
        model = train_model(model, X_train, y_train, epochs=params["EPOCHS"], lr=params["LEARNING_RATE"])

        # Evaluate and plot results
        train_preds, test_preds, actual = evaluate_model(model, X_train, y_train, X_test, y_test, scaler)
        plot_results(actual, train_preds, test_preds)

        # Future predictions
        last_sequence = torch.tensor(scaled_data[-params["SEQ_LENGTH"]:], dtype=torch.float32).unsqueeze(0).unsqueeze(
            -1).to(device)
        future_preds = future_predictions(model, last_sequence, params["future_days"], scaler)

        # Plot future predictions
        plt.figure(figsize=(12, 6))
        plt.plot(scaler.inverse_transform(scaled_data), label="Historical Prices")
        plt.plot(range(len(scaled_data), len(scaled_data) + params["future_days"]), future_preds,
                 label="Future Predictions", color="red")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
