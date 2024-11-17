from data import fetch_data, preprocess_data
import numpy as np
import torch
from models.model_pytorch import StockLSTM, train_model
from models.save_model import save_model_local
from predict.predict_pytorch import future_predictions
from predict.evaluate_model import evaluate_model

from utils.plot_utils import (
    plot_all, plot_future_predictions, plot_results,
    plot_residuals, plot_residual_distribution,
    plot_train_test_predictions, plot_confidence_interval,
    plot_autocorrelation, plot_historical_and_future
)
from utils.sequence_utils import create_sequences
from utils.tensor_utils import prepare_tensors_pytorch
from utils.device_utils import get_device
from mlflow_setup import init_mlflow, log_params


def main():
    # Initialize MLflow tracking
    init_mlflow()

    # Set model parameters and hyperparameters
    params = {
        "yfinance_ticker": "BTC-USD",   # Ticker symbol for Bitcoin in Yahoo Finance
        "yfinance_period": "max",       # Maximum period available for data collection
        "framework": "pytorch",         # Model framework choice (can be "pytorch" or "keras")
        "seq_length": 20,               # Sequence length for LSTM input
        "epochs": 300,                  # Number of training epochs
        "learning_rate": 0.008,         # Learning rate for model training
        "hidden_layer_size": 140,       # Size of the hidden layer in the LSTM
        "future_days": 15               # Number of days to predict into the future
    }

    # Log parameters to MLflow for tracking experiment configurations
    log_params(params)

    # Determine and set device based on the framework (CPU or GPU)
    device = get_device(params["framework"])

    # Fetch and preprocess data
    data = fetch_data(ticker=params["yfinance_ticker"], period=params["yfinance_period"])
    scaled_data, scaler = preprocess_data(data)

    # Model framework-specific setup
    if params["framework"] == "pytorch":
        # Create input sequences and targets for training
        sequences, targets = create_sequences(scaled_data, params["seq_length"])
        train_size = int(0.8 * len(sequences))  # Split data into 80% training and 20% testing

        # Prepare PyTorch tensors for training and testing sets
        X_train, y_train = prepare_tensors_pytorch(sequences[:train_size], targets[:train_size], device)
        X_test, y_test = prepare_tensors_pytorch(sequences[train_size:], targets[train_size:], device)

        # Initialize and train the PyTorch LSTM model
        model = StockLSTM(input_size=1, hidden_layer_size=params["hidden_layer_size"], output_size=1).to(device)
        model = train_model(model, X_train, y_train, epochs=params["epochs"], lr=params["learning_rate"])

        # Save the trained model locally
        save_model_local(model, path=r'src\models\saved\trained_model.pth')

        # Evaluate the model on both training and testing sets
        train_preds, test_preds, actual = evaluate_model(model, X_train, y_train, X_test, y_test, scaler)

        # Plot the results
        plot_results(actual, train_preds, test_preds)  # Actual, training, and testing predictions

        # Residuals and additional plots
        residuals = actual - np.concatenate((train_preds, test_preds))
        plot_residuals(actual, np.concatenate((train_preds, test_preds)))  # Residual plot
        plot_residual_distribution(residuals)  # Distribution of residuals
        plot_train_test_predictions(actual, train_preds, test_preds)  # Train vs. test predictions
        plot_confidence_interval(actual, np.concatenate((train_preds, test_preds)), residuals)  # Confidence interval
        plot_autocorrelation(residuals)  # Autocorrelation of residuals

        # Prepare the last sequence for future predictions
        last_sequence = torch.tensor(scaled_data[-params["seq_length"]:], dtype=torch.float32).unsqueeze(0).unsqueeze(
            -1).to(device)
        future_preds = future_predictions(model, last_sequence, params["future_days"], scaler)  # Generate future predictions

        # Plot the future predictions alongside the historical data
        plot_future_predictions(scaled_data, future_preds, scaler, future_days=params["future_days"])
        plot_historical_and_future(actual, future_preds)  # Historical data and future predictions

        # Plot all predictions in a single chart for a complete overview
        plot_all(actual, train_preds, test_preds, future_preds, seq_length=params["seq_length"],
                 future_days=params["future_days"])


# Execute main function if script is run directly
if __name__ == '__main__':
    main()
