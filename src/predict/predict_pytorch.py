import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mlflow_setup import log_metrics


def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).cpu().numpy()
        test_preds = model(X_test).cpu().numpy()

    # Inverse transform the predictions
    train_preds = scaler.inverse_transform(train_preds)
    test_preds = scaler.inverse_transform(test_preds)

    # Inverse transform the actual values for comparison
    actual_train = scaler.inverse_transform(y_train.cpu().numpy())
    actual_test = scaler.inverse_transform(y_test.cpu().numpy())

    # Calculate the MSE for train and test
    train_mse = mean_squared_error(actual_train, train_preds)
    test_mse = mean_squared_error(actual_test, test_preds)

    # Log the metrics to MLflow
    log_metrics({"train_mse": train_mse, "test_mse": test_mse})

    return train_preds, test_preds, np.concatenate([actual_train, actual_test], axis=0)

# def plot_results(actual, train_preds, test_preds):
#     plt.figure(figsize=(12, 6))
#     plt.plot(actual, label="Actual Prices")
#     plt.plot(np.arange(len(train_preds)), train_preds, label="Training Predictions")
#     plt.plot(np.arange(len(train_preds), len(train_preds) + len(test_preds)), test_preds, label="Testing Predictions")
#     plt.legend()
#     plt.show()


def future_predictions(model, last_sequence, future_days, scaler):
    """
    Generate future predictions using the trained model.

    Parameters:
        model (torch.nn.Module): The trained LSTM model.
        last_sequence (torch.Tensor): The last sequence of data points to use as input for predictions.
        future_days (int): The number of days to predict into the future.
        scaler (scaler object): The scaler used for inverse transformation.

    Returns:
        np.array: Future predictions, inverse transformed to the original scale.
    """
    predictions = []
    model.eval()

    # Garantir que last_sequence tenha exatamente 3 dimensões: (1, sequence_length, 1)
    last_sequence = last_sequence.view(1, -1, 1)  # (1, sequence_length, 1)

    with torch.no_grad():
        for _ in range(future_days):
            # Fazer a previsão
            future_pred = model(last_sequence)

            # Extrair apenas o último valor do vetor de previsões como a próxima previsão
            next_pred = future_pred[:, -1, 0] if future_pred.dim() == 3 else future_pred[-1, 0]

            # Armazenar a previsão como um escalar
            predictions.append(next_pred.item())

            # Atualizar `last_sequence` para manter o comprimento correto
            # Remover o primeiro elemento e adicionar `next_pred` no final para manter o comprimento constante
            last_sequence = torch.cat((last_sequence[:, 1:, :], next_pred.view(1, 1, 1)), dim=1)

    # Transformar as previsões de volta para a escala original
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))