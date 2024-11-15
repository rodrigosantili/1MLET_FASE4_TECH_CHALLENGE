import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

    # Calculate evaluation metrics
    train_mae = mean_absolute_error(actual_train, train_preds)
    test_mae = mean_absolute_error(actual_test, test_preds)
    train_rmse = mean_squared_error(actual_train, train_preds) ** 0.5  # Manual sqrt for RMSE
    test_rmse = mean_squared_error(actual_test, test_preds) ** 0.5     # Manual sqrt for RMSE
    train_mape = np.mean(np.abs((actual_train - train_preds) / actual_train)) * 100
    test_mape = np.mean(np.abs((actual_test - test_preds) / actual_test)) * 100

    # Log metrics to MLflow
    log_metrics({
        "train_mae": train_mae, "test_mae": test_mae,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "train_mape": train_mape, "test_mape": test_mape
    })

    return train_preds, test_preds, np.concatenate([actual_train, actual_test], axis=0)