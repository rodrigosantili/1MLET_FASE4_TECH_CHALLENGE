# import torch
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.metrics import root_mean_squared_error  # Use esta função se disponível
# from mlflow_setup import log_metrics
#
# def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
#     model.eval()  # Coloca o modelo em modo de avaliação
#     with torch.no_grad():
#         # Certifica se os dados estão no formato correto (torch.Tensor)
#         X_train = X_train.clone().detach().to(next(model.parameters()).device)
#         y_train = y_train.clone().detach().to(next(model.parameters()).device)
#         X_test = X_test.clone().detach().to(next(model.parameters()).device)
#         y_test = y_test.clone().detach().to(next(model.parameters()).device)
#
#         # Previsões
#         train_preds = model(X_train).cpu().numpy()
#         test_preds = model(X_test).cpu().numpy()
#
#     # Transformação inversa para escala original
#     train_preds = scaler.inverse_transform(train_preds)
#     test_preds = scaler.inverse_transform(test_preds)
#
#     # Transformação inversa dos valores reais
#     actual_train = scaler.inverse_transform(y_train.cpu().numpy())
#     actual_test = scaler.inverse_transform(y_test.cpu().numpy())
#
#     # Cálculo de métricas
#     train_mae = mean_absolute_error(actual_train, train_preds)
#     test_mae = mean_absolute_error(actual_test, test_preds)
#
#     # Usar root_mean_squared_error se disponível
#     try:
#         train_rmse = root_mean_squared_error(actual_train, train_preds)
#         test_rmse = root_mean_squared_error(actual_test, test_preds)
#     except ImportError:  # Calcular manualmente
#         train_rmse = np.sqrt(mean_squared_error(actual_train, train_preds))
#         test_rmse = np.sqrt(mean_squared_error(actual_test, test_preds))
#
#     train_mape = np.mean(np.abs((actual_train - train_preds) / actual_train)) * 100
#     test_mape = np.mean(np.abs((actual_test - test_preds) / actual_test)) * 100
#
#     # Registro no MLflow
#     log_metrics({
#         "train_mae": train_mae, "test_mae": test_mae,
#         "train_rmse": train_rmse, "test_rmse": test_rmse,
#         "train_mape": train_mape, "test_mape": test_mape
#     })
#
#     return train_preds, test_preds, np.concatenate([actual_train, actual_test], axis=0)
#


import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow_setup import log_metrics

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    model.eval()  # Evaluation mode
    with torch.no_grad():
        # Ensure tensors are moved to the correct device
        X_train = X_train.clone().detach().to(next(model.parameters()).device)
        y_train = y_train.clone().detach().to(next(model.parameters()).device)
        X_test = X_test.clone().detach().to(next(model.parameters()).device)
        y_test = y_test.clone().detach().to(next(model.parameters()).device)

        # Predictions
        train_preds = model(X_train).cpu().numpy()
        test_preds = model(X_test).cpu().numpy()

    # Inverse transform
    train_preds = scaler.inverse_transform(train_preds)
    test_preds = scaler.inverse_transform(test_preds)
    actual_train = scaler.inverse_transform(y_train.cpu().numpy())
    actual_test = scaler.inverse_transform(y_test.cpu().numpy())

    # Metrics
    train_mae = mean_absolute_error(actual_train, train_preds)
    test_mae = mean_absolute_error(actual_test, test_preds)
    train_rmse = np.sqrt(mean_squared_error(actual_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(actual_test, test_preds))
    train_mape = np.mean(np.abs((actual_train - train_preds) / actual_train)) * 100
    test_mape = np.mean(np.abs((actual_test - test_preds) / actual_test)) * 100

    log_metrics({
        "train_mae": train_mae, "test_mae": test_mae,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "train_mape": train_mape, "test_mape": test_mape
    })

    return torch.tensor(train_preds), torch.tensor(test_preds), torch.tensor(np.concatenate([actual_train, actual_test]))
