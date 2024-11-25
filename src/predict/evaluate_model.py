import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow_setup import log_metrics


def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    """
    Avalia o modelo usando os conjuntos de treino e teste.

    Parâmetros:
        model: Modelo treinado.
        X_train: Tensor de entrada de treino.
        y_train: Tensor de saída real de treino.
        X_test: Tensor de entrada de teste.
        y_test: Tensor de saída real de teste.
        scaler: Objeto MinMaxScaler usado para normalizar os dados.

    Retorna:
        train_preds: Previsões para o conjunto de treino (desnormalizadas).
        test_preds: Previsões para o conjunto de teste (desnormalizadas).
        actual: Valores reais desnormalizados de treino e teste.
    """
    model.eval()  # Modo de avaliação
    with torch.no_grad():
        # Garantir que os tensores estejam no dispositivo correto
        X_train = X_train.to(next(model.parameters()).device)
        y_train = y_train.to(next(model.parameters()).device)
        X_test = X_test.to(next(model.parameters()).device)
        y_test = y_test.to(next(model.parameters()).device)

        # Previsões
        train_preds = model(X_train).cpu().numpy()
        test_preds = model(X_test).cpu().numpy()

    # Ajustar as previsões para o formato esperado pelo scaler
    num_features = scaler.min_.shape[0]  # Número total de features usadas no scaler
    train_preds_full = np.zeros((train_preds.shape[0], num_features))
    test_preds_full = np.zeros((test_preds.shape[0], num_features))

    train_preds_full[:, 0] = train_preds.flatten()  # Inserir previsões na coluna correspondente ao 'Close'
    test_preds_full[:, 0] = test_preds.flatten()

    # Inverter a normalização apenas para a coluna 'Close'
    train_preds = scaler.inverse_transform(train_preds_full)[:, 0]
    test_preds = scaler.inverse_transform(test_preds_full)[:, 0]

    # Inverter a normalização para os valores reais
    y_train_full = np.zeros((y_train.shape[0], num_features))
    y_test_full = np.zeros((y_test.shape[0], num_features))
    y_train_full[:, 0] = y_train.cpu().numpy().flatten()
    y_test_full[:, 0] = y_test.cpu().numpy().flatten()

    actual_train = scaler.inverse_transform(y_train_full)[:, 0]
    actual_test = scaler.inverse_transform(y_test_full)[:, 0]

    # Métricas
    train_mae = mean_absolute_error(actual_train, train_preds)
    test_mae = mean_absolute_error(actual_test, test_preds)
    train_rmse = np.sqrt(mean_squared_error(actual_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(actual_test, test_preds))
    train_mape = np.mean(np.abs((actual_train - train_preds) / actual_train)) * 100
    test_mape = np.mean(np.abs((actual_test - test_preds) / actual_test)) * 100

    # Registrar métricas no MLflow
    log_metrics({
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mape": train_mape,
        "test_mape": test_mape
    })

    # Retornar previsões e valores reais
    return torch.tensor(train_preds), torch.tensor(test_preds), torch.tensor(np.concatenate([actual_train, actual_test]))

