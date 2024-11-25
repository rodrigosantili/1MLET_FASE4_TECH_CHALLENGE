import mlflow
import mlflow.pytorch
import numpy as np
import torch


def init_mlflow(experiment_name="Stock_Price_Prediction"):
    """Inicializa o experimento no MLflow com o nome especificado."""
    mlflow.set_experiment(experiment_name)


def log_params(params):
    """Registra parâmetros no MLflow."""
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics):
    """Registra métricas no MLflow."""
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_pytorch_model(model, model_name="Stock_LSTM_Model", seq_length=30, num_features=4):
    """
    Salva um modelo PyTorch no MLflow, incluindo um input_example para assinatura.

    Parâmetros:
        model: O modelo PyTorch a ser salvo.
        model_name: Nome do modelo para registro no MLflow.
        seq_length: Comprimento da sequência usada no treinamento.
        num_features: Número de features usadas no modelo.
    """
    # Criando um exemplo de entrada baseado no tamanho de entrada esperado pelo modelo
    input_example = np.random.rand(1, seq_length, num_features).astype(np.float32)  # Convertendo para np.ndarray

    # Log do modelo no MLflow
    mlflow.pytorch.log_model(model, model_name, input_example=input_example)
