import mlflow
import mlflow.pytorch
# import mlflow.keras
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


def log_pytorch_model(model, model_name="Stock_LSTM_Model"):
    """Salva um modelo PyTorch no MLflow, incluindo um input_example para assinatura."""
    # Criando um exemplo de entrada baseado no tamanho de entrada esperado pelo modelo
    input_example = torch.rand(1, 60, 1).numpy()
    mlflow.pytorch.log_model(model, model_name, input_example=input_example)


# def log_keras_model(model, model_name="Stock_LSTM_Keras_Model"):
#     """Salva um modelo Keras no MLflow."""
#     mlflow.keras.log_model(model, model_name)
