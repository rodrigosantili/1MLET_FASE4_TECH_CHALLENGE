import mlflow
import mlflow.pytorch
import mlflow.keras


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
    """Salva um modelo PyTorch no MLflow."""
    mlflow.pytorch.log_model(model, model_name)


def log_keras_model(model, model_name="Stock_LSTM_Keras_Model"):
    """Salva um modelo Keras no MLflow."""
    mlflow.keras.log_model(model, model_name)
