import mlflow
import mlflow.pytorch
import numpy as np


def init_mlflow(experiment_name="Stock_Price_Prediction"):
    """
    Initializes an MLflow experiment with the specified name.

    Parameters:
        experiment_name (str): The name of the experiment to be set in MLflow.

    Returns:
        None
    """
    mlflow.set_experiment(experiment_name)


def log_params(params):
    """
    Logs parameters in MLflow.

    Parameters:
        params (dict): A dictionary of parameters to log, where keys are parameter names and values are parameter values.

    Returns:
        None
    """
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics):
    """
    Logs metrics in MLflow.

    Parameters:
        metrics (dict): A dictionary of metrics to log, where keys are metric names and values are metric values.

    Returns:
        None
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_pytorch_model(model, model_name="Stock_LSTM_Model", seq_length=30, num_features=4):
    """
    Saves a PyTorch model in MLflow, including an input example for signature.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_name (str): The name of the model for registration in MLflow.
        seq_length (int): The sequence length used in training.
        num_features (int): The number of features used in the model.

    Returns:
        None
    """
    # Create an input example based on the expected input size of the model
    input_example = np.random.rand(1, seq_length, num_features).astype(np.float32)  # Convert to np.ndarray

    # Log the model in MLflow
    mlflow.pytorch.log_model(model, model_name, input_example=input_example)