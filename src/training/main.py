import torch
import time
import psutil

from torch.utils.data import DataLoader, TensorDataset
from data.feature_engineering import add_technical_indicators
from data.fetch_data import fetch_yfinance_data
from data.preprocess_data import preprocess_data
from ml.model_pytorch import StockLSTM
from ml.train_pytorch import train_model
from utils.device_utils import get_device
from utils.save_utils import save_model_local, save_scaler_torch
from ml.predict_utils import future_predictions
from ml.evaluate_model import evaluate_model
from utils.plot_utils import (
    plot_residual_distribution, plot_residuals, plot_all,
    plot_train_test_predictions, plot_confidence_interval,
    plot_autocorrelation, plot_historical_and_future
)
from utils.sequence_utils import create_sequences
from utils.tensor_utils import prepare_tensors_pytorch
from mlflow_setup import init_mlflow, log_params, log_pytorch_model, log_metrics


def monitor_performance(model, X_test, y_test, scaler):
    """
    Monitora o desempenho do modelo em termos de tempo de resposta e utilização de recursos.
    """
    model.eval()
    inference_times = []
    cpu_usages = []
    memory_usages = []

    with torch.no_grad():
        for i in range(len(X_test)):
            start_time = time.time()

            # Inferência do modelo
            _ = model(X_test[i:i+1])

            # Medir o tempo de inferência
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Capturar métricas de recursos
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)

    # Calcular métricas médias
    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    # Logar métricas no MLflow
    log_metrics({
        "avg_inference_time": avg_inference_time,
        "avg_cpu_usage": avg_cpu_usage,
        "avg_memory_usage": avg_memory_usage
    })

    print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f}%")
    print(f"Average Memory Usage: {avg_memory_usage:.2f}%")


def main():
    # Initialize MLflow tracking
    init_mlflow()

    # Set model parameters and hyperparameters
    params = {
        # Parâmetros de coleta de dados
        "yfinance_ticker": "BTC-USD",  # Nome do ativo para coletar os dados
        "yfinance_period": "10y",
        # Período disponível para coleta de dados

        # Parâmetros do modelo
        "framework": "pytorch",
        "hidden_layer_size": 350,
        "num_layers": 3,
        "dropout": 0.35,

        # Parâmetros de treinamento
        "seq_length": 60,
        "epochs": 200,
        #"epochs": 1,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "batch_size": 32,

        # Parâmetros do scheduler de aprendizado
        "scheduler_type": "step",

        # Parâmetros para ReduceLROnPlateau
        "patience": 10,
        "factor": 0.3,

        # Parâmetros para CyclicLR
        "cyclic_base_lr": 1e-5,
        "cyclic_max_lr": 0.0005,
        "step_size_up": 10,

        # Parâmetros para StepLR
        "step_size": 40,
        "gamma": 0.75,

        # Parâmetros para CosineAnnealingLR
        "t_max": 50,

        # Parâmetros de early stopping
        "early_stopping_patience": 20,

        # Parâmetros de predição futura
        "future_days": 7
    }

    log_params(params)
    device = get_device(params["framework"])

    # Fetch data
    data = fetch_yfinance_data(ticker=params["yfinance_ticker"], period=params["yfinance_period"])

    # Aplicar engenharia de features
    feature_columns = ['Close']
    data, feature_columns = add_technical_indicators(data, feature_columns=feature_columns)

    # Split data: 80% training, 20% validation
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Normalize data
    scaled_train_data, scaler = preprocess_data(train_data, feature_columns=feature_columns)
    scaled_test_data = scaler.transform(test_data[feature_columns])

    # Salvar o scaler após o preprocessamento
    save_scaler_torch(scaler, "../../res/models/saved/scaler.pt")

    # Prepare sequences
    train_sequences, train_targets = create_sequences(scaled_train_data, params["seq_length"])
    test_sequences, test_targets = create_sequences(scaled_test_data, params["seq_length"])

    # Convert to tensors
    X_train, y_train = prepare_tensors_pytorch(train_sequences, train_targets, device)
    X_test, y_test = prepare_tensors_pytorch(test_sequences, test_targets, device)

    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    # Initialize model
    num_features = X_train.shape[2]  # Número de features no tensor de entrada
    model = StockLSTM(
        input_size=num_features,
        hidden_layer_size=params["hidden_layer_size"],
        output_size=1,
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    ).to(device)

    # Train the model
    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        epochs=params["epochs"],
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
        scheduler_type=params["scheduler_type"],

        # Parâmetros específicos para cada scheduler
        step_size=params["step_size"],
        gamma=params["gamma"],
        t_max=params["t_max"],
        cyclic_base_lr=params["cyclic_base_lr"],
        cyclic_max_lr=params["cyclic_max_lr"],
        step_size_up=params["step_size_up"],
        patience=params["patience"],
        factor=params["factor"],

        # Parâmetros para early stopping
        early_stopping_patience=params["early_stopping_patience"]
    )

    save_model_local(trained_model, path="../../res/models/saved/trained_model.pth")

    # Logar o modelo no MLflow
    log_pytorch_model(
        trained_model,
        model_name="Stock_LSTM_Model",
        seq_length=params["seq_length"],
        num_features=num_features
    )

    # Evaluate model
    train_preds, val_preds, actual = evaluate_model(trained_model, X_train, y_train, X_test, y_test, scaler)

    # Plot residuals, predictions, and other analyses
    plot_train_test_predictions(actual, train_preds, val_preds)
    residuals = actual - torch.cat((train_preds, val_preds))
    plot_residuals(actual.numpy(), torch.cat((train_preds, val_preds)).numpy())
    plot_residual_distribution(residuals.numpy())
    plot_confidence_interval(actual.numpy(), torch.cat((train_preds, val_preds)).numpy(), residuals.numpy())
    plot_autocorrelation(residuals.numpy())

    # Future predictions
    last_sequence = torch.tensor(scaled_test_data[-params["seq_length"]:], dtype=torch.float32).to(device)
    future_preds = future_predictions(trained_model, last_sequence, params["future_days"], scaler)

    plot_historical_and_future(actual.numpy(), future_preds)

    # Plot all predictions for complete overview
    plot_all(actual, train_preds, val_preds, future_preds,
             seq_length=params["seq_length"], future_days=params["future_days"])

    # Call monitor_performance
    monitor_performance(model, X_test, y_test, scaler)

    
if __name__ == "__main__":
    main()
