import torch
from torch.utils.data import DataLoader, TensorDataset
from data import fetch_data, preprocess_data
from models.model_pytorch import StockLSTM
from models.train_pytorch import train_model
from utils.save_utils import save_model_local, save_scaler_torch
from predict.predict_pytorch import future_predictions
from predict.evaluate_model import evaluate_model

from utils.plot_utils import (
    plot_residual_distribution, plot_residuals, plot_all,
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
        # Parâmetros de coleta de dados
        "yfinance_ticker": "NVDA",       # Nome do ativo para coletar os dados
        "yfinance_period": "10y",           # Período disponível para coleta de dados  ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

        # Parâmetros do modelo
        "framework": "pytorch",             # Framework utilizado (e.g., 'pytorch')
        "hidden_layer_size": 250,            # Tamanho da camada oculta do LSTM
        "num_layers": 3,                    # Número de camadas LSTM empilhadas
        "dropout": 0.3,                     # Dropout para regularização

        # Parâmetros de treinamento
        "seq_length": 30,                   # Comprimento da sequência de entrada
        "epochs": 200,                       # Número de épocas de treinamento
        "learning_rate": 0.001,            # Taxa de aprendizado
        "weight_decay": 1e-4,             # Regularização L2 (Adam)
        "batch_size": 32,                   # Tamanho do lote

        # Parâmetros do scheduler de aprendizado
        "scheduler_type": "step",           # Tipo de scheduler ('reduce_on_plateau', 'cyclic', 'step', 'cosine_annealing')

        # Parâmetros para ReduceLROnPlateau
        "patience": 10,                     # Paciência para scheduler (ReduceLROnPlateau)
        "factor": 0.3,                      # Fator de redução da taxa de aprendizado (ReduceLROnPlateau)

        # Parâmetros para CyclicLR
        "cyclic_base_lr": 1e-5,             # Taxa de aprendizado mínima (CyclicLR)
        "cyclic_max_lr": 0.0005,            # Taxa de aprendizado máxima (CyclicLR)
        "step_size_up": 10,                 # Número de iterações para aumentar a LR até o valor máximo (CyclicLR)

        # Parâmetros para StepLR
        "step_size": 50,                    # Número de épocas após as quais a LR será reduzida (StepLR)
        "gamma": 0.7,                       # Fator de redução da taxa de aprendizado (StepLR)

        # Parâmetros para CosineAnnealingLR
        "t_max": 50,                        # Número de épocas para a taxa de aprendizado reduzir até zero (CosineAnnealingLR)

        # Parâmetros de early stopping
        "early_stopping_patience": 30,      # Paciência para early stopping

        # Parâmetros de predição futura
        "future_days": 7                   # Dias para prever
    }

    log_params(params)
    device = get_device(params["framework"])

    # Fetch data
    data = fetch_data(ticker=params["yfinance_ticker"], period=params["yfinance_period"])
    train_size = int(0.8 * len(data))  # Split data: 80% training, 20% validation

    # Normalize train data only to prevent data leakage
    train_data = data[:train_size]
    test_data = data[train_size:]
    scaled_train_data, scaler = preprocess_data(train_data)
    scaled_test_data = scaler.transform(test_data)

    # Salvar o scaler após o preprocessamento
    save_scaler_torch(scaler, "src/models/saved/scaler.pt")

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
    model = StockLSTM(
        input_size=1,
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
        step_size=params["step_size"],              # Para StepLR
        gamma=params["gamma"],                      # Para StepLR
        t_max=params["t_max"],                      # Para CosineAnnealingLR
        cyclic_base_lr=params["cyclic_base_lr"],    # Para CyclicLR
        cyclic_max_lr=params["cyclic_max_lr"],      # Para CyclicLR
        step_size_up=params["step_size_up"],        # Para CyclicLR
        patience=params["patience"],                # Para ReduceLROnPlateau
        factor=params["factor"],                    # Para ReduceLROnPlateau

        # Parâmetros para early stopping
        early_stopping_patience=params["early_stopping_patience"]
    )

    save_model_local(trained_model, path="src/models/saved/trained_model.pth")

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
    last_sequence = torch.tensor(scaled_test_data[-params["seq_length"]:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    future_preds = future_predictions(trained_model, last_sequence, params["future_days"], scaler)

    plot_historical_and_future(actual.numpy(), future_preds)

    # Plot all predictions for complete overview
    plot_all(actual, train_preds, val_preds, future_preds,
             seq_length=params["seq_length"], future_days=params["future_days"])


if __name__ == "__main__":
    main()
