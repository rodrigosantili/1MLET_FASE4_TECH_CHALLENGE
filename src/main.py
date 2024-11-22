# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from data import fetch_data, preprocess_data
# from models.model_pytorch import StockLSTM
# from models.train_pytorch import train_model
# from models.save_model import save_model_local
# from predict.predict_pytorch import future_predictions
# from predict.evaluate_model import evaluate_model
#
# from utils.plot_utils import (
#     plot_all, plot_residuals, plot_residual_distribution,
#     plot_train_test_predictions, plot_confidence_interval,
#     plot_autocorrelation, plot_historical_and_future
# )
# from utils.sequence_utils import create_sequences
# from utils.tensor_utils import prepare_tensors_pytorch
# from utils.device_utils import get_device
# from mlflow_setup import init_mlflow, log_params
#
#
# def main():
#     # Initialize MLflow tracking
#     init_mlflow()
#
#     # Set model parameters and hyperparameters
#     params = {
#         # Parâmetros de coleta de dados
#         "yfinance_ticker": "NVDA",       # Nome do ativo para coletar os dados
#         "yfinance_period": "10y",           # Período disponível para coleta de dados (e.g., '1d', '1mo', '10y', 'max')
#
#         # Parâmetros do modelo
#         "framework": "pytorch",             # Framework utilizado (e.g., 'pytorch')
#         "hidden_layer_size": 420,           # Tamanho da camada oculta do LSTM
#         "num_layers": 2,                    # Número de camadas LSTM empilhadas
#         "dropout": 0.2,                     # Dropout para regularização
#
#         # Parâmetros de treinamento
#         "seq_length": 60,                   # Comprimento da sequência de entrada
#         "epochs": 300,                      # Número de épocas de treinamento
#         "learning_rate": 0.0002,             # Taxa de aprendizado
#         "weight_decay": 0,                  # Regularização L2 (Adam)
#         "batch_size": 32,                   # Tamanho do lote
#         "tolerance": 0.02,                  # Tolerância para calcular acurácia baseada em erro relativo
#
#         # Parâmetros do scheduler de aprendizado
#         # "step_size": 50,                    # Intervalo de épocas para reduzir a taxa de aprendizado
#         # "gamma": 0.8,                       # Fator de redução da taxa de aprendizado
#         "t_max": 150,
#         # Parâmetros de predição futura
#         "future_days": 7                    # Número de dias para prever no futuro
#     }
#
#     # Log parameters to MLflow for tracking experiment configurations
#     log_params(params)
#
#     # Determine and set device based on the framework (CPU or GPU)
#     device = get_device(params["framework"])
#
#     # Fetch and preprocess data
#     data = fetch_data(ticker=params["yfinance_ticker"], period=params["yfinance_period"])
#     scaled_data, scaler = preprocess_data(data)
#
#     # Model framework-specific setup
#     if params["framework"] == "pytorch":
#         # Create input sequences and targets for training
#         sequences, targets = create_sequences(scaled_data, params["seq_length"])
#         train_size = int(0.8 * len(sequences))  # 80% training, 20% validation
#
#         # # Prepare DataLoaders for training and testing
#         X_train, y_train = sequences[:train_size], targets[:train_size]
#         X_val, y_val = sequences[train_size:], targets[train_size:]
#
#         X_train, y_train = prepare_tensors_pytorch(X_train, y_train, device)
#         X_val, y_val = prepare_tensors_pytorch(X_val, y_val, device)
#         train_dataset = TensorDataset(X_train, y_train)
#         val_dataset = TensorDataset(X_val, y_val)
#
#         train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)
#
#         # Initialize the PyTorch LSTM model
#         model = StockLSTM(
#             input_size=1,
#             hidden_layer_size=params["hidden_layer_size"],
#             output_size=1,
#             num_layers=params["num_layers"],
#             dropout=params["dropout"]
#         ).to(device)
#
#         # Train the model
#         trained_model = train_model(
#             model,
#             train_loader,
#             val_loader,
#             epochs=params["epochs"],
#             lr=params["learning_rate"],
#             weight_decay=params["weight_decay"],
#             tolerance=params["tolerance"],
#             step_size=params["step_size"],
#             gamma=params["gamma"],
#             t_max=params["t_max"]
#         )
#
#         # Save the trained model locally
#         save_model_local(trained_model, path=r'src\models\saved\trained_model.pth')
#
#         # Avaliação e predições
#         train_preds, val_preds, actual = evaluate_model(trained_model, X_train, y_train, X_val, y_val, scaler)
#
#         # Converter predições para tensores antes de concatenar
#         train_preds_tensor = torch.tensor(train_preds, dtype=torch.float32)
#         val_preds_tensor = torch.tensor(val_preds, dtype=torch.float32)
#         actual_tensor = torch.tensor(actual, dtype=torch.float32)
#
#         # Calcular os resíduos como tensores
#         residuals = actual_tensor - torch.cat((train_preds_tensor, val_preds_tensor))
#
#         # Plotagem dos gráficos
#         plot_train_test_predictions(actual, train_preds, val_preds)
#         plot_residuals(actual_tensor.numpy(), torch.cat((train_preds_tensor, val_preds_tensor)).numpy())
#         plot_residual_distribution(residuals.numpy())
#         plot_confidence_interval(actual_tensor.numpy(), torch.cat((train_preds_tensor, val_preds_tensor)).numpy(), residuals.numpy())
#         plot_autocorrelation(residuals.numpy())
#
#         # Prepare the last sequence for future predictions
#         last_sequence = torch.tensor(scaled_data[-params["seq_length"]:], dtype=torch.float32).unsqueeze(0).unsqueeze(
#             -1).to(device)
#         future_preds = future_predictions(trained_model, last_sequence, params["future_days"], scaler)  # Generate future predictions
#
#         # # Plot historical and future predictions
#         # plot_historical_and_future(actual, future_preds)  # Historical data and future predictions
#         #
#         # # Plot all predictions for complete overview
#         # plot_all(actual, train_preds, val_preds, future_preds,
#         #          seq_length=params["seq_length"], future_days=params["future_days"])
#
#
# # Execute main function if script is run directly
# if __name__ == '__main__':
#     main()

import torch
from torch.utils.data import DataLoader, TensorDataset
from data import fetch_data, preprocess_data
from models.model_pytorch import StockLSTM
from models.train_pytorch import train_model
from utils.save_utils import save_model_local, save_scaler_torch
from predict.predict_pytorch import future_predictions
from predict.evaluate_model import evaluate_model

from utils.plot_utils import (
    plot_residual_distribution,
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
        "yfinance_ticker": "NVDA",
        "yfinance_period": "10y",

        # Parâmetros do modelo
        "framework": "pytorch",
        "hidden_layer_size": 100,
        "num_layers": 2,
        "dropout": 0.3,

        # Parâmetros de treinamento
        "seq_length": 90,
        "epochs": 300,
        "learning_rate": 0.0001,
        "weight_decay": 0.001,
        "batch_size": 32,
        "tolerance": 0.02,

        # Parâmetros do scheduler de aprendizado
        # "step_size": 50,
        # "gamma": 0.8,
        "t_max": 100,

        # Parâmetros de predição futura
        "future_days": 7
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
        tolerance=params["tolerance"],
        t_max=params["t_max"]
    )

    save_model_local(trained_model, path="src/models/saved/trained_model.pth")

    # Evaluate model
    train_preds, val_preds, actual = evaluate_model(trained_model, X_train, y_train, X_test, y_test, scaler)

    # Plot residuals, predictions, and other analyses
    plot_train_test_predictions(actual, train_preds, val_preds)
    residuals = actual - torch.cat((train_preds, val_preds))
    plot_residual_distribution(residuals.numpy())
    plot_confidence_interval(actual, torch.cat((train_preds, val_preds)).numpy(), residuals.numpy())
    plot_autocorrelation(residuals.numpy())

    # Future predictions
    last_sequence = torch.tensor(scaled_test_data[-params["seq_length"]:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    future_preds = future_predictions(trained_model, last_sequence, params["future_days"], scaler)

    plot_historical_and_future(actual.numpy(), future_preds)


if __name__ == "__main__":
    main()
