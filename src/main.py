import torch
from torch.utils.data import DataLoader, TensorDataset
from data.fetch_data import fetch_data
from data.preprocess_data import preprocess_data
from data.feature_engineering import add_technical_indicators
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
from mlflow_setup import init_mlflow, log_params, log_pytorch_model


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
        "num_layers": 2,
        "dropout": 0.3,

        # Parâmetros de treinamento
        "seq_length": 60,
        "epochs": 200,
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
        "step_size": 50,
        "gamma": 0.85,

        # Parâmetros para CosineAnnealingLR
        "t_max": 50,

        # Parâmetros de early stopping
        "early_stopping_patience": 30,

        # Parâmetros de predição futura
        "future_days": 7
    }

    log_params(params)
    device = get_device(params["framework"])

    # Fetch data
    data = fetch_data(ticker=params["yfinance_ticker"], period=params["yfinance_period"])

    # Aplicar engenharia de features
    data = add_technical_indicators(data)

    # Remover linhas com valores NaN causados pelos cálculos de indicadores técnicos
    data = data.dropna()

    # Verificar se não há valores NaN restantes
    if data.isnull().any().any():
        raise ValueError("Os dados ainda contêm valores NaN após a remoção.")

    # Split data: 80% training, 20% validation
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Normalize data
    feature_columns = ['Close', 'sma_20', 'rsi', 'macd']
    scaled_train_data, scaler = preprocess_data(train_data, feature_columns=feature_columns)
    scaled_test_data = scaler.transform(test_data[feature_columns])

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

    save_model_local(trained_model, path="src/models/saved/trained_model.pth")

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


if __name__ == "__main__":
    main()
