import numpy as np
import matplotlib.pyplot as plt


def plot_all(actual, train_preds, test_preds, future_preds, seq_length, future_days):
    """
    Plota os valores reais, as previsões de treino, teste e as previsões futuras.

    Parâmetros:
    - actual (np.array): Série de preços reais.
    - train_preds (np.array): Previsões do conjunto de treino.
    - test_preds (np.array): Previsões do conjunto de teste.
    - future_preds (np.array): Previsões para os dias futuros.
    - seq_length (int): Comprimento da sequência usada no modelo.
    - future_days (int): Número de dias para previsões futuras.
    """
    # Configurar o tamanho do gráfico
    plt.figure(figsize=(14, 7))

    # Plotar valores reais (ajustado para começar no índice `seq_length`)
    plt.plot(np.arange(seq_length, len(actual)), actual[seq_length:], label="Actual Prices", color="blue")

    # Plotar previsões de treino
    plt.plot(np.arange(seq_length, seq_length + len(train_preds)), train_preds, label="Training Predictions", color="orange")

    # Plotar previsões de teste
    test_start_idx = seq_length + len(train_preds)
    plt.plot(np.arange(test_start_idx, test_start_idx + len(test_preds)), test_preds, label="Testing Predictions", color="green")

    # Plotar previsões futuras, logo após o índice final dos dados de teste
    future_start_idx = test_start_idx + len(test_preds)
    plt.plot(np.arange(future_start_idx, future_start_idx + future_days), future_preds, label="Future Predictions", color="red", linestyle="--")

    # Adicionar título e legendas
    plt.title("Stock Price Predictions")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()



def plot_future_predictions(scaled_data, future_preds, scaler, future_days):
    """
    Plota os preços históricos e as previsões futuras.

    Parâmetros:
    - scaled_data (np.array): Dados escalados dos preços históricos.
    - future_preds (np.array): Previsões para os dias futuros.
    - scaler: Objeto de escalador usado para inversão da escala.
    - future_days (int): Número de dias para previsões futuras.
    """
    plt.figure(figsize=(12, 6))

    # Plotar preços históricos (transformados de volta à escala original)
    plt.plot(scaler.inverse_transform(scaled_data), label="Historical Prices", color="blue")

    # Plotar previsões futuras
    plt.plot(range(len(scaled_data), len(scaled_data) + future_days), future_preds, label="Future Predictions", color="red")

    plt.title("Future Stock Price Predictions")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_results(actual, train_preds, test_preds):
    """
    Plota os valores reais junto com as previsões de treino e teste.

    Parâmetros:
    - actual (np.array): Série de preços reais.
    - train_preds (np.array): Previsões do conjunto de treino.
    - test_preds (np.array): Previsões do conjunto de teste.
    """
    plt.figure(figsize=(12, 6))

    # Plotar preços reais
    plt.plot(actual, label="Actual Prices", color="blue")

    # Plotar previsões de treino
    plt.plot(np.arange(len(train_preds)), train_preds, label="Training Predictions", color="orange")

    # Plotar previsões de teste
    plt.plot(np.arange(len(train_preds), len(train_preds) + len(test_preds)), test_preds, label="Testing Predictions", color="green")

    plt.title("Training and Testing Predictions")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
