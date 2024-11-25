from sklearn.preprocessing import MinMaxScaler
import numpy as np


def preprocess_data(data):
    """
    Normalize data using MinMaxScaler and ensure data consistency.

    Parameters:
        data (array-like): Input data to normalize. Should be a 1D or 2D array.

    Returns:
        scaled_data (np.array): Normalized data.
        scaler (MinMaxScaler): Fitted scaler for inverse transformations.
    """
    # Garantir que os dados sejam um array numpy e fazer uma cópia para evitar alterações externas
    data = np.array(data, copy=True)

    # Verificar dimensões e ajustar para 2D, se necessário
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Garantir que os dados não contenham valores NaN ou infinitos
    if np.any(np.isnan(data)):
        raise ValueError(f"Os dados contêm valores NaN. Índices problemáticos: {np.where(np.isnan(data))}")
    if np.any(np.isinf(data)):
        raise ValueError(f"Os dados contêm valores infinitos. Índices problemáticos: {np.where(np.isinf(data))}")

    # Inicializar e ajustar o escalonador
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler
