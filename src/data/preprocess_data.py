from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def preprocess_data(data, feature_columns):
    """
    Normalize data with multiple features using MinMaxScaler and ensure data consistency.

    Parameters:
        data (pd.DataFrame): Input data to normalize. Should be a DataFrame with multiple features.
        feature_columns (list): List of columns to normalize.

    Returns:
        scaled_data (np.array): Normalized data (NumPy array with shape [n_samples, n_features]).
        scaler (MinMaxScaler): Fitted scaler for inverse transformations.
    """
    # Selecionar apenas as colunas relevantes (features)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("O parâmetro 'data' deve ser um DataFrame do pandas.")
    if not all(col in data.columns for col in feature_columns):
        raise ValueError("Algumas colunas especificadas em 'feature_columns' não estão presentes no DataFrame.")

    data = data[feature_columns].copy()

    # Garantir que os dados não contenham valores NaN ou infinitos
    if data.isnull().any().any():
        raise ValueError(f"Os dados contêm valores NaN. Índices problemáticos:\n{data.isnull().sum()}")
    if np.isinf(data.values).any():
        raise ValueError("Os dados contêm valores infinitos.")

    # Inicializar e ajustar o escalonador
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler

