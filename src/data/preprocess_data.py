from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(data):
    """
    Normalize data using MinMaxScaler and ensure data consistency.

    Parameters:
        data (array-like): Input data to normalize.

    Returns:
        scaled_data (np.array): Normalized data.
        scaler (MinMaxScaler): Fitted scaler for inverse transformations.
    """
    # Garantir que os dados não contenham valores NaN ou infinitos
    assert not np.any(np.isnan(data)), "Os dados contêm valores NaN."
    assert not np.any(np.isinf(data)), "Os dados contêm valores infinitos."

    # Inicializar e ajustar o escalonador
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler
