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
    # Select only the relevant columns (features)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The 'data' parameter must be a pandas DataFrame.")
    if not all(col in data.columns for col in feature_columns):
        raise ValueError("Some columns specified in 'feature_columns' are not present in the DataFrame.")

    data = data[feature_columns].copy()

    # Ensure the data does not contain NaN or infinite values
    if data.isnull().any().any():
        raise ValueError(f"The data contains NaN values. Problematic indices:\n{data.isnull().sum()}")
    if np.isinf(data.values).any():
        raise ValueError("The data contains infinite values.")

    # Initialize and fit the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler
