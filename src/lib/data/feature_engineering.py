import pandas as pd


def add_technical_indicators(df, feature_columns=None, sma_period=20, rsi_period=14,
                             macd_span_short=12, macd_span_long=26):
    """
    Adds technical indicators to the DataFrame with dynamic parameter calculation
    and preserves only the desired columns.

    Parameters:
        df (pd.DataFrame): DataFrame with historical data (must contain the 'Close' column).
        feature_columns (list): List of columns to be retained from the original DataFrame.
        sma_period (int): Period for calculating the Simple Moving Average (SMA).
        rsi_period (int): Period for calculating the Relative Strength Index (RSI).
        macd_span_short (int): Short period for calculating the Exponential Moving Average (EMA) of the MACD.
        macd_span_long (int): Long period for calculating the EMA of the MACD.

    Returns:
        pd.DataFrame: DataFrame with the added technical indicators.
    """
    if 'Close' not in df.columns:
        raise KeyError("The 'Close' column was not found in the DataFrame.")

    df['Close'] = df['Close'].astype(float)

    # Ensure 'Close' is one-dimensional
    df['Close'] = pd.Series(df['Close'].values.flatten(), index=df.index)

    # SMA (Simple Moving Average)
    df[f'sma_{sma_period}'] = df['Close'].rolling(window=sma_period).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff(1)
    gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(window=rsi_period).mean()
    rs = gain / loss
    df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_short = df['Close'].ewm(span=macd_span_short, adjust=False).mean()
    ema_long = df['Close'].ewm(span=macd_span_long, adjust=False).mean()
    df['macd'] = ema_short - ema_long

    df = df.dropna()

    # Retain only the specified columns and the new indicators
    if feature_columns is not None:
        columns_to_keep = feature_columns + [f'sma_{sma_period}', f'rsi_{rsi_period}', 'macd']
        df = df[columns_to_keep]

    new_features = df.columns

    return df, new_features
