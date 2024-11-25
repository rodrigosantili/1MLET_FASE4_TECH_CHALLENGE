import pandas as pd


def add_technical_indicators(df):
    """
    Adiciona indicadores técnicos ao DataFrame sem usar bibliotecas externas.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados históricos (deve conter a coluna 'Close').

    Retorna:
        pd.DataFrame: DataFrame com os indicadores técnicos adicionados.
    """
    if 'Close' not in df.columns:
        raise KeyError("A coluna 'Close' não foi encontrada no DataFrame.")

    # Garantir que 'Close' seja unidimensional
    df['Close'] = pd.Series(df['Close'].values.flatten(), index=df.index)

    # SMA (Média Móvel Simples)
    df['sma_20'] = df['Close'].rolling(window=20).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff(1)
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26

    return df
