import pandas as pd


def add_technical_indicators(df, feature_columns=None, sma_period=20, rsi_period=14, macd_span_short=12, macd_span_long=26):
    """
    Adiciona indicadores técnicos ao DataFrame com cálculo dinâmico dos parâmetros e preserva apenas as colunas desejadas.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados históricos (deve conter a coluna 'Close').
        feature_columns (list): Lista de colunas a serem mantidas do DataFrame original.
        sma_period (int): Período para o cálculo da Média Móvel Simples (SMA).
        rsi_period (int): Período para o cálculo do Índice de Força Relativa (RSI).
        macd_span_short (int): Período curto para o cálculo da Média Móvel Exponencial (EMA) do MACD.
        macd_span_long (int): Período longo para o cálculo da EMA do MACD.

    Retorna:
        pd.DataFrame: DataFrame com os indicadores técnicos adicionados.
    """
    if 'Close' not in df.columns:
        raise KeyError("A coluna 'Close' não foi encontrada no DataFrame.")

    df['Close'] = df['Close'].astype(float)

    # Garantir que 'Close' seja unidimensional
    df['Close'] = pd.Series(df['Close'].values.flatten(), index=df.index)

    # SMA (Média Móvel Simples)
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

    # Manter apenas as colunas especificadas e os novos indicadores
    if feature_columns is not None:
        columns_to_keep = feature_columns + [f'sma_{sma_period}', f'rsi_{rsi_period}', 'macd']
        df = df[columns_to_keep]

    new_features = df.columns

    return df, new_features
