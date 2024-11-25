import yfinance as yf
import pandas as pd


def fetch_data(ticker, period):
    """
    Faz a coleta dos dados do ativo financeiro usando o yfinance.

    Parâmetros:
        ticker (str): Símbolo do ativo financeiro.
        period (str): Período de coleta (ex: '1mo', '1y', '10y').

    Retorna:
        pd.DataFrame: DataFrame com os dados históricos contendo a coluna 'Close'.
    """
    data = yf.download(tickers=ticker, period=period)

    # Garantir que 'Close' seja uma série unidimensional com tipo numérico
    if 'Close' in data.columns:
        data['Close'] = data['Close'].astype(float)
    else:
        raise KeyError("A coluna 'Close' não está presente nos dados baixados.")

    return data
