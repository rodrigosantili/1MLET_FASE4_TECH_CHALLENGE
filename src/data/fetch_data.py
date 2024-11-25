import yfinance as yf
import pandas as pd


def fetch_data(ticker, period):
    """
    Faz a coleta dos dados do ativo financeiro usando o yfinance, com tratamento de erros.

    Parâmetros:
        ticker (str): Símbolo do ativo financeiro.
        period (str): Período de coleta (ex: '1mo', '1y', '10y').

    Retorna:
        pd.DataFrame: DataFrame com os dados históricos contendo a coluna 'Close'.
    """
    try:
        data = yf.download(tickers=ticker, period=period)

        # Garantir que o dataframe não esteja vazio
        if data.empty:
            raise ValueError(f"Nenhum dado foi baixado para o ticker '{ticker}' no período '{period}'.")

    except Exception as e:
        raise ValueError(f"Erro ao coletar dados: {e}")

    return data
