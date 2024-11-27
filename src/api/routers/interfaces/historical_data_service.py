from abc import ABC, abstractmethod

from pandas import DataFrame


class HistoricalDataService(ABC):
    @abstractmethod
    async def fetch_historical_data(self, ticker: str = "BTC-USD") -> DataFrame:
        raise NotImplementedError
