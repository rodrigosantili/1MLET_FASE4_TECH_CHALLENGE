from abc import ABC, abstractmethod

from domain.responses import GetStockPricePredictionResponse


class PredictionService(ABC):
    @abstractmethod
    async def get_stock_price_prediction(self, historical_data: [float]) -> GetStockPricePredictionResponse:
        raise NotImplementedError
