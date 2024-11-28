from abc import ABC, abstractmethod

from pandas import DataFrame

from ...domain.responses import GetPredictionResponse


class PredictionService(ABC):
    @abstractmethod
    async def get_prediction(self, historical_data: DataFrame, days_forward: int = 1) -> GetPredictionResponse:
        raise NotImplementedError
