from pydantic import BaseModel


class GetStockPricePredictionRequest(BaseModel):
    historical_data: list[float]
