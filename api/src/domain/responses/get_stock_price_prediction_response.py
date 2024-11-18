from pydantic import BaseModel


class GetStockPricePredictionResponse(BaseModel):
    predicted_price: float
