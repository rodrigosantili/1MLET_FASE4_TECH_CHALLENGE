from pydantic import BaseModel


class GetPredictionResponse(BaseModel):
    predicted_price: list[float]
