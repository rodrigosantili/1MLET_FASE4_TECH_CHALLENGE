from pydantic import BaseModel


class GetPredictionRequest(BaseModel):
    days_forward: int
