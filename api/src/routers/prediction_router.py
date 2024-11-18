import logging
from fastapi import APIRouter, Depends

from domain.requests import GetStockPricePredictionRequest
from domain.responses import GetStockPricePredictionResponse
from routers.di import injector
from routers.interfaces import PredictionService

logging.basicConfig(level=logging.INFO)


def get_prediction_service() -> PredictionService:
    return injector.get(PredictionService)


prediction_router = APIRouter()


@prediction_router.post("/predict/",
                        response_model=GetStockPricePredictionResponse,
                        summary="Predict stock price",
                        description="Predicts stock price based on historical data.",
                        tags=["Prediction"])
async def predict(
        data: GetStockPricePredictionRequest,
        prediction_service: PredictionService = Depends(get_prediction_service)
) -> GetStockPricePredictionResponse:
    """
    Predicts the next stock price based on historical data.

    - **historical_data**: List of historical stock prices.
    """
    return await prediction_service.get_stock_price_prediction(data.historical_data)

