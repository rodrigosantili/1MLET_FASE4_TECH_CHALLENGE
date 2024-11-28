import logging
from fastapi import APIRouter, Depends

from ..domain.requests import GetPredictionRequest
from ..domain.responses import GetPredictionResponse
from ..routers.di import injector
from ..routers.interfaces import PredictionService, HistoricalDataService

logging.basicConfig(level=logging.INFO)


def get_historical_data_service() -> HistoricalDataService:
    return injector.get(HistoricalDataService)


def get_prediction_service() -> PredictionService:
    return injector.get(PredictionService)


prediction_router = APIRouter()


@prediction_router.post("/predict/",
                        response_model=GetPredictionResponse,
                        summary="Predict BTC price in USD",
                        description="Predicts BTC price in USD based on historical data.",
                        tags=["Prediction", "Finance", "BTC"])
async def predict(
        request: GetPredictionRequest,
        historical_data_service: HistoricalDataService = Depends(get_historical_data_service),
        prediction_service: PredictionService = Depends(get_prediction_service)
) -> GetPredictionResponse:
    fetch_historical_data_response = await historical_data_service.fetch_historical_data(ticker="BTC-USD")
    return await prediction_service.get_prediction(fetch_historical_data_response, request.days_forward)

