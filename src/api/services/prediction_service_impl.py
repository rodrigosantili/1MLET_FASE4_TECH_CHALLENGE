import logging
import torch

from fastapi import HTTPException
from pandas import DataFrame

from ...lib.data import add_technical_indicators
from ...lib.ml.pytorch_predict import future_predictions
from ..domain.responses import GetPredictionResponse
from ..ml import MLLoader
from ..routers.interfaces import PredictionService

logging.basicConfig(level=logging.INFO)


class PredictionServiceImpl(PredictionService):
    def __init__(self):
        ml_loader = MLLoader()
        self.__model = ml_loader.model
        self.__scaler = ml_loader.scaler
        self.__device = ml_loader.device

    async def get_prediction(self, historical_data: DataFrame, days_forward: int = 1) -> GetPredictionResponse:
        """
        Predicts the next closing price based on historical data.
        :param historical_data: The historical data to use for prediction.
        :param days_forward: The number of days forward to predict (default: 1).
        :return: The predicted price as a JSON response.
        """
        logging.info("Predicting the next closing price based on historical data.")
        try:
            # Transform data
            df = DataFrame(historical_data)
            df, feature_columns = add_technical_indicators(df, feature_columns=['Close'])
            scaled_data = self.__scaler.transform(df[feature_columns])

            # Convert to PyTorch tensor and move to the appropriate device
            X_tensor = torch.tensor(scaled_data[-60:], dtype=torch.float32).unsqueeze(0).to(self.__device)

            prediction = future_predictions(self.__model, X_tensor, days_forward, self.__scaler)

            # Return the predicted price as a JSON response
            return GetPredictionResponse(predicted_price=prediction)

        except Exception as ex:
            logging.error(f"Error: {ex}")
            raise HTTPException(status_code=500, detail=str(ex))
