import logging

import numpy as np
from fastapi import HTTPException
from scipy.interpolate import interp1d

from domain.responses import GetStockPricePredictionResponse
from ml import MLLoader
from routers.interfaces import PredictionService

logging.basicConfig(level=logging.INFO)


class PredictionServiceImpl(PredictionService):
    def __init__(self):
        ml_loader = MLLoader()
        self.__model = ml_loader.model
        self.__scaler = ml_loader.scaler

    async def get_stock_price_prediction(self, historical_data: [float]) -> GetStockPricePredictionResponse:
        try:
            # Extrapolate to create a sequence of the required length (60)
            x = np.arange(len(historical_data))
            f = interp1d(x, historical_data, fill_value="extrapolate")
            extrapolated_prices = f(np.linspace(0, len(historical_data) - 1, 60))

            # Scale the extrapolated data using the scaler (assume the scaler is preloaded)
            scaled_prices = self.__scaler.transform(np.array(extrapolated_prices).reshape(-1, 1))

            # Reshape the data to the format expected by the model (1 sample, 60 sequence length, 1 feature)
            X_input = np.reshape(scaled_prices, (1, 60, 1))

            # Make the prediction using the loaded model
            predicted_price_scaled = self.__model.predict(X_input)

            # Inverse transform the prediction to get the original scale
            predicted_price = self.__scaler.inverse_transform(predicted_price_scaled)

            # Return the predicted price as a JSON response
            return GetStockPricePredictionResponse(predicted_price=predicted_price[0][0])

        except Exception as ex:
            logging.error(f"Error: {ex}")
            raise HTTPException(status_code=500, detail=str(ex))